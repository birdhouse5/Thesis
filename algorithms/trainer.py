# trainer.py - Optimized with fixed-length trajectory batching

from collections import deque
from datetime import datetime
from typing import Dict, List
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

# Needed to clone environments for batched rollouts
from environments.env import MetaEnv


class PerformanceDiagnostic:
    def __init__(self):
        self.timings = {}
    
    def time_section(self, name):
        return TimingContext(self, name)
    
    def add_timing(self, name, duration):
        if name not in self.timings:
            self.timings[name] = []
        self.timings[name].append(duration)
    
    def report(self):
        print("\n=== PERFORMANCE BREAKDOWN ===")
        for name, times in self.timings.items():
            avg_time = sum(times) / len(times)
            total_time = sum(times)
            print(f"{name:30s}: {avg_time:.4f}s avg, {total_time:.2f}s total ({len(times)} calls)")


class TimingContext:
    def __init__(self, diagnostic, name):
        self.diagnostic = diagnostic
        self.name = name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        duration = time.time() - self.start_time
        self.diagnostic.add_timing(self.name, duration)


class PPOTrainer:
    """
    PPO Trainer for VariBAD Portfolio Optimization.
    OPTIMIZED: Fixed-length trajectory batching for maximum speedup.
    """

    def __init__(self, env, policy, vae, config):
        self.env = env
        self.policy = policy
        self.vae = vae
        self.config = config

        self.episode_count = 0
        self.total_steps = 0
        self.device = torch.device(config.device)

        self.sequential_mode = False

        # Vectorization knob
        self.num_envs = max(1, int(getattr(config, "num_envs", 1)))

        # Check if we can use fixed-length optimization
        self.use_fixed_length = (config.min_horizon == config.max_horizon)

        # Optimizer: include VAE params only if enabled/present
        param_groups = [
            {"params": policy.parameters(), "lr": config.policy_lr}
        ]
        if vae is not None and not getattr(config, "disable_vae", False):
            param_groups.append({"params": vae.parameters(), "lr": config.vae_lr})
        self.optimizer = Adam(param_groups)
        # Experience buffers
        self.experience_buffer = ExperienceBuffer(config.batch_size)  # for PPO
        self.vae_buffer = deque(maxlen=1000)  # recent trajectories for VAE

        # Rolling stats (store Python floats to avoid CUDA logging issues)
        self.policy_losses = deque(maxlen=100)
        self.vae_losses = deque(maxlen=100)
        self.episode_rewards = deque(maxlen=100)

        # Extra tracking (kept for potential analysis hooks)
        self.episode_details = []
        self.training_start_time = time.time()
        self.episode_start_time = None
        self.portfolio_metrics_history = []


    # ---------------------------------------------------------------------
    # Public training entry point
    # ---------------------------------------------------------------------

    def train_episode(self) -> Dict[str, float]:
        """
        Train for one episode and return metrics for MLflow logging.
        """
        diag = PerformanceDiagnostic()

        # === Collect trajectory ===
        with diag.time_section("collect_single_trajectory"):
            tr = self.collect_trajectory()
            self.vae_buffer.append(tr)
            self.experience_buffer.add_trajectory(tr)

        # Aggregate reward for episode
        episode_reward_sum = float(tr["rewards"].sum().item()) if torch.is_tensor(tr["rewards"]) else float(sum(tr["rewards"]))

        # === PPO update ===
        with diag.time_section("ppo_update"):
            loss, update_info = self.ppo_loss(tr)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

        # === Extract logging data ===
        detailed_logging = getattr(self.env, "eval_mode", False) and ("step_info_list" in tr)
        episode_data = None
        if detailed_logging:
            episode_data = {
                "step_rewards": [],
                "step_returns": [],
                "step_excess_returns": [],
                "step_capital": [],
                "step_concentrations": [],
                "step_active_positions": [],
                "step_cash_positions": [],
                "step_transaction_costs": [],
                "step_long_exposures": [],
                "step_short_exposures": [],
                "step_net_exposures": [],
                "step_gross_exposures": [],
                "step_rel_excess_returns": [],
                "step_dsr_alpha": [],
                "step_dsr_beta": [],
            }
            for step_info in tr["step_info_list"]:
                episode_data["step_rewards"].append(step_info.get("sharpe_reward", 0.0))
                episode_data["step_returns"].append(step_info.get("log_return", 0.0))
                episode_data["step_excess_returns"].append(step_info.get("excess_log_return", 0.0))
                episode_data["step_capital"].append(step_info.get("capital", 0.0))
                episode_data["step_concentrations"].append(step_info.get("portfolio_concentration", 0.0))
                episode_data["step_active_positions"].append(step_info.get("num_active_positions", 0))
                episode_data["step_cash_positions"].append(step_info.get("cash_pct", 0.0))
                episode_data["step_transaction_costs"].append(step_info.get("transaction_cost", 0.0))
                episode_data["step_long_exposures"].append(step_info.get("weights_long", 0.0))
                episode_data["step_short_exposures"].append(step_info.get("weights_short", 0.0))
                episode_data["step_net_exposures"].append(step_info.get("net_exposure", 0.0))
                episode_data["step_gross_exposures"].append(step_info.get("gross_exposure", 0.0))
                episode_data["step_rel_excess_returns"].append(step_info.get("relative_excess_log_return", 0.0))
                episode_data["step_dsr_alpha"].append(step_info.get("dsr_alpha", 0.0))
                episode_data["step_dsr_beta"].append(step_info.get("dsr_beta", 0.0))

        # === Final episode-level values ===
        # Aggregates available regardless of detailed logging
        final_capital = float(getattr(self.env, "current_capital", 0.0))
        cumulative_return = final_capital / self.env.initial_capital - 1.0 if final_capital else 0.0
        final_weights_tensor = getattr(self.env, "prev_weights", None)
        if torch.is_tensor(final_weights_tensor):
            long_exposure = float(final_weights_tensor[final_weights_tensor > 0].sum().item())
            short_exposure = float(torch.abs(final_weights_tensor[final_weights_tensor < 0]).sum().item())
            net_exposure = float(final_weights_tensor.sum().item())
            gross_exposure = float(torch.sum(torch.abs(final_weights_tensor)).item())
        else:
            long_exposure = 0.0
            short_exposure = 0.0
            net_exposure = 0.0
            gross_exposure = 0.0

        # === Loss tracking ===
        policy_loss = update_info.get("policy_loss", 0.0)
        vae_loss = update_info.get("vae_loss", 0.0)
        vae_loss_components = {k: v for k, v in update_info.items() if k.startswith("vae_")}

        self.episode_rewards.append(episode_reward_sum)
        self.policy_losses.append(policy_loss)
        if vae_loss > 0:
            self.vae_losses.append(vae_loss)

        self.episode_count += 1

        # === Build results ===
        # Optional: include final_weights for logger consumers
        final_weights = final_weights_tensor.detach().cpu().tolist() if torch.is_tensor(final_weights_tensor) else None

        results = {
            "episode_reward": episode_reward_sum,
            "policy_loss": policy_loss,
            "vae_loss": vae_loss,
            "total_steps": int(self.total_steps),

            # Portfolio metrics
            "episode_final_capital": final_capital,
            "episode_total_return": cumulative_return,
            # Aggregates
            "episode_sum_reward": episode_reward_sum,
            "episode_long_exposure": long_exposure,
            "episode_short_exposure": short_exposure,
            "episode_net_exposure": net_exposure,
            "episode_gross_exposure": gross_exposure,
            "final_weights": final_weights,

            # Rolling stats
            "rolling_avg_episode_reward": float(np.mean(list(self.episode_rewards))) if self.episode_rewards else 0.0,
            "rolling_std_episode_reward": float(np.std(list(self.episode_rewards))) if len(self.episode_rewards) > 1 else 0.0,
            "rolling_avg_policy_loss": float(np.mean(list(self.policy_losses))) if self.policy_losses else 0.0,
            "rolling_avg_vae_loss": float(np.mean(list(self.vae_losses))) if self.vae_losses else 0.0,

            # VAE breakdown
            **vae_loss_components,

            # Misc
            "episode_count": int(self.episode_count),
            "steps_per_episode": len(episode_data["step_rewards"]),
            "num_episodes_in_batch": 1,
            # Include detailed step data only in eval mode
            "step_data": episode_data if detailed_logging else None,
        }

        return results


    # ---------------------------------------------------------------------
    # Trajectory collection (single)
    # ---------------------------------------------------------------------
    def collect_trajectory(self):
        # Reset and prepare shapes
        obs0 = self.env.reset()  # tensor [N, F] on env.device
        obs0 = obs0.to(self.device).to(torch.float32)
        obs_tensor = obs0.unsqueeze(0)  # [1, N, F]

        max_horizon = int(self.config.max_horizon)
        obs_shape = tuple(obs0.shape)

        # Preallocate tensors
        observations = torch.zeros((max_horizon,) + obs_shape, dtype=torch.float32, device=self.device)
        actions      = torch.zeros((max_horizon, self.config.num_assets), dtype=torch.float32, device=self.device)
        values       = torch.zeros((max_horizon,), dtype=torch.float32, device=self.device)
        log_probs    = torch.zeros((max_horizon,), dtype=torch.float32, device=self.device)
        latents      = torch.zeros((max_horizon, self.config.latent_dim), dtype=torch.float32, device=self.device)
        rewards      = torch.zeros((max_horizon,), dtype=torch.float32, device=self.device)
        dones        = torch.zeros((max_horizon,), dtype=torch.bool, device=self.device)

        context = {"observations": [], "actions": [], "rewards": []}
        done, step = False, 0

        while not done and step < max_horizon:
            # === Latent context ===
            latent = self._get_latent_for_step(obs_tensor, context)

            # === Sample action ===
            with torch.no_grad():
                actions_raw, value_t, log_prob_t = self.policy.act(obs_tensor, latent, deterministic=False)

            # === Environment step with tensor action ===
            next_obs, reward_scalar, done_flag, info = self.env.step(actions_raw.squeeze(0))

            # === Write step data ===
            observations[step] = obs_tensor.squeeze(0)
            actions[step]      = actions_raw.squeeze(0).to(self.device)
            values[step]       = value_t.squeeze(0)
            log_probs[step]    = log_prob_t.squeeze(0)
            latents[step]      = latent.squeeze(0)
            rewards[step]      = float(reward_scalar)
            dones[step]        = bool(done_flag)

            # Update context for VAE (keep as tensors)
            context["observations"].append(obs_tensor.squeeze(0).detach())
            context["actions"].append(actions_raw.squeeze(0).detach())
            context["rewards"].append(torch.tensor(reward_scalar, dtype=torch.float32, device=self.device))

            # Advance
            done = bool(done_flag)
            if not done:
                obs_tensor = next_obs.to(self.device, dtype=torch.float32).unsqueeze(0)
            step += 1

        T = step if step > 0 else 1
        traj = {
            "observations": observations[:T],
            "actions": actions[:T],
            "rewards": rewards[:T],
            "values": values[:T],
            "log_probs": log_probs[:T],
            "latents": latents[:T],
            "dones": dones[:T],
        }
        return traj



    # ---------------------------------------------------------------------
    # Environment cloning
    # ---------------------------------------------------------------------
    def _clone_env(self) -> MetaEnv:
        # Rebuild MetaEnv with same dataset + parameters
        src = self.env
        return MetaEnv(
            dataset={"features": src.dataset["features"], "raw_prices": src.dataset["raw_prices"]},
            feature_columns=src.feature_columns,
            seq_len=src.seq_len,
            min_horizon=src.min_horizon,
            max_horizon=src.max_horizon,
        )


    def _batch_vae_encode(self, ctx_obs_tensor, ctx_act_tensor, ctx_rew_tensor, ctx_lengths, done):
        """VAE encoding for variable-length sequences"""
        B = ctx_obs_tensor.shape[0]
        active_mask = (ctx_lengths > 0) & (~torch.from_numpy(done).to(ctx_lengths.device))
        
        if not active_mask.any():
            return torch.zeros(B, self.config.latent_dim, device=self.device)
        
        max_len = ctx_lengths[active_mask].max().item()
        if max_len == 0:
            return torch.zeros(B, self.config.latent_dim, device=self.device)
        
        batch_obs = ctx_obs_tensor[:, :max_len]
        batch_acts = ctx_act_tensor[:, :max_len]
        batch_rews = ctx_rew_tensor[:, :max_len]
        
        for i in range(B):
            if not active_mask[i] or ctx_lengths[i] < max_len:
                actual_len = ctx_lengths[i] if active_mask[i] else 0
                if actual_len < max_len:
                    batch_obs[i, actual_len:] = 0
                    batch_acts[i, actual_len:] = 0
                    batch_rews[i, actual_len:] = 0
        
        try:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                mu, logvar, _ = self.vae.encode(batch_obs, batch_acts, batch_rews)
                latents = self.vae.reparameterize(mu, logvar)
        except Exception as e:
            print(f"VAE batch encode failed: {e}")
            latents = torch.zeros(B, self.config.latent_dim, device=self.device)
        
        latents[~active_mask] = 0
        return latents

    def _create_empty_trajectory(self):
        """Helper to create empty trajectory for failed environments"""
        empty_obs_shape = (1,) + tuple(self.env.reset().shape)
        traj = {
            "observations": torch.zeros(empty_obs_shape, dtype=torch.float32, device=self.device),
            "actions": torch.zeros((1, self.config.num_assets), dtype=torch.float32, device=self.device),
            "values": torch.zeros(1, dtype=torch.float32, device=self.device),
            "log_probs": torch.zeros(1, dtype=torch.float32, device=self.device),
            "latents": torch.zeros((1, self.config.latent_dim), dtype=torch.float32, device=self.device),
            "rewards": torch.zeros(1, dtype=torch.float32, device=self.device),
            "dones": [True]
        }
        
        # === NEW: Add empty step info list ===
        traj["step_info_list"] = [{}]
        # Single empty step info
        
        return traj

    # ---------------------------------------------------------------------
    # PPO / VAE updates (unchanged)
    # ---------------------------------------------------------------------
    
    def compute_gae(self, rewards, values, dones, last_value=0.0):
        """
        Generalized Advantage Estimation (Schulman et al., 2016).
        Computes normalized advantages and returns.

        Args:
            rewards: tensor (T,)
            values: tensor (T,)
            dones: tensor (T,) of 0/1 or bool
            last_value: bootstrap value for final step (default 0)

        Returns:
            advantages, returns (both tensors shape (T,))
        """
        gamma = self.config.discount_factor
        lam = self.config.gae_lambda
        T = len(rewards)

        advantages = torch.zeros_like(rewards, device=self.device)
        returns = torch.zeros_like(rewards, device=self.device)

        gae = 0.0
        next_value = last_value
        for t in reversed(range(T)):
            mask = 1.0 - dones[t].float()
            delta = rewards[t] + gamma * next_value * mask - values[t]
            gae = delta + gamma * lam * mask * gae
            advantages[t] = gae
            returns[t] = gae + values[t]
            next_value = values[t]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns


    def ppo_loss(self, traj):
        """
        PPO clipped surrogate objective + value loss + entropy bonus.
        """
        obs = torch.stack(traj["observations"]).to(self.device)
        actions = torch.stack(traj["actions"]).to(self.device)
        latents = torch.stack(traj["latents"]).to(self.device)
        rewards = torch.as_tensor(traj["rewards"], dtype=torch.float32, device=self.device)
        values = torch.stack(traj["values"]).to(self.device)
        dones = torch.as_tensor(traj["dones"], dtype=torch.float32, device=self.device)
        old_logp = torch.stack(traj["log_probs"]).to(self.device)



        # Bootstrap if provided
        last_value = traj.get("last_value", 0.0)
        if isinstance(last_value, torch.Tensor):
            last_value = last_value.to(self.device)

        # Compute GAE
        advantages, returns = self.compute_gae(rewards, values, dones, last_value)

        # Evaluate current policy
        new_values, new_logp, entropy = self.policy.evaluate_actions(obs, latents, actions)
        new_values = new_values.squeeze(-1)
        new_logp = new_logp.squeeze(-1)
        entropy = entropy.squeeze(-1)

        # Probability ratio
        ratio = torch.exp(new_logp - old_logp)

        # Clipped surrogate loss
        eps = self.config.ppo_clip_ratio
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = F.mse_loss(new_values, returns) * self.config.value_loss_coef

        # Entropy bonus
        entropy_loss = -self.config.entropy_coef * entropy.mean()

        total_loss = policy_loss + value_loss + entropy_loss

        return total_loss, {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.mean().item(),
            "advantages_mean": advantages.mean().item(),
            "ratio_mean": ratio.mean().item()
        }

    
    
    
    def compute_advantages(self, trajectory):
        rewards = trajectory["rewards"]
        values = trajectory["values"]
        dones = trajectory["dones"]

        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        gae = 0.0
        next_value = 0.0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.config.discount_factor * next_value * (1 - int(dones[t])) - values[t]
            gae = delta + self.config.discount_factor * self.config.gae_lambda * (1 - int(dones[t])) * gae
            advantages[t] = gae
            returns[t] = gae + values[t]
            next_value = values[t]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def update_joint(self, trajectory):
        """
        Update PPO (policy + value) and VAE (if enabled).
        trajectory: dict with observations, actions, rewards, values, log_probs, dones
        """
        obs = torch.stack(trajectory["observations"]).to(self.device)
        actions = torch.stack(trajectory["actions"]).to(self.device)
        rewards = torch.tensor(trajectory["rewards"], dtype=torch.float32, device=self.device)
        values = torch.stack(trajectory["values"]).to(self.device).squeeze(-1)
        old_log_probs = torch.stack(trajectory["log_probs"]).to(self.device).squeeze(-1)

        # === Compute returns & advantages ===
        returns = []
        G = 0
        for r in reversed(rewards.tolist()):
            G = r + self.config.discount_factor * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # === PPO update ===
        policy_loss, value_loss, entropy_loss = 0.0, 0.0, 0.0

        for _ in range(self.config.ppo_epochs):
            mean, logstd, value_preds = self.policy.forward(obs, torch.stack(trajectory["latents"]).to(self.device))
            dist = torch.distributions.Normal(mean, logstd.exp())
            new_log_probs = dist.log_prob(actions).sum(-1)

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config.ppo_clip_ratio, 1 + self.config.ppo_clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(value_preds.squeeze(-1), returns)

            entropy_loss = -dist.entropy().sum(-1).mean()

            total_loss = (policy_loss
                        + self.config.value_loss_coef * value_loss
                        + self.config.entropy_coef * entropy_loss)

            # === Optimize ===
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

        # === VAE update ===
        vae_loss_val = 0.0
        if self.vae is not None and not self.config.disable_vae:
            obs_seq = obs.unsqueeze(0)  # (1, T, N, F)
            act_seq = actions.unsqueeze(0)  # (1, T, N)
            rew_seq = rewards.unsqueeze(0).unsqueeze(-1)  # (1, T, 1)
            vae_loss, vae_info = self.vae.compute_loss(
                obs_seq, act_seq, rew_seq, beta=self.config.vae_beta
            )
            self.optimizer.zero_grad()
            vae_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.vae.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            vae_loss_val = vae_loss.item()

        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy_loss.item()),
            "vae_loss": float(vae_loss_val),
        }



    # ---------------------------------------------------------------------
    # Checkpoint helpers (unchanged)
    # ---------------------------------------------------------------------
    def get_state(self):
        return {
            "episode_count": self.episode_count,
            "total_steps": self.total_steps,
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state(self, state):
        self.episode_count = state.get("episode_count", 0)
        self.total_steps = state.get("total_steps", 0)
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])

    def _get_latent_for_step(self, obs_tensor, trajectory_context):
        """Get latent encoding for current step, supporting both episodic and sequential modes."""
        if getattr(self.config, "disable_vae", False):
            return torch.zeros(1, self.config.latent_dim, device=self.device)
        
        # Choose context source based on mode
        if self.sequential_mode and len(self.rolling_context) > 0:
            # Sequential mode: use rolling context
            context_list = list(self.rolling_context)
            obs_seq = torch.stack([ctx['observations'] for ctx in context_list]).unsqueeze(0)
            act_seq = torch.stack([ctx['actions'] for ctx in context_list]).unsqueeze(0)
            rew_seq = torch.stack([ctx['rewards'] for ctx in context_list]).unsqueeze(0).unsqueeze(-1)
        elif len(trajectory_context["observations"]) == 0:
            # No context available
            return torch.zeros(1, self.config.latent_dim, device=self.device)
        else:
            # Episodic mode: use trajectory context
            obs_seq = torch.stack(trajectory_context["observations"]).unsqueeze(0)
            act_seq = torch.stack(trajectory_context["actions"]).unsqueeze(0)
            rew_seq = torch.stack(trajectory_context["rewards"]).unsqueeze(0).unsqueeze(-1)
        
        try:
            mu, logvar, _ = self.vae.encode(obs_seq, act_seq, rew_seq)
            return self.vae.reparameterize(mu, logvar)
        except Exception as e:
            logger.warning(f"VAE encoding failed: {e}, using zero latent")
            return torch.zeros(1, self.config.latent_dim, device=self.device)


# ---------------------------------------------------------------------
# Experience buffer (unchanged)
# ---------------------------------------------------------------------
class ExperienceBuffer:
    """Buffer for storing PPO training data."""

    def __init__(self, min_batch_size=64):
        self.min_batch_size = min_batch_size
        self.trajectories = []
        self.total_steps = 0

    def add_trajectory(self, trajectory):
        self.trajectories.append(trajectory)
        self.total_steps += len(trajectory["rewards"])

    def is_ready(self):
        return self.total_steps >= self.min_batch_size

    def get_all(self):
        return self.trajectories

    def clear(self):
        self.trajectories = []
        self.total_steps = 0