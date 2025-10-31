import gc
from collections import deque
from datetime import datetime
from typing import Dict, List
import time
import logging
logger = logging.getLogger(__name__)

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from environments.env import MetaEnv

def log_memory(label):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3

def diagnose_gpu_tensors():
    """Find all tensors on GPU"""
    import gc
    tensors = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                tensors.append((type(obj), obj.size(), obj.element_size() * obj.nelement() / 1024**2))
        except:
            pass
    
    # Group by size
    tensors.sort(key=lambda x: x[2], reverse=True)
    total_mb = sum(t[2] for t in tensors)


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
    """PPO Trainer for VariBAD Portfolio Optimization."""

    def __init__(self, env, policy, vae, config):
        self.env = env
        self.policy = policy
        self.vae = vae
        self.config = config

        self.episode_count = 0
        self.total_steps = 0
        self.device = torch.device(config.device)

        self.sequential_mode = False
        self.num_envs = max(1, int(getattr(config, "num_envs", 1)))
        self.use_fixed_length = (config.min_horizon == config.max_horizon)
        self.vae_enabled = (vae is not None) and (not getattr(config, "disable_vae", False))

        self.policy_optimizer = Adam(policy.parameters(), lr=config.policy_lr)
        self.vae_optimizer = Adam(vae.parameters(), lr=config.vae_lr) if self.vae_enabled else None

        # Experience buffers
        self.experience_buffer = ExperienceBuffer(config.batch_size)
        self.vae_buffer = deque(maxlen=1000)

        self.policy_losses = deque(maxlen=100)
        self.vae_losses = deque(maxlen=100)
        self.episode_rewards = deque(maxlen=100)

        self.episode_details = []
        self.training_start_time = time.time()
        self.episode_start_time = None
        self.portfolio_metrics_history = []

        self.current_task_id = None
        self.persistent_context = None
        self.task_count = 0
    


    def train_on_task(self) -> Dict[str, float]:
        """Train over multiple episodes on same task (BAMDP)"""
        
        # Sample task once
        task = self.env.sample_task()
        task_id = task.get("task_id", self.task_count)
        self.env.set_task(task)
        self.task_count += 1
        
        context_start_step = 0
        context_obs_list = []  # Will store on CPU
        context_act_list = []
        context_rew_list = []
        
        # Accumulate all transitions across episodes
        all_transitions = {
            "observations": [],
            "actions": [],
            "raw_actions": [],
            "rewards": [],
            "values": [],
            "log_probs": [],
            "latents": [],
            "dones": [],
            "episode_boundaries": [],
            "prior_mu": [],
            "prior_logvar": []
        }
        
        episode_rewards = []
        
        # Multiple episodes on same task
        for episode_idx in range(self.config.episodes_per_task):
            trajectory = self.collect_trajectory_with_context_v2(
                context_obs_list, context_act_list, context_rew_list
            )
            
            episode_reward = float(trajectory["rewards"].sum().item())
            episode_rewards.append(episode_reward)

            self.vae_buffer.append(trajectory)
            
            traj_length = len(trajectory["observations"])
            current_total = sum(len(t) for t in all_transitions["observations"])
            all_transitions["episode_boundaries"].append(current_total + traj_length - 1)
            
            all_transitions["prior_mu"].append(trajectory["prior_mu"])
            all_transitions["prior_logvar"].append(trajectory["prior_logvar"])
            
            for key in all_transitions.keys():
                if key in trajectory and key not in ["prior_mu", "prior_logvar"]:
                    all_transitions[key].append(trajectory[key])
            
            T = len(trajectory["observations"])
            for t in range(T):
                context_obs_list.append(trajectory["observations"][t].detach().cpu())
                context_act_list.append(trajectory["actions"][t].detach().cpu())
                context_rew_list.append(trajectory["rewards"][t].detach().cpu())
            
            del trajectory
            torch.cuda.empty_cache()
        
        # Stack accumulated transitions from all episodes
        for key in all_transitions.keys():
            if key not in ["episode_boundaries", "prior_mu", "prior_logvar"]:
                all_transitions[key] = torch.cat(all_transitions[key], dim=0)
        
        episode_boundaries = all_transitions["episode_boundaries"]
        for boundary_idx in episode_boundaries[:-1]:  # All except last episode boundary
            all_transitions["dones"][boundary_idx] = False
        
        total_loss, update_info = self.update_ppo_and_vae(all_transitions)

        self.episode_count += self.config.episodes_per_task
        
        # Build results
        final_capital = self.env.current_capital
        cumulative_return = final_capital / self.env.initial_capital - 1.0
        total_steps = int(all_transitions["rewards"].shape[0])

        results = {
            "policy_loss": update_info.get("policy_loss", 0.0),
            "vae_loss": update_info.get("vae_loss", 0.0),
            "value_loss": update_info.get("value_loss", 0.0),
            "entropy": update_info.get("entropy", 0.0),
            "total_steps": total_steps,
            "task_total_reward": sum(episode_rewards),
            "task_avg_reward_per_episode": sum(episode_rewards) / len(episode_rewards),
            "task_final_capital": final_capital,
            "task_cumulative_return": cumulative_return,
            "episode_rewards": episode_rewards,
            "episodes_per_task": self.config.episodes_per_task,
            "task_count": self.task_count,
            **update_info
        }
        
        del all_transitions
        del context_obs_list
        del context_act_list
        del context_rew_list
        
        if self.task_count % 10 == 0:
            self.policy_optimizer.zero_grad(set_to_none=True)
            for group in self.policy_optimizer.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad = None
                    state = self.policy_optimizer.state.get(p, None)
                    if state is not None:
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                state[k] = v.detach()
            
            if self.vae_optimizer:
                self.vae_optimizer.zero_grad(set_to_none=True)
                for group in self.vae_optimizer.param_groups:
                    for p in group['params']:
                        if p.grad is not None:
                            p.grad.detach_()
                            p.grad = None
                        state = self.vae_optimizer.state.get(p, None)
                        if state is not None:
                            for k, v in state.items():
                                if torch.is_tensor(v):
                                    state[k] = v.detach()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()
        
        return results

    def collect_trajectory_with_context_v2(self, context_obs, context_act, context_rew):
        """Collect trajectory with context for VAE encoding."""
        obs0 = self.env.reset()
        obs0 = obs0.to(self.device).to(torch.float32)
        obs_tensor = obs0.unsqueeze(0)
        
        max_horizon = int(self.config.max_horizon)
        obs_shape = tuple(obs0.shape)
        
        if len(context_obs) == 0 or getattr(self.config, "disable_vae", False) or self.vae is None:
            prior_mu = torch.zeros(1, self.config.latent_dim, device=self.device)
            prior_logvar = torch.zeros(1, self.config.latent_dim, device=self.device)
        else:
            with torch.no_grad():
                obs_seq = torch.stack(context_obs).to(self.device).unsqueeze(0)
                act_seq = torch.stack(context_act).to(self.device).unsqueeze(0)
                rew_seq = torch.stack(context_rew).to(self.device).unsqueeze(0).unsqueeze(-1)
                
                try:
                    prior_mu, prior_logvar, _ = self.vae.encode(obs_seq, act_seq, rew_seq)
                    prior_mu = prior_mu.detach()
                    prior_logvar = prior_logvar.detach()
                except Exception as e:
                    logger.warning(f"Prior computation failed: {e}, using N(0,I)")
                    prior_mu = torch.zeros(1, self.config.latent_dim, device=self.device)
                    prior_logvar = torch.zeros(1, self.config.latent_dim, device=self.device)
                finally:
                    del obs_seq, act_seq, rew_seq
        
        # Preallocate
        observations = torch.zeros((max_horizon,) + obs_shape, dtype=torch.float32, device=self.device)
        actions = torch.zeros((max_horizon, self.config.num_assets), dtype=torch.float32, device=self.device)
        raw_actions = torch.zeros((max_horizon, self.config.num_assets), dtype=torch.float32, device=self.device) 
        values = torch.zeros((max_horizon,), dtype=torch.float32, device=self.device)
        log_probs = torch.zeros((max_horizon,), dtype=torch.float32, device=self.device)
        latents = torch.zeros((max_horizon, self.config.latent_dim), dtype=torch.float32, device=self.device)
        rewards = torch.zeros((max_horizon,), dtype=torch.float32, device=self.device)
        dones = torch.zeros((max_horizon,), dtype=torch.bool, device=self.device)
        
        done, step = False, 0
        
        while not done and step < max_horizon:
            latent = self._get_latent_from_context_v2(obs_tensor, context_obs, context_act, context_rew)
            
            with torch.no_grad():
                weights, value_t, log_prob_t, raw_actions_t = self.policy.act(obs_tensor, latent, deterministic=False)
            
            next_obs, reward_scalar, done_flag, info = self.env.step(weights.squeeze(0))
            normalized_weights = info['weights'].detach()

            # Store step data
            observations[step] = obs_tensor.squeeze(0)
            actions[step] = normalized_weights.to(self.device)
            raw_actions[step] = raw_actions_t.squeeze(0).to(self.device)
            values[step] = value_t.squeeze(0)
            log_probs[step] = log_prob_t.squeeze(0)
            latents[step] = latent.squeeze(0)
            rewards[step] = float(reward_scalar)
            dones[step] = bool(done_flag)

            done = bool(done_flag)
            if not done:
                obs_tensor = next_obs.to(self.device, dtype=torch.float32).unsqueeze(0)
            step += 1
            
            del latent, weights, value_t, log_prob_t, raw_actions_t
        
        T = step if step > 0 else 1
        return {
            "observations": observations[:T],
            "actions": actions[:T],
            "raw_actions": raw_actions[:T],
            "rewards": rewards[:T],
            "values": values[:T],
            "log_probs": log_probs[:T],
            "latents": latents[:T],
            "dones": dones[:T],
            "prior_mu": prior_mu.cpu(),
            "prior_logvar": prior_logvar.cpu(),
        }

    def _get_latent_from_context_v2(self, obs_tensor, context_obs, context_act, context_rew):
        """Get latent encoding from context."""
        if self.vae is None:
            return torch.zeros(1, self.config.latent_dim, device=self.device)
        
        if len(context_obs) == 0:
            return torch.zeros(1, self.config.latent_dim, device=self.device)
        
        obs_seq = torch.stack(context_obs).to(self.device).unsqueeze(0)
        act_seq = torch.stack(context_act).to(self.device).unsqueeze(0)
        rew_seq = torch.stack(context_rew).to(self.device).unsqueeze(0).unsqueeze(-1)

        try:
            with torch.no_grad():
                if self.config.encoder == "hmm":
                    regime_probs = self.vae.encode(obs_seq, act_seq, rew_seq)
                    latent = regime_probs.detach()
                elif getattr(self.config, "disable_vae", False):
                    latent = torch.zeros(1, self.config.latent_dim, device=self.device)
                else:
                    mu, logvar, _ = self.vae.encode(obs_seq, act_seq, rew_seq)
                    latent = self.vae.reparameterize(mu, logvar).clone()
            
            return latent
            
        except Exception as e:
            logger.warning(f"VAE encoding failed: {e}, using zero latent")
            return torch.zeros(1, self.config.latent_dim, device=self.device)
        finally:
            del obs_seq, act_seq, rew_seq
            if 'mu' in locals():
                del mu, logvar

    def collect_trajectory_with_context(self, persistent_context):
        """Collect trajectory with persistent context from previous episodes."""
        # Reset environment
        obs0 = self.env.reset()
        obs0 = obs0.to(self.device).to(torch.float32)
        obs_tensor = obs0.unsqueeze(0)
        
        max_horizon = int(self.config.max_horizon)
        obs_shape = tuple(obs0.shape)
        
        # Preallocate
        observations = torch.zeros((max_horizon,) + obs_shape, dtype=torch.float32, device=self.device)
        actions = torch.zeros((max_horizon, self.config.num_assets), dtype=torch.float32, device=self.device)
        values = torch.zeros((max_horizon,), dtype=torch.float32, device=self.device)
        log_probs = torch.zeros((max_horizon,), dtype=torch.float32, device=self.device)
        latents = torch.zeros((max_horizon, self.config.latent_dim), dtype=torch.float32, device=self.device)
        rewards = torch.zeros((max_horizon,), dtype=torch.float32, device=self.device)
        dones = torch.zeros((max_horizon,), dtype=torch.bool, device=self.device)
        
        done, step = False, 0
        
        while not done and step < max_horizon:
            latent = self._get_latent_with_persistent_context(obs_tensor, persistent_context)
            
            with torch.no_grad():
                actions_raw, value_t, log_prob_t = self.policy.act(obs_tensor, latent, deterministic=False)
            
            next_obs, reward_scalar, done_flag, info = self.env.step(actions_raw.squeeze(0))
            
            # Store step data
            observations[step] = obs_tensor.squeeze(0)
            actions[step] = actions_raw.squeeze(0).to(self.device)
            values[step] = value_t.squeeze(0)
            log_probs[step] = log_prob_t.squeeze(0)
            latents[step] = latent.squeeze(0)
            rewards[step] = float(reward_scalar)
            dones[step] = bool(done_flag)
            
            persistent_context["observations"].append(obs_tensor.squeeze(0).detach().cpu())
            persistent_context["actions"].append(actions_raw.squeeze(0).detach().cpu())
            persistent_context["rewards"].append(torch.tensor(reward_scalar, dtype=torch.float32))
            
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


    def _get_latent_with_persistent_context(self, obs_tensor, persistent_context):
        """Get latent encoding using persistent context."""
        if getattr(self.config, "disable_vae", False) or self.vae is None:
            return torch.zeros(1, self.config.latent_dim, device=self.device)
        
        if len(persistent_context["observations"]) == 0:
            return torch.zeros(1, self.config.latent_dim, device=self.device)
        
        max_context_len = 200
        start_idx = max(0, len(persistent_context["observations"]) - max_context_len)
        obs_seq = torch.stack(persistent_context["observations"][start_idx:]).to(self.device).unsqueeze(0)
        act_seq = torch.stack(persistent_context["actions"][start_idx:]).to(self.device).unsqueeze(0)
        rew_seq = torch.stack(persistent_context["rewards"][start_idx:]).to(self.device).unsqueeze(0).unsqueeze(-1)
        
        try:
            mu, logvar, _ = self.vae.encode(obs_seq, act_seq, rew_seq)
            latent = self.vae.reparameterize(mu, logvar)
            return latent
        except Exception as e:
            logger.warning(f"VAE encoding failed: {e}, using zero latent")
            return torch.zeros(1, self.config.latent_dim, device=self.device)

    def train_episode(self) -> Dict[str, float]:
        """
        Train for one episode and return metrics.
        """
        diag = PerformanceDiagnostic()

        with diag.time_section("collect_single_trajectory"):
            tr = self.collect_trajectory()
            self.vae_buffer.append(tr)
            self.experience_buffer.add_trajectory(tr)

        episode_reward_sum = float(tr["rewards"].sum().item()) if torch.is_tensor(tr["rewards"]) else float(sum(tr["rewards"]))

        with diag.time_section("ppo_update"):
            _, update_info = self.update_ppo_and_vae(tr)
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

        policy_loss = update_info.get("policy_loss", 0.0)
        vae_loss = update_info.get("vae_loss", 0.0)
        value_loss = update_info.get("value_loss", 0.0)
        entropy = update_info.get("entropy", 0.0)
        vae_loss_components = {k: v for k, v in update_info.items() if k.startswith("vae_")}

        self.episode_rewards.append(episode_reward_sum)
        self.policy_losses.append(policy_loss)
        if vae_loss > 0:
            self.vae_losses.append(vae_loss)

        self.episode_count += 1

        final_weights = final_weights_tensor.detach().cpu().tolist() if torch.is_tensor(final_weights_tensor) else None

        results = {
            "policy_loss": policy_loss,
            "vae_loss": vae_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "total_steps": int(self.total_steps),
            "episode_final_capital": final_capital,
            "episode_total_return": cumulative_return,
            "episode_sum_reward": episode_reward_sum,
            "episode_long_exposure": long_exposure,
            "episode_short_exposure": short_exposure,
            "episode_net_exposure": net_exposure,
            "episode_gross_exposure": gross_exposure,
            "final_weights": final_weights,
            "recent_avg_episode_reward": float(np.mean(list(self.episode_rewards))) if self.episode_rewards else 0.0,
            "recent_std_episode_reward": float(np.std(list(self.episode_rewards))) if len(self.episode_rewards) > 1 else 0.0,
            "recent_avg_policy_loss": float(np.mean(list(self.policy_losses))) if self.policy_losses else 0.0,
            "recent_avg_vae_loss": float(np.mean(list(self.vae_losses))) if self.vae_losses else 0.0,
            **vae_loss_components,
            "episode_count": int(self.episode_count),
            "steps_per_episode": (int(tr["rewards"].shape[0]) if isinstance(tr["rewards"], torch.Tensor) else len(tr["rewards"])),
            "num_episodes_in_batch": 1,
            "step_data": episode_data if detailed_logging else None,
        }

        return results


    def collect_trajectory(self):
        obs0 = self.env.reset()
        obs0 = obs0.to(self.device).to(torch.float32)
        obs_tensor = obs0.unsqueeze(0)

        max_horizon = int(self.config.max_horizon)
        obs_shape = tuple(obs0.shape)
        observations = torch.zeros((max_horizon,) + obs_shape, dtype=torch.float32, device=self.device)
        actions      = torch.zeros((max_horizon, self.config.num_assets), dtype=torch.float32, device=self.device)
        raw_actions  = torch.zeros((max_horizon, self.config.num_assets), dtype=torch.float32, device=self.device)
        values       = torch.zeros((max_horizon,), dtype=torch.float32, device=self.device)
        log_probs    = torch.zeros((max_horizon,), dtype=torch.float32, device=self.device)
        latents      = torch.zeros((max_horizon, self.config.latent_dim), dtype=torch.float32, device=self.device)
        rewards      = torch.zeros((max_horizon,), dtype=torch.float32, device=self.device)
        dones        = torch.zeros((max_horizon,), dtype=torch.bool, device=self.device)

        context = {"observations": [], "actions": [], "rewards": []}
        done, step = False, 0

        while not done and step < max_horizon:
            latent = self._get_latent_for_step(obs_tensor, context)

            with torch.no_grad():
                actions_raw, value_t, log_prob_t, raw_actions_t = self.policy.act(obs_tensor, latent, deterministic=False)

            next_obs, reward_scalar, done_flag, info = self.env.step(actions_raw.squeeze(0))
            observations[step] = obs_tensor.squeeze(0)
            actions[step]      = actions_raw.squeeze(0).to(self.device)
            raw_actions[step]  = raw_actions_t.squeeze(0).to(self.device)
            values[step]       = value_t.squeeze(0)
            log_probs[step]    = log_prob_t.squeeze(0)
            latents[step]      = latent.squeeze(0)
            rewards[step]      = float(reward_scalar)
            dones[step]        = bool(done_flag)

            context["observations"].append(obs_tensor.squeeze(0).detach())
            context["actions"].append(actions_raw.squeeze(0).detach())
            context["rewards"].append(torch.tensor(reward_scalar, dtype=torch.float32, device=self.device))

            done = bool(done_flag)
            if not done:
                obs_tensor = next_obs.to(self.device, dtype=torch.float32).unsqueeze(0)
            step += 1

        T = step if step > 0 else 1
        traj = {
            "observations": observations[:T],
            "actions": actions[:T],
            "raw_actions": raw_actions[:T],
            "rewards": rewards[:T],
            "values": values[:T],
            "log_probs": log_probs[:T],
            "latents": latents[:T],
            "dones": dones[:T],
        }
        return traj



    def _clone_env(self) -> MetaEnv:
        src = self.env
        return MetaEnv(
            dataset={"features": src.dataset["features"], "raw_prices": src.dataset["raw_prices"]},
            feature_columns=src.feature_columns,
            seq_len=src.seq_len,
            min_horizon=src.min_horizon,
            max_horizon=src.max_horizon,
        )


    def _batch_vae_encode(self, ctx_obs_tensor, ctx_act_tensor, ctx_rew_tensor, ctx_lengths, done):
        """VAE encoding for variable-length sequences."""
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
        
        traj["step_info_list"] = [{}]
        
        return traj
    
    def compute_gae(self, rewards, values, dones, last_value=0.0):
        """Generalized Advantage Estimation."""
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


        if torch.isnan(advantages).any() or torch.isinf(advantages).any():
            logger.error("NaN/Inf detected in raw advantages!")
            advantages = torch.zeros_like(advantages)
            returns = values.clone()

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        adv_std = advantages.std()
        if adv_std < 1e-4:
            logger.warning(f"Advantage std too small ({adv_std:.2e}), using zeros")
            advantages = torch.zeros_like(advantages)
        else:
            advantages = (advantages - advantages.mean()) / (adv_std + 1e-6)
            advantages = torch.clamp(advantages, -10.0, 10.0)
        

        adv_scale = 1.0
        advantages = advantages * adv_scale



        return advantages, returns


    def update_ppo_and_vae(self, traj):
        """PPO with mini-batching + VAE update."""
        
        obs = traj["observations"].detach()
        actions = traj["actions"].detach()
        raw_actions = traj["raw_actions"].detach()
        latents = traj["latents"].detach()
        rewards = traj["rewards"].detach()
        dones = traj["dones"].detach()
        values = traj["values"].detach()
        old_logp = traj["log_probs"].detach()
        
        last_value = traj.get("last_value", 0.0)
        if isinstance(last_value, torch.Tensor):
            last_value = last_value.detach().to(self.device)
        

        # Compute advantages once
        advantages, returns = self.compute_gae(rewards, values, dones, last_value)

        advantages = advantages.detach()
        returns = returns.detach()

        # Safely compute logstd statistics depending on policy type
        with torch.no_grad():
            if hasattr(self.policy, "actor_logstd"):  # old global version
                logstd = self.policy.actor_logstd
            else:  # new state-dependent version
                # Use a representative batch of recent states to estimate the range
                sample_obs = torch.randn(8, *self.policy.obs_shape, device=self.device)
                sample_latent = torch.randn(8, self.policy.latent_dim if self.policy.latent_dim > 0 else 1, device=self.device)
                _, logstd, _ = self.policy.forward(sample_obs, sample_latent)


        first_epoch_metrics = {
            "advantages_mean": float(advantages.mean().item()),
            "advantages_std": float(advantages.std().item()),
            "advantages_min": float(advantages.min().item()),
            "advantages_max": float(advantages.max().item()),
        }
        
        # Mini-batch setup
        batch_size = min(self.config.ppo_minibatch_size, len(obs))
        num_samples = len(obs)
        
        final_ratio_mean = 1.0
        
        for epoch in range(self.config.ppo_epochs):
            indices = torch.randperm(num_samples, device=self.device)
            
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                batch_indices = indices[start_idx:end_idx]
                
                # Extract mini-batch
                batch_obs = obs[batch_indices]
                batch_actions = actions[batch_indices]
                batch_raw_actions = raw_actions[batch_indices]
                batch_latents = latents[batch_indices]
                batch_old_logp = old_logp[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                new_values, new_logp, entropy = self.policy.evaluate_actions(
                    batch_obs, batch_latents, batch_raw_actions
                )
                new_values = new_values.squeeze(-1)
                new_logp = new_logp.squeeze(-1)
                entropy = entropy.squeeze(-1)
                
                ratio = torch.exp(new_logp - batch_old_logp)
                ratio = torch.clamp(ratio, 0.1, 10.0)
                eps = self.config.ppo_clip_ratio
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(new_values, batch_returns)
                entropy_loss = -entropy.mean()
                
                current_entropy_coef = self.config.entropy_coef

                ppo_loss = (policy_loss +
                self.config.value_loss_coef * value_loss +
                current_entropy_coef * entropy_loss)
                
                # Capture first batch metrics
                    if epoch == 0 and start_idx == 0:
                        first_epoch_metrics.update({
                        "policy_loss": float(policy_loss.item()),
                        "value_loss": float(value_loss.item()),
                        "entropy": float(entropy.mean().item()),
                        "advantages_mean": float(advantages.mean().item()),
                        "advantages_std": float(advantages.std().item()),
                        "advantages_min": float(advantages.min().item()),
                        "advantages_max": float(advantages.max().item()),
                    })
                
                final_ratio_mean = float(ratio.mean().item())
                
                # Update
                self.policy_optimizer.zero_grad()
                ppo_loss.backward()

                policy_grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), float('inf'))

                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.policy_optimizer.step()

                del new_values, new_logp, entropy, surr1, surr2, ratio
                del policy_loss, value_loss, entropy_loss, ppo_loss
                del batch_obs, batch_actions, batch_latents, batch_old_logp
                del batch_advantages, batch_returns
            
            del indices
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # VAE UPDATE with aggressive cleanup
        vae_loss_val = 0.0
        if self.vae_enabled and (self.episode_count % self.config.vae_update_freq == 0):
            min_buffer_size = 8
            vae_batch_size = min(16, len(self.vae_buffer))
            
            if len(self.vae_buffer) >= min_buffer_size:
                try:
                    indices = np.random.choice(len(self.vae_buffer), vae_batch_size, replace=False)
                    
                    # Collect priors and data from sampled trajectories
                    batch_obs, batch_act, batch_rew = [], [], []
                    batch_prior_mu, batch_prior_logvar = [], []
                    max_len = max(self.vae_buffer[idx]["observations"].shape[0] for idx in indices)
                    
                    for idx in indices:
                        traj = self.vae_buffer[idx]
                        T = traj["observations"].shape[0]
                        
                        # Pad observations
                        obs_padded = torch.zeros((max_len,) + traj["observations"].shape[1:], 
                                                dtype=torch.float32, device=self.device)
                        obs_padded[:T] = traj["observations"].to(self.device)
                        batch_obs.append(obs_padded)
                        
                        # Pad actions
                        act_padded = torch.zeros((max_len,) + traj["actions"].shape[1:],
                                                dtype=torch.float32, device=self.device)
                        act_padded[:T] = traj["actions"].to(self.device)
                        batch_act.append(act_padded)
                        
                        # Pad rewards
                        rew_padded = torch.zeros(max_len, dtype=torch.float32, device=self.device)
                        rew_padded[:T] = traj["rewards"].to(self.device)
                        batch_rew.append(rew_padded)
                        
                        # Extract priors (stored as CPU tensors, move to GPU)
                        batch_prior_mu.append(traj["prior_mu"].to(self.device))
                        batch_prior_logvar.append(traj["prior_logvar"].to(self.device))
                        
                        # Delete temporary padded tensors
                        del obs_padded, act_padded, rew_padded
                    
                    # Stack into batches
                    obs_batch = torch.stack(batch_obs)
                    act_batch = torch.stack(batch_act)
                    rew_batch = torch.stack(batch_rew).unsqueeze(-1)
                    prior_mu_batch = torch.stack(batch_prior_mu)
                    prior_logvar_batch = torch.stack(batch_prior_logvar)
                    
                    del batch_obs, batch_act, batch_rew, batch_prior_mu, batch_prior_logvar
                    
                    vae_loss, vae_info = self.vae.compute_loss(
                        obs_batch, act_batch, rew_batch, 
                        beta=self.config.vae_beta,
                        num_elbo_terms=getattr(self.config, 'vae_num_elbo_terms', 8),
                        prior_mu=prior_mu_batch,
                        prior_logvar=prior_logvar_batch
                    )
                    
                    self.vae_optimizer.zero_grad()
                    vae_loss.backward()

                    vae_grad_norm = torch.nn.utils.clip_grad_norm_(self.vae.parameters(), float('inf'))

                    torch.nn.utils.clip_grad_norm_(self.vae.parameters(), self.config.max_grad_norm)
                    self.vae_optimizer.step()

                    vae_loss_val = float(vae_loss.item())
                    first_epoch_metrics.update({f"vae_{k}": v for k, v in vae_info.items()})
                    
                    # CRITICAL: Delete VAE tensors
                    del obs_batch, act_batch, rew_batch, vae_loss
                    
                except Exception as e:
                    logger.warning(f"VAE update failed: {e}")
                finally:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        else:
            if self.vae_enabled:
                logger.debug(f"VAE update skipped (episode {self.episode_count} % {self.config.vae_update_freq} = {self.episode_count % self.config.vae_update_freq})")

        # Final cleanup
        del obs, actions, latents, rewards, dones, values, old_logp, advantages, returns
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        first_epoch_metrics["vae_loss"] = vae_loss_val
        first_epoch_metrics["ratio_mean"] = final_ratio_mean
        
        # Map VAE-specific metrics for logger compatibility
        if "vae_num_elbo_terms" in first_epoch_metrics:
            first_epoch_metrics["vae_context_len"] = first_epoch_metrics["vae_num_elbo_terms"]


        return torch.tensor(0.0), first_epoch_metrics

    def compute_advantages(self, trajectory):
        rewards = trajectory["rewards"]
        values = trajectory["values"]
        dones = trajectory["dones"]

        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        pass

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


    # ---------------------------------------------------------------------
    # Checkpoint helpers (unchanged)
    # ---------------------------------------------------------------------
    def get_state(self):
        return {
            "episode_count": self.episode_count,
            "total_steps": self.total_steps,
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "vae_optimizer": self.vae_optimizer.state_dict() if self.vae_optimizer else None,
        }

    def load_state(self, state):
        self.episode_count = state.get("episode_count", 0)
        self.total_steps = state.get("total_steps", 0)
        if "policy_optimizer" in state:
            self.policy_optimizer.load_state_dict(state["policy_optimizer"])
        if "vae_optimizer" in state and self.vae_optimizer:
            self.vae_optimizer.load_state_dict(state["vae_optimizer"])

    def _get_latent_for_step(self, obs_tensor, trajectory_context):
        """
        Get latent encoding for current step, supporting both VAE and HMM.
        For HMM we want to use the frozen encoder outputs even though disable_vae=True.
        """
        # If no encoder at all, just zeros
        if self.vae is None:
            return torch.zeros(1, self.config.latent_dim, device=self.device)

        # Build a short context from current trajectory (same as before)
        if self.sequential_mode and len(self.rolling_context) > 0:
            context_list = list(self.rolling_context)
            obs_seq = torch.stack([ctx['observations'] for ctx in context_list]).unsqueeze(0)
            act_seq = torch.stack([ctx['actions'] for ctx in context_list]).unsqueeze(0)
            rew_seq = torch.stack([ctx['rewards'] for ctx in context_list]).unsqueeze(0).unsqueeze(-1)
        elif len(trajectory_context["observations"]) == 0:
            return torch.zeros(1, self.config.latent_dim, device=self.device)
        else:
            obs_seq = torch.stack(trajectory_context["observations"]).unsqueeze(0)
            act_seq = torch.stack(trajectory_context["actions"]).unsqueeze(0)
            rew_seq = torch.stack(trajectory_context["rewards"]).unsqueeze(0).unsqueeze(-1)

        try:
            if self.config.encoder == "hmm":
                # Use the pre-trained HMM encoder as a frozen feature extractor
                with torch.no_grad():
                    regime_probs = self.vae.encode(obs_seq, act_seq, rew_seq)  # (1, 4)
                    latent = regime_probs.detach()
            else:
                # VAE path (only when not disabled)
                if getattr(self.config, "disable_vae", False):
                    return torch.zeros(1, self.config.latent_dim, device=self.device)
                mu, logvar, _ = self.vae.encode(obs_seq, act_seq, rew_seq)
                latent = self.vae.reparameterize(mu, logvar)
            return latent
        except Exception as e:
            logger.warning(f"Latent encoding failed: {e}, using zero latent")
            return torch.zeros(1, self.config.latent_dim, device=self.device)

 

# ---------------------------------------------------------------------
# Experience buffer
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