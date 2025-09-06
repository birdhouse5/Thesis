# trainer.py - Optimized with fixed-length trajectory batching
import logging
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

logger = logging.getLogger(__name__)


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

        # Vectorization knob
        self.num_envs = max(1, int(getattr(config, "num_envs", 1)))

        # Check if we can use fixed-length optimization
        self.use_fixed_length = (config.min_horizon == config.max_horizon)
        if self.use_fixed_length and self.num_envs > 1:
            logger.info(f"Fixed-length optimization enabled: length={config.min_horizon}")
        elif self.num_envs > 1:
            logger.warning(f"Variable-length trajectories detected: min={config.min_horizon}, max={config.max_horizon}")
            logger.warning("Consider setting min_horizon = max_horizon for better performance")

        # Optimizers
        self.policy_optimizer = Adam(policy.parameters(), lr=config.policy_lr)
        self.vae_optimizer = Adam(vae.parameters(), lr=config.vae_lr)

        # Experience buffers
        self.experience_buffer = ExperienceBuffer(config.batch_size)  # for PPO
        self.vae_buffer = deque(maxlen=1000)  # recent trajectories for VAE

        # Rolling stats (store Python floats to avoid CUDA logging issues)
        self.policy_losses = deque(maxlen=100)
        self.vae_losses = deque(maxlen=100)
        self.episode_rewards = deque(maxlen=100)

        # Extra tracking (kept for potential analysis hooks)
        self.episode_details = []
        self.training_start_time = datetime.now()
        self.episode_start_time = None
        self.portfolio_metrics_history = []

        logger.info("PPO Trainer initialized")
        logger.info(f"Policy LR: {config.policy_lr}, VAE LR: {config.vae_lr}")
        logger.info(f"PPO epochs: {config.ppo_epochs}, clip ratio: {config.ppo_clip_ratio}")
        if self.num_envs > 1:
            logger.info(f"Trainer batched rollouts enabled: num_envs={self.num_envs}")

    # ---------------------------------------------------------------------
    # Public training entry point
    # ---------------------------------------------------------------------
    def train_episode(self) -> Dict[str, float]:
        """
        Train for one (or many) episode(s) with performance diagnostics.
        Uses fixed-length optimization when min_horizon == max_horizon.
        """
        diag = PerformanceDiagnostic()
        
        if self.num_envs > 1:
            if self.use_fixed_length:
                with diag.time_section("collect_fixed_length_batched"):
                    trajectories = self.collect_trajectories_batched_fixed_length(self.num_envs)
            else:
                with diag.time_section("collect_trajectories_batched"):
                    trajectories = self.collect_trajectories_batched(self.num_envs)
            
            with diag.time_section("add_to_buffers"):
                for tr in trajectories:
                    self.vae_buffer.append(tr)
                    self.experience_buffer.add_trajectory(tr)
            
            episode_reward_mean = float(np.mean([float(sum(tr["rewards"])) for tr in trajectories]))
        else:
            with diag.time_section("collect_single_trajectory"):
                tr = self.collect_trajectory()
                self.vae_buffer.append(tr)
                self.experience_buffer.add_trajectory(tr)
            episode_reward_mean = float(sum(tr["rewards"]))

        # Updates
        policy_loss = 0.0
        vae_loss = 0.0

        if self.experience_buffer.is_ready():
            with diag.time_section("update_policy"):
                policy_loss = float(self.update_policy())
            self.experience_buffer.clear()

        if (
            not getattr(self.config, "disable_vae", False)
            and self.episode_count % self.config.vae_update_freq == 0
            and len(self.vae_buffer) >= self.config.vae_batch_size
        ):
            with diag.time_section("update_vae"):
                vae_loss = float(self.update_vae())

        self.episode_rewards.append(float(episode_reward_mean))
        if policy_loss > 0:
            self.policy_losses.append(float(policy_loss))
        if vae_loss > 0:
            self.vae_losses.append(float(vae_loss))

        # Report timing every 50 episodes
        if self.episode_count % 50 == 0:
            diag.report()

        self.episode_count += 1

        return {
            "episode_reward": float(episode_reward_mean),
            "policy_loss": float(policy_loss),
            "vae_loss": float(vae_loss),
            "total_steps": int(self.total_steps),
        }

    # ---------------------------------------------------------------------
    # Trajectory collection (single)
    # ---------------------------------------------------------------------
    def collect_trajectory(self):
        traj = {"observations": [], "actions": [], "rewards": [], "values": [], "log_probs": [], "latents": [], "dones": []}

        obs = self.env.reset()
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        context = {"observations": [], "actions": [], "rewards": []}
        done = False
        step = 0

        while not done and step < self.config.max_horizon:
            latent = self._get_latent_for_step(obs_tensor, context)

            with torch.no_grad():
                action, _ = self.policy.act(obs_tensor, latent, deterministic=False)
                values, log_prob, _ = self.policy.evaluate_actions(obs_tensor, latent, action)

            action_cpu = action.squeeze(0).detach().cpu().numpy()
            next_obs, reward, done, _ = self.env.step(action_cpu)

            # Store (CPU)
            traj["observations"].append(obs_tensor.squeeze(0).detach().cpu())
            traj["actions"].append(action.squeeze(0).detach().cpu())
            traj["latents"].append(latent.squeeze(0).detach().cpu())
            traj["rewards"].append(float(reward))
            traj["values"].append(values.squeeze().cpu())
            traj["log_probs"].append(log_prob.squeeze().cpu())
            traj["dones"].append(bool(done))

            # Update context (GPU)
            context["observations"].append(obs_tensor.squeeze(0).detach())
            context["actions"].append(action.squeeze(0).detach())
            context["rewards"].append(torch.tensor(reward, device=self.device))

            if not done:
                obs_tensor = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)

            step += 1
            self.total_steps += 1

        # Stack to device
        traj["observations"] = torch.stack(traj["observations"]).to(self.device)
        traj["actions"] = torch.stack(traj["actions"]).to(self.device)
        traj["values"] = torch.stack(traj["values"]).to(self.device)
        traj["log_probs"] = torch.stack(traj["log_probs"]).to(self.device)
        traj["latents"] = torch.stack(traj["latents"]).to(self.device)
        traj["rewards"] = torch.tensor(traj["rewards"], dtype=torch.float32, device=self.device)
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

    # ---------------------------------------------------------------------
    # OPTIMIZED: Fixed-length trajectory collection
    # ---------------------------------------------------------------------
    def collect_trajectories_batched_fixed_length(self, B: int) -> List[Dict]:
        """
        OPTIMIZED: Fixed-length trajectories where all environments finish simultaneously.
        This eliminates synchronization bottlenecks and enables true VAE batching.
        """
        B = max(1, int(B))
        envs = [self._clone_env() for _ in range(B)]
        for e in envs:
            e.set_task(e.sample_task())
        obs_np = [e.reset() for e in envs]
        obs = torch.as_tensor(np.stack(obs_np, axis=0), dtype=torch.float32, device=self.device)

        # For fixed lengths, we know exactly when all environments will finish
        fixed_length = self.config.min_horizon  # Since min_horizon == max_horizon
        
        # Pre-allocate for exact trajectory length - this is key for performance
        ctx_obs_tensor = torch.zeros(B, fixed_length, *obs.shape[1:], device=self.device)
        ctx_act_tensor = torch.zeros(B, fixed_length, self.config.num_assets, device=self.device)
        ctx_rew_tensor = torch.zeros(B, fixed_length, 1, device=self.device)
        
        # Pre-allocate trajectory storage
        all_observations = torch.zeros(B, fixed_length, *obs.shape[1:], device=self.device)
        all_actions = torch.zeros(B, fixed_length, self.config.num_assets, device=self.device)
        all_latents = torch.zeros(B, fixed_length, self.config.latent_dim, device=self.device)
        all_rewards = torch.zeros(B, fixed_length, device=self.device)
        all_values = torch.zeros(B, fixed_length, device=self.device)
        all_log_probs = torch.zeros(B, fixed_length, device=self.device)

        for step in range(fixed_length):
            # VAE processing - efficient because we know the step count
            if getattr(self.config, "disable_vae", False) or step == 0:
                latent = torch.zeros(B, self.config.latent_dim, device=self.device)
            else:
                # Use data from previous steps (all environments have same history length)
                ctx_len = step
                batch_obs = ctx_obs_tensor[:, :ctx_len]
                batch_acts = ctx_act_tensor[:, :ctx_len]  
                batch_rews = ctx_rew_tensor[:, :ctx_len]
                
                # TRUE BATCHED VAE CALL - no masking needed since all have same length!
                try:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                        mu, logvar, _ = self.vae.encode(batch_obs, batch_acts, batch_rews)
                        latent = self.vae.reparameterize(mu, logvar)
                except Exception as e:
                    print(f"VAE batch encode failed: {e}")
                    latent = torch.zeros(B, self.config.latent_dim, device=self.device)

            # Policy step (already batched)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                action, _ = self.policy.act(obs, latent, deterministic=False)
                values, log_probs, _ = self.policy.evaluate_actions(obs, latent, action)

            # Store everything in pre-allocated tensors
            all_observations[:, step] = obs
            all_actions[:, step] = action
            all_latents[:, step] = latent
            all_values[:, step] = values.squeeze(-1)
            all_log_probs[:, step] = log_probs.squeeze(-1)
            
            # Environment steps
            action_np = action.detach().cpu().numpy()
            next_obs_list = []
            rewards = []
            
            for i, e in enumerate(envs):
                o2, r, d, _ = e.step(action_np[i])
                next_obs_list.append(o2)
                rewards.append(r)
                
                # Store context for next VAE call
                if not getattr(self.config, "disable_vae", False):
                    ctx_obs_tensor[i, step] = obs[i]
                    ctx_act_tensor[i, step] = action[i]
                    ctx_rew_tensor[i, step, 0] = r

            all_rewards[:, step] = torch.tensor(rewards, device=self.device)
            
            # Update observations for next step (except on final step)
            if step < fixed_length - 1:
                obs = torch.as_tensor(np.stack(next_obs_list), dtype=torch.float32, device=self.device)

        # Convert to list of trajectories (expected format for PPO)
        trajs = []
        for i in range(B):
            trajs.append({
                "observations": all_observations[i],      # [fixed_length, ...]
                "actions": all_actions[i],               # [fixed_length, num_assets]
                "latents": all_latents[i],               # [fixed_length, latent_dim]
                "rewards": all_rewards[i],               # [fixed_length]
                "values": all_values[i],                 # [fixed_length]
                "log_probs": all_log_probs[i],           # [fixed_length]
                "dones": [False] * (fixed_length - 1) + [True]  # Only last step is done
            })

        self.total_steps += B * fixed_length
        return trajs

    # ---------------------------------------------------------------------
    # FALLBACK: Variable-length trajectory collection (original approach)
    # ---------------------------------------------------------------------
    def collect_trajectories_batched(self, B: int) -> List[Dict]:
        """Fallback method for variable-length trajectories"""
        B = max(1, int(B))
        envs = [self._clone_env() for _ in range(B)]
        for e in envs:
            e.set_task(e.sample_task())
        obs_np = [e.reset() for e in envs]
        obs = torch.as_tensor(np.stack(obs_np, axis=0), dtype=torch.float32, device=self.device)

        done = np.zeros(B, dtype=bool)
        step = 0

        # Context storage for variable lengths
        max_context_len = 100
        ctx_obs_tensor = torch.zeros(B, max_context_len, *obs.shape[1:], device=self.device)
        ctx_act_tensor = torch.zeros(B, max_context_len, self.config.num_assets, device=self.device)
        ctx_rew_tensor = torch.zeros(B, max_context_len, 1, device=self.device)
        ctx_lengths = torch.zeros(B, dtype=torch.long)

        trajs = [
            {"observations": [], "actions": [], "rewards": [], "values": [], "log_probs": [], "latents": [], "dones": []}
            for _ in range(B)
        ]

        while not np.all(done) and step < self.config.max_horizon:
            if getattr(self.config, "disable_vae", False):
                latent = torch.zeros(B, self.config.latent_dim, device=self.device)
            elif step == 0:
                latent = torch.zeros(B, self.config.latent_dim, device=self.device)
            else:
                latent = self._batch_vae_encode(ctx_obs_tensor, ctx_act_tensor, ctx_rew_tensor, ctx_lengths, done)

            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                action, _ = self.policy.act(obs, latent, deterministic=False)
                values, log_probs, _ = self.policy.evaluate_actions(obs, latent, action)

            action_np = action.detach().cpu().numpy()
            next_obs_list = []
            
            for i, e in enumerate(envs):
                if done[i]:
                    next_obs_list.append(obs_np[i])
                    continue

                o2, r, d, _ = e.step(action_np[i])
                
                trajs[i]["observations"].append(obs[i].detach().cpu())
                trajs[i]["actions"].append(action[i].detach().cpu())
                trajs[i]["latents"].append(latent[i].detach().cpu())
                trajs[i]["rewards"].append(float(r))
                trajs[i]["values"].append(values[i].detach().cpu())
                trajs[i]["log_probs"].append(log_probs[i].detach().cpu())
                trajs[i]["dones"].append(bool(d))

                if not getattr(self.config, "disable_vae", False) and ctx_lengths[i] < max_context_len:
                    idx = ctx_lengths[i]
                    ctx_obs_tensor[i, idx] = obs[i]
                    ctx_act_tensor[i, idx] = action[i]
                    ctx_rew_tensor[i, idx, 0] = r
                    ctx_lengths[i] += 1

                done[i] = d
                next_obs_list.append(o2)

            obs_np = next_obs_list
            obs = torch.as_tensor(np.stack(obs_np, axis=0), dtype=torch.float32, device=self.device)
            step += 1
            self.total_steps += int(np.sum(~done))

        # Stack to device per env
        for i in range(B):
            if len(trajs[i]["rewards"]) == 0:
                trajs[i] = self._create_empty_trajectory()
                continue

            for k in ["observations", "actions", "values", "log_probs", "latents"]:
                trajs[i][k] = torch.stack(trajs[i][k]).to(self.device)
            trajs[i]["rewards"] = torch.tensor(trajs[i]["rewards"], dtype=torch.float32, device=self.device)

        return trajs

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
        return {
            "observations": torch.zeros(empty_obs_shape, dtype=torch.float32, device=self.device),
            "actions": torch.zeros((1, self.config.num_assets), dtype=torch.float32, device=self.device),
            "values": torch.zeros(1, dtype=torch.float32, device=self.device),
            "log_probs": torch.zeros(1, dtype=torch.float32, device=self.device),
            "latents": torch.zeros((1, self.config.latent_dim), dtype=torch.float32, device=self.device),
            "rewards": torch.zeros(1, dtype=torch.float32, device=self.device),
            "dones": [True]
        }

    # ---------------------------------------------------------------------
    # PPO / VAE updates (unchanged)
    # ---------------------------------------------------------------------
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

    def update_policy(self):
        all_traj = self.experience_buffer.get_all()
        if not all_traj:
            return 0.0

        b_obs, b_act, b_lat, b_adv, b_ret, b_logp_old = [], [], [], [], [], []
        for tr in all_traj:
            adv, ret = self.compute_advantages(tr)
            b_obs.append(tr["observations"])
            b_act.append(tr["actions"])
            b_lat.append(tr["latents"])
            b_adv.append(adv)
            b_ret.append(ret)
            b_logp_old.append(tr["log_probs"])

        batch_obs = torch.cat(b_obs, dim=0)
        batch_actions = torch.cat(b_act, dim=0)
        batch_latents = torch.cat(b_lat, dim=0)
        batch_adv = torch.cat(b_adv, dim=0)
        batch_ret = torch.cat(b_ret, dim=0)
        batch_logp_old = torch.cat(b_logp_old, dim=0)

        total_loss = 0.0
        for _ in range(self.config.ppo_epochs):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                values, log_probs, entropy = self.policy.evaluate_actions(batch_obs, batch_latents, batch_actions)
                values = values.squeeze(-1)
                log_probs = log_probs.squeeze(-1)
                entropy = entropy.mean()

                ratio = torch.exp(log_probs - batch_logp_old)
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1 - self.config.ppo_clip_ratio, 1 + self.config.ppo_clip_ratio) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, batch_ret)
                loss = policy_loss + self.config.value_loss_coef * value_loss - self.config.entropy_coef * entropy

            self.policy_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
            self.policy_optimizer.step()
            total_loss += float(loss.item())

        return total_loss / max(self.config.ppo_epochs, 1)

    def update_vae(self):
        if len(self.vae_buffer) < self.config.vae_batch_size:
            return 0.0

        indices = np.random.choice(len(self.vae_buffer), self.config.vae_batch_size, replace=False)
        batch_traj = [self.vae_buffer[i] for i in indices]
        total_loss_value = 0.0
        loss_count = 0
        self.vae_optimizer.zero_grad()

        for tr in batch_traj:
            seq_len = len(tr["rewards"])
            if seq_len < 2:
                continue

            max_t = min(seq_len - 1, 20)
            t = np.random.randint(1, max_t + 1)

            obs_ctx = tr["observations"][:t].detach().clone().unsqueeze(0)
            act_ctx = tr["actions"][:t].detach().clone().unsqueeze(0)
            rew_ctx = tr["rewards"][:t].detach().clone().unsqueeze(0).unsqueeze(-1)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                vae_loss, _ = self.vae.compute_loss(obs_ctx, act_ctx, rew_ctx, beta=self.config.vae_beta, context_len=t)

            vae_loss.backward(retain_graph=False)
            total_loss_value += float(vae_loss.item())
            loss_count += 1

        if loss_count == 0:
            return 0.0

        torch.nn.utils.clip_grad_norm_(self.vae.parameters(), self.config.max_grad_norm)
        self.vae_optimizer.step()
        return total_loss_value / loss_count

    # ---------------------------------------------------------------------
    # Checkpoint helpers (unchanged)
    # ---------------------------------------------------------------------
    def get_state(self):
        return {
            "episode_count": self.episode_count,
            "total_steps": self.total_steps,
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "vae_optimizer": self.vae_optimizer.state_dict(),
        }

    def load_state(self, state):
        self.episode_count = state.get("episode_count", 0)
        self.total_steps = state.get("total_steps", 0)
        if "policy_optimizer" in state:
            self.policy_optimizer.load_state_dict(state["policy_optimizer"])
        if "vae_optimizer" in state:
            self.vae_optimizer.load_state_dict(state["vae_optimizer"])

    def _get_latent_for_step(self, obs_tensor, trajectory_context):
        if getattr(self.config, "disable_vae", False):
            return torch.zeros(1, self.config.latent_dim, device=self.device)
        if len(trajectory_context["observations"]) == 0:
            return torch.zeros(1, self.config.latent_dim, device=self.device)
        obs_seq = torch.stack(trajectory_context["observations"]).unsqueeze(0)
        act_seq = torch.stack(trajectory_context["actions"]).unsqueeze(0)
        rew_seq = torch.stack(trajectory_context["rewards"]).unsqueeze(0).unsqueeze(-1)
        mu, logvar, _ = self.vae.encode(obs_seq, act_seq, rew_seq)
        return self.vae.reparameterize(mu, logvar)


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