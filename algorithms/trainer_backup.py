# trainer.py
import logging
from collections import deque
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

# Needed to clone environments for batched rollouts
from environments.env import MetaEnv

logger = logging.getLogger(__name__)


class PPOTrainer:
    """
    PPO Trainer for VariBAD Portfolio Optimization.
    Now supports optional batched rollouts (num_envs > 1) to improve GPU utilization.
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
        Train for one (or many) episode(s).
        If num_envs > 1, this collects *num_envs* trajectories in parallel and
        pushes each into PPO/vae buffers.
        """
        if self.num_envs > 1:
            trajectories = self.collect_trajectories_batched(self.num_envs)
            # Add to buffers
            for tr in trajectories:
                self.vae_buffer.append(tr)
                self.experience_buffer.add_trajectory(tr)
            # Produce a compact summary (mean across the batch)
            episode_reward_mean = float(np.mean([float(sum(tr["rewards"])) for tr in trajectories]))
        else:
            tr = self.collect_trajectory()
            self.vae_buffer.append(tr)
            self.experience_buffer.add_trajectory(tr)
            episode_reward_mean = float(sum(tr["rewards"]))

        # Updates
        policy_loss = 0.0
        vae_loss = 0.0

        if self.experience_buffer.is_ready():
            policy_loss = float(self.update_policy())
            self.experience_buffer.clear()

        if (
            not getattr(self.config, "disable_vae", False)
            and self.episode_count % self.config.vae_update_freq == 0
            and len(self.vae_buffer) >= self.config.vae_batch_size
        ):
            vae_loss = float(self.update_vae())

        self.episode_rewards.append(float(episode_reward_mean))
        if policy_loss > 0:
            self.policy_losses.append(float(policy_loss))
        if vae_loss > 0:
            self.vae_losses.append(float(vae_loss))

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
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device, non_blocking=True).unsqueeze(0)

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
                obs_tensor = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device, non_blocking=True).unsqueeze(0)

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
    # Trajectory collection (batched)
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

    def collect_trajectories_batched(self, B: int) -> List[Dict]:
        B = max(1, int(B))
        envs = [self._clone_env() for _ in range(B)]
        for e in envs:
            e.set_task(e.sample_task())
        obs_np = [e.reset() for e in envs]
        obs = torch.as_tensor(np.stack(obs_np, axis=0), dtype=torch.float32, device=self.device)

        done = np.zeros(B, dtype=bool)
        step = 0

        # Per-env context for VAE
        ctx_obs = [[] for _ in range(B)]
        ctx_act = [[] for _ in range(B)]
        ctx_rew = [[] for _ in range(B)]

        # Per-env trajectories
        trajs = [
            {"observations": [], "actions": [], "rewards": [], "values": [], "log_probs": [], "latents": [], "dones": []}
            for _ in range(B)
        ]

        while not np.all(done) and step < self.config.max_horizon:
            # Build latent per env
            latents = []
            for i in range(B):
                if getattr(self.config, "disable_vae", False) or len(ctx_obs[i]) == 0 or done[i]:
                    latents.append(torch.zeros(1, self.config.latent_dim, device=self.device))
                else:
                    o = torch.stack(ctx_obs[i]).unsqueeze(0)
                    a = torch.stack(ctx_act[i]).unsqueeze(0)
                    r = torch.stack(ctx_rew[i]).unsqueeze(0).unsqueeze(-1)
                    mu, logvar, _ = self.vae.encode(o, a, r)
                    latents.append(self.vae.reparameterize(mu, logvar))
            latent = torch.cat(latents, dim=0)  # [B, latent]

            # Policy step (batched)
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
                # Store transition for env i
                trajs[i]["observations"].append(obs[i].detach().cpu())
                trajs[i]["actions"].append(action[i].detach().cpu())
                trajs[i]["latents"].append(latent[i].detach().cpu())
                trajs[i]["rewards"].append(float(r))
                trajs[i]["values"].append(values[i].detach().cpu())
                trajs[i]["log_probs"].append(log_probs[i].detach().cpu())
                trajs[i]["dones"].append(bool(d))

                if not getattr(self.config, "disable_vae", False):
                    ctx_obs[i].append(obs[i].detach())
                    ctx_act[i].append(action[i].detach())
                    ctx_rew[i].append(torch.tensor(r, device=self.device))

                done[i] = d
                next_obs_list.append(o2)

            obs_np = next_obs_list
            obs = torch.as_tensor(np.stack(obs_np, axis=0), dtype=torch.float32, device=self.device)

            step += 1
            self.total_steps += int(np.sum(~done))  # count active steps

        # Stack to device per env
        for i in range(B):
            if len(trajs[i]["rewards"]) == 0:
                # Ensure non-empty trajectories
                trajs[i]["observations"] = torch.zeros((1,) + self.env.reset().shape, dtype=torch.float32, device=self.device)
                trajs[i]["actions"] = torch.zeros((1, self.config.num_assets), dtype=torch.float32, device=self.device)
                trajs[i]["values"] = torch.zeros(1, dtype=torch.float32, device=self.device)
                trajs[i]["log_probs"] = torch.zeros(1, dtype=torch.float32, device=self.device)
                trajs[i]["latents"] = torch.zeros((1, self.config.latent_dim), dtype=torch.float32, device=self.device)
                trajs[i]["rewards"] = torch.zeros(1, dtype=torch.float32, device=self.device)
                trajs[i]["dones"] = [True]
                continue

            for k in ["observations", "actions", "values", "log_probs", "latents"]:
                trajs[i][k] = torch.stack(trajs[i][k]).to(self.device)
            trajs[i]["rewards"] = torch.tensor(trajs[i]["rewards"], dtype=torch.float32, device=self.device)

        return trajs

    # ---------------------------------------------------------------------
    # PPO / VAE updates
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
# Early stopping extension state getters (optional)
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
