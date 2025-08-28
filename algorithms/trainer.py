# trainer.py
import logging
from collections import deque
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

logger = logging.getLogger(__name__)


class PPOTrainer:
    """
    PPO Trainer for VariBAD Portfolio Optimization.
    Handles both policy updates and VAE training.
    """

    def __init__(self, env, policy, vae, config):
        """
        NOTE: `logger` and `csv_logger` args are kept only for backward compatibility.
        They are ignored in this refactor to keep the trainer decoupled from I/O.
        """
        self.env = env
        self.policy = policy
        self.vae = vae
        self.config = config

        # Training state
        self.episode_count = 0
        self.total_steps = 0
        self.device = torch.device(config.device)

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

    # ----------------------------
    # Public training entry point
    # ----------------------------
    def train_episode(self):
        """Train for one episode and return a rich dict of episode metrics."""
        # 1) Collect on-policy trajectory (with online latent)
        trajectory = self.collect_trajectory()

        # 2) Add to buffers
        self.vae_buffer.append(trajectory)
        self.experience_buffer.add_trajectory(trajectory)

        # 3) Updates
        policy_loss = 0.0
        vae_loss = 0.0

        if self.experience_buffer.is_ready():
            policy_loss = float(self.update_policy())
            self.experience_buffer.clear()

        if (not getattr(self.config, 'disable_vae', False) and 
            self.episode_count % self.config.vae_update_freq == 0 and 
            len(self.vae_buffer) >= self.config.vae_batch_size):
            vae_loss = float(self.update_vae())

        # 4) Episode-level metrics
        episode_reward = sum(trajectory["rewards"])
        if torch.is_tensor(episode_reward):
            episode_reward = episode_reward.item()

        episode_length = len(trajectory["rewards"])
        initial_capital = self.env.initial_capital
        final_capital = self.env.current_capital
        cumulative_return = (final_capital - initial_capital) / initial_capital

        # Portfolio metrics computed from env episode trajectory
        portfolio_allocations = []
        significant_changes = 0
        for i, step in enumerate(getattr(self.env, "episode_trajectory", [])):
            if "action" in step:
                portfolio_allocations.append(step["action"])
                if i > 0 and "action" in self.env.episode_trajectory[i - 1]:
                    prev = self.env.episode_trajectory[i - 1]["action"]
                    change = np.linalg.norm(np.array(step["action"]) - np.array(prev))
                    if change > 0.1:
                        significant_changes += 1

        if len(portfolio_allocations) > 0:
            allocations_np = np.array(portfolio_allocations)
            avg_allocation = np.mean(allocations_np, axis=0)
            allocation_variance = float(np.var(allocations_np, axis=0).mean())
            portfolio_concentration = float(np.max(avg_allocation))
            portfolio_turnover = float(significant_changes / episode_length) if episode_length > 0 else 0.0
        else:
            avg_allocation = np.zeros(self.config.num_assets)
            allocation_variance = 0.0
            portfolio_concentration = 0.0
            portfolio_turnover = 0.0

        # Risk metrics from capital history
        if hasattr(self.env, "capital_history") and len(self.env.capital_history) > 1:
            capital_history = np.array(self.env.capital_history)
            returns_series = np.diff(capital_history) / capital_history[:-1]

            running_max = np.maximum.accumulate(capital_history)
            drawdown = (capital_history - running_max) / running_max
            max_drawdown = float(np.min(drawdown))

            returns_volatility = float(np.std(returns_series)) if len(returns_series) > 1 else 0.0
            sharpe_ratio = float(episode_reward)  # your reward is Sharpe per step aggregated as episode sum/mean
        else:
            max_drawdown = 0.0
            returns_volatility = 0.0
            sharpe_ratio = float(episode_reward)

        # Update rolling stats
        self.episode_rewards.append(float(episode_reward))
        if policy_loss > 0:
            self.policy_losses.append(float(policy_loss))
        if vae_loss > 0:
            self.vae_losses.append(float(vae_loss))

        self.episode_count += 1

        # 5) Pack episode details (returned to caller; caller logs via RunLogger)
        episode_details = {
            "episode_reward": float(episode_reward),
            "policy_loss": float(policy_loss),
            "vae_loss": float(vae_loss),
            "episode_length": int(episode_length),
            "initial_capital": float(initial_capital),
            "final_capital": float(final_capital),
            "cumulative_return": float(cumulative_return),
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": float(max_drawdown),
            "returns_volatility": float(returns_volatility),
            "portfolio_concentration": float(portfolio_concentration),
            "portfolio_turnover": float(portfolio_turnover),
            "allocation_variance": float(allocation_variance),
            "num_trades": int(significant_changes),
            "avg_cash_position": float(
                1.0 - np.mean([np.sum(a) for a in portfolio_allocations])
            ) if len(portfolio_allocations) > 0 else 0.0,
            "task_id": getattr(self.env, "task_id", None),
            "total_steps": int(self.total_steps),
        }

        # Optional latent magnitude stats (kept for analysis; not required)
        if hasattr(self, "latent_magnitudes") and self.latent_magnitudes:
            lm = list(self.latent_magnitudes)
            episode_details.update(
                {
                    "avg_latent_magnitude": float(np.mean(lm)),
                    "std_latent_magnitude": float(np.std(lm)),
                    "max_latent_magnitude": float(np.max(lm)),
                }
            )

        return episode_details

    # ----------------------------
    # Trajectory collection
    # ----------------------------
    def collect_trajectory(self):
        """Collect a single episode trajectory with online latent updates."""
        traj = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "values": [],
            "log_probs": [],
            "latents": [],
            "dones": [],
        }

        # Reset env and prepare tensors
        obs = self.env.reset()
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Context τ:t for VAE encoder
        context = {"observations": [], "actions": [], "rewards": []}

        done = False
        step = 0

        while not done and step < self.config.max_horizon:
            # Encode context to get latent
            latent = self._get_latent_for_step(obs_tensor, context)

            # Track latent norms (for optional analysis)
            if not hasattr(self, "latent_magnitudes"):
                self.latent_magnitudes = deque(maxlen=1000)
            self.latent_magnitudes.append(torch.norm(latent).item())

            # Policy step
            with torch.no_grad():
                action, value = self.policy.act(obs_tensor, latent, deterministic=False)
                _, log_prob, _ = self.policy.evaluate_actions(obs_tensor, latent, action)

            # Env step (requires numpy)
            action_cpu = action.squeeze(0).detach().cpu().numpy()
            next_obs, reward, done, info = self.env.step(action_cpu)

            # Store transition (CPU where appropriate)
            traj["observations"].append(obs_tensor.squeeze(0).cpu())
            traj["actions"].append(action.squeeze(0).cpu())
            traj["rewards"].append(float(reward))
            traj["values"].append(value.squeeze().cpu())
            traj["log_probs"].append(log_prob.squeeze().cpu())
            traj["latents"].append(latent.squeeze(0).cpu())
            traj["dones"].append(bool(done))

            # Update context for VAE (keep on device)
            context["observations"].append(obs_tensor.squeeze(0).detach())
            context["actions"].append(action.squeeze(0).detach())
            context["rewards"].append(torch.tensor(reward, device=self.device))

            # Next step
            if not done:
                obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)

            step += 1
            self.total_steps += 1

        # Stack to tensors on device
        traj["observations"] = torch.stack(traj["observations"]).to(self.device)  # (T, N, F)
        traj["actions"] = torch.stack(traj["actions"]).to(self.device)            # (T, N)
        traj["values"] = torch.stack(traj["values"]).to(self.device)              # (T,)
        traj["log_probs"] = torch.stack(traj["log_probs"]).to(self.device)        # (T,)
        traj["latents"] = torch.stack(traj["latents"]).to(self.device)            # (T, latent_dim)
        traj["rewards"] = torch.tensor(traj["rewards"], dtype=torch.float32, device=self.device)  # (T,)
        # dones remain list[bool] for GAE logic
        return traj

    # ----------------------------
    # PPO / VAE updates
    # ----------------------------
    def compute_advantages(self, trajectory):
        """Compute GAE advantages."""
        rewards = trajectory["rewards"]  # (T,)
        values = trajectory["values"]    # (T,)
        dones = trajectory["dones"]

        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        gae = 0.0
        next_value = 0.0  # terminal value = 0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.config.discount_factor * next_value * (1 - int(dones[t])) - values[t]
            gae = delta + self.config.discount_factor * self.config.gae_lambda * (1 - int(dones[t])) * gae
            advantages[t] = gae
            returns[t] = gae + values[t]
            next_value = values[t]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def update_policy(self):
        """Update policy using PPO."""
        all_traj = self.experience_buffer.get_all()
        total_loss = 0.0

        # Build batch
        b_obs, b_act, b_lat, b_adv, b_ret, b_logp_old = [], [], [], [], [], []
        for tr in all_traj:
            adv, ret = self.compute_advantages(tr)
            b_obs.append(tr["observations"])
            b_act.append(tr["actions"])
            b_lat.append(tr["latents"])
            b_adv.append(adv)
            b_ret.append(ret)
            b_logp_old.append(tr["log_probs"])

        batch_obs = torch.cat(b_obs, dim=0)         # (B, N, F)
        batch_actions = torch.cat(b_act, dim=0)     # (B, N)
        batch_latents = torch.cat(b_lat, dim=0)     # (B, latent_dim)
        batch_adv = torch.cat(b_adv, dim=0)         # (B,)
        batch_ret = torch.cat(b_ret, dim=0)         # (B,)
        batch_logp_old = torch.cat(b_logp_old, dim=0)  # (B,)

        # PPO epochs
        for _ in range(self.config.ppo_epochs):
            values, log_probs, entropy = self.policy.evaluate_actions(batch_obs, batch_latents, batch_actions)
            values = values.squeeze(-1)        # (B,)
            log_probs = log_probs.squeeze(-1)  # (B,)
            entropy = entropy.mean()           # scalar

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
        """Update VAE using random trajectory prefixes."""
        if len(self.vae_buffer) < self.config.vae_batch_size:
            return 0.0

        indices = np.random.choice(len(self.vae_buffer), self.config.vae_batch_size, replace=False)
        batch_traj = [self.vae_buffer[i] for i in indices]

        total_loss = 0.0
        loss_count = 0

        for tr in batch_traj:
            seq_len = len(tr["rewards"])
            if seq_len < 2:
                continue

            max_t = min(seq_len - 1, 20)  # cap for efficiency
            t = np.random.randint(1, max_t + 1)

            obs_ctx = tr["observations"][:t].unsqueeze(0)        # (1, t, N, F)
            act_ctx = tr["actions"][:t].unsqueeze(0)             # (1, t, N)
            rew_ctx = tr["rewards"][:t].unsqueeze(0).unsqueeze(-1)  # (1, t, 1)

            # Full trajectory (if your VAE decoder needs it)
            # obs_full = tr["observations"].unsqueeze(0)
            # act_full = tr["actions"].unsqueeze(0)
            # rew_full = tr["rewards"].unsqueeze(0).unsqueeze(-1)

            vae_loss, _ = self.vae.compute_loss(
                obs_ctx, act_ctx, rew_ctx, beta=self.config.vae_beta, context_len=t
            )
            total_loss += float(vae_loss)
            loss_count += 1

        if loss_count == 0:
            return 0.0

        avg_loss = total_loss / loss_count
        self.vae_optimizer.zero_grad()
        # If compute_loss returns a tensor on device, backprop it:
        torch.tensor(avg_loss, device=self.device, dtype=torch.float32).backward()
        torch.nn.utils.clip_grad_norm_(self.vae.parameters(), self.config.max_grad_norm)
        self.vae_optimizer.step()
        return avg_loss

    # ----------------------------
    # Checkpoint helpers
    # ----------------------------
    def get_state(self):
        """Get trainer state for checkpointing."""
        return {
            "episode_count": self.episode_count,
            "total_steps": self.total_steps,
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "vae_optimizer": self.vae_optimizer.state_dict(),
        }

    def load_state(self, state):
        """Load trainer state from checkpoint."""
        self.episode_count = state.get("episode_count", 0)
        self.total_steps = state.get("total_steps", 0)
        if "policy_optimizer" in state:
            self.policy_optimizer.load_state_dict(state["policy_optimizer"])
        if "vae_optimizer" in state:
            self.vae_optimizer.load_state_dict(state["vae_optimizer"])
    
    def _get_latent_for_step(self, obs_tensor, trajectory_context):
        """Get latent embedding - supports VAE ablation"""
        if getattr(self.config, 'disable_vae', False):
            # Ablation: use zero latent instead of VAE
            return torch.zeros(1, self.config.latent_dim, device=self.device)
        
        # Normal VAE path (existing logic)
        if len(trajectory_context['observations']) == 0:
            return torch.zeros(1, self.config.latent_dim, device=self.device)
        else:
            obs_seq = torch.stack(trajectory_context['observations']).unsqueeze(0)
            act_seq = torch.stack(trajectory_context['actions']).unsqueeze(0)  
            rew_seq = torch.stack(trajectory_context['rewards']).unsqueeze(0).unsqueeze(-1)
            mu, logvar, _ = self.vae.encode(obs_seq, act_seq, rew_seq)
            return self.vae.reparameterize(mu, logvar)

# ----------------------------
    # PHASE 3: Early stopping extension (add to end of PPOTrainer class)
    # ----------------------------
    
    def __init_early_stopping(self):
        """Initialize early stopping state (call this once)"""
        if not hasattr(self, '_early_stopping_initialized'):
            self.validation_scores = []
            self.best_val_score = float('-inf')
            self.patience_counter = 0
            self.early_stopped = False
            
            # Get config params (with defaults)
            self.es_patience = getattr(self.config, 'early_stopping_patience', 5)
            self.es_min_delta = getattr(self.config, 'early_stopping_min_delta', 0.01)
            self.es_min_episodes = getattr(self.config, 'min_episodes_before_stopping', 1000)
            
            self._early_stopping_initialized = True
    
    def add_validation_score(self, score: float) -> bool:
        """
        Add validation score and check if should stop early.
        
        Args:
            score: Current validation score (higher is better)
            
        Returns:
            bool: True if training should stop, False otherwise
        """
        self.__init_early_stopping()
        
        # Don't stop too early
        if self.episode_count < self.es_min_episodes:
            self.validation_scores.append(score)
            return False
        
        self.validation_scores.append(score)
        
        # Check for improvement
        if score > self.best_val_score + self.es_min_delta:
            # Significant improvement
            self.best_val_score = score
            self.patience_counter = 0
            logger.info(f"New best validation: {self.best_val_score:.4f}")
            return False
        else:
            # No improvement
            self.patience_counter += 1
            logger.info(f"No improvement. Patience: {self.patience_counter}/{self.es_patience}")
            
            if self.patience_counter >= self.es_patience:
                logger.info(f"Early stopping at episode {self.episode_count}")
                self.early_stopped = True
                return True
            
            return False
    
    def should_stop_early(self) -> bool:
        """Check if training should stop early"""
        self.__init_early_stopping()
        return self.early_stopped
    
    def get_early_stopping_state(self) -> dict:
        """Get early stopping state for checkpointing"""
        self.__init_early_stopping()
        return {
            'validation_scores': self.validation_scores,
            'best_val_score': self.best_val_score,
            'patience_counter': self.patience_counter,
            'early_stopped': self.early_stopped
        }

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
