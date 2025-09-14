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

        # Optimizers
        params = list(policy.parameters())
        if vae is not None and not config.disable_vae:
            params += list(vae.parameters())
        self.optimizer = Adam([
            {"params": policy.parameters(), "lr": config.policy_lr},
            {"params": vae.parameters(), "lr": config.vae_lr},
        ])
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
        Train for one (or many) episode(s) with performance diagnostics.
        Enhanced training with comprehensive step-level tracking
        """
        diag = PerformanceDiagnostic()
        
        # === NEW: Initialize episode tracking ===
        episode_data = {
            'step_rewards': [],
            'step_capital': [],
            'step_weights': [],
            'step_returns': [],
            'step_excess_returns': [],
            'step_dsr_alpha': [],
            'step_dsr_beta': [],
            'step_transaction_costs': [],
            'step_concentrations': [],
            'step_active_positions': [],
            'step_cash_positions': [],
            'step_turnovers': [],
            'step_long_exposures': [],
            'step_short_exposures': [],
            'step_net_exposures': [],
            'step_gross_exposures': [],
            'step_rel_excess_returns': [],
            'final_capital': 0.0,
            'num_episodes': 0
        }

    
        with diag.time_section("collect_single_trajectory"):
            tr = self.collect_trajectory()
            self.vae_buffer.append(tr)
            self.experience_buffer.add_trajectory(tr)
        episode_reward_mean = float(sum(tr["rewards"]))
        
        # === Extract step-level data from single trajectory ===
        episode_data['num_episodes'] = 1
        if "step_info_list" in tr:
            for step_info in tr["step_info_list"]:
                episode_data['step_rewards'].append(step_info.get('sharpe_reward', 0.0))
                episode_data['step_capital'].append(step_info.get('capital', 0.0))
                episode_data['step_weights'].append(step_info.get('weights', []))
                episode_data['step_returns'].append(step_info.get('log_return', 0.0))
                episode_data['step_excess_returns'].append(step_info.get('excess_log_return', 0.0))
                episode_data['step_dsr_alpha'].append(step_info.get('dsr_alpha', 0.0))
                episode_data['step_dsr_beta'].append(step_info.get('dsr_beta', 0.0))
                episode_data['step_transaction_costs'].append(step_info.get('transaction_cost', 0.0))
                episode_data['step_concentrations'].append(step_info.get('portfolio_concentration', 0.0))
                episode_data['step_active_positions'].append(step_info.get('num_active_positions', 0))
                episode_data['step_cash_positions'].append(step_info.get('cash_pct', 0.0))
                episode_data['step_turnovers'].append(step_info.get('turnover', 0.0))
                episode_data['step_long_exposures'].append(step_info.get('weights_long', 0.0))
                episode_data['step_short_exposures'].append(step_info.get('weights_short', 0.0))
                episode_data['step_net_exposures'].append(step_info.get('net_exposure', 0.0))
                episode_data['step_gross_exposures'].append(step_info.get('gross_exposure', 0.0))
                episode_data['step_rel_excess_returns'].append(step_info.get('relative_excess_log_return', 0.0))

        
        # Get final capital
        if len(episode_data['step_capital']) > 0:
            episode_data['final_capital'] = episode_data['step_capital'][-1]

        # Updates
        policy_loss = 0.0
        vae_loss = 0.0
        vae_loss_components = {}  # NEW: Store VAE loss components
        
        with diag.time_section("updating joint objective"):
            update_info = self.update_joint(tr)   # update right away with the current trajectory


        # --- extract update info (if any) ---
        policy_loss = update_info.get("policy_loss", 0.0)
        vae_loss = update_info.get("vae_loss", 0.0)
        vae_loss_components = {k.replace("vae_", ""): v for k, v in update_info.items() if k.startswith("vae_")}

        self.episode_rewards.append(float(episode_reward_mean))
        if policy_loss > 0:
            self.policy_losses.append(float(policy_loss))
        if vae_loss > 0:
            self.vae_losses.append(float(vae_loss))

        # Report timing every 50 episodes
        if self.episode_count % 50 == 0:
            diag.report()

        self.episode_count += 1

        # === NEW: Enhanced results with comprehensive tracking ===
        results = {
            # Existing core metrics
            "episode_reward": float(episode_reward_mean),
            "policy_loss": float(policy_loss),
            "vae_loss": float(vae_loss),
            "total_steps": int(self.total_steps),

            # === NEW: Episode-level portfolio aggregates ===
            "episode_final_capital": float(episode_data['final_capital']),
            "episode_total_return": float(sum(episode_data['step_returns'])) if episode_data['step_returns'] else 0.0,
            "episode_total_excess_return": float(sum(episode_data['step_excess_returns'])) if episode_data['step_excess_returns'] else 0.0,
            "episode_avg_concentration": float(np.mean(episode_data['step_concentrations'])) if episode_data['step_concentrations'] else 0.0,
            "episode_max_concentration": float(np.max(episode_data['step_concentrations'])) if episode_data['step_concentrations'] else 0.0,
            "episode_avg_active_positions": float(np.mean(episode_data['step_active_positions'])) if episode_data['step_active_positions'] else 0.0,
            "episode_avg_cash_position": float(np.mean(episode_data['step_cash_positions'])) if episode_data['step_cash_positions'] else 0.0,
            "episode_total_transaction_costs": float(sum(episode_data['step_transaction_costs'])) if episode_data['step_transaction_costs'] else 0.0,
            "episode_volatility": float(np.std(episode_data['step_returns'])) if len(episode_data['step_returns']) > 1 else 0.0,
            "episode_excess_volatility": float(np.std(episode_data['step_excess_returns'])) if len(episode_data['step_excess_returns']) > 1 else 0.0,

            # === NEW: aggregated info metrics ===
            "episode_avg_reward": float(np.mean(episode_data['step_rewards'])) if episode_data['step_rewards'] else 0.0,
            "episode_sum_reward": float(np.sum(episode_data['step_rewards'])) if episode_data['step_rewards'] else 0.0,
            "episode_avg_long_exposure": float(np.mean(episode_data['step_long_exposures'])) if episode_data['step_long_exposures'] else 0.0,
            "episode_avg_short_exposure": float(np.mean(episode_data['step_short_exposures'])) if episode_data['step_short_exposures'] else 0.0,
            "episode_avg_net_exposure": float(np.mean(episode_data['step_net_exposures'])) if episode_data['step_net_exposures'] else 0.0,
            "episode_avg_gross_exposure": float(np.mean(episode_data['step_gross_exposures'])) if episode_data['step_gross_exposures'] else 0.0,
            "episode_max_active_positions": float(np.max(episode_data['step_active_positions'])) if episode_data['step_active_positions'] else 0.0,
            "episode_sum_transaction_costs": float(np.sum(episode_data['step_transaction_costs'])) if episode_data['step_transaction_costs'] else 0.0,
            "episode_avg_turnover": float(np.mean(episode_data['step_turnovers'])) if episode_data['step_turnovers'] else 0.0,
            "episode_sum_rel_excess_return": float(np.sum(episode_data['step_rel_excess_returns'])) if episode_data['step_rel_excess_returns'] else 0.0,

            # === NEW: DSR tracking ===
            "episode_final_dsr_alpha": float(episode_data['step_dsr_alpha'][-1]) if episode_data['step_dsr_alpha'] else 0.0,
            "episode_final_dsr_beta": float(episode_data['step_dsr_beta'][-1]) if episode_data['step_dsr_beta'] else 0.0,
            "episode_dsr_variance": float(max(episode_data['step_dsr_beta'][-1] - episode_data['step_dsr_alpha'][-1]**2, 1e-8)) if episode_data['step_dsr_beta'] and episode_data['step_dsr_alpha'] else 0.0,

            # === NEW: Portfolio composition tracking ===
            "episode_long_exposure": float(np.mean([np.sum(np.maximum(w, 0)) for w in episode_data['step_weights']])) if episode_data['step_weights'] else 0.0,
            "episode_short_exposure": float(np.mean([np.sum(np.abs(np.minimum(w, 0))) for w in episode_data['step_weights']])) if episode_data['step_weights'] else 0.0,
            "episode_net_exposure": float(np.mean([np.sum(w) for w in episode_data['step_weights']])) if episode_data['step_weights'] else 0.0,
            "episode_gross_exposure": float(np.mean([np.sum(np.abs(w)) for w in episode_data['step_weights']])) if episode_data['step_weights'] else 0.0,


            # === NEW: VAE loss components (when available) ===
            **{f"vae_{k}": float(v) for k, v in vae_loss_components.items()},

            # === NEW: Rolling statistics ===
            "rolling_avg_episode_reward": float(np.mean(list(self.episode_rewards))) if self.episode_rewards else 0.0,
            "rolling_std_episode_reward": float(np.std(list(self.episode_rewards))) if len(self.episode_rewards) > 1 else 0.0,
            "rolling_avg_policy_loss": float(np.mean(list(self.policy_losses))) if self.policy_losses else 0.0,
            "rolling_avg_vae_loss": float(np.mean(list(self.vae_losses))) if self.vae_losses else 0.0,

            # === NEW: Step-by-step data for artifact logging ===
            "step_data": episode_data,

            # === NEW: Training diagnostics ===
            "num_episodes_in_batch": int(episode_data['num_episodes']),
            "episode_count": int(self.episode_count),
            "steps_per_episode": float(len(episode_data['step_returns'])) if episode_data['step_returns'] else 0.0,

        }

        return results

    # ---------------------------------------------------------------------
    # Trajectory collection (single)
    # ---------------------------------------------------------------------
    def collect_trajectory(self):
        traj = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "values": [],
            "log_probs": [],
            "latents": [],
            "dones": []
        }

        obs = self.env.reset()
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        context = {"observations": [], "actions": [], "rewards": []}
        done, step = False, 0

        while not done and step < self.config.max_horizon:
            # === Latent context ===
            latent = self._get_latent_for_step(obs_tensor, context)

            # === Sample action from policy ===
            with torch.no_grad():
                actions_raw, value, log_prob = self.policy.act(obs_tensor, latent, deterministic=False)

            # Store before env normalization
            traj["observations"].append(obs_tensor.squeeze(0).cpu())
            traj["actions"].append(actions_raw.squeeze(0).cpu())
            traj["values"].append(value.squeeze(0).cpu())
            traj["log_probs"].append(log_prob.squeeze(0).cpu())
            traj["latents"].append(latent.squeeze(0).cpu())

            # === Step environment ===
            next_obs, reward, done, info = self.env.step(actions_raw.squeeze(0).cpu().numpy())
            traj["rewards"].append(reward)
            traj["dones"].append(done)

            # Update context for VAE
            context["observations"].append(obs_tensor.squeeze(0).detach())
            context["actions"].append(actions_raw.squeeze(0).detach())
            context["rewards"].append(torch.tensor(reward, dtype=torch.float32, device=self.device))

            # Advance
            if not done:
                obs_tensor = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)

            step += 1

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
        OPTIMIZED: Fixed-length trajectories with comprehensive step info collection.
        """
        B = max(1, int(B))
        envs = [self._clone_env() for _ in range(B)]
        for e in envs:
            e.set_task(e.sample_task())
        obs_np = [e.reset() for e in envs]
        obs = torch.as_tensor(np.stack(obs_np, axis=0), dtype=torch.float32, device=self.device)

        # For fixed lengths, we know exactly when all environments will finish
        fixed_length = self.config.min_horizon  # Since min_horizon == max_horizon
        
        # Pre-allocate for exact trajectory length
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
        
        # === NEW: Pre-allocate step info storage ===
        all_step_info = [[{} for _ in range(fixed_length)] for _ in range(B)]

        for step in range(fixed_length):
            # VAE processing
            if getattr(self.config, "disable_vae", False) or step == 0:
                latent = torch.zeros(B, self.config.latent_dim, device=self.device)
            else:
                ctx_len = step
                batch_obs = ctx_obs_tensor[:, :ctx_len]
                batch_acts = ctx_act_tensor[:, :ctx_len]  
                batch_rews = ctx_rew_tensor[:, :ctx_len]
                
                try:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                        mu, logvar, _ = self.vae.encode(batch_obs, batch_acts, batch_rews)
                        latent = self.vae.reparameterize(mu, logvar)
                except Exception as e:
                    print(f"VAE batch encode failed: {e}")
                    latent = torch.zeros(B, self.config.latent_dim, device=self.device)

            # Policy step
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                action, _ = self.policy.act(obs, latent, deterministic=False)
                values, log_probs, _ = self.policy.evaluate_actions(obs, latent, action)

            # Store everything in pre-allocated tensors
            all_observations[:, step] = obs
            all_actions[:, step] = action
            all_latents[:, step] = latent
            all_values[:, step] = values.squeeze(-1)
            all_log_probs[:, step] = log_probs.squeeze(-1)
            
            # Environment steps with step info collection
            action_np = action.detach().cpu().numpy()
            next_obs_list = []
            rewards = []
            
            for i, e in enumerate(envs):
                o2, r, d, info = e.step(action_np[i])
                next_obs_list.append(o2)
                rewards.append(r)
                
                # === NEW: Store comprehensive step info ===
                all_step_info[i][step] = info.copy()
                
                # Store context for next VAE call
                if not getattr(self.config, "disable_vae", False):
                    ctx_obs_tensor[i, step] = obs[i]
                    ctx_act_tensor[i, step] = action[i]
                    ctx_rew_tensor[i, step, 0] = r

            all_rewards[:, step] = torch.tensor(rewards, device=self.device)
            
            # Update observations for next step (except on final step)
            if step < fixed_length - 1:
                obs = torch.as_tensor(np.stack(next_obs_list), dtype=torch.float32, device=self.device)

        # Convert to list of trajectories with step info
        trajs = []
        for i in range(B):
            traj = {
                "observations": all_observations[i],      # [fixed_length, ...]
                "actions": all_actions[i],               # [fixed_length, num_assets]
                "latents": all_latents[i],               # [fixed_length, latent_dim]
                "rewards": all_rewards[i],               # [fixed_length]
                "values": all_values[i],                 # [fixed_length]
                "log_probs": all_log_probs[i],           # [fixed_length]
                "dones": [False] * (fixed_length - 1) + [True]  # Only last step is done
            }
            
            # === NEW: Attach step info list ===
            traj["step_info_list"] = all_step_info[i]
            
            trajs.append(traj)

        self.total_steps += B * fixed_length
        return trajs

    # ---------------------------------------------------------------------
    # FALLBACK: Variable-length trajectory collection (original approach)
    # ---------------------------------------------------------------------
    def collect_trajectories_batched(self, B: int) -> List[Dict]:
        """Fallback method for variable-length trajectories with step info collection"""
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
        
        # === NEW: Initialize step info lists ===
        step_info_lists = [[] for _ in range(B)]

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

                o2, r, d, info = e.step(action_np[i])
                
                trajs[i]["observations"].append(obs[i].detach().cpu())
                trajs[i]["actions"].append(action[i].detach().cpu())
                trajs[i]["latents"].append(latent[i].detach().cpu())
                trajs[i]["rewards"].append(float(r))
                trajs[i]["values"].append(values[i].detach().cpu())
                trajs[i]["log_probs"].append(log_probs[i].detach().cpu())
                trajs[i]["dones"].append(bool(d))
                
                # === NEW: Collect step info ===
                step_info_lists[i].append(info.copy())

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

        # Stack to device per env and attach step info
        for i in range(B):
            if len(trajs[i]["rewards"]) == 0:
                trajs[i] = self._create_empty_trajectory()
                trajs[i].step_info_list = []
                continue

            for k in ["observations", "actions", "values", "log_probs", "latents"]:
                trajs[i][k] = torch.stack(trajs[i][k]).to(self.device)
            trajs[i]["rewards"] = torch.tensor(trajs[i]["rewards"], dtype=torch.float32, device=self.device)
            
            # === NEW: Attach step info list ===
            trajs[i].step_info_list = step_info_lists[i]

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
        References:
        - Schulman et al., 2017 (PPO paper)
        - Spinning Up PPO implementation
        """
        obs = torch.tensor(traj["observations"], dtype=torch.float32, device=self.device)
        actions = torch.tensor(traj["actions"], dtype=torch.long, device=self.device)
        latents = torch.tensor(traj["latents"], dtype=torch.float32, device=self.device)
        rewards = torch.tensor(traj["rewards"], dtype=torch.float32, device=self.device)
        values = torch.tensor(traj["values"], dtype=torch.float32, device=self.device)
        dones = torch.tensor(traj["dones"], dtype=torch.float32, device=self.device)
        old_logp = torch.tensor(traj["log_probs"], dtype=torch.float32, device=self.device)


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