# trainer.py - Optimized with fixed-length trajectory batching
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

# Needed to clone environments for batched rollouts
from environments.env import MetaEnv

def log_memory(label):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"[{label}] GPU: {allocated:.2f}GB alloc, {reserved:.2f}GB reserved")

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
    
    logger.info(f"Found {len(tensors)} GPU tensors, total {total_mb:.2f}MB")
    logger.info("Top 10 largest:")
    for i, (typ, size, mb) in enumerate(tensors[:10]):
        logger.info(f"  {i+1}. {typ} {size} = {mb:.2f}MB")


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

        # Track VAE enablement
        self.vae_enabled = (vae is not None) and (not getattr(config, "disable_vae", False))

        # Optimizer: include VAE params only if enabled/present
        param_groups = [
            {"params": policy.parameters(), "lr": config.policy_lr}
        ]
        if self.vae_enabled:
            param_groups.append({"params": vae.parameters(), "lr": config.vae_lr})
        self.optimizer = Adam(param_groups)
        # Experience buffers
        self.experience_buffer = ExperienceBuffer(config.batch_size)  # for PPO
        self.vae_buffer = deque(maxlen=10)  # TODO recent trajectories for VAE

        # Rolling stats (store Python floats to avoid CUDA logging issues)
        self.policy_losses = deque(maxlen=100)
        self.vae_losses = deque(maxlen=100)
        self.episode_rewards = deque(maxlen=100)

        # Extra tracking (kept for potential analysis hooks)
        self.episode_details = []
        self.training_start_time = time.time()
        self.episode_start_time = None
        self.portfolio_metrics_history = []

        # BAMDP tracking
        self.current_task_id = None
        self.persistent_context = None
        self.task_count = 0
    
    logger.info("PPOTrainer initialized with BAMDP multi-episode support")

    # CORRECTED TASK TRAINING

    # def train_on_task(self) -> Dict[str, float]:
    #     """
    #     Train over multiple episodes on same task (BAMDP).
    #     Context persists across episodes to enable belief refinement.
    #     """
    #     import gc
    #     log_memory("Task start")
    #     # Sample task once
    #     task = self.env.sample_task()
    #     task_id = task.get("task_id", self.task_count)
    #     self.env.set_task(task)
    #     self.task_count += 1
        
    #     # Initialize context for this task
    #     persistent_context = {
    #         "observations": [],
    #         "actions": [],
    #         "rewards": []
    #     }
        
    #     # Accumulate all transitions across episodes
    #     all_transitions = {
    #         "observations": [],
    #         "actions": [],
    #         "rewards": [],
    #         "values": [],
    #         "log_probs": [],
    #         "latents": [],
    #         "dones": []
    #     }
        
    #     episode_rewards = []
    #     context_sizes = []
    #     latent_norms = []
        
    #     #logger.info(f"=== Starting BAMDP Task {task_id} ({self.config.episodes_per_task} episodes) ===")
        
    #     # Multiple episodes on same task
    #     for episode_idx in range(self.config.episodes_per_task):
    #         #logger.info(f"  Episode {episode_idx+1}/{self.config.episodes_per_task} - Context size: {len(persistent_context['observations'])}")
            
    #         # Collect trajectory with persistent context
    #         log_memory(f"Episode {episode_idx} start")
    #         trajectory = self.collect_trajectory_with_context(persistent_context)
    #         log_memory(f"Episode {episode_idx} collected")
            
    #         # Track metrics
    #         episode_reward = float(trajectory["rewards"].sum().item())
    #         episode_rewards.append(episode_reward)
            
    #         # Validation check 1: Context grows across episodes
    #         context_size_after = len(persistent_context['observations'])
    #         context_sizes.append(context_size_after)
    #         #logger.info(f"    Episode reward: {episode_reward:.4f}")
    #         #logger.info(f"    Context grew: {context_sizes[-1] - (context_sizes[-2] if len(context_sizes) > 1 else 0)} steps")
            
    #         # Validation check 2: Latent beliefs change across episodes
    #         if len(trajectory["latents"]) > 0:
    #             avg_latent_norm = trajectory["latents"].norm(dim=1).mean().item()
    #             latent_norms.append(avg_latent_norm)
    #             #logger.info(f"    Avg latent norm: {avg_latent_norm:.4f}")
    #             #if episode_idx > 0:
    #                 #logger.info(f"    Latent change: {abs(latent_norms[-1] - latent_norms[-2]):.4f}")
            
    #         # Accumulate transitions (all are tensors)
    #         for key in all_transitions.keys():
    #             if key in trajectory:
    #                 all_transitions[key].append(trajectory[key])
            
    #         # Add to episode trajectory buffer for VAE
    #         self.vae_buffer.append(trajectory)
    #         # 🔥 NEW: Delete trajectory after copying to all_transitions
    #         del trajectory

    #     # Stack accumulated transitions from all episodes
    #     for key in all_transitions.keys():
    #         all_transitions[key] = torch.cat(all_transitions[key], dim=0)

    #     # Validation check 3: Single update per task
    #     #logger.info(f"  Performing single PPO+VAE update on full BAMDP trajectory ({all_transitions['rewards'].shape[0]} steps)")
        
    #     # Single update on entire BAMDP trajectory
    #     log_memory("Before update")
    #     total_loss, update_info = self.update_ppo_and_vae(all_transitions)
    #     log_memory("After update")
    #     # 🔥 NEW: Aggressive cleanup
    #     del all_transitions
    #     del persistent_context
    #     if torch.cuda.is_available():
    #         torch.cuda.synchronize()  # Wait for all ops to complete
    #         torch.cuda.empty_cache()
    #     gc.collect()
    #     log_memory("After aggressive cleanup")
        
    #     # Aggregate metrics
    #     final_capital = self.env.current_capital
    #     cumulative_return = final_capital / self.env.initial_capital - 1.0
        
    #     results = {
    #         "policy_loss": update_info.get("policy_loss", 0.0),
    #         "vae_loss": update_info.get("vae_loss", 0.0),
    #         "value_loss": update_info.get("value_loss", 0.0),
    #         "entropy": update_info.get("entropy", 0.0),
    #         "total_steps": int(all_transitions["rewards"].shape[0]),
            
    #         # Task-level metrics
    #         "task_total_reward": sum(episode_rewards),
    #         "task_avg_reward_per_episode": sum(episode_rewards) / len(episode_rewards),
    #         "task_final_capital": final_capital,
    #         "task_cumulative_return": cumulative_return,
            
    #         # Validation metrics
    #         "context_size_final": context_sizes[-1],
    #         "context_growth": context_sizes[-1] - context_sizes[0] if len(context_sizes) > 1 else context_sizes[0],
    #         "latent_norm_change": abs(latent_norms[-1] - latent_norms[0]) if len(latent_norms) > 1 else 0,
            
    #         # Episode breakdown
    #         "episode_rewards": episode_rewards,
    #         "episodes_per_task": self.config.episodes_per_task,
    #         "task_count": self.task_count,
            
    #         **update_info
    #     }
        
    #     # logger.info(f"=== Task {task_id} Complete ===")
    #     # logger.info(f"  Total reward: {results['task_total_reward']:.4f}")
    #     # logger.info(f"  Context growth: {results['context_growth']} steps")
    #     # logger.info(f"  Latent refinement: {results['latent_norm_change']:.4f}")
        
    #     return results


    # def collect_trajectory_with_context(self, persistent_context):
    #     """
    #     Collect trajectory with persistent context from previous episodes.
    #     Context accumulates across episodes within same task.
    #     """
    #     # Reset environment (position only, task remains)
    #     obs0 = self.env.reset()
    #     obs0 = obs0.to(self.device).to(torch.float32)
    #     obs_tensor = obs0.unsqueeze(0)
        
    #     max_horizon = int(self.config.max_horizon)
    #     obs_shape = tuple(obs0.shape)
        
    #     # Preallocate
    #     observations = torch.zeros((max_horizon,) + obs_shape, dtype=torch.float32, device=self.device)
    #     actions = torch.zeros((max_horizon, self.config.num_assets), dtype=torch.float32, device=self.device)
    #     values = torch.zeros((max_horizon,), dtype=torch.float32, device=self.device)
    #     log_probs = torch.zeros((max_horizon,), dtype=torch.float32, device=self.device)
    #     latents = torch.zeros((max_horizon, self.config.latent_dim), dtype=torch.float32, device=self.device)
    #     rewards = torch.zeros((max_horizon,), dtype=torch.float32, device=self.device)
    #     dones = torch.zeros((max_horizon,), dtype=torch.bool, device=self.device)
        
    #     done, step = False, 0
        
    #     while not done and step < max_horizon:
    #         # Get latent using persistent context
    #         latent = self._get_latent_with_persistent_context(obs_tensor, persistent_context)
            
    #         # Sample action
    #         with torch.no_grad():
    #             actions_raw, value_t, log_prob_t = self.policy.act(obs_tensor, latent, deterministic=False)
            
    #         # Environment step
    #         next_obs, reward_scalar, done_flag, info = self.env.step(actions_raw.squeeze(0))
            
    #         # Store step data
    #         observations[step] = obs_tensor.squeeze(0)
    #         actions[step] = actions_raw.squeeze(0).to(self.device)
    #         values[step] = value_t.squeeze(0)
    #         log_probs[step] = log_prob_t.squeeze(0)
    #         latents[step] = latent.squeeze(0)
    #         rewards[step] = float(reward_scalar)
    #         dones[step] = bool(done_flag)
            
    #         # Update persistent context (accumulates across episodes)
    #         persistent_context["observations"].append(obs_tensor.squeeze(0).detach())
    #         persistent_context["actions"].append(actions_raw.squeeze(0).detach())
    #         persistent_context["rewards"].append(torch.tensor(reward_scalar, dtype=torch.float32, device=self.device))
            
    #         # Advance
    #         done = bool(done_flag)
    #         if not done:
    #             obs_tensor = next_obs.to(self.device, dtype=torch.float32).unsqueeze(0)
    #         step += 1
        
    #     T = step if step > 0 else 1
    #     traj = {
    #         "observations": observations[:T],
    #         "actions": actions[:T],
    #         "rewards": rewards[:T],
    #         "values": values[:T],
    #         "log_probs": log_probs[:T],
    #         "latents": latents[:T],
    #         "dones": dones[:T],
    #     }
    #     return traj


    # def _get_latent_with_persistent_context(self, obs_tensor, persistent_context):
    #     """Get latent encoding using persistent context from all episodes."""
    #     if getattr(self.config, "disable_vae", False) or self.vae is None:
    #         return torch.zeros(1, self.config.latent_dim, device=self.device)
        
    #     if len(persistent_context["observations"]) == 0:
    #         # First step of first episode - use prior
    #         return torch.zeros(1, self.config.latent_dim, device=self.device)
        
    #     # 🔥 NEW: Limit context to last N steps to prevent OOM
    #     max_context_len = 200  # Only use last 200 steps
    #     start_idx = max(0, len(persistent_context["observations"]) - max_context_len)
        
    #     # Encode full persistent context
    #     obs_seq = torch.stack(persistent_context["observations"]).unsqueeze(0)
    #     act_seq = torch.stack(persistent_context["actions"]).unsqueeze(0)
    #     rew_seq = torch.stack(persistent_context["rewards"]).unsqueeze(0).unsqueeze(-1)
        
    #     try:
    #         mu, logvar, _ = self.vae.encode(obs_seq, act_seq, rew_seq)
    #         latent = self.vae.reparameterize(mu, logvar)
    #         return latent
    #     except Exception as e:
    #         logger.warning(f"VAE encoding failed: {e}, using zero latent")
    #         return torch.zeros(1, self.config.latent_dim, device=self.device)

    def train_on_task(self) -> Dict[str, float]:
        """Train over multiple episodes on same task (BAMDP) - MEMORY OPTIMIZED"""
        
        log_memory("Task start")
        
        # Sample task once
        task = self.env.sample_task()
        task_id = task.get("task_id", self.task_count)
        self.env.set_task(task)
        self.task_count += 1
        
        # Initialize context for this task (STORE ONLY INDICES, NOT TENSORS)
        context_start_step = 0
        context_obs_list = []  # Will store on CPU
        context_act_list = []
        context_rew_list = []
        
        # Accumulate all transitions across episodes
        all_transitions = {
            "observations": [],
            "actions": [],
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
            log_memory(f"Episode {episode_idx} start")
            
            # Collect trajectory with MINIMAL context
            trajectory = self.collect_trajectory_with_context_v2(
                context_obs_list, context_act_list, context_rew_list
            )
            log_memory(f"Episode {episode_idx} collected")
            
            episode_reward = float(trajectory["rewards"].sum().item())
            episode_rewards.append(episode_reward)

            # NEW: Track where this episode ends in the accumulated trajectory
            traj_length = len(trajectory["observations"])
            current_total = sum(len(t) for t in all_transitions["observations"])
            all_transitions["episode_boundaries"].append(current_total + traj_length - 1)
            
            # Store priors (one per trajectory/episode)
            all_transitions["prior_mu"].append(trajectory["prior_mu"])
            all_transitions["prior_logvar"].append(trajectory["prior_logvar"])
            
            # Accumulate transitions (keep on GPU for now)
            for key in all_transitions.keys():
                if key in trajectory and key not in ["prior_mu", "prior_logvar"]:
                    all_transitions[key].append(trajectory[key])
            
            # Update context (move to CPU immediately)
            T = len(trajectory["observations"])
            for t in range(T):
                context_obs_list.append(trajectory["observations"][t].detach().cpu())
                context_act_list.append(trajectory["actions"][t].detach().cpu())
                context_rew_list.append(trajectory["rewards"][t].detach().cpu())
            
            # CRITICAL: Delete trajectory immediately after copying
            del trajectory
            torch.cuda.empty_cache()
        
        # Stack accumulated transitions from all episodes
        for key in all_transitions.keys():
            if key not in ["episode_boundaries", "prior_mu", "prior_logvar"]:
                all_transitions[key] = torch.cat(all_transitions[key], dim=0)
        
        episode_boundaries = all_transitions["episode_boundaries"]
        for boundary_idx in episode_boundaries[:-1]:  # All except last episode boundary
            all_transitions["dones"][boundary_idx] = False
        
        # Single update on entire BAMDP trajectory
        log_memory("Before update")
        total_loss, update_info = self.update_ppo_and_vae(all_transitions)
        log_memory("After update")
        
        # Build results BEFORE cleanup
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
        
        # AGGRESSIVE CLEANUP
        del all_transitions
        del context_obs_list
        del context_act_list  
        del context_rew_list
        
        # Clear optimizer state periodically (every 10 tasks)
        if self.task_count % 10 == 0:
            self.optimizer.zero_grad(set_to_none=True)
            # Force optimizer state reset
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad = None
                    state = self.optimizer.state.get(p, None)
                    if state is not None:
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                state[k] = v.detach()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()
        
        log_memory("After aggressive cleanup")
        return results

    def collect_trajectory_with_context_v2(self, context_obs, context_act, context_rew):
        """
        Collect trajectory with MINIMAL context usage.
        Context lists are CPU-only and only used for VAE encoding.
        """
        obs0 = self.env.reset()
        obs0 = obs0.to(self.device).to(torch.float32)
        obs_tensor = obs0.unsqueeze(0)
        
        max_horizon = int(self.config.max_horizon)
        obs_shape = tuple(obs0.shape)
        
        # Compute prior from existing context
        if len(context_obs) == 0 or getattr(self.config, "disable_vae", False) or self.vae is None:
            # First episode of task or no VAE - use N(0,I)
            prior_mu = torch.zeros(1, self.config.latent_dim, device=self.device)
            prior_logvar = torch.zeros(1, self.config.latent_dim, device=self.device)
        else:
            # Encode existing context to get prior
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
        values = torch.zeros((max_horizon,), dtype=torch.float32, device=self.device)
        log_probs = torch.zeros((max_horizon,), dtype=torch.float32, device=self.device)
        latents = torch.zeros((max_horizon, self.config.latent_dim), dtype=torch.float32, device=self.device)
        rewards = torch.zeros((max_horizon,), dtype=torch.float32, device=self.device)
        dones = torch.zeros((max_horizon,), dtype=torch.bool, device=self.device)
        
        done, step = False, 0
        
        while not done and step < max_horizon:
            # Get latent with EXPLICIT cleanup
            latent = self._get_latent_from_context_v2(obs_tensor, context_obs, context_act, context_rew)
            
            # Sample action
            with torch.no_grad():
                actions_raw, value_t, log_prob_t = self.policy.act(obs_tensor, latent, deterministic=False)
            
            # Environment step
            next_obs, reward_scalar, done_flag, info = self.env.step(actions_raw.squeeze(0))
            
            # Store step data
            observations[step] = obs_tensor.squeeze(0)
            actions[step] = actions_raw.squeeze(0).to(self.device)
            values[step] = value_t.squeeze(0)
            log_probs[step] = log_prob_t.squeeze(0)
            latents[step] = latent.squeeze(0)
            rewards[step] = float(reward_scalar)
            dones[step] = bool(done_flag)
            
            # Advance
            done = bool(done_flag)
            if not done:
                obs_tensor = next_obs.to(self.device, dtype=torch.float32).unsqueeze(0)
            step += 1
            
            # CRITICAL: Delete intermediate tensors
            del latent, actions_raw, value_t, log_prob_t
        
        T = step if step > 0 else 1
        return {
            "observations": observations[:T],
            "actions": actions[:T],
            "rewards": rewards[:T],
            "values": values[:T],
            "log_probs": log_probs[:T],
            "latents": latents[:T],
            "dones": dones[:T],
            "prior_mu": prior_mu.cpu(),
            "prior_logvar": prior_logvar.cpu(),
        }

    def _get_latent_from_context_v2(self, obs_tensor, context_obs, context_act, context_rew):
        """Get latent with EXPLICIT tensor lifecycle management."""
        if getattr(self.config, "disable_vae", False) or self.vae is None:
            return torch.zeros(1, self.config.latent_dim, device=self.device)
        
        if len(context_obs) == 0:
            return torch.zeros(1, self.config.latent_dim, device=self.device)
        
        # # Limit context window (CRITICAL for memory)
        # max_context_len = 200  # Reduced from 200
        # start_idx = max(0, len(context_obs) - max_context_len)
        
        # # Create GPU tensors ONLY for encoding, then DELETE immediately
        # obs_seq = torch.stack(context_obs[start_idx:]).to(self.device).unsqueeze(0)
        # act_seq = torch.stack(context_act[start_idx:]).to(self.device).unsqueeze(0)
        # rew_seq = torch.stack(context_rew[start_idx:]).to(self.device).unsqueeze(0).unsqueeze(-1)
        
        obs_seq = torch.stack(context_obs).to(self.device).unsqueeze(0)
        act_seq = torch.stack(context_act).to(self.device).unsqueeze(0)
        rew_seq = torch.stack(context_rew).to(self.device).unsqueeze(0).unsqueeze(-1)

        try:
            with torch.no_grad():  # CRITICAL: No gradients for context encoding
                mu, logvar, _ = self.vae.encode(obs_seq, act_seq, rew_seq)
                latent = self.vae.reparameterize(mu, logvar).clone()  # Clone to detach from encoder
        except Exception as e:
            logger.warning(f"VAE encoding failed: {e}, using zero latent")
            latent = torch.zeros(1, self.config.latent_dim, device=self.device)
        finally:
            # EXPLICIT cleanup of temporary tensors
            del obs_seq, act_seq, rew_seq
            if 'mu' in locals():
                del mu, logvar
        
        return latent

    def collect_trajectory_with_context(self, persistent_context):
        """
        Collect trajectory with persistent context from previous episodes.
        Context stored on CPU to prevent GPU memory accumulation.
        """
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
            # Get latent using persistent context
            latent = self._get_latent_with_persistent_context(obs_tensor, persistent_context)
            
            # Sample action
            with torch.no_grad():
                actions_raw, value_t, log_prob_t = self.policy.act(obs_tensor, latent, deterministic=False)
            
            # Environment step
            next_obs, reward_scalar, done_flag, info = self.env.step(actions_raw.squeeze(0))
            
            # Store step data
            observations[step] = obs_tensor.squeeze(0)
            actions[step] = actions_raw.squeeze(0).to(self.device)
            values[step] = value_t.squeeze(0)
            log_probs[step] = log_prob_t.squeeze(0)
            latents[step] = latent.squeeze(0)
            rewards[step] = float(reward_scalar)
            dones[step] = bool(done_flag)
            
            # Update persistent context (CPU storage)
            persistent_context["observations"].append(obs_tensor.squeeze(0).detach().cpu())
            persistent_context["actions"].append(actions_raw.squeeze(0).detach().cpu())
            persistent_context["rewards"].append(torch.tensor(reward_scalar, dtype=torch.float32))
            
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


    def _get_latent_with_persistent_context(self, obs_tensor, persistent_context):
        """Get latent encoding using persistent context (moves from CPU to GPU)."""
        if getattr(self.config, "disable_vae", False) or self.vae is None:
            return torch.zeros(1, self.config.latent_dim, device=self.device)
        
        if len(persistent_context["observations"]) == 0:
            return torch.zeros(1, self.config.latent_dim, device=self.device)
        
        # Limit context window
        max_context_len = 200
        start_idx = max(0, len(persistent_context["observations"]) - max_context_len)
        
        # Move context from CPU to GPU for encoding
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

    # ---------------------------------------------------------------------
    # Public training entry point
    # ---------------------------------------------------------------------

    def train_episode(self) -> Dict[str, float]:
        """
        Train for one episode and return metrics.
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
            # update_ppo_and_vae performs its own optimization/update and returns metrics
            _, update_info = self.update_ppo_and_vae(tr)

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
        value_loss = update_info.get("value_loss", 0.0)
        entropy = update_info.get("entropy", 0.0)
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
            "policy_loss": policy_loss,
            "vae_loss": vae_loss,
            "value_loss": value_loss,     
            "entropy": entropy,        
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
            "recent_avg_episode_reward": float(np.mean(list(self.episode_rewards))) if self.episode_rewards else 0.0,
            "recent_std_episode_reward": float(np.std(list(self.episode_rewards))) if len(self.episode_rewards) > 1 else 0.0,
            "recent_avg_policy_loss": float(np.mean(list(self.policy_losses))) if self.policy_losses else 0.0,
            "recent_avg_vae_loss": float(np.mean(list(self.vae_losses))) if self.vae_losses else 0.0,

            # VAE breakdown
            **vae_loss_components,

            # Misc
            "episode_count": int(self.episode_count),
            "steps_per_episode": (int(tr["rewards"].shape[0]) if isinstance(tr["rewards"], torch.Tensor) else len(tr["rewards"])),
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

    # def update_ppo_and_vae(self, traj):
    #     """
    #     PPO with mini-batching + separate VAE update.
    #     """
    #     def ensure_stacked(x):
    #         if isinstance(x, torch.Tensor):
    #             return x.to(self.device)
    #         return torch.stack(x).to(self.device)

    #     # === Prepare data ===
    #     obs = ensure_stacked(traj["observations"])
    #     actions = ensure_stacked(traj["actions"])
    #     latents = ensure_stacked(traj["latents"])
    #     rewards = traj["rewards"] if isinstance(traj["rewards"], torch.Tensor) \
    #         else torch.as_tensor(traj["rewards"], dtype=torch.float32, device=self.device)
    #     dones = traj["dones"] if isinstance(traj["dones"], torch.Tensor) \
    #         else torch.as_tensor(traj["dones"], dtype=torch.float32, device=self.device)
    #     values = ensure_stacked(traj["values"])
    #     old_logp = ensure_stacked(traj["log_probs"])

    #     # === DETACH ALL INPUTS ===
    #     obs = obs.detach()
    #     actions = actions.detach()
    #     latents = latents.detach()
    #     rewards = rewards.detach()
    #     dones = dones.detach()
    #     values = values.detach()
    #     old_logp = old_logp.detach()

    #     last_value = traj.get("last_value", 0.0)
    #     if isinstance(last_value, torch.Tensor):
    #         last_value = last_value.detach().to(self.device)

    #     # === Compute advantages once ===
    #     advantages, returns = self.compute_gae(rewards, values, dones, last_value)
    #     advantages = advantages.detach()
    #     returns = returns.detach()

    #     # === Mini-batch setup ===
    #     batch_size = min(self.config.ppo_minibatch_size, len(obs))  # Don't exceed trajectory length
    #     num_samples = len(obs)
    #     logger.info(f"PPO update: trajectory size={len(obs)}, mini-batch={batch_size}")
        
    #     # === Initialize tracking ===
    #     first_epoch_metrics = {}
    #     final_ratio_mean = 1.0
    #     final_ppo_loss = torch.tensor(0.0, device=self.device)

    #     # === MULTI-EPOCH MINI-BATCH PPO UPDATE ===
    #     for epoch in range(self.config.ppo_epochs):
    #         # Shuffle indices
    #         indices = torch.randperm(num_samples, device=self.device)
            
    #         # Process mini-batches
    #         for start_idx in range(0, num_samples, batch_size):
    #             end_idx = min(start_idx + batch_size, num_samples)
    #             batch_indices = indices[start_idx:end_idx]
                
    #             # Extract mini-batch
    #             batch_obs = obs[batch_indices]
    #             batch_actions = actions[batch_indices]
    #             batch_latents = latents[batch_indices]
    #             batch_old_logp = old_logp[batch_indices]
    #             batch_advantages = advantages[batch_indices]
    #             batch_returns = returns[batch_indices]
                
    #             # Forward pass
    #             new_values, new_logp, entropy = self.policy.evaluate_actions(
    #                 batch_obs, batch_latents, batch_actions
    #             )
    #             new_values = new_values.squeeze(-1)
    #             new_logp = new_logp.squeeze(-1)
    #             entropy = entropy.squeeze(-1)

    #             # PPO losses
    #             ratio = torch.exp(new_logp - batch_old_logp)
    #             eps = self.config.ppo_clip_ratio
    #             surr1 = ratio * batch_advantages
    #             surr2 = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * batch_advantages
    #             policy_loss = -torch.min(surr1, surr2).mean()
    #             value_loss = F.mse_loss(new_values, batch_returns)
    #             entropy_loss = -entropy.mean()

    #             ppo_loss = (policy_loss + 
    #                         self.config.value_loss_coef * value_loss + 
    #                         self.config.entropy_coef * entropy_loss)

    #             # Capture first batch of first epoch metrics
    #             if epoch == 0 and start_idx == 0:
    #                 first_epoch_metrics = {
    #                     "policy_loss": float(policy_loss.item()),
    #                     "value_loss": float(value_loss.item()),
    #                     "entropy": float(entropy.mean().item()),
    #                     "advantages_mean": float(batch_advantages.mean().item()),
    #                 }

    #             final_ratio_mean = float(ratio.mean().item())
    #             final_ppo_loss = ppo_loss.detach()

    #             # Update
    #             self.optimizer.zero_grad()
    #             ppo_loss.backward()
    #             torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
    #             self.optimizer.step()

    #             # for param in self.policy.parameters():
    #             #     if param.grad is not None:
    #             #         param.grad.detach_()
    #             #         param.grad.zero_()
    #             self.optimizer.zero_grad(set_to_none=True)

    #             # Cleanup mini-batch
    #             del new_values, new_logp, entropy, surr1, surr2, ratio
    #             del policy_loss, value_loss, entropy_loss, ppo_loss
    #             del batch_obs, batch_actions, batch_latents, batch_old_logp
    #             del batch_advantages, batch_returns

    #         # Cleanup after each epoch
    #         if torch.cuda.is_available():
    #             torch.cuda.empty_cache()

    #     # === VAE update (sample from buffer) ===
    #     vae_loss_val = 0.0
    #     if self.vae_enabled and (self.episode_count % self.config.vae_update_freq == 0):
    #         # Minimum buffer size before training
    #         min_buffer_size = 8
    #         vae_batch_size = min(32, len(self.vae_buffer))
            
    #         if len(self.vae_buffer) >= min_buffer_size:
    #             try:
    #                 # Sample mini-batch from buffer
    #                 indices = np.random.choice(len(self.vae_buffer), vae_batch_size, replace=False)
            
    #                 # Collect trajectories (move from CPU to GPU) and find max length for padding
    #                 sampled_trajs = [
    #                     {k: v.to(self.device) if torch.is_tensor(v) else v 
    #                     for k, v in self.vae_buffer[idx].items()}
    #                     for idx in indices
    #                 ]
    #                 max_len = max(traj["observations"].shape[0] for traj in sampled_trajs)
                    
    #                 # Pad and stack trajectories
    #                 batch_obs, batch_act, batch_rew = [], [], []
    #                 for traj in sampled_trajs:
    #                     T = traj["observations"].shape[0]
                        
    #                     # Pad observations
    #                     obs_padded = torch.zeros((max_len,) + traj["observations"].shape[1:], 
    #                                             dtype=torch.float32, device=self.device)
    #                     obs_padded[:T] = traj["observations"]
    #                     batch_obs.append(obs_padded)
                        
    #                     # Pad actions
    #                     act_padded = torch.zeros((max_len,) + traj["actions"].shape[1:],
    #                                             dtype=torch.float32, device=self.device)
    #                     act_padded[:T] = traj["actions"]
    #                     batch_act.append(act_padded)
                        
    #                     # Pad rewards
    #                     rew_padded = torch.zeros(max_len, dtype=torch.float32, device=self.device)
    #                     rew_padded[:T] = traj["rewards"]
    #                     batch_rew.append(rew_padded)
                    
    #                 # Stack into batches
    #                 obs_batch = torch.stack(batch_obs)  # (batch, max_len, N, F)
    #                 act_batch = torch.stack(batch_act)  # (batch, max_len, N)
    #                 rew_batch = torch.stack(batch_rew).unsqueeze(-1)  # (batch, max_len, 1)
                    
    #                 # Train VAE on batch
    #                 vae_loss, vae_info = self.vae.compute_loss(
    #                     obs_batch, act_batch, rew_batch, beta=self.config.vae_beta
    #                 )

    #                 self.optimizer.zero_grad()
    #                 vae_loss.backward()
    #                 torch.nn.utils.clip_grad_norm_(self.vae.parameters(), self.config.max_grad_norm)
    #                 self.optimizer.step()

    #                 vae_loss_val = float(vae_loss.item())
    #                 first_epoch_metrics.update({f"vae_{k}": v for k, v in vae_info.items()})

    #             except Exception as e:
    #                 logger.warning(f"VAE update failed: {e}")
    #             finally:
    #                 # Cleanup
    #                 if 'obs_batch' in locals():
    #                     del obs_batch, act_batch, rew_batch
    #                 if 'vae_loss' in locals():
    #                     del vae_loss
    #                 if torch.cuda.is_available():
    #                     torch.cuda.empty_cache()
    #         else:
    #             # Not enough data in buffer yet - skip VAE update
    #             logger.debug(f"VAE buffer has {len(self.vae_buffer)} trajectories, need {min_buffer_size}")

    #     # === Final cleanup ===
    #     del obs, actions, latents, rewards, dones, values, old_logp, advantages, returns
    #     if torch.cuda.is_available():
    #         torch.cuda.empty_cache()

    #     # === Build final metrics ===
    #     first_epoch_metrics["vae_loss"] = vae_loss_val
    #     first_epoch_metrics["ratio_mean"] = final_ratio_mean

    #     return final_ppo_loss, first_epoch_metrics
    
    def update_ppo_and_vae(self, traj):
        """PPO with mini-batching + VAE update - MEMORY OPTIMIZED"""
        
        # Prepare data with DETACH
        obs = traj["observations"].detach()
        actions = traj["actions"].detach()
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
        
        # Mini-batch setup
        batch_size = min(self.config.ppo_minibatch_size, len(obs))
        num_samples = len(obs)
        logger.info(f"PPO update: trajectory size={len(obs)}, mini-batch={batch_size}")
        
        first_epoch_metrics = {}
        final_ratio_mean = 1.0
        
        # PPO UPDATE with explicit cleanup
        for epoch in range(self.config.ppo_epochs):
            indices = torch.randperm(num_samples, device=self.device)
            
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                batch_indices = indices[start_idx:end_idx]
                
                # Extract mini-batch
                batch_obs = obs[batch_indices]
                batch_actions = actions[batch_indices]
                batch_latents = latents[batch_indices]
                batch_old_logp = old_logp[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Forward pass
                new_values, new_logp, entropy = self.policy.evaluate_actions(
                    batch_obs, batch_latents, batch_actions
                )
                new_values = new_values.squeeze(-1)
                new_logp = new_logp.squeeze(-1)
                entropy = entropy.squeeze(-1)
                
                # PPO losses
                ratio = torch.exp(new_logp - batch_old_logp)
                eps = self.config.ppo_clip_ratio
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(new_values, batch_returns)
                entropy_loss = -entropy.mean()
                
                ppo_loss = (policy_loss + 
                        self.config.value_loss_coef * value_loss + 
                        self.config.entropy_coef * entropy_loss)
                
                # Capture first batch metrics
                if epoch == 0 and start_idx == 0:
                    first_epoch_metrics = {
                        "policy_loss": float(policy_loss.item()),
                        "value_loss": float(value_loss.item()),
                        "entropy": float(entropy.mean().item()),
                    }
                
                final_ratio_mean = float(ratio.mean().item())
                
                # Update
                self.optimizer.zero_grad()
                ppo_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                
                # CRITICAL: Explicit cleanup of batch tensors
                del new_values, new_logp, entropy, surr1, surr2, ratio
                del policy_loss, value_loss, entropy_loss, ppo_loss
                del batch_obs, batch_actions, batch_latents, batch_old_logp
                del batch_advantages, batch_returns
            
            # Cleanup after epoch
            del indices
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # VAE UPDATE with aggressive cleanup
        vae_loss_val = 0.0
        if self.vae_enabled and (self.episode_count % self.config.vae_update_freq == 0):
            min_buffer_size = 8
            vae_batch_size = min(16, len(self.vae_buffer))  # Reduced from 32
            
            if len(self.vae_buffer) >= min_buffer_size:
                logger.info(f"=== VAE Update (Episode {self.episode_count}) ===")
                logger.info(f"  Buffer size: {len(self.vae_buffer)}, Batch size: {vae_batch_size}")
                try:
                    indices = np.random.choice(len(self.vae_buffer), vae_batch_size, replace=False)
                    
                    # Move to GPU batch-by-batch to avoid OOM
                    batch_obs, batch_act, batch_rew = [], [], []
                    max_len = max(self.vae_buffer[idx]["observations"].shape[0] for idx in indices)
                    
                    for idx in indices:
                        traj = self.vae_buffer[idx]
                        T = traj["observations"].shape[0]
                        
                        # Pad on GPU
                        obs_padded = torch.zeros((max_len,) + traj["observations"].shape[1:], 
                                                dtype=torch.float32, device=self.device)
                        obs_padded[:T] = traj["observations"].to(self.device)
                        batch_obs.append(obs_padded)
                        
                        act_padded = torch.zeros((max_len,) + traj["actions"].shape[1:],
                                                dtype=torch.float32, device=self.device)
                        act_padded[:T] = traj["actions"].to(self.device)
                        batch_act.append(act_padded)
                        
                        rew_padded = torch.zeros(max_len, dtype=torch.float32, device=self.device)
                        rew_padded[:T] = traj["rewards"].to(self.device)
                        batch_rew.append(rew_padded)
                        
                        # CRITICAL: Delete temporary padded tensors
                        del obs_padded, act_padded, rew_padded
                    
                    # Stack
                    obs_batch = torch.stack(batch_obs)
                    act_batch = torch.stack(batch_act)
                    rew_batch = torch.stack(batch_rew).unsqueeze(-1)
                    
                    # Delete intermediate lists
                    del batch_obs, batch_act, batch_rew
                    
                    # Train VAE
                    vae_loss, vae_info = self.vae.compute_loss(
                        obs_batch, act_batch, rew_batch, beta=self.config.vae_beta,
                        num_elbo_terms=getattr(self.config, 'vae_num_elbo_terms', 8)
                    )
                    
                    logger.info(f"  Loss breakdown:")
                    logger.info(f"    Total: {vae_info['total']:.4f}")
                    logger.info(f"    Recon obs: {vae_info['recon_obs']:.4f}")
                    logger.info(f"    Recon reward: {vae_info['recon_reward']:.4f}")
                    logger.info(f"    KL: {vae_info['kl']:.4f}")
                    logger.info(f"  Timesteps sampled: {vae_info['num_elbo_terms']} -> {vae_info['timesteps_sampled']}")
                    
                    self.optimizer.zero_grad()
                    vae_loss.backward()

                    vae_grad_norm = torch.nn.utils.clip_grad_norm_(self.vae.parameters(), self.config.max_grad_norm)
                    logger.info(f"  VAE grad norm: {vae_grad_norm:.4f}")
                    
                    self.optimizer.step()
                    logger.info(f"  ✓ VAE weights updated") 

                    vae_loss_val = float(vae_loss.item())
                    first_epoch_metrics.update({f"vae_{k}": v for k, v in vae_info.items()})
                    
                    # CRITICAL: Delete VAE tensors
                    del obs_batch, act_batch, rew_batch, vae_loss
                    
                except Exception as e:
                    logger.warning(f"VAE update failed: {e}")
                finally:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        # Final cleanup
        del obs, actions, latents, rewards, dones, values, old_logp, advantages, returns
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        first_epoch_metrics["vae_loss"] = vae_loss_val
        first_epoch_metrics["ratio_mean"] = final_ratio_mean
        
        return torch.tensor(0.0), first_epoch_metrics

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
        if self.vae_enabled:
            obs_seq = obs.unsqueeze(0)  # (1, T, N, F)
            # Normalize actions to portfolio weights for VAE (sum(|w|)+w_cash=1 budget)
            denom = actions.abs().sum(dim=-1, keepdim=True) + 1.0 + 1e-8
            vae_actions = actions / denom
            act_seq = vae_actions.unsqueeze(0)  # (1, T, N)
            rew_seq = rewards.unsqueeze(0).unsqueeze(-1)  # (1, T, 1)
            try:
                vae_loss, vae_info = self.vae.compute_loss(
                    obs_seq, act_seq, rew_seq, beta=self.config.vae_beta
                )
                self.optimizer.zero_grad()
                vae_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.vae.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                vae_loss_val = float(vae_loss.item())
            except Exception as e:
                logger.warning(f"VAE loss computation failed: {e}")
                vae_loss_val = 0.0

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