#!/usr/bin/env python3
"""
Diagnostic script to find what's causing negative Sharpe ratios in Phase 3.
Compares working Phase 2 config vs Phase 3 config step by step.
"""

import torch
import logging
import numpy as np
from pathlib import Path
import time

# Import modules
from environments.dataset import create_split_datasets
from environments.env import MetaEnv
from models.vae import VAE
from models.policy import PortfolioPolicy
from algorithms.trainer import PPOTrainer
from run_logger import RunLogger, seed_everything

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Phase2Config:
    """EXACT Phase 2 Trial 46 config that worked"""
    def __init__(self):
        # Phase 2 Trial 46 parameters (KNOWN TO WORK)
        self.latent_dim = 512
        self.hidden_dim = 1024
        self.vae_lr = 0.0010748206602172
        self.policy_lr = 0.0020289998766945
        self.vae_beta = 0.0125762666385515
        self.vae_update_freq = 5
        self.seq_len = 120
        self.episodes_per_task = 3
        self.batch_size = 8192
        self.vae_batch_size = 1024
        self.ppo_epochs = 8
        self.entropy_coef = 0.0013141391952945
        
        # Environment
        self.data_path = "environments/data/sp500_rl_ready_cleaned.parquet"
        self.train_end = '2015-12-31'
        self.val_end = '2020-12-31'
        self.num_assets = 30
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Training
        self.max_episodes = 3000
        self.val_interval = 600
        self.val_episodes = 75
        self.log_interval = 150
        
        # PPO
        self.ppo_clip_ratio = 0.2
        self.value_loss_coef = 0.5
        self.max_grad_norm = 0.5
        self.gae_lambda = 0.95
        self.discount_factor = 0.99
        
        # Derived
        self.max_horizon = min(self.seq_len - 10, int(self.seq_len * 0.8))
        self.min_horizon = max(self.max_horizon - 15, self.max_horizon // 2)
        
        self.seed = 42
        self.exp_name = "phase2_baseline_test"

class Phase3Config:
    """Phase 3 config that's failing"""
    def __init__(self):
        # SAME as Phase 2
        self.latent_dim = 512
        self.hidden_dim = 1024
        self.vae_lr = 0.0010748206602172
        self.policy_lr = 0.0020289998766945
        self.vae_beta = 0.0125762666385515
        self.vae_update_freq = 5
        self.seq_len = 120
        self.episodes_per_task = 3
        self.batch_size = 8192
        self.vae_batch_size = 1024
        self.ppo_epochs = 8
        self.entropy_coef = 0.0013141391952945
        
        # Environment
        self.data_path = "environments/data/sp500_rl_ready_cleaned.parquet"
        self.train_end = '2015-12-31'
        self.val_end = '2020-12-31'
        self.num_assets = 30
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Training (smaller for testing)
        self.max_episodes = 800
        self.val_interval = 200
        self.val_episodes = 50
        self.log_interval = 100
        
        # PPO
        self.ppo_clip_ratio = 0.2
        self.value_loss_coef = 0.5
        self.max_grad_norm = 0.5
        self.gae_lambda = 0.95
        self.discount_factor = 0.99
        
        # NEW: Early stopping parameters
        self.early_stopping_patience = 5
        self.early_stopping_min_delta = 0.01
        self.min_episodes_before_stopping = 400
        
        # Derived
        self.max_horizon = min(self.seq_len - 10, int(self.seq_len * 0.8))
        self.min_horizon = max(self.max_horizon - 15, self.max_horizon // 2)
        
        self.seed = 42
        self.exp_name = "phase3_diagnostic_test"

def update_vae_fixed_original(trainer):
    """Original VAE update from Phase 2 (known to work)"""
    if len(trainer.vae_buffer) < trainer.config.vae_batch_size:
        return 0.0

    indices = np.random.choice(len(trainer.vae_buffer), trainer.config.vae_batch_size, replace=False)
    batch_traj = [trainer.vae_buffer[i] for i in indices]

    total_loss = None
    loss_count = 0

    trainer.vae_optimizer.zero_grad()

    for tr in batch_traj:
        seq_len = len(tr["rewards"])
        if seq_len < 2:
            continue

        max_t = min(seq_len - 1, 20)  
        t = np.random.randint(1, max_t + 1)

        obs_ctx = tr["observations"][:t].unsqueeze(0)        
        act_ctx = tr["actions"][:t].unsqueeze(0)             
        rew_ctx = tr["rewards"][:t].unsqueeze(0).unsqueeze(-1)  

        vae_loss, _ = trainer.vae.compute_loss(
            obs_ctx, act_ctx, rew_ctx, beta=trainer.config.vae_beta, context_len=t
        )
        
        if total_loss is None:
            total_loss = vae_loss
        else:
            total_loss = total_loss + vae_loss
        loss_count += 1

    if loss_count == 0:
        return 0.0

    avg_loss = total_loss / loss_count
    avg_loss.backward()
    torch.nn.utils.clip_grad_norm_(trainer.vae.parameters(), trainer.config.max_grad_norm)
    trainer.vae_optimizer.step()
    
    return float(avg_loss.item())

def prepare_environments(config):
    """Setup environments exactly like Phase 2"""
    if not Path(config.data_path).exists():
        raise FileNotFoundError(f"Dataset not found: {config.data_path}")

    datasets = create_split_datasets(
        data_path=config.data_path,
        train_end=config.train_end,
        val_end=config.val_end
    )

    config.num_assets = datasets['train'].num_assets
    envs = {}

    for split_name, dataset in [('train', datasets['train']), ('val', datasets['val'])]:
        features_list = []
        prices_list = []
        num_windows = max(1, (len(dataset) - config.seq_len) // config.seq_len)

        for i in range(num_windows):
            start_idx = i * config.seq_len
            end_idx = start_idx + config.seq_len

            if end_idx <= len(dataset):
                window = dataset.get_window(start_idx, end_idx)
                features_list.append(torch.tensor(window['features'], dtype=torch.float32))
                prices_list.append(torch.tensor(window['raw_prices'], dtype=torch.float32))

        if features_list:
            all_features = torch.stack(features_list)
            all_prices = torch.stack(prices_list)

            dataset_tensor = {
                'features': all_features.view(-1, config.num_assets, dataset.num_features),
                'raw_prices': all_prices.view(-1, config.num_assets)
            }

            envs[split_name] = MetaEnv(
                dataset=dataset_tensor,
                feature_columns=dataset.feature_cols,
                seq_len=config.seq_len,
                min_horizon=config.min_horizon,
                max_horizon=config.max_horizon
            )

    return envs

def quick_validation(env, policy, vae, config, episodes=5):
    """Quick validation test"""
    device = torch.device(config.device)
    rewards = []
    
    vae.eval()
    policy.eval()
    
    with torch.no_grad():
        for ep in range(episodes):
            task = env.sample_task()
            env.set_task(task)
            obs = env.reset()
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            
            episode_reward = 0
            done = False
            trajectory_context = {'observations': [], 'actions': [], 'rewards': []}
            
            step_count = 0
            while not done and step_count < 10:  # Limit steps for diagnostics
                if len(trajectory_context['observations']) == 0:
                    latent = torch.zeros(1, config.latent_dim, device=device)
                else:
                    obs_seq = torch.stack(trajectory_context['observations']).unsqueeze(0)
                    action_seq = torch.stack(trajectory_context['actions']).unsqueeze(0)
                    reward_seq = torch.stack(trajectory_context['rewards']).unsqueeze(0)
                    # Ensure reward has correct shape: (batch, seq_len, 1)
                    if reward_seq.dim() == 2:
                        reward_seq = reward_seq.unsqueeze(-1)
                    mu, logvar, _ = vae.encode(obs_seq, action_seq, reward_seq)
                    latent = vae.reparameterize(mu, logvar)
                
                action, _ = policy.act(obs_tensor, latent, deterministic=True)
                action_cpu = action.squeeze(0).detach().cpu().numpy()
                
                # DIAGNOSTIC: Check action values
                if ep == 0 and step_count < 3:
                    print(f"  Step {step_count}: action_sum={action_cpu.sum():.4f}, action_max={action_cpu.max():.4f}")
                
                next_obs, reward, done, info = env.step(action_cpu)
                episode_reward += reward
                
                # DIAGNOSTIC: Check reward
                if ep == 0 and step_count < 3:
                    print(f"  Step {step_count}: reward={reward:.4f}, capital={getattr(env, 'current_capital', 'N/A')}")
                
                trajectory_context['observations'].append(obs_tensor.squeeze(0).detach())
                trajectory_context['actions'].append(action.squeeze(0).detach())
                trajectory_context['rewards'].append(torch.tensor(reward, device=device))
                
                if not done:
                    obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
                
                step_count += 1
            
            rewards.append(episode_reward)
            if ep == 0:
                print(f"  Episode {ep}: total_reward={episode_reward:.4f}")
    
    vae.train()
    policy.train()
    
    return np.mean(rewards)

def test_config(config, test_name, use_early_stopping=False):
    """Test a specific configuration"""
    print(f"\n{'='*50}")
    print(f"TESTING: {test_name}")
    print(f"{'='*50}")
    
    try:
        # Setup
        seed_everything(config.seed)
        run_dir = Path("diagnostic_results") / test_name.lower().replace(" ", "_")
        run_dir.mkdir(parents=True, exist_ok=True)
        run = RunLogger(run_dir, config.__dict__, name=config.exp_name)
        
        print("1. Setting up environments...")
        envs = prepare_environments(config)
        train_env = envs['train']
        val_env = envs['val']
        
        print("2. Initializing models...")
        task = train_env.sample_task()
        train_env.set_task(task)
        initial_obs = train_env.reset()
        obs_shape = initial_obs.shape
        
        device = torch.device(config.device)
        vae = VAE(obs_shape, config.num_assets, config.latent_dim, config.hidden_dim).to(device)
        policy = PortfolioPolicy(obs_shape, config.latent_dim, config.num_assets, config.hidden_dim).to(device)
        
        print("3. Setting up trainer...")
        trainer = PPOTrainer(env=train_env, policy=policy, vae=vae, config=config)
        
        # Use original VAE update
        trainer.update_vae = lambda: update_vae_fixed_original(trainer)
        
        print("4. Testing initial validation...")
        initial_val = quick_validation(val_env, policy, vae, config, episodes=3)
        print(f"   Initial validation: {initial_val:.4f}")
        
        print("5. Training a few episodes...")
        for i in range(5):
            episode_result = trainer.train_episode()
            reward = episode_result.get('episode_reward', 0)
            print(f"   Episode {i+1}: reward={reward:.4f}")
            
            # Check if early stopping methods work (if enabled)
            if use_early_stopping and hasattr(trainer, 'add_validation_score'):
                test_score = reward  # Use episode reward as mock validation
                should_stop = trainer.add_validation_score(test_score)
                print(f"   Early stopping test: should_stop={should_stop}")
        
        print("6. Testing validation after training...")
        final_val = quick_validation(val_env, policy, vae, config, episodes=3)
        print(f"   Final validation: {final_val:.4f}")
        
        run.close()
        
        print(f"✅ {test_name} COMPLETED")
        print(f"   Initial val: {initial_val:.4f}")
        print(f"   Final val: {final_val:.4f}")
        print(f"   Change: {final_val - initial_val:+.4f}")
        
        return True, initial_val, final_val
        
    except Exception as e:
        print(f"❌ {test_name} FAILED: {e}")
        import traceback
        traceback.print_exc()
        try:
            run.close()
        except:
            pass
        return False, None, None

def main():
    """Run diagnostic tests"""
    print("🔍 PHASE 3 DIAGNOSTIC ANALYSIS")
    print("Comparing working Phase 2 vs failing Phase 3...")
    
    results = {}
    
    # Test 1: Exact Phase 2 config (should work)
    success, init_val, final_val = test_config(Phase2Config(), "Phase 2 Baseline", use_early_stopping=False)
    results['phase2'] = (success, init_val, final_val)
    
    # Test 2: Phase 3 config without early stopping (isolate the issue)
    success, init_val, final_val = test_config(Phase3Config(), "Phase 3 Without Early Stopping", use_early_stopping=False)
    results['phase3_no_es'] = (success, init_val, final_val)
    
    # Test 3: Phase 3 config with early stopping (full test)
    success, init_val, final_val = test_config(Phase3Config(), "Phase 3 With Early Stopping", use_early_stopping=True)
    results['phase3_with_es'] = (success, init_val, final_val)
    
    # Analysis
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)
    
    for test_name, (success, init_val, final_val) in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        if success:
            print(f"{test_name}: {status} (init: {init_val:.4f}, final: {final_val:.4f})")
        else:
            print(f"{test_name}: {status}")
    
    # Diagnosis
    print("\n🔍 DIAGNOSIS:")
    
    if results['phase2'][0] and not results['phase3_no_es'][0]:
        print("❌ Issue is in Phase 3 config differences (not early stopping)")
        print("   Check: seq_len, batch sizes, training parameters")
    elif results['phase2'][0] and results['phase3_no_es'][0] and not results['phase3_with_es'][0]:
        print("❌ Issue is in early stopping implementation")
        print("   Check: trainer extension methods")
    elif not results['phase2'][0]:
        print("❌ Issue is more fundamental - even Phase 2 config fails")
        print("   Check: environment, models, dataset")
    else:
        print("✅ All configs work - issue might be in Optuna integration")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if all(r[0] for r in results.values()):
        print("   All tests pass - issue is likely in Optuna trial setup")
        print("   Check: parameter passing, trial isolation, parallel conflicts")
    else:
        failed_tests = [name for name, (success, _, _) in results.items() if not success]
        print(f"   Failed tests: {', '.join(failed_tests)}")
        print("   Focus debugging on these configurations")

if __name__ == "__main__":
    main()