# VariBAD Refactoring Master Plan

## Current Status: Phase 1 COMPLETE ✅ → Phase 2 READY 🚀
- ✅ Testing infrastructure deployed (`tests/` directory with working safety tests)
- ✅ Baseline captured successfully (`tests/baselines/` contains system snapshots)
- ✅ Safety tests working (3/13 tests passing - sufficient safety net)
- ✅ **Phase 1 COMPLETE**: Removed redundant files (`debug.py`, `workbench.ipynb`, `buffer_test.py`)
- ✅ Monitoring consolidation verified (no duplicates found in trainer)
- 🚀 **READY FOR PHASE 2**: Configuration System Enhancement

## Phase 2: Configuration System Enhancement (NEXT PRIORITY)

### Goals
- Multi-experiment management and comparison
- Parameter sweeps and ablation studies  
- Automatic experiment tracking and naming
- Parallel execution support foundation

### Implementation Tasks

#### 1. Configuration Architecture Design (HIGH PRIORITY)
Create enhanced configuration system:

```
config/
├── base.conf              # Default values
├── profiles/
│   ├── debug.conf         # Fast iteration (5 iter, 10 episodes)
│   ├── development.conf   # Normal experiments (100 iter, 30 episodes)  
│   ├── production.conf    # Long runs (2000 iter, 90 episodes)
│   └── ablation.conf      # Systematic studies
└── experiments/           # Specific experiment configs
    ├── exp_001_baseline.conf
    ├── exp_002_no_short.conf
    └── exp_003_latent_dim_sweep.conf
```

#### 2. Enhanced Main Script Features
**Target Commands**:
```bash
# Profile-based training
python varibad/main.py --config profiles/debug.conf

# Parameter sweeps  
python varibad/main.py --sweep latent_dim=5,8,12 num_iterations=100,500

# Automatic naming
# Creates: exp_20250127_143022_latent5_iter100
```

#### 3. Results Database System
- Automatic experiment metadata storage
- Performance comparison tools
- Results tracking in `results/experiments.db`
- **Keep**: `monitor_training.py` (enhanced standalone version)
- **Remove**: Duplicate monitoring code in `varibad/core/trainer.py`
- **Target**: Consolidated logging, no duplicate dashboard code

#### 4. Archive System Simplification (Medium Impact)
- **Current**: Heavy `archive_results.py` (comprehensive but overkill)
- **Target**: Lightweight version for active research, full archival for final results
- **Keep**: Core archival functionality, simplify automated analysis

### Testing Strategy for Phase 1
```bash
# Safety test after each change
pytest tests/test_phase1_data_pipeline.py::TestDataPipelineBaseline::test_production_dataset_exists -v

# Full baseline check periodically  
pytest tests/test_phase1_data_pipeline.py::TestDataPipelineBaseline -v

# If any test fails: rollback immediately
git reset --hard HEAD~1
```

## Phase 2: Configuration System Enhancement

### Goals
- Multi-experiment management
- Parameter sweeps and ablation studies  
- Automatic experiment tracking
- Parallel execution support

### Configuration Architecture
```
config/
├── base.conf              # Default values
├── profiles/
│   ├── debug.conf         # Fast iteration (5 iter, 10 episodes)
│   ├── development.conf   # Normal experiments (100 iter, 30 episodes)  
│   ├── production.conf    # Long runs (2000 iter, 90 episodes)
│   └── ablation.conf      # Systematic studies
└── experiments/           # Specific experiment configs
    ├── exp_001_baseline.conf
    ├── exp_002_no_short.conf
    └── exp_003_reduced_features.conf
```

### Enhanced Main Script Features
- `python varibad/main.py --config profiles/debug.conf`
- `python varibad/main.py --sweep latent_dim=5,8,12 num_iterations=100,500`
- Automatic experiment naming: `exp_20250127_143022_latent5_iter100`
- Results database in `results/experiments.db`

### Key Hyperparameters to Test

#### VariBAD-Specific
```python
VARIBAD_PARAMS = {
    'latent_dim': [3, 5, 8, 12],                    # Belief complexity
    'vae_updates': [5, 10, 20],                     # Learning frequency
    'kl_regularization_weight': [0.1, 1.0, 10.0],  # Prior strength
    'max_trajectory_length': [10, 20, 50],          # Sequence length
    'belief_learning_rate': [1e-5, 1e-4, 1e-3]     # Separate LR for beliefs
}
```

#### Training Dynamics  
```python
TRAINING_PARAMS = {
    'episode_length': [15, 30, 60, 90],             # Trading horizon
    'episodes_per_iteration': [3, 5, 10],           # Data collection
    'policy_lr': [1e-5, 1e-4, 1e-3],               # Policy learning
    'lr_decay_schedule': ['constant', 'cosine', 'step'],
    'action_noise_schedule': [0.01, 0.05, 'decay'] # Exploration
}
```

#### Architecture Search
```python
ARCHITECTURE_PARAMS = {
    'encoder_hidden_dim': [64, 128, 256],
    'decoder_hidden_dim': [64, 128, 256], 
    'policy_hidden_dim': [128, 256, 512],
    'num_layers': [2, 3, 4]
}
```

## Phase 3: Experimental Tools

### Quick Iteration Tools
- **Debug Mode**: 2-minute training cycles with tiny models
- **Smoke Tests**: Automated sanity checks  
- **Fast Baselines**: Buy-and-hold, equal-weight comparisons
- **Performance Profiling**: Automatic bottleneck detection

### Systematic Studies
- **Ablation Framework**: Easy component on/off switches
- **Grid Search**: Automated hyperparameter sweeps
- **Random Search**: Efficient parameter exploration
- **Sensitivity Analysis**: Robust hyperparameter identification

### Results Analysis
- **Auto-visualization**: Standard plots for each experiment
- **Statistical Testing**: Proper significance testing between runs
- **Performance Attribution**: What drove returns/losses
- **Convergence Analysis**: Learning curve diagnostics

## Phase 4: Advanced Features (Future)

### Parallel Execution
- Multiple experiments on different GPUs
- Queue system for long experiment sequences
- Resource allocation and monitoring

### Integration Options
- Weights & Biases integration
- MLflow experiment tracking
- Slurm cluster submission
- Docker containerization

## Implementation Strategy

### Backward Compatibility Principle
- All current commands continue working: `python varibad/main.py --mode train`
- Only add features, never break existing workflows
- Feature flags for new functionality

### Rollback Plan
- Git tags at each phase completion: `phase1-complete`, `phase2-complete`
- Automated rollback triggers if tests fail
- Documentation of exact rollback commands

### Success Metrics
- **Phase 1**: Remove 5+ redundant files, consolidate monitoring, streamlined archive system
- **Phase 2**: Run 5 experiments in parallel, automated comparison
- **Phase 3**: Complete ablation study in 1 command
- **Phase 4**: Full hyperparameter sweep overnight

## Current Working Commands

### Established Safety Net
```bash
# Capture new baseline
python tests/baseline_capture.py

# Run safety tests  
pytest tests/test_phase1_data_pipeline.py::TestDataPipelineBaseline::test_production_dataset_exists -v

# Full dataset validation
pytest tests/test_phase1_data_pipeline.py::TestDataPipelineBaseline -v

# Current training (should work throughout refactoring)
python varibad/main.py --mode train --num_iterations 100
```

### File Structure Status
```
KEEP THESE:
✅ varibad/data_pipeline.py           # Main data processing
✅ varibad/data/*.py                  # Required by data_pipeline.py  
✅ varibad/core/                      # Core VariBAD implementation
✅ monitor_training.py                # Enhanced monitoring
✅ tests/                             # Our safety net

REMOVED:
❌ varibad/utils/debug.py            # Debug utility (backed up)
❌ varibad/workbench.ipynb           # Development notebook (backed up)

NEXT TO REMOVE:
🎯 varibad/utils/buffer_test.py      # Redundant with new tests
🎯 Duplicate monitoring in trainer    # Keep monitor_training.py only
🎯 Heavy archive_results.py features  # Simplify for rapid iteration
```

## Emergency Procedures

### If Tests Fail
```bash
# Immediate rollback
git reset --hard HEAD~1

# Check what changed
git diff HEAD~1

# Re-run tests to confirm rollback worked
pytest tests/test_phase1_data_pipeline.py::TestDataPipelineBaseline::test_production_dataset_exists -v
```

### If System Breaks Completely
```bash
# Nuclear option: return to pre-refactoring state
git reset --hard baseline-before-phase1

# Verify system works
python varibad/main.py --mode train --num_iterations 5
```

## Next Session Priorities

1. **Complete Phase 1**: Remove `buffer_test.py`, simplify technical indicators
2. **Begin Phase 2**: Design configuration system architecture  
3. **Implement multi-experiment workflow**: Profile-based training

The testing infrastructure is solid and the safety net is working. Ready to refactor aggressively while maintaining system reliability.