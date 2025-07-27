# VariBAD Testing Infrastructure

This testing infrastructure provides a comprehensive safety net for refactoring the VariBAD portfolio optimization system. It's designed to ensure we can safely consolidate code, simplify components, and improve experimental workflows without breaking functionality.

## Quick Start

### 1. Set Up Testing Infrastructure

```bash
# Run from project root directory
python setup_testing.py
```

This creates the testing directory structure and basic scripts.

### 2. Install Test Dependencies

```bash
pip install pytest
```

### 3. Copy Test Files

Copy these files to your `tests/` directory:
- `test_infrastructure.py` (test utilities and baseline capture)
- `conftest.py` (pytest fixtures and configuration)
- `test_phase1_data_pipeline.py` (Phase 1 specific tests)

### 4. Capture Baseline

```bash
# Capture current system state before refactoring
python tests/baseline_capture.py
```

This will:
- Create/validate your dataset
- Run a quick training test
- Capture all current system metrics
- Create baseline files for comparison

### 5. Run Tests

```bash
# Run all tests
pytest tests/ -v

# Or use the simple runner
python tests/run_tests.py
```

## Testing Strategy

### Four Levels of Testing

1. **Unit Tests** - Individual functions (< 1 second)
2. **Integration Tests** - Component interaction (< 30 seconds)  
3. **End-to-End Tests** - Full pipeline (2-10 minutes)
4. **Regression Tests** - Compare against baseline (10+ minutes)

### Test Organization

```
tests/
├── conftest.py                     # Pytest fixtures
├── test_infrastructure.py          # Test utilities
├── baseline_capture.py             # Baseline capture script
├── run_tests.py                   # Simple test runner
├── test_phase1_data_pipeline.py   # Phase 1 tests
├── data/                          # Test data
├── baselines/                     # Reference baselines
└── reports/                       # Test reports
```

## Phase 1: Data Pipeline Consolidation

### Safety Net Tests

**Before refactoring**, these tests establish what should remain constant:

- `test_production_dataset_exists()` - Verify dataset is available
- `test_dataset_integrity()` - Check data quality and structure
- `test_technical_indicators_present()` - Verify all expected features exist
- `test_data_pipeline_reproducibility()` - Ensure consistent outputs

**During refactoring**, these tests provide safety:

- `test_simplified_pipeline_equivalence()` - New pipeline = old pipeline output
- `test_indicator_subset_validity()` - Reduced indicators still meaningful
- `test_performance_regression()` - No major slowdowns

**After refactoring**, these tests validate success:

- `test_simplified_data_trains_successfully()` - Training still works
- `test_full_pipeline_integration()` - End-to-end workflow intact

### Running Phase 1 Tests

```bash
# Run the complete Phase 1 test suite
./tests/phase1_test_runner.sh

# Or run specific test categories
pytest tests/test_phase1_data_pipeline.py::TestDataPipelineBaseline -v
pytest tests/test_phase1_data_pipeline.py::TestTechnicalIndicatorValidation -v
pytest tests/test_phase1_data_pipeline.py::TestDataPipelinePerformance -v
```

## Key Features

### Baseline Capture System

Captures complete system state including:
- Dataset properties (shape, columns, ranges, checksums)
- Training performance metrics
- System environment info
- Git repository state
- File structure

### Deterministic Testing

- Fixed random seeds for reproducible results
- Sample datasets with known properties  
- Reference values for technical indicators
- Tolerance-based comparisons for floating point

### Performance Monitoring

- Execution time tracking
- Memory usage monitoring
- Regression detection (>10% slowdown triggers warning)
- Baseline performance comparison

### Isolated Test Environments

- Temporary directories for each test
- No interference between tests
- Automatic cleanup after test completion
- Mock data generators for consistent testing

## Usage Patterns

### Safe Refactoring Workflow

1. **Capture baseline** before any changes
2. **Run safety tests** to establish current behavior
3. **Make small changes** (one file at a time)
4. **Run tests after each change**
5. **Compare outputs** against baseline
6. **Rollback if tests fail**

### Example: Removing Redundant Code

```bash
# 1. Capture baseline
python tests/baseline_capture.py

# 2. Run all tests (should pass)
pytest tests/ -v

# 3. Make change (e.g., remove duplicate function)
# ... edit code ...

# 4. Test immediately
pytest tests/test_phase1_data_pipeline.py::test_data_pipeline_reproducibility -v

# 5. If passes, commit; if fails, investigate
git add . && git commit -m "Remove duplicate function - tests pass"

# 6. Periodically run full test suite
pytest tests/ -v
```

### Baseline Comparison

```python
# Compare current dataset against baseline
from tests.test_infrastructure import BaselineCapture

baseline = BaselineCapture("tests/baselines")
baseline.compare_dataset_baseline("data/sp500_rl_ready_cleaned.parquet", "baseline_20250127_143022")
```

## Test Configuration

### Environment Variables

```bash
export PYTEST_CURRENT_TEST="true"  # Activates test mode
export VARIBAD_TEST_DATA_DIR="tests/data"  # Test data location
```

### Pytest Configuration

Edit `pytest.ini` to customize:

```ini
[tool:pytest]
testpaths = tests
addopts = -v --tb=short --color=yes
markers = 
    slow: marks tests as slow
    integration: marks integration tests
    baseline: marks baseline establishment tests
```

### Running Specific Test Types

```bash
# Only fast tests
pytest tests/ -m "not slow" -v

# Only integration tests  
pytest tests/ -m "integration" -v

# Skip baseline tests (for rapid iteration)
pytest tests/ -m "not baseline" -v
```

## Troubleshooting

### Common Issues

**"Production dataset not found"**
```bash
# Create the dataset first
python varibad/main.py --mode data_only
```

**"Baseline file not found"**
```bash
# Capture baseline first
python tests/baseline_capture.py
```

**"Tests fail after refactoring"**
```bash
# Check what changed
python tests/compare_baseline.py

# If expected, update baseline
python tests/baseline_capture.py
```

**"ImportError in tests"**
```bash
# Make sure you're in project root
pwd  # Should show your project directory

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"
```

### Debugging Test Failures

1. **Run with verbose output**:
   ```bash
   pytest tests/test_phase1_data_pipeline.py::test_failing_test -vvv --tb=long
   ```

2. **Use debugging mode**:
   ```bash
   pytest tests/test_phase1_data_pipeline.py::test_failing_test --pdb
   ```

3. **Check baseline differences**:
   ```python
   # In test or debug session
   from tests.test_infrastructure import compare_datasets_with_tolerance
   report = compare_datasets_with_tolerance(old_df, new_df)
   print(report)
   ```

## Next Steps

After Phase 1 (Data Pipeline Consolidation):

1. **Phase 2**: Configuration system enhancement
2. **Phase 3**: Experimental tools development  
3. **Phase 4**: Advanced features (parallel execution, etc.)

Each phase will have its own test suite building on this foundation.

---

**Remember**: The goal is to refactor confidently while maintaining a working system at all times. When in doubt, run the tests!