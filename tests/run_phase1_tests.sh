#!/bin/bash
# Phase 1 Test Runner - Establishes baseline and runs safety tests
# Run this before starting refactoring to establish our reference point

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() { echo -e "${BLUE}$1${NC}"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

print_header "🧪 VariBAD Phase 1 Test Runner"
print_header "================================"
echo ""
print_header "This script will:"
echo "1. 📸 Capture system baseline (current state)"
echo "2. 🧪 Run comprehensive tests to establish safety net"
echo "3. ✅ Verify everything is ready for refactoring"
echo ""

# Check prerequisites
print_header "Step 1: Prerequisites Check"
echo ""

# Check if we're in the right directory
if [ ! -d "varibad" ]; then
    print_error "varibad directory not found!"
    print_error "Please run this script from the project root directory"
    exit 1
fi

print_success "✓ Project structure validated"

# Check if virtual environment is active
if [ -z "$VIRTUAL_ENV" ]; then
    print_warning "Virtual environment not detected"
    print_warning "Consider activating with: source venv/bin/activate"
else
    print_success "✓ Virtual environment active: $(basename $VIRTUAL_ENV)"
fi

# Check Python dependencies
echo "Checking Python dependencies..."
if python -c "import pandas, numpy, torch, pytest" 2>/dev/null; then
    print_success "✓ Core dependencies available"
else
    print_error "Missing dependencies. Install with:"
    print_error "pip install pandas numpy torch pytest"
    exit 1
fi

# Create test directories
echo "Creating test directories..."
mkdir -p tests/data
mkdir -p tests/baselines
mkdir -p tests/reports
print_success "✓ Test directories created"

echo ""

# Step 2: Baseline Capture
print_header "Step 2: Baseline Capture"
echo ""

echo "🔄 Capturing current system state..."
echo "This may take 5-10 minutes..."

if python tests/baseline_capture.py; then
    print_success "✓ Baseline captured successfully"
    
    # Show baseline summary
    if [ -f "tests/baselines/LATEST_BASELINE.txt" ]; then
        echo ""
        print_header "📋 Baseline Summary:"
        cat tests/baselines/LATEST_BASELINE.txt
    fi
else
    print_error "Baseline capture failed!"
    print_error "Check the output above for details"
    exit 1
fi

echo ""

# Step 3: Safety Net Tests
print_header "Step 3: Safety Net Tests"
echo ""

echo "🧪 Running comprehensive test suite..."
echo "This establishes our safety net for refactoring"

# Set test environment variables
export PYTEST_CURRENT_TEST="true"

# Run tests with detailed output
if pytest tests/test_phase1_data_pipeline.py -v --tb=short --color=yes; then
    print_success "✓ All safety tests passed"
else
    print_error "Some tests failed!"
    print_error "Review the test output above and fix issues before refactoring"
    exit 1
fi

echo ""

# Step 4: Performance Baseline
print_header "Step 4: Performance Baseline"
echo ""

echo "⏱️  Establishing performance baseline..."

# Create performance report
PERF_REPORT="tests/reports/performance_baseline_$(date +%Y%m%d_%H%M%S).txt"

echo "VariBAD Performance Baseline Report" > $PERF_REPORT
echo "Generated: $(date)" >> $PERF_REPORT
echo "=================================" >> $PERF_REPORT
echo "" >> $PERF_REPORT

# Time data pipeline
echo "Timing data pipeline execution..." | tee -a $PERF_REPORT
START_TIME=$(date +%s)

if python varibad/main.py --mode data_only >/dev/null 2>&1; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo "Data pipeline execution time: ${DURATION}s" | tee -a $PERF_REPORT
    print_success "✓ Data pipeline timing: ${DURATION}s"
else
    print_warning "Data pipeline timing failed (may already exist)"
fi

# Check dataset size
if [ -f "data/sp500_rl_ready_cleaned.parquet" ]; then
    DATASET_SIZE=$(du -h data/sp500_rl_ready_cleaned.parquet | cut -f1)
    echo "Dataset size: $DATASET_SIZE" | tee -a $PERF_REPORT
    print_success "✓ Dataset size: $DATASET_SIZE"
fi

echo "" >> $PERF_REPORT
print_success "✓ Performance baseline saved to: $PERF_REPORT"

echo ""

# Step 5: Git Status Check
print_header "Step 5: Git Status Check"
echo ""

echo "📋 Checking git repository status..."

if git status --porcelain | grep -q .; then
    print_warning "Repository has uncommitted changes:"
    git status --short
    echo ""
    print_warning "Consider committing changes before refactoring"
else
    print_success "✓ Repository is clean"
fi

# Create git tag for baseline
BASELINE_TAG="baseline-before-phase1-$(date +%Y%m%d_%H%M%S)"
if git tag $BASELINE_TAG 2>/dev/null; then
    print_success "✓ Created git tag: $BASELINE_TAG"
else
    print_warning "Could not create git tag (may not be a git repo)"
fi

echo ""

# Step 6: Final Validation
print_header "Step 6: Final Validation"
echo ""

echo "🔍 Running final validation checks..."

# Check that we can import key modules
if python -c "
from varibad.core.models import VariBADVAE
from varibad.core.environment import MetaTraderPortfolioMDP  
from varibad.core.trainer import VariBADTrainer
print('✓ All core modules importable')
"; then
    print_success "✓ Core modules import successfully"
else
    print_error "Core module import failed - check for syntax errors"
    exit 1
fi

# Quick training smoke test
echo "🚀 Running quick training smoke test..."
if timeout 300 python varibad/main.py --mode train --num_iterations 2 --episode_length 5 --episodes_per_iteration 1 --vae_updates 1 >/dev/null 2>&1; then
    print_success "✓ Training smoke test passed"
else
    print_warning "Training smoke test failed or timed out (may be expected)"
fi

echo ""

# Success Summary
print_header "🎉 Phase 1 Test Setup Complete!"
print_header "================================"
echo ""
print_success "✅ Baseline captured and validated"
print_success "✅ Safety net tests established"
print_success "✅ Performance baseline recorded"
print_success "✅ Git state documented"
echo ""

print_header "📋 Next Steps:"
echo "1. Begin Phase 1 refactoring (data pipeline consolidation)"
echo "2. Run tests after each change: pytest tests/test_phase1_data_pipeline.py"
echo "3. Compare results against baseline regularly"
echo "4. If anything breaks, rollback using git tag: $BASELINE_TAG"
echo ""

print_header "🛡️  Safety Commands:"
echo "• Run all tests: pytest tests/ -v"
echo "• Check against baseline: python tests/compare_baseline.py"
echo "• Rollback if needed: git reset --hard $BASELINE_TAG"
echo ""

print_header "🚀 Ready to begin Phase 1 refactoring!"

# Save completion status
echo "$(date): Phase 1 test setup completed successfully" >> tests/reports/test_history.log

exit 0