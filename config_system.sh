#!/bin/bash
# Test Phase 2 Configuration System
# This script tests the new enhanced configuration features

echo "🧪 Testing Phase 2 Configuration System"
echo "========================================"

# Step 1: Setup configuration system
echo "1. Setting up configuration system..."
python config_structure_setup.py

echo ""
echo "2. Testing configuration loading..."

# Step 2: Test basic profile usage
echo "Testing debug profile..."
python varibad/main.py --config profiles/debug.conf --mode train

echo ""
echo "3. Testing parameter sweep..."

# Step 3: Test parameter sweep
echo "Testing latent dimension sweep..."
python varibad/main.py --config profiles/debug.conf --sweep latent_dim=3,5,8

echo ""
echo "4. Testing specific experiment..."

# Step 4: Test specific experiment
echo "Testing baseline experiment..."
python varibad/main.py --config experiments/exp_001_baseline.conf --mode train

echo ""
echo "✅ Phase 2 Configuration System Test Complete!"
echo ""
echo "Available commands:"
echo "  python varibad/main.py --config profiles/debug.conf"
echo "  python varibad/main.py --config profiles/development.conf"
echo "  python varibad/main.py --config experiments/exp_001_baseline.conf"
echo "  python varibad/main.py --sweep latent_dim=3,5,8 episode_length=15,30"
echo ""
echo "Check config/README.md for full documentation"