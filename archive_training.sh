#!/bin/bash
# Simple script to archive your VariBAD training results

echo "📦 VariBAD Training Results Archiver"
echo "===================================="

# Check if training files exist
if [ ! -d "logs" ] && [ ! -d "checkpoints" ]; then
    echo "❌ No training files found (no logs/ or checkpoints/ directories)"
    echo "Make sure you're in the directory where training was run"
    exit 1
fi

echo "Found training files ✅"

# Get run description from user
echo ""
read -p "Enter a name for this training run (optional): " RUN_NAME
read -p "Enter a description for this run (optional): " RUN_DESC

# Create the archive
echo ""
echo "Creating complete training archive..."

if [ ! -z "$RUN_NAME" ] && [ ! -z "$RUN_DESC" ]; then
    python archive_results.py --name "$RUN_NAME" --description "$RUN_DESC"
elif [ ! -z "$RUN_NAME" ]; then
    python archive_results.py --name "$RUN_NAME"
elif [ ! -z "$RUN_DESC" ]; then
    python archive_results.py --description "$RUN_DESC"
else
    python archive_results.py
fi

echo ""
echo "🎉 Archive created successfully!"
echo ""
echo "📋 What's included in your archive:"
echo "• Complete system information and package versions"
echo "• Training configuration and parameters"
echo "• Dataset information and statistics"
echo "• All training metrics and logs"
echo "• Visual analysis plots"
echo "• Executive summary with recommendations"
echo "• All model checkpoints and artifacts"
echo ""
echo "📁 Archives are stored in: archives/"
echo "📦 Zip files can be shared or stored long-term"
echo ""
echo "💡 To view results later, just extract the zip and read executive_summary.md"