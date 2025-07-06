#!/bin/bash

set -e  # Exit on error

# ====== CONFIGURATION ======
VIDEO_NAME="seq1"  # Change this to the video you want to debug
MATCHERS=("ensemble" )  
# MATCHERS=("ensemble") 
RESULTS_DIR="results"
STEP=1  # Data is already downsampled to 1fps
PYTHON_SCRIPT="scripts/matching_analysis.py"

echo "🔍 Starting debug analysis for: $VIDEO_NAME"
echo "📁 Results will be saved to: $RESULTS_DIR"

# ====== RUN MATCHERS ======
for matcher in "${MATCHERS[@]}"; do
    echo ""
    echo "⚙️  Running matcher: $matcher"
    PYTHONPATH=$(pwd) python "$PYTHON_SCRIPT" \
        --video_name "$VIDEO_NAME" \
        --matcher "$matcher" \
        --out_dir "$RESULTS_DIR" \
        --step "$STEP"\
        --top_frac "0.5" \
        --dedup_thresh "3"
done

echo ""
echo "✅ Debugging complete for $VIDEO_NAME."
