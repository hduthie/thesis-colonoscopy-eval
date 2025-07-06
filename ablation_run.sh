#!/bin/bash

# VIDEO="cecum_t2_a"
OUT_DIR="results"
STEP=1

VIDEOS=(
  # "cecum_t1_a"
  # "cecum_t1_b"
  # "cecum_t2_a"
  "cecum_t2_b"
  # "cecum_t2_c"
  # "cecum_t4_a"
  # "cecum_t4_b"
  # "sigmoid_t2_a"
  # "trans_t2_a"
  # "trans_t2_b"
  # "trans_t2_c"
  # "trans_t4_a"
  # "cecum_t3_a"
  # "desc_t4_a"
  # "sigmoid_t1_a"
  # "sigmoid_t3_a"
  # "sigmoid_t3_b"
  # "trans_t1_a"
  # "trans_t1_b"
  # "trans_t3_a"
  # "trans_t4_b"
  # "seq1"
  # "trans_t3_b"
  # "seq2"
  # "seq3"
  # "seq4"
)


TOP_FRACS=(0.2 0.3 0.5 0.7)
THRESHOLDS=(1 3 5)
for VIDEO in "${VIDEOS[@]}"; do
  echo "🔍 Starting analysis for video: $VIDEO"
    for TOP_FRAC in "${TOP_FRACS[@]}"; do
        for THRESH in "${THRESHOLDS[@]}"; do
            echo "🧪 Running ensemble with top_frac=${TOP_FRAC}, dedup_thresh=${THRESH}"
            
            TOP_FRAC=$TOP_FRAC DEDUP_THRESH=$THRESH \
            PYTHONPATH=$(pwd) python scripts/matching_analysis.py \
                --video_name "$VIDEO" \
                --matcher "ensemble" \
                --out_dir "$OUT_DIR/Ensemble_ablation" \
                --step "$STEP"\
                --top_frac "$TOP_FRAC" \
                --dedup_thresh "$THRESH"
            echo "✅ Completed analysis for top_frac=${TOP_FRAC}, dedup_thresh=${THRESH}"
        echo "-----------------------------"
        done
    done
done