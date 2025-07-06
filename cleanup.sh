#!/bin/bash

# List of folder names to delete
names=(
  "topfrac_0.2_thresh_1"
  "topfrac_0.2_thresh_3"
  "topfrac_0.2_thresh_5"
  "topfrac_0.2_thresh_10"
  "topfrac_0.3_thresh_1"
  "topfrac_0.3_thresh_3"
  "topfrac_0.3_thresh_5"
  "topfrac_0.3_thresh_10"
  "topfrac_0.5_thresh_1"
  "topfrac_0.5_thresh_3"
  "topfrac_0.5_thresh_5"
  "topfrac_0.5_thresh_10"
  "topfrac_0.7_thresh_1"
  "topfrac_0.7_thresh_3"
  "topfrac_0.7_thresh_5"
  "topfrac_0.7_thresh_10"
  "topfrac_1.0_thresh_1"
  "topfrac_1.0_thresh_3"
  "topfrac_1.0_thresh_5"
  "topfrac_1.0_thresh_10"
)

# Delete each one from results/
for name in "${names[@]}"; do
  path="results/$name"
  if [ -e "$path" ]; then
    rm -rf "$path"
    echo "Deleted: $path"
  else
    echo "Not found: $path"
  fi
done
