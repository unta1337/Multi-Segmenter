#!/bin/bash

execute=$1
modes=("cuda")
tolerances=("15.0")
obj_files=(assets/tests/*.obj)

for obj_file in "${obj_files[@]}"; do
  for tolerance in "${tolerances[@]}"; do
    for mode in "${modes[@]}"; do
      filename=$(basename -- "$obj_file")
      filename="${filename%.*}"
      profile_result_path="assets/tests/Profile_${mode}_${tolerance}_${filename}"
      ncu --export "$profile_result_path" --force-overwrite --target-processes application-only --replay-mode kernel --kernel-name-base function --launch-skip-before-match 0 --filter-mode global --section LaunchStats --section Occupancy --section SpeedOfLight --sampling-interval auto --sampling-max-passes 5 --sampling-buffer-size 33554432 --profile-from-start 1 --cache-control all --clock-control base --apply-rules yes --import-source no --check-exit-code yes "$execute" "$mode" "$tolerance" "$obj_file"
    done
  done
done
