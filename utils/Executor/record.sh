#!/bin/bash

execute=$1

modes=("serial" "parallel" "cuda")
tolerances=("15.0" "30.0" "45.0")
obj_files=(assets/tests/*.obj)

for obj_file in "${obj_files[@]}"; do
  for tolerance in "${tolerances[@]}"; do
    for mode in "${modes[@]}"; do
      $execute "$mode" "$tolerance" "$obj_file"
      filename=$(basename -- "$obj_file")
      filename="${filename%.*}"
      segmented_file="assets/tests/Segmented_${mode}_${tolerance}_${filename}.txt"
      node utils/LogTool/parser.js --f "$segmented_file" -t result.json &
    done
  done
done
