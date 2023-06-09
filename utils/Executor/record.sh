#!/bin/bash

execute=$1
email=$2
key=$3
doc=$4
modes=("serial" "parallel" "cuda")
tolerances=(15.0 30.0)
obj_files=(assets/tests/*.obj) # obj 파일 경로에 맞게 수정해주세요

for mode in "${modes[@]}"; do
  for tolerance in "${tolerances[@]}"; do
    for obj_file in "${obj_files[@]}"; do
      $execute "$mode" "$tolerance" "$obj_file"
      segmented_file="assets/tests/Segmented_${mode}_${tolerance}_${obj_file}.txt"
      node utils/LogTools/parser.js --f "$segmented_file" -e "$email" -k "$key" -d "$doc"
    done
  done
done
