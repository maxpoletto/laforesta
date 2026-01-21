#!/bin/bash

: "${2:?usage: $0 <input_dir> <output_dir>}"
INPUT_DIR="$1"
OUTPUT_DIR="$2"
mkdir -p "$OUTPUT_DIR"
# Output: size, mtime (epoch), full path — tab-separated
gfind "$INPUT_DIR" -ipath '*.dwg' -printf '%s\t%T@\t%p\n' \
  | sort -t$'\t' -k1,1n -k2,2rn \
  | sort -t$'\t' -k1,1n -u \
  | sort -t$'\t' -k3 \
  | {
    x=1
    while IFS=$'\t' read -r size mtime filepath; do
      base="${filepath##*/}"
      f2=$(printf "%03d-%s" "$x" "$base")
      cp -f "$filepath" "$OUTPUT_DIR"/"$f2"
      x=$((x+1))
    done
  }
