#!/bin/bash

BASE_DIR="saved/models"
OUT_DIR="images"

for run_name in "$BASE_DIR"/*; do
    if [[ -d "$run_name" ]]; then
        for run_time in "$run_name"/*; do
            if [[ -d "$run_time" ]]; then
                model_path=""
                if [[ -f "$run_time/model_best.pth" ]]; then
                    model_path="$run_time/model_best.pth"
                else
                    checkpoint_file=$(ls "$run_time"/checkpoint-epoch*.pth 2>/dev/null | sort -V | tail -n 1)
                    if [[ -n "$checkpoint_file" ]]; then
                        model_path="$checkpoint_file"
                    fi
                fi

                if [[ -n "$model_path" ]]; then
                    # Construct output directory
                    rel_run_name=$(basename "$run_name")
                    rel_run_time=$(basename "$run_time")
                    out_path="$OUT_DIR/$rel_run_name/$rel_run_time"

                    # Make sure output directory exists
                    mkdir -p "$out_path"

                    # Call the plotting script
                    echo "Processing $model_path -> $out_path"
                    python test_tze.py --resume "$model_path" --output_dir "$out_path" || continue
                fi
            fi
        done
    fi
done
