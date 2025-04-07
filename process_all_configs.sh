#!/bin/bash

set -e

process_config() {
    local config_file=$1
    echo "Processing config: $config_file"
    python main.py --config_path "$config_file"
}

for dataset_dir in cfgs/*_data/; do
    if [ -d "$dataset_dir" ]; then
        echo "Processing configs in $dataset_dir"
        for config_file in "$dataset_dir"*.json; do
            if [ -f "$config_file" ]; then
                process_config "$config_file"
            fi
        done
    fi
done

echo "All configs processed successfully!" 