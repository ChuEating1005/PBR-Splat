#!/bin/bash

OUTPUT_DIR="output/car"
VISUALS_DIR="$OUTPUT_DIR/visuals"

# Loop through all iter_* directories
# for iter_dir in "$VISUALS_DIR"/iter_*; do
#     if [ -d "$iter_dir" ]; then
#         echo "Processing: $iter_dir"
#         python create_comparison_figure.py "$iter_dir"
#     fi
# done
python create_comparison_figure.py "$VISUALS_DIR/iter_000"