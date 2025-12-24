#!/bin/bash

# Render G-buffer video from trained model
# Usage: bash scripts/render_video.sh

python render_gbuffer_video.py \
    --checkpoint output/car/point_cloud/iteration_30000/point_cloud.ply \
    --sparse_dir data/refnerf/car/sparse/0 \
    --output_dir output/car/rendered_video \
    --num_frames 120 \
    --sh_degree 3

