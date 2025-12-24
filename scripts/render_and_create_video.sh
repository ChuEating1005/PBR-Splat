#!/bin/bash

# Complete workflow: Render G-buffer frames and create videos
# Usage: bash scripts/render_and_create_video.sh

set -e  # Exit on error

echo "========================================="
echo "G-buffer Video Rendering Workflow"
echo "========================================="
echo ""

# Configuration
SCENE_NAME="coffee"
CHECKPOINT="output/${SCENE_NAME}_regularized/point_cloud/iteration_5000/point_cloud.ply"
SPARSE_DIR="data/refnerf/${SCENE_NAME}/sparse/0"
OUTPUT_DIR="output/${SCENE_NAME}_regularized/rendered_video"
NUM_FRAMES=120
FPS=30
QUALITY="high"  # Options: high, medium, low

# Step 1: Render frames
echo "Step 1: Rendering G-buffer frames..."
echo "  Checkpoint: $CHECKPOINT"
echo "  Sparse dir: $SPARSE_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  Frames: $NUM_FRAMES"
echo ""

python render_gbuffer_video.py \
    --checkpoint "$CHECKPOINT" \
    --sparse_dir "$SPARSE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --num_frames $NUM_FRAMES \
    --sh_degree 3

echo ""
echo "========================================="
echo ""

# Step 2: Create videos from frames
echo "Step 2: Creating MP4 videos..."
echo "  FPS: $FPS"
echo "  Quality: $QUALITY"
echo ""

python create_video_from_frames.py \
    --input_dir "$OUTPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --fps $FPS \
    --quality "$QUALITY" \
    --create_combined

echo ""
echo "========================================="
echo "✅ Complete! Videos saved to: $OUTPUT_DIR"
echo "========================================="

