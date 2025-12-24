#!/bin/bash

# Create videos for all relighting results
# Usage: bash scripts/create_relight_videos.sh

set -e

echo "========================================="
echo "Relight Video Creation Workflow"
echo "========================================="
echo ""

# Configuration
SCENE_NAME="car_regularized"
RELIGHT_DIR="output/${SCENE_NAME}/relight"
FPS=30
QUALITY="high"
GRID_OUTPUT="${RELIGHT_DIR}/relight_grid_2x3.mp4"

# HDR list from relight.sh
hdr_list=( "flower_road_no_sun_2k.hdr" 
           "lightroom_14b.hdr"
           "pillars_2k.hdr"
           "studio_small_02_2k.hdr"
           "syferfontein_18d_clear_2k.hdr"
           "the_sky_is_on_fire_2k.hdr"
         )

# Track generated videos (same order as hdr_list)
video_paths=()

for hdr in "${hdr_list[@]}"
do
    echo "Processing HDR: $hdr"
    
    # Path to the renders directory
    # Structure: output/car/relight/HDR_NAME/ours_5000/renders/
    RENDER_DIR="${RELIGHT_DIR}/${hdr}/ours_10000/renders"
    VIDEO_OUTPUT="${RELIGHT_DIR}/${hdr}.mp4"
    
    if [ ! -d "$RENDER_DIR" ]; then
        echo "❌ Warning: Render directory not found: $RENDER_DIR"
        continue
    fi
    
    echo "  Input: $RENDER_DIR"
    echo "  Output: $VIDEO_OUTPUT"
    
    # We use a simplified python command just to create video from one sequence
    # Since create_video_from_frames.py is designed for G-buffer folders (albedo/normal/etc)
    # we can use a small python snippet or ffmpeg directly here.
    # But to reuse existing tools, we can call ffmpeg directly since it's simple.
    
    # Quality presets (matching create_video_from_frames.py)
    CRF=18 # High quality
    
    # Use ffmpeg to create video
    ffmpeg -y \
        -framerate $FPS \
        -i "${RENDER_DIR}/%05d.png" \
        -c:v libx264 \
        -crf $CRF \
        -pix_fmt yuv420p \
        "$VIDEO_OUTPUT" > /dev/null 2>&1
        
    if [ $? -eq 0 ]; then
        echo "✅ Video created: $VIDEO_OUTPUT"
        video_paths+=( "$VIDEO_OUTPUT" )
    else
        echo "❌ Failed to create video for $hdr"
    fi
    echo ""
done

echo "========================================="
echo "Creating 2x3 grid concat video..."
echo "  Output: $GRID_OUTPUT"

# We expect exactly 6 videos (one per HDR). If some are missing, skip grid creation.
if [ ${#video_paths[@]} -ne 6 ]; then
    echo "⚠️  Skipping grid video: expected 6 videos, found ${#video_paths[@]}."
else
    # Build ffmpeg inputs
    ffmpeg_cmd=(ffmpeg -y)
    for vp in "${video_paths[@]}"; do
        ffmpeg_cmd+=( -i "$vp" )
    done

    # xstack layout for 2 rows x 3 cols:
    # [0][1][2]
    # [3][4][5]
    #
    # Ensure all streams are same even dimensions, then stack.
    filter_complex="\
[0:v]scale=trunc(iw/2)*2:trunc(ih/2)*2,setsar=1[v0];\
[1:v]scale=trunc(iw/2)*2:trunc(ih/2)*2,setsar=1[v1];\
[2:v]scale=trunc(iw/2)*2:trunc(ih/2)*2,setsar=1[v2];\
[3:v]scale=trunc(iw/2)*2:trunc(ih/2)*2,setsar=1[v3];\
[4:v]scale=trunc(iw/2)*2:trunc(ih/2)*2,setsar=1[v4];\
[5:v]scale=trunc(iw/2)*2:trunc(ih/2)*2,setsar=1[v5];\
[v0][v1][v2][v3][v4][v5]xstack=inputs=6:layout=0_0|w0_0|w0+w1_0|0_h0|w0_h0|w0+w1_h0:fill=black[out]"

    "${ffmpeg_cmd[@]}" \
        -filter_complex "$filter_complex" \
        -map "[out]" \
        -c:v libx264 \
        -crf 18 \
        -pix_fmt yuv420p \
        "$GRID_OUTPUT" > /dev/null 2>&1

    if [ $? -eq 0 ]; then
        echo "✅ Grid video created: $GRID_OUTPUT"
    else
        echo "❌ Failed to create grid video: $GRID_OUTPUT"
    fi
fi

echo "========================================="
echo "All tasks finished!"

