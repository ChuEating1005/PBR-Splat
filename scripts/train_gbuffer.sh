#!/bin/bash

# regularization weights
smooth_normal_weight=0.5
smooth_albedo_weight=0.5
smooth_metallic_weight=0.5
smooth_roughness_weight=0.3

object_list=( "car" )
for obj in ${object_list[@]}; do
    python train_gbuffer.py \
        -s data/refnerf/$obj \
        -m output/${obj}_regularized \
        --start_checkpoint data/refnerf/$obj/point_cloud/iteration_30000/point_cloud.ply \
        --iterations 5000 \
        --smooth_normal_weight $smooth_normal_weight \
        --smooth_albedo_weight $smooth_albedo_weight \
        --smooth_metallic_weight $smooth_metallic_weight \
        --smooth_roughness_weight $smooth_roughness_weight
done