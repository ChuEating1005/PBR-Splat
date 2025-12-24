#!/bin/bash  

# List of tasks  
scan_list=( "coffee_regularized" "helmet_regularized" "teapot_regularized" )   

hdr_list=( "flower_road_no_sun_2k.hdr" 
           "lightroom_14b.hdr"
           "pillars_2k.hdr"
           "studio_small_02_2k.hdr"
           "syferfontein_18d_clear_2k.hdr"
           "the_sky_is_on_fire_2k.hdr"
         )
  
  
# Generate the bash commands for each GPU  
for scan_name in ${scan_list[@]}; do
    for hdr_list_name in ${hdr_list[@]}; do
        model_dir="output/${scan_name}"
        hdr_path="hdri/${hdr_list_name}"
        python render.py -m ${model_dir} --skip_test --save_name ${hdr_list_name} --environment_texture $hdr_path --render_relight --hdr_rotation 
    done
done

echo "All tasks are finished!" 