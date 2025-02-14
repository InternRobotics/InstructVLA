gpu_id=$2

declare -a arr=($1)

# lr_switch=laying horizontally but flipped left-right to match real eval; upright=standing; laid_vertically=laying vertically
declare -a coke_can_options_arr=("lr_switch=True" "upright=True" "laid_vertically=True")

for ckpt_path in "${arr[@]}"; do echo "$ckpt_path"; done


# # base setup

# env_name=GraspSingleOpenedCokeCanInScene-v0
# scene_name=google_pick_coke_can_1_v4

# for coke_can_option in "${coke_can_options_arr[@]}";

# do for ckpt_path in "${arr[@]}";

# do CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
#   --robot google_robot_static \
#   --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
#   --env-name ${env_name} --scene-name ${scene_name} \
#   --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
#   --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
#   --additional-env-build-kwargs ${coke_can_option};

# done

# done



# # table textures

# env_name=GraspSingleOpenedCokeCanInScene-v0

# declare -a scene_arr=("Baked_sc1_staging_objaverse_cabinet1_h870" \
#                       "Baked_sc1_staging_objaverse_cabinet2_h870")


# for coke_can_option in "${coke_can_options_arr[@]}";

# do for scene_name in "${scene_arr[@]}";

# do for ckpt_path in "${arr[@]}";

# do CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
#   --robot google_robot_static \
#   --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
#   --env-name ${env_name} --scene-name ${scene_name} \
#   --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
#   --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
#   --additional-env-build-kwargs ${coke_can_option};

# done

# done

# done




# # distractors

# env_name=GraspSingleOpenedCokeCanDistractorInScene-v0
# scene_name=google_pick_coke_can_1_v4

# for coke_can_option in "${coke_can_options_arr[@]}";

# do for ckpt_path in "${arr[@]}";

# do CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
#   --robot google_robot_static \
#   --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
#   --env-name ${env_name} --scene-name ${scene_name} \
#   --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
#   --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
#   --additional-env-build-kwargs ${coke_can_option};

# CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
#   --robot google_robot_static \
#   --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
#   --env-name ${env_name} --scene-name ${scene_name} \
#   --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
#   --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
#   --additional-env-build-kwargs ${coke_can_option} distractor_config=more;

# done

# done




# # backgrounds

# env_name=GraspSingleOpenedCokeCanInScene-v0
# declare -a scene_arr=("google_pick_coke_can_1_v4_alt_background" \
#                       "google_pick_coke_can_1_v4_alt_background_2")

# for coke_can_option in "${coke_can_options_arr[@]}";

# do for scene_name in "${scene_arr[@]}";

# do for ckpt_path in "${arr[@]}";

# do CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
#   --robot google_robot_static \
#   --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
#   --env-name ${env_name} --scene-name ${scene_name} \
#   --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
#   --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
#   --additional-env-build-kwargs ${coke_can_option};

# done

# done

# done



# # lightings

# env_name=GraspSingleOpenedCokeCanInScene-v0
# scene_name=google_pick_coke_can_1_v4

# for coke_can_option in "${coke_can_options_arr[@]}";

# do for ckpt_path in "${arr[@]}";

# do CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
#   --robot google_robot_static \
#   --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
#   --env-name ${env_name} --scene-name ${scene_name} \
#   --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
#   --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
#   --additional-env-build-kwargs ${coke_can_option} slightly_darker_lighting=True;

# CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
#   --robot google_robot_static \
#   --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
#   --env-name ${env_name} --scene-name ${scene_name} \
#   --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
#   --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
#   --additional-env-build-kwargs ${coke_can_option} slightly_brighter_lighting=True;

# done

# done




# camera orientations

declare -a env_arr=("GraspSingleOpenedCokeCanAltGoogleCameraInScene-v0" \
                   "GraspSingleOpenedCokeCanAltGoogleCamera2InScene-v0")
scene_name=google_pick_coke_can_1_v4

for coke_can_option in "${coke_can_options_arr[@]}";

do for env_name in "${env_arr[@]}";

do for ckpt_path in "${arr[@]}";

do CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name ${env_name} --scene-name ${scene_name} \
  --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  --additional-env-build-kwargs ${coke_can_option};

done

done

done



# shader_dir=rt means that we turn on ray-tracing rendering; this is quite crucial for the open / close drawer task as policies often rely on shadows to infer depth
gpu_id=$2

declare -a ckpt_paths=(
$1
)

declare -a env_names=(
OpenTopDrawerCustomInScene-v0
OpenMiddleDrawerCustomInScene-v0
OpenBottomDrawerCustomInScene-v0
# CloseTopDrawerCustomInScene-v0
# CloseMiddleDrawerCustomInScene-v0
# CloseBottomDrawerCustomInScene-v0
)

# URDF variations
declare -a urdf_version_arr=("recolor_cabinet_visual_matching_1" "recolor_tabletop_visual_matching_1" "recolor_tabletop_visual_matching_2" None)

for urdf_version in "${urdf_version_arr[@]}"; do

EXTRA_ARGS="--enable-raytracing --additional-env-build-kwargs station_name=mk_station_recolor light_mode=simple disable_bad_material=True urdf_version=${urdf_version}"

EvalOverlay() {
# A0
CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 113 \
  --env-name ${env_name} --scene-name dummy_drawer \
  --robot-init-x 0.644 0.644 1 --robot-init-y -0.179 -0.179 1 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.03 -0.03 1 \
  --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 \
  --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/open_drawer_a0.png \
  ${EXTRA_ARGS}

# A1
CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 113 \
  --env-name ${env_name} --scene-name dummy_drawer \
  --robot-init-x 0.765 0.765 1 --robot-init-y -0.182 -0.182 1 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.02 -0.02 1 \
  --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 \
  --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/open_drawer_a1.png \
  ${EXTRA_ARGS}

# A2
CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 113 \
  --env-name ${env_name} --scene-name dummy_drawer \
  --robot-init-x 0.889 0.889 1 --robot-init-y -0.203 -0.203 1 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.06 -0.06 1 \
  --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 \
  --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/open_drawer_a2.png \
  ${EXTRA_ARGS}

# B0
CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 113 \
  --env-name ${env_name} --scene-name dummy_drawer \
  --robot-init-x 0.652 0.652 1 --robot-init-y 0.009 0.009 1 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 \
  --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/open_drawer_b0.png \
  ${EXTRA_ARGS}

# B1
CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 113 \
  --env-name ${env_name} --scene-name dummy_drawer \
  --robot-init-x 0.752 0.752 1 --robot-init-y 0.009 0.009 1 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 \
  --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/open_drawer_b1.png \
  ${EXTRA_ARGS}

# B2
CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 113 \
  --env-name ${env_name} --scene-name dummy_drawer \
  --robot-init-x 0.851 0.851 1 --robot-init-y 0.035 0.035 1 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 \
  --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/open_drawer_b2.png \
  ${EXTRA_ARGS}

# C0
CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 113 \
  --env-name ${env_name} --scene-name dummy_drawer \
  --robot-init-x 0.665 0.665 1 --robot-init-y 0.224 0.224 1 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 \
  --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/open_drawer_c0.png \
  ${EXTRA_ARGS}

# C1
CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 113 \
  --env-name ${env_name} --scene-name dummy_drawer \
  --robot-init-x 0.765 0.765 1 --robot-init-y 0.222 0.222 1 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.025 -0.025 1 \
  --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 \
  --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/open_drawer_c1.png \
  ${EXTRA_ARGS}

# C2
CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 113 \
  --env-name ${env_name} --scene-name dummy_drawer \
  --robot-init-x 0.865 0.865 1 --robot-init-y 0.222 0.222 1 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.025 -0.025 1 \
  --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 \
  --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/open_drawer_c2.png \
  ${EXTRA_ARGS}
}


for ckpt_path in "${ckpt_paths[@]}"; do
  for env_name in "${env_names[@]}"; do
    EvalOverlay
  done
done



done
