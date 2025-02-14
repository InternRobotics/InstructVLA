# shader_dir=rt means that we turn on ray-tracing rendering; this is quite crucial for the open / close drawer task as policies often rely on shadows to infer depth
gpu_id=$2

declare -a ckpt_paths=(
$1
)
# CogACT/CogACT-Large CogACT/CogACT-Small
declare -a env_names=(
OpenTopDrawerCustomInScene-v0
OpenMiddleDrawerCustomInScene-v0
OpenBottomDrawerCustomInScene-v0
CloseTopDrawerCustomInScene-v0
CloseMiddleDrawerCustomInScene-v0
CloseBottomDrawerCustomInScene-v0
)

EXTRA_ARGS="--enable-raytracing"


# base setup
scene_name=frl_apartment_stage_simple

EvalSim() {
  echo ${ckpt_path} ${env_name}

  CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
    --robot google_robot_static \
    --control-freq 3 --sim-freq 513 --max-episode-steps 113 \
    --env-name ${env_name} --scene-name ${scene_name} \
    --robot-init-x 0.65 0.85 3 --robot-init-y -0.2 0.2 3 \
    --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.0 0.0 1 \
    --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 \
    ${EXTRA_ARGS}
}


# for ckpt_path in "${ckpt_paths[@]}"; do
#   for env_name in "${env_names[@]}"; do
#     EvalSim
#   done
# done


# # backgrounds

# declare -a scene_names=(
# "modern_bedroom_no_roof"
# "modern_office_no_roof"
# )

# for scene_name in "${scene_names[@]}"; do
#   for ckpt_path in "${ckpt_paths[@]}"; do
#     for env_name in "${env_names[@]}"; do
#       EXTRA_ARGS="--additional-env-build-kwargs shader_dir=rt"
#       EvalSim
#     done
#   done
# done


# # lightings
# scene_name=frl_apartment_stage_simple

# for ckpt_path in "${ckpt_paths[@]}"; do
#   for env_name in "${env_names[@]}"; do
#     EXTRA_ARGS="--additional-env-build-kwargs shader_dir=rt light_mode=brighter"
#     EvalSim
#     EXTRA_ARGS="--additional-env-build-kwargs shader_dir=rt light_mode=darker"
#     EvalSim
#   done
# done


# new cabinets
scene_name=frl_apartment_stage_simple

for ckpt_path in "${ckpt_paths[@]}"; do
  for env_name in "${env_names[@]}"; do
    EXTRA_ARGS="--additional-env-build-kwargs shader_dir=rt station_name=mk_station2"
    EvalSim
    EXTRA_ARGS="--additional-env-build-kwargs shader_dir=rt station_name=mk_station3"
    EvalSim
  done
done


gpu_id=$2

declare -a arr=($1)

# lr_switch=laying horizontally but flipped left-right to match real eval; upright=standing; laid_vertically=laying vertically
declare -a coke_can_options_arr=("lr_switch=True" "upright=True" "laid_vertically=True")

# URDF variations
declare -a urdf_version_arr=(None "recolor_tabletop_visual_matching_1" "recolor_tabletop_visual_matching_2" "recolor_cabinet_visual_matching_1")

env_name=GraspSingleOpenedCokeCanInScene-v0
scene_name=google_pick_coke_can_1_v4
rgb_overlay_path=./ManiSkill2_real2sim/data/real_inpainting/google_coke_can_real_eval_1.png

for ckpt_path in "${arr[@]}"; do echo "$ckpt_path"; done



for urdf_version in "${urdf_version_arr[@]}";

do for coke_can_option in "${coke_can_options_arr[@]}";

do for ckpt_path in "${arr[@]}";

do CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name ${env_name} --scene-name ${scene_name} \
  --rgb-overlay-path ${rgb_overlay_path} \
  --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  --additional-env-build-kwargs ${coke_can_option} urdf_version=${urdf_version};

done

done

done
