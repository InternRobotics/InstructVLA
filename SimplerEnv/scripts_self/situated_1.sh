# shader_dir=rt means that we turn on ray-tracing rendering; this is quite crucial for the open / close drawer task as policies often rely on shadows to infer depth
gpu_id=$2

declare -a ckpt_paths=(
$1
)

declare -a env_names=(
AltGraspSpongeDistractorInSceneEnv-v0
AltGraspOrangeDistractorInSceneEnv-v0
AltGraspOrange2DistractorInSceneEnv-v0
AltGraspEggplantDistractorInSceneEnv-v0
AltGraspMugDistractorInSceneEnv-v0
AltGraspMugDistractorInSceneEnv-v1
)

# declare -a coke_can_options_arr=("lr_switch=True" "upright=True" "laid_vertically=True")

EvalOverlay() {

CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name ${env_name} --scene-name google_pick_coke_can_1_v4_alt_background \
  --robot-init-x 0.35 0.35 1 --robot-init-y 0.21 0.21 1 --obj-variation-mode xy\
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.0 0.0 1 \
  --obj-init-x-range -0.1 -0.3 4 --obj-init-y-range 0.0 0.4 4\
  --additional-env-build-kwargs laid_vertically=True

CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name ${env_name} --scene-name google_pick_coke_can_1_v4_alt_background_2 \
  --robot-init-x 0.35 0.35 1 --robot-init-y 0.21 0.21 1 --obj-variation-mode xy\
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.0 0.0 1 \
  --obj-init-x-range -0.1 -0.3 4 --obj-init-y-range 0.0 0.4 4\
  --additional-env-build-kwargs laid_vertically=True distractor_config=more

}


for ckpt_path in "${ckpt_paths[@]}"; do
  for env_name in "${env_names[@]}"; do
    EvalOverlay
  done

done