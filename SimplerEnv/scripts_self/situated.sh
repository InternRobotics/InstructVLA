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


# ============================== long horizon =====================================
CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 150 \
  --env-name AltCloseDrawerInLongStep2HorizonInSceneEnv-v0 --scene-name frl_apartment_stage_simple \
  --robot-init-x 0.55 0.75 3 --robot-init-y -0.2 0.2 3 --obj-variation-mode xy\
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.0 0.0 1 \
  --obj-init-x-range -0.1 -0.3 1 --obj-init-y-range 0.0 0.2 1\
  --additional-env-build-kwargs model_ids=orange shader_dir=rt

CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 150 \
  --env-name AltCloseDrawerInLongStep2HorizonInSceneEnv-v0 --scene-name frl_apartment_stage_simple \
  --robot-init-x 0.55 0.75 3 --robot-init-y -0.2 0.2 3 --obj-variation-mode xy\
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.0 0.0 1 \
  --obj-init-x-range -0.1 -0.3 1 --obj-init-y-range 0.0 0.2 1\
  --additional-env-build-kwargs model_ids=apple shader_dir=rt

CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 150 \
  --env-name AltCloseDrawerInLongStep2HorizonInSceneEnv-v0 --scene-name frl_apartment_stage_simple \
  --robot-init-x 0.55 0.75 3 --robot-init-y -0.2 0.2 3 --obj-variation-mode xy\
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.0 0.0 1 \
  --obj-init-x-range -0.1 -0.3 1 --obj-init-y-range 0.0 0.2 1\
  --additional-env-build-kwargs model_ids=coffee_mug shader_dir=rt

CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 150 \
  --env-name AltCloseMiddleDrawerInLongStep2HorizonInSceneEnv-v0 --scene-name frl_apartment_stage_simple \
  --robot-init-x 0.55 0.75 3 --robot-init-y -0.2 0.2 3 --obj-variation-mode xy\
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.0 0.0 1 \
  --obj-init-x-range -0.1 -0.3 1 --obj-init-y-range 0.0 0.2 1\
  --additional-env-build-kwargs model_ids=orange shader_dir=rt

CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 150 \
  --env-name AltCloseMiddleDrawerInLongStep2HorizonInSceneEnv-v0 --scene-name frl_apartment_stage_simple \
  --robot-init-x 0.55 0.75 3 --robot-init-y -0.2 0.2 3 --obj-variation-mode xy\
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.0 0.0 1 \
  --obj-init-x-range -0.1 -0.3 1 --obj-init-y-range 0.0 0.2 1\
  --additional-env-build-kwargs model_ids=apple shader_dir=rt


CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 150 \
  --env-name AltCloseDrawerInLongStep1HorizonInSceneEnv-v0 --scene-name frl_apartment_stage_simple \
  --robot-init-x 0.55 0.75 3 --robot-init-y -0.2 0.2 3 --obj-variation-mode xy\
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.0 0.0 1 \
  --obj-init-x-range -0.1 -0.3 1 --obj-init-y-range 0.0 0.2 1\
  --additional-env-build-kwargs model_ids=opened_coke_can shader_dir=rt

CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 150 \
  --env-name AltCloseDrawerInLongStep1HorizonInSceneEnv-v0 --scene-name modern_bedroom_no_roof \
  --robot-init-x 0.55 0.75 3 --robot-init-y -0.2 0.2 3 --obj-variation-mode xy\
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.0 0.0 1 \
  --obj-init-x-range -0.1 -0.3 1 --obj-init-y-range 0.0 0.2 1\
  --additional-env-build-kwargs model_ids=opened_coke_can shader_dir=rt

# ================================== drawer alter =====================================
CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 113 \
  --env-name AltOpenBottomDrawerCustomInSceneEnv-v0 --scene-name modern_bedroom_no_roof \
  --robot-init-x 0.65 0.80 3 --robot-init-y -0.2 0.2 3\
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.0 0.0 1 \
  --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 \
  --additional-env-build-kwargs shader_dir=rt

CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 113 \
  --env-name AltOpenTopDrawerCustomInSceneEnv-v0 --scene-name modern_bedroom_no_roof \
  --robot-init-x 0.65 0.80 3 --robot-init-y -0.2 0.2 3\
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.0 0.0 1 \
  --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 \
  --additional-env-build-kwargs shader_dir=rt

CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 113 \
  --env-name AltOpenMiddleDrawerCustomInSceneEnv-v0 --scene-name modern_bedroom_no_roof \
  --robot-init-x 0.65 0.80 3 --robot-init-y -0.2 0.2 3\
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.0 0.0 1 \
  --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 \
  --additional-env-build-kwargs shader_dir=rt

CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 113 \
  --env-name AltOpenBottomDrawerCustomInSceneEnv2-v0 --scene-name modern_bedroom_no_roof \
  --robot-init-x 0.65 0.80 3 --robot-init-y -0.2 0.2 3\
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.0 0.0 1 \
  --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 \
  --additional-env-build-kwargs shader_dir=rt

CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 113 \
  --env-name AltOpenTopDrawerCustomInSceneEnv2-v0 --scene-name modern_bedroom_no_roof \
  --robot-init-x 0.65 0.80 3 --robot-init-y -0.2 0.2 3\
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.0 0.0 1 \
  --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 \
  --additional-env-build-kwargs shader_dir=rt

CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 113 \
  --env-name AltOpenMiddleDrawerCustomInSceneEnv2-v0 --scene-name modern_bedroom_no_roof \
  --robot-init-x 0.65 0.80 3 --robot-init-y -0.2 0.2 3\
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.0 0.0 1 \
  --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 \
  --additional-env-build-kwargs shader_dir=rt

# ================================ move near ===============================
CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name AltPlaceAppleNearOrangeCanEnv-v0 --scene-name google_pick_coke_can_1_v4 \
  --robot-init-x 0.35 0.35 1 --robot-init-y 0.21 0.21 1 --obj-variation-mode episode --obj-episode-range 0 12\
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.0 0.0 1

CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name AltPlaceBottleNearSpongeEnv-v0 --scene-name google_pick_coke_can_1_v4_alt_background \
  --robot-init-x 0.35 0.35 1 --robot-init-y 0.21 0.21 1 --obj-variation-mode episode --obj-episode-range 0 12\
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.0 0.0 1

CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name AltPlaceBottleNearOrangeEnv-v0 --scene-name google_pick_coke_can_1_v4_alt_background \
  --robot-init-x 0.35 0.35 1 --robot-init-y 0.21 0.21 1 --obj-variation-mode episode --obj-episode-range 0 12\
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.0 0.0 1

# ================================ bridge ================================
robot=widowx_sink_camera_setup
rgb_overlay_path=ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png
robot_init_x=0.127
robot_init_y=0.06

CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot ${robot} --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 120 \
  --env-name AltDryEgglpantInSceneEnv-v0 --scene-name bridge_table_1_v2 \
  --rgb-overlay-path ${rgb_overlay_path} \
  --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 --obj-variation-mode episode --obj-episode-range 0 24 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1;

CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot ${robot} --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 120 \
  --env-name AltDryOrangeInSceneEnv-v0 --scene-name bridge_table_1_v2 \
  --rgb-overlay-path ${rgb_overlay_path} \
  --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 --obj-variation-mode episode --obj-episode-range 0 24 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1;


done