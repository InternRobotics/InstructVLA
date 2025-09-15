# shader_dir=rt means that we turn on ray-tracing rendering; this is quite crucial for the open / close drawer task as policies often rely on shadows to infer depth
gpu_id=$2

declare -a ckpt_paths=(
$1
)


for ckpt_path in "${ckpt_paths[@]}"; do


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
  
done