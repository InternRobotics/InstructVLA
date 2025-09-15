# shader_dir=rt means that we turn on ray-tracing rendering; this is quite crucial for the open / close drawer task as policies often rely on shadows to infer depth
gpu_id=$2

declare -a ckpt_paths=(
$1
)


# declare -a coke_can_options_arr=("lr_switch=True" "upright=True" "laid_vertically=True")


for ckpt_path in "${ckpt_paths[@]}"; do


CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name FreeGraspSingleBananaInScene-v0 --scene-name google_pick_coke_can_1_v4 \
  --robot-init-x 0.35 0.35 1 --robot-init-y  0.20  0.20 1 --obj-variation-mode xy\
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.0 0.0 1 \
  --obj-init-x-range -0.35 -0.196 4 --obj-init-y-range 0.12 0.46 4 \
  --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/google_coke_can_real_eval_1.png \
  --additional-env-build-kwargs urdf_version=recolor_cabinet_visual_matching_1

CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name FreeGraspSinglePearInScene-v0 --scene-name google_pick_coke_can_1_v4 \
  --robot-init-x 0.35 0.35 1 --robot-init-y  0.20  0.20 1 --obj-variation-mode xy\
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.0 0.0 1 \
  --obj-init-x-range -0.25 -0.1 4 --obj-init-y-range 0.06 0.36 4 \
  --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/google_coke_can_real_eval_1.png \
  --additional-env-build-kwargs urdf_version=recolor_cabinet_visual_matching_1

CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name FreeGraspSinglePlumInScene-v0 --scene-name google_pick_coke_can_1_v4 \
  --robot-init-x 0.35 0.35 1 --robot-init-y  0.20  0.20 1 --obj-variation-mode xy\
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.0 0.0 1 \
  --obj-init-x-range -0.25 -0.1 4 --obj-init-y-range 0.06 0.36 4 \
  --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/google_coke_can_real_eval_1.png \
  --additional-env-build-kwargs urdf_version=recolor_cabinet_visual_matching_1

CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name FreeGraspSingleLemonInScene-v0 --scene-name google_pick_coke_can_1_v4 \
  --robot-init-x 0.35 0.35 1 --robot-init-y  0.20  0.20 1 --obj-variation-mode xy\
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.0 0.0 1 \
  --obj-init-x-range -0.25 -0.1 4 --obj-init-y-range 0.06 0.36 4 \
  --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/google_coke_can_real_eval_1.png \
  --additional-env-build-kwargs urdf_version=recolor_cabinet_visual_matching_1

CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name FreeGraspSinglePumpkinInScene-v0 --scene-name google_pick_coke_can_1_v4 \
  --robot-init-x 0.35 0.35 1 --robot-init-y  0.20  0.20 1 --obj-variation-mode xy\
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.0 0.0 1 \
  --obj-init-x-range -0.25 -0.1 4 --obj-init-y-range 0.06 0.36 4 \
  --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/google_coke_can_real_eval_1.png \
  --additional-env-build-kwargs urdf_version=recolor_cabinet_visual_matching_1

CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name FreeGraspSingleOrangeInScene-v0 --scene-name google_pick_coke_can_1_v4 \
  --robot-init-x 0.35 0.35 1 --robot-init-y  0.20  0.20 1 --obj-variation-mode xy\
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.0 0.0 1 \
  --obj-init-x-range -0.25 -0.1 4 --obj-init-y-range 0.06 0.36 4 \
  --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/google_coke_can_real_eval_1.png \
  --additional-env-build-kwargs urdf_version=recolor_cabinet_visual_matching_1
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name FreeGraspSingleBananaInScene-v1 --scene-name google_pick_coke_can_1_v4_alt_background \
  --robot-init-x 0.35 0.35 1 --robot-init-y  0.20  0.20 1 --obj-variation-mode xy\
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.0 0.0 1 \
  --obj-init-x-range -0.35 -0.196 4 --obj-init-y-range 0.12 0.46 4 \
  --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/google_coke_can_real_eval_1.png \
  --additional-env-build-kwargs urdf_version=recolor_cabinet_visual_matching_1


CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name FreeGraspSingleBananaInScene-v2 --scene-name google_pick_coke_can_1_v4_alt_background \
  --robot-init-x 0.35 0.35 1 --robot-init-y  0.20  0.20 1 --obj-variation-mode xy\
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.0 0.0 1 \
  --obj-init-x-range -0.35 -0.196 4 --obj-init-y-range 0.12 0.46 4 \
  --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/google_coke_can_real_eval_1.png \
  --additional-env-build-kwargs urdf_version=recolor_cabinet_visual_matching_1

CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name FreeGraspSinglePearInScene-v1 --scene-name google_pick_coke_can_1_v4_alt_background \
  --robot-init-x 0.35 0.35 1 --robot-init-y  0.20  0.20 1 --obj-variation-mode xy\
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.0 0.0 1 \
  --obj-init-x-range -0.25 -0.1 4 --obj-init-y-range 0.06 0.36 4 \
  --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/google_coke_can_real_eval_1.png \
  --additional-env-build-kwargs urdf_version=recolor_cabinet_visual_matching_1



CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name FreeGraspSinglePearInScene-v2 --scene-name google_pick_coke_can_1_v4_alt_background \
  --robot-init-x 0.35 0.35 1 --robot-init-y  0.20  0.20 1 --obj-variation-mode xy\
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.0 0.0 1 \
  --obj-init-x-range -0.25 -0.1 4 --obj-init-y-range 0.06 0.36 4 \
  --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/google_coke_can_real_eval_1.png \
  --additional-env-build-kwargs urdf_version=recolor_cabinet_visual_matching_1


done