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

CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name FreeGraspSinglePlumInScene-v1 --scene-name google_pick_coke_can_1_v4_alt_background \
  --robot-init-x 0.35 0.35 1 --robot-init-y  0.20  0.20 1 --obj-variation-mode xy\
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.0 0.0 1 \
  --obj-init-x-range -0.25 -0.1 4 --obj-init-y-range 0.06 0.36 4 \
  --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/google_coke_can_real_eval_1.png \
  --additional-env-build-kwargs urdf_version=recolor_cabinet_visual_matching_1

CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name FreeGraspSingleLemonInScene-v1 --scene-name google_pick_coke_can_1_v4_alt_background \
  --robot-init-x 0.35 0.35 1 --robot-init-y  0.20  0.20 1 --obj-variation-mode xy\
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.0 0.0 1 \
  --obj-init-x-range -0.25 -0.1 4 --obj-init-y-range 0.06 0.36 4 \
  --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/google_coke_can_real_eval_1.png \
  --additional-env-build-kwargs urdf_version=recolor_cabinet_visual_matching_1

CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name FreeGraspSingleLemonInScene-v2 --scene-name google_pick_coke_can_1_v4_alt_background \
  --robot-init-x 0.35 0.35 1 --robot-init-y  0.20  0.20 1 --obj-variation-mode xy\
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.0 0.0 1 \
  --obj-init-x-range -0.25 -0.1 4 --obj-init-y-range 0.06 0.36 4 \
  --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/google_coke_can_real_eval_1.png \
  --additional-env-build-kwargs urdf_version=recolor_cabinet_visual_matching_1

CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name FreeGraspSinglePumpkinInScene-v1 --scene-name google_pick_coke_can_1_v4_alt_background \
  --robot-init-x 0.35 0.35 1 --robot-init-y  0.20  0.20 1 --obj-variation-mode xy\
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.0 0.0 1 \
  --obj-init-x-range -0.25 -0.1 4 --obj-init-y-range 0.06 0.36 4 \
  --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/google_coke_can_real_eval_1.png \
  --additional-env-build-kwargs urdf_version=recolor_cabinet_visual_matching_1

CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name FreeGraspSingleOrangeInScene-v1 --scene-name google_pick_coke_can_1_v4_alt_background \
  --robot-init-x 0.35 0.35 1 --robot-init-y  0.20  0.20 1 --obj-variation-mode xy\
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.0 0.0 1 \
  --obj-init-x-range -0.25 -0.1 4 --obj-init-y-range 0.06 0.36 4 \
  --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/google_coke_can_real_eval_1.png \
  --additional-env-build-kwargs urdf_version=recolor_cabinet_visual_matching_1

# =============================================================


CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name FreeGraspSingleOpenedPepsiCanInScene-v0 --scene-name google_pick_coke_can_1_v4 \
  --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  --additional-env-build-kwargs upright=True;


CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name FreeGraspSingleOpened7upCanInScene-v0 --scene-name google_pick_coke_can_1_v4 \
  --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  --additional-env-build-kwargs upright=True;



CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name FreeGraspSingleSpriteCanInScene-v0 --scene-name google_pick_coke_can_1_v4 \
  --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  --additional-env-build-kwargs upright=True;


CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name FreeGraspSingle7upCanInScene-v0 --scene-name google_pick_coke_can_1_v4 \
  --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1


#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name FreeGraspSingleOpenedPepsiCanInScene-v1 --scene-name google_pick_coke_can_1_v4_alt_background \
  --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  --additional-env-build-kwargs upright=True;


CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name FreeGraspSingleOpened7upCanInScene-v1 --scene-name google_pick_coke_can_1_v4_alt_background \
  --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  --additional-env-build-kwargs upright=True;


CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name FreeGraspSingleSpriteCanInScene-v1 --scene-name google_pick_coke_can_1_v4_alt_background \
  --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  --additional-env-build-kwargs upright=True;


CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name FreeGraspSingleSpriteCanInScene-v2 --scene-name google_pick_coke_can_1_v4_alt_background \
  --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  --additional-env-build-kwargs upright=True


CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name FreeGraspSingle7upCanInScene-v1 --scene-name google_pick_coke_can_1_v4_alt_background \
  --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1


CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name FreeGraspSingle7upCanInScene-v2 --scene-name google_pick_coke_can_1_v4_alt_background \
  --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1
# ===============================================================

CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 113 \
  --env-name FreeOpenTopDrawerCustomInScene-v0 --scene-name modern_bedroom_no_roof \
  --robot-init-x 0.65 0.80 3 --robot-init-y -0.2 0.2 3\
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.0 0.0 1 \
  --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 \
  --additional-env-build-kwargs shader_dir=rt

CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 113 \
  --env-name FreeOpenMiddleDrawerCustomInScene-v0 --scene-name modern_bedroom_no_roof \
  --robot-init-x 0.65 0.80 3 --robot-init-y -0.2 0.2 3\
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.0 0.0 1 \
  --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 \
  --additional-env-build-kwargs shader_dir=rt

CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 113 \
  --env-name FreeOpenBottomDrawerCustomInScene-v0 --scene-name modern_bedroom_no_roof \
  --robot-init-x 0.65 0.80 3 --robot-init-y -0.2 0.2 3\
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.0 0.0 1 \
  --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 \
  --additional-env-build-kwargs shader_dir=rt

CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 113 \
  --env-name FreeCloseTopDrawerCustomInScene-v0 --scene-name modern_bedroom_no_roof \
  --robot-init-x 0.65 0.80 3 --robot-init-y -0.2 0.2 3\
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.0 0.0 1 \
  --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 \
  --additional-env-build-kwargs shader_dir=rt

CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 113 \
  --env-name FreeCloseMiddleDrawerCustomInScene-v0 --scene-name modern_bedroom_no_roof \
  --robot-init-x 0.65 0.80 3 --robot-init-y -0.2 0.2 3\
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.0 0.0 1 \
  --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 \
  --additional-env-build-kwargs shader_dir=rt

CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 113 \
  --env-name FreeCloseBottomDrawerCustomInScene-v0 --scene-name modern_bedroom_no_roof \
  --robot-init-x 0.65 0.80 3 --robot-init-y -0.2 0.2 3\
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.0 0.0 1 \
  --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 \
  --additional-env-build-kwargs shader_dir=rt

#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 113 \
  --env-name FreeOpenTopDrawerCustomInScene-v1 --scene-name frl_apartment_stage_simple \
  --robot-init-x 0.65 0.85 3 --robot-init-y -0.2 0.2 3\
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.0 0.0 1 \
  --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 \
  --enable-raytracing

CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 113 \
  --env-name FreeOpenMiddleDrawerCustomInScene-v1 --scene-name frl_apartment_stage_simple \
  --robot-init-x 0.65 0.85 3 --robot-init-y -0.2 0.2 3\
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.0 0.0 1 \
  --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 \
  --enable-raytracing

CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 113 \
  --env-name FreeOpenBottomDrawerCustomInScene-v1 --scene-name frl_apartment_stage_simple \
  --robot-init-x 0.65 0.85 3 --robot-init-y -0.2 0.2 3\
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.0 0.0 1 \
  --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 \
  --enable-raytracing

CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 113 \
  --env-name FreeCloseTopDrawerCustomInScene-v1 --scene-name frl_apartment_stage_simple \
  --robot-init-x 0.65 0.85 3 --robot-init-y -0.2 0.2 3\
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.0 0.0 1 \
  --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 \
  --enable-raytracing

CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 113 \
  --env-name FreeCloseMiddleDrawerCustomInScene-v1 --scene-name frl_apartment_stage_simple \
  --robot-init-x 0.65 0.85 3 --robot-init-y -0.2 0.2 3\
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.0 0.0 1 \
  --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 \
  --enable-raytracing

CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 113 \
  --env-name FreeCloseBottomDrawerCustomInScene-v1 --scene-name frl_apartment_stage_simple \
  --robot-init-x 0.65 0.85 3 --robot-init-y -0.2 0.2 3\
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.0 0.0 1 \
  --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 \
  --enable-raytracing

# =================================================================

CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name FreePlaceLemonNearPearEnv-v0 --scene-name google_pick_coke_can_1_v4 \
  --robot-init-x 0.35 0.35 1 --robot-init-y 0.21 0.21 1 --obj-variation-mode episode --obj-episode-range 0 12\
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.0 0.0 1 \
  --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/google_coke_can_real_eval_1.png \
  --additional-env-build-kwargs urdf_version=recolor_cabinet_visual_matching_1

CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name FreePlaceLemonNearPearEnv-v1 --scene-name google_pick_coke_can_1_v4 \
  --robot-init-x 0.35 0.35 1 --robot-init-y 0.21 0.21 1 --obj-variation-mode episode --obj-episode-range 0 12\
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.0 0.0 1 \
  --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/google_coke_can_real_eval_1.png \
  --additional-env-build-kwargs urdf_version=recolor_cabinet_visual_matching_1

CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name FreePlaceLemonNearPearEnv-v2 --scene-name google_pick_coke_can_1_v4 \
  --robot-init-x 0.35 0.35 1 --robot-init-y 0.21 0.21 1 --obj-variation-mode episode --obj-episode-range 0 12\
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.0 0.0 1 \
  --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/google_coke_can_real_eval_1.png \
  --additional-env-build-kwargs urdf_version=recolor_cabinet_visual_matching_1

CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name FreePlaceLNearVEnv-v0 --scene-name google_pick_coke_can_1_v4 \
  --robot-init-x 0.35 0.35 1 --robot-init-y 0.21 0.21 1 --obj-variation-mode episode --obj-episode-range 0 12\
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/google_coke_can_real_eval_1.png \
  --additional-env-build-kwargs urdf_version=recolor_cabinet_visual_matching_1
  
CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name FreePlaceLNearAEnv-v0 --scene-name google_pick_coke_can_1_v4 \
  --robot-init-x 0.35 0.35 1 --robot-init-y 0.21 0.21 1 --obj-variation-mode episode --obj-episode-range 0 12\
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/google_coke_can_real_eval_1.png \
  --additional-env-build-kwargs urdf_version=recolor_cabinet_visual_matching_1

CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name FreePlaceBottleNearOrangeEnv-v0 --scene-name google_pick_coke_can_1_v4 \
  --robot-init-x 0.35 0.35 1 --robot-init-y 0.21 0.21 1 --obj-variation-mode episode --obj-episode-range 0 12\
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/google_coke_can_real_eval_1.png \
  --additional-env-build-kwargs urdf_version=recolor_cabinet_visual_matching_1

CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name FreePlaceMugNearPlaystationEnv-v0 --scene-name google_pick_coke_can_1_v4 \
  --robot-init-x 0.35 0.35 1 --robot-init-y 0.21 0.21 1 --obj-variation-mode episode --obj-episode-range 0 12\
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/google_coke_can_real_eval_1.png \
  --additional-env-build-kwargs urdf_version=recolor_cabinet_visual_matching_1


CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name FreePlaceMugNearPlaystationEnv-v1 --scene-name google_pick_coke_can_1_v4 \
  --robot-init-x 0.35 0.35 1 --robot-init-y 0.21 0.21 1 --obj-variation-mode episode --obj-episode-range 0 12\
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/google_coke_can_real_eval_1.png \
  --additional-env-build-kwargs urdf_version=recolor_cabinet_visual_matching_1

CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name FreePlaceMugNearPlaystationEnv-v0 --scene-name google_pick_coke_can_1_v4 \
  --robot-init-x 0.35 0.35 1 --robot-init-y 0.21 0.21 1 --obj-variation-mode episode --obj-episode-range 0 12\
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1


CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name FreePlaceMugNearPlaystationEnv-v1 --scene-name google_pick_coke_can_1_v4 \
  --robot-init-x 0.35 0.35 1 --robot-init-y 0.21 0.21 1 --obj-variation-mode episode --obj-episode-range 0 12\
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1

# =============================================================================================

CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 200 \
  --env-name FreePlaceIntoTopDrawerEnv-v0 --scene-name dummy_drawer \
  --robot-init-x 0.665 0.665 1 --robot-init-y 0.224 0.224 1 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  --obj-init-x-range -0.08 -0.02 3 --obj-init-y-range -0.02 0.08 3 \
  --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/open_drawer_c0.png \
  --enable-raytracing --additional-env-build-kwargs station_name=mk_station_recolor light_mode=simple disable_bad_material=True urdf_version=recolor_tabletop_visual_matching_1 model_ids=baked_apple_v2


CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 200 \
  --env-name FreePlaceIntoTopDrawerEnv-v1 --scene-name dummy_drawer \
  --robot-init-x 0.652 0.652 1 --robot-init-y 0.009 0.009 1 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  --obj-init-x-range -0.08 -0.02 3 --obj-init-y-range -0.02 0.08 3 \
  --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/open_drawer_b0.png \
  --enable-raytracing --additional-env-build-kwargs station_name=mk_station_recolor light_mode=simple disable_bad_material=True urdf_version=recolor_tabletop_visual_matching_1 model_ids=baked_apple_v2


CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 200 \
  --env-name FreePlaceIntoTopDrawerEnv-v2 --scene-name dummy_drawer \
  --robot-init-x 0.644 0.644 1 --robot-init-y -0.179 -0.179 1 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.03 -0.03 1 \
  --obj-init-x-range -0.08 -0.02 3 --obj-init-y-range -0.02 0.08 3 \
  --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/open_drawer_a0.png \
  --enable-raytracing --additional-env-build-kwargs station_name=mk_station_recolor light_mode=simple disable_bad_material=True urdf_version=recolor_tabletop_visual_matching_1 model_ids=baked_apple_v2


done