gpu_id=$2

declare -a arr=($1)

START_PORT=$3
END_PORT=$4

get_random_port() {
    local port=$((RANDOM % (END_PORT - START_PORT + 1) + START_PORT))
    while lsof -i:$port &>/dev/null; do
        port=$((RANDOM % (END_PORT - START_PORT + 1) + START_PORT))
    done
    echo $port
}

AVAILABLE_PORT=$(get_random_port)
AVAILABLE_HOST='127.0.0.1'

wait_for_server() {
    while ! lsof -i:$AVAILABLE_PORT > /dev/null 2>&1; do
        echo "Waiting for server to start..."
        sleep 5
    done
    echo "Server is ready, starting client..."
}

echo "Using available host: $AVAILABLE_HOST ,available port: $AVAILABLE_PORT"
CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference_server.py --host $AVAILABLE_HOST --port $AVAILABLE_PORT --policy-model cogact --ckpt-path $1 --env-name None &
SERVER_PID=$!
echo "Server started with PID $SERVER_PID"

wait_for_server

# lr_switch=laying horizontally but flipped left-right to match real eval; upright=standing; laid_vertically=laying vertically
declare -a coke_can_options_arr=("lr_switch=True" "upright=True" "laid_vertically=True")

for ckpt_path in "${arr[@]}"; do echo "$ckpt_path"; done


# # base setup

# env_name=GraspSingleOpenedCokeCanInScene-v0
# scene_name=google_pick_coke_can_1_v4

# for coke_can_option in "${coke_can_options_arr[@]}";

# do for ckpt_path in "${arr[@]}";

# do CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference_client.py --host $AVAILABLE_HOST --port $AVAILABLE_PORT --policy-model cogact --ckpt-path ${ckpt_path} \
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

# do CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference_client.py --host $AVAILABLE_HOST --port $AVAILABLE_PORT --policy-model cogact --ckpt-path ${ckpt_path} \
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

# do CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference_client.py --host $AVAILABLE_HOST --port $AVAILABLE_PORT --policy-model cogact --ckpt-path ${ckpt_path} \
#   --robot google_robot_static \
#   --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
#   --env-name ${env_name} --scene-name ${scene_name} \
#   --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
#   --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
#   --additional-env-build-kwargs ${coke_can_option};

# CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference_client.py --host $AVAILABLE_HOST --port $AVAILABLE_PORT --policy-model cogact --ckpt-path ${ckpt_path} \
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

# do CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference_client.py --host $AVAILABLE_HOST --port $AVAILABLE_PORT --policy-model cogact --ckpt-path ${ckpt_path} \
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

# do CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference_client.py --host $AVAILABLE_HOST --port $AVAILABLE_PORT --policy-model cogact --ckpt-path ${ckpt_path} \
#   --robot google_robot_static \
#   --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
#   --env-name ${env_name} --scene-name ${scene_name} \
#   --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
#   --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
#   --additional-env-build-kwargs ${coke_can_option} slightly_darker_lighting=True;

# CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference_client.py --host $AVAILABLE_HOST --port $AVAILABLE_PORT --policy-model cogact --ckpt-path ${ckpt_path} \
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

do CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference_client.py --host $AVAILABLE_HOST --port $AVAILABLE_PORT --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name ${env_name} --scene-name ${scene_name} \
  --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  --additional-env-build-kwargs ${coke_can_option};

done

done

done





gpu_id=$2

declare -a arr=($1)

env_name=MoveNearGoogleBakedTexInScene-v0
# env_name=MoveNearGoogleBakedTexInScene-v1
scene_name=google_pick_coke_can_1_v4
rgb_overlay_path=./ManiSkill2_real2sim/data/real_inpainting/google_move_near_real_eval_1.png

# URDF variations
declare -a urdf_version_arr=(None "recolor_tabletop_visual_matching_1" "recolor_tabletop_visual_matching_2" "recolor_cabinet_visual_matching_1")

for ckpt_path in "${arr[@]}"; do echo "$ckpt_path"; done


for urdf_version in "${urdf_version_arr[@]}";

do for ckpt_path in "${arr[@]}";

do CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference_client.py --host $AVAILABLE_HOST --port $AVAILABLE_PORT --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name ${env_name} --scene-name ${scene_name} \
  --rgb-overlay-path ${rgb_overlay_path} \
  --robot-init-x 0.35 0.35 1 --robot-init-y 0.21 0.21 1 --obj-variation-mode episode --obj-episode-range 0 60 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.09 -0.09 1 \
  --additional-env-build-kwargs urdf_version=${urdf_version} \
  --additional-env-save-tags baked_except_bpb_orange;

done

done


gpu_id=$2

declare -a arr=($1)
# CogACT/CogACT-Large CogACT/CogACT-Small
for ckpt_path in "${arr[@]}"; do echo "$ckpt_path"; done


# base setup

env_name=MoveNearGoogleInScene-v0
scene_name=google_pick_coke_can_1_v4

for ckpt_path in "${arr[@]}";

do CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference_client.py --host $AVAILABLE_HOST --port $AVAILABLE_PORT --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name ${env_name} --scene-name ${scene_name} \
  --robot-init-x 0.35 0.35 1 --robot-init-y 0.21 0.21 1 --obj-variation-mode episode --obj-episode-range 0 60 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.09 -0.09 1;

done



# distractor

for ckpt_path in "${arr[@]}";

do CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference_client.py --host $AVAILABLE_HOST --port $AVAILABLE_PORT --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name ${env_name} --scene-name ${scene_name} \
  --robot-init-x 0.35 0.35 1 --robot-init-y 0.21 0.21 1 --obj-variation-mode episode --obj-episode-range 0 60 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.09 -0.09 1 \
  --additional-env-build-kwargs no_distractor=True;

done


# backgrounds

env_name=MoveNearGoogleInScene-v0
declare -a scene_arr=("google_pick_coke_can_1_v4_alt_background" \
                      "google_pick_coke_can_1_v4_alt_background_2")

for scene_name in "${scene_arr[@]}";

do for ckpt_path in "${arr[@]}";

do CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference_client.py --host $AVAILABLE_HOST --port $AVAILABLE_PORT --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name ${env_name} --scene-name ${scene_name} \
  --robot-init-x 0.35 0.35 1 --robot-init-y 0.21 0.21 1 --obj-variation-mode episode --obj-episode-range 0 60 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.09 -0.09 1;

done

done





# lighting

env_name=MoveNearGoogleInScene-v0
scene_name=google_pick_coke_can_1_v4

for ckpt_path in "${arr[@]}";

do CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference_client.py --host $AVAILABLE_HOST --port $AVAILABLE_PORT --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name ${env_name} --scene-name ${scene_name} \
  --robot-init-x 0.35 0.35 1 --robot-init-y 0.21 0.21 1 --obj-variation-mode episode --obj-episode-range 0 60 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.09 -0.09 1 \
  --additional-env-build-kwargs slightly_darker_lighting=True;

CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference_client.py --host $AVAILABLE_HOST --port $AVAILABLE_PORT --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name ${env_name} --scene-name ${scene_name} \
  --robot-init-x 0.35 0.35 1 --robot-init-y 0.21 0.21 1 --obj-variation-mode episode --obj-episode-range 0 60 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.09 -0.09 1 \
  --additional-env-build-kwargs slightly_brighter_lighting=True;

done





# table textures

env_name=MoveNearGoogleInScene-v0
declare -a scene_arr=("Baked_sc1_staging_objaverse_cabinet1_h870" \
                      "Baked_sc1_staging_objaverse_cabinet2_h870")

for scene_name in "${scene_arr[@]}";

do for ckpt_path in "${arr[@]}";

do CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference_client.py --host $AVAILABLE_HOST --port $AVAILABLE_PORT --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name ${env_name} --scene-name ${scene_name} \
  --robot-init-x 0.35 0.35 1 --robot-init-y 0.21 0.21 1 --obj-variation-mode episode --obj-episode-range 0 60 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.09 -0.09 1;

done

done




# camera orientations

declare -a env_arr=("MoveNearAltGoogleCameraInScene-v0" \
                   "MoveNearAltGoogleCamera2InScene-v0")
scene_name=google_pick_coke_can_1_v4

for env_name in "${env_arr[@]}";

do for ckpt_path in "${arr[@]}";

do CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference_client.py --host $AVAILABLE_HOST --port $AVAILABLE_PORT --policy-model cogact --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name ${env_name} --scene-name ${scene_name} \
  --robot-init-x 0.35 0.35 1 --robot-init-y 0.21 0.21 1 --obj-variation-mode episode --obj-episode-range 0 60 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.09 -0.09 1;

done

done


echo "Shutting down server with PID $SERVER_PID..."
kill $SERVER_PID
echo "Server shut down."