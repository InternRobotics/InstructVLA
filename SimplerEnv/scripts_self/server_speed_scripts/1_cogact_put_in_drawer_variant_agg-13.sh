# shader_dir=rt means that we turn on ray-tracing rendering; this is quite crucial for the open / close drawer task as policies often rely on shadows to infer depth
gpu_id=$2


declare -a ckpt_paths=(
$1
)

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

declare -a env_names=(
PlaceIntoClosedTopDrawerCustomInScene-v0
)

EXTRA_ARGS="--enable-raytracing  --additional-env-build-kwargs model_ids=apple"


# base setup
scene_name=frl_apartment_stage_simple

EvalSim() {
  echo ${ckpt_path} ${env_name}

  CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference_client.py --host $AVAILABLE_HOST --port $AVAILABLE_PORT --policy-model cogact --ckpt-path ${ckpt_path} \
    --robot google_robot_static \
    --control-freq 3 --sim-freq 513 --max-episode-steps 200 \
    --env-name ${env_name} --scene-name ${scene_name} \
    --robot-init-x 0.65 0.65 1 --robot-init-y -0.2 0.2 3 \
    --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.0 0.0 1 \
    --obj-init-x-range -0.08 -0.02 3 --obj-init-y-range -0.02 0.08 3 \
    ${EXTRA_ARGS}
}


for ckpt_path in "${ckpt_paths[@]}"; do
  for env_name in "${env_names[@]}"; do
    EvalSim
  done
done


# backgrounds

declare -a scene_names=(
"modern_bedroom_no_roof"
"modern_office_no_roof"
)

for scene_name in "${scene_names[@]}"; do
  for ckpt_path in "${ckpt_paths[@]}"; do
    for env_name in "${env_names[@]}"; do
      EXTRA_ARGS="--additional-env-build-kwargs shader_dir=rt model_ids=apple"
      EvalSim
    done
  done
done


# lightings
scene_name=frl_apartment_stage_simple

for ckpt_path in "${ckpt_paths[@]}"; do
  for env_name in "${env_names[@]}"; do
    EXTRA_ARGS="--additional-env-build-kwargs shader_dir=rt light_mode=brighter model_ids=apple"
    EvalSim
    EXTRA_ARGS="--additional-env-build-kwargs shader_dir=rt light_mode=darker model_ids=apple"
    EvalSim
  done
done


# # new cabinets
# scene_name=frl_apartment_stage_simple

# for ckpt_path in "${ckpt_paths[@]}"; do
#   for env_name in "${env_names[@]}"; do
#     EXTRA_ARGS="--additional-env-build-kwargs shader_dir=rt station_name=mk_station2 model_ids=apple"
#     EvalSim
#     EXTRA_ARGS="--additional-env-build-kwargs shader_dir=rt station_name=mk_station3 model_ids=apple"
#     EvalSim
#   done
# done

echo "Shutting down server with PID $SERVER_PID..."
kill $SERVER_PID
echo "Server shut down."