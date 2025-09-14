# you only need to change ckpt_path
ckpt_path="/mnt/petrelfs/yangshuai1/rep/InstructVLA_official/outputs/release_ckpts/instructvla_finetune_v2_xlora_freeze_head_instruction_state--image_aug/checkpoints/step-006000-epoch-01-loss=0.1724_instruct_cot_3.pt"

cd ./SimplerEnv
base_dir=$(dirname "$ckpt_path")
base_dir=$(dirname "$base_dir")
file_name=$(basename "$ckpt_path" .pt)
result_path="${base_dir}/results_${file_name}"

export LD_LIBRARY_PATH=~/miniconda3/envs/openvla/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=~/miniconda3/envs/openvla/lib/python3.10/site-packages/nvidia/cudnn/lib/libcudnn_ops_infer.so.8
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

if [ ! -d "$result_path" ]; then
    mkdir -p "$result_path"
fi
log_path="${result_path}/log/"
if [ ! -d "$log_path" ]; then
    mkdir -p "$log_path"
fi


export PYTHONPATH="/mnt/petrelfs/yangshuai1/rep/InstructVLA_official/SimplerEnv:$PYTHONPATH" 

bash ./scripts_self/situated_1.sh $ckpt_path 0 > "${log_path}/log1.log" 2>&1 &
pid1=$!
echo "1: $pid1"

bash ./scripts_self/situated_2.sh $ckpt_path 1 > "${log_path}/log2.log" 2>&1 &
pid2=$!
echo "2: $pid2"

bash ./scripts_self/situated_3.sh $ckpt_path 2 > "${log_path}/log3.log" 2>&1 &
pid3=$!
echo "3: $pid3"

bash ./scripts_self/aggregation_1.sh $ckpt_path 3 > "${log_path}/log4.log" 2>&1 &
pid4=$!
echo "4: $pid4"

bash ./scripts_self/aggregation_2.sh $ckpt_path 4 > "${log_path}/log5.log" 2>&1 &
pid5=$!
echo "5: $pid5"

bash ./scripts_self/aggregation_3.sh $ckpt_path 5 > "${log_path}/log6.log" 2>&1 &
pid6=$!
echo "6: $pid6"

bash ./scripts_self/aggregation_4.sh $ckpt_path 6 > "${log_path}/log7.log" 2>&1 &
pid7=$!
echo "7: $pid7"

bash ./scripts_self/aggregation_5.sh $ckpt_path 7 > "${log_path}/log8.log" 2>&1 &
pid8=$!
echo "8: $pid8"

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8

echo "Done!"

echo "computing final_results_instruct.log ... "

full_path="${result_path}/$(basename "$ckpt_path")"

python calc_instruction_folliowing.py --results-dir $full_path  > "${log_path}/final_results_instruct.log" 2>&1