# you only need to change ckpt_path
ckpt_path="/mnt/petrelfs/yangshuai1/rep/InstructVLA_official/outputs/head_balation/sys12_meta_query_full_finetune_sync_cotraining_v2_xlora_freeze_head_instruction_long--image_augstage2/checkpoints/step-013500-epoch-01-loss=0.1093_simpler_1.pt"

cd ./SimplerEnv
base_dir=$(dirname "$ckpt_path")
base_dir=$(dirname "$base_dir")
file_name=$(basename "$ckpt_path" .pt)
result_path="${base_dir}/results_${file_name}"

export LD_LIBRARY_PATH=~/miniconda3/envs/openvla-simpler/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=~/miniconda3/envs/openvla-simpler/lib/python3.10/site-packages/nvidia/cudnn/lib/libcudnn_ops_infer.so.8
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

if [ ! -d "$result_path" ]; then
    mkdir -p "$result_path"
fi
log_path="${result_path}/log/"
if [ ! -d "$log_path" ]; then
    mkdir -p "$log_path"
fi
export PYTHONPATH="/mnt/petrelfs/yangshuai1/rep/InstructVLA_official/SimplerEnv:$PYTHONPATH"
bash ./scripts_self/cogact_bridge_v2.sh $ckpt_path 4
# bash  ./scripts_self/cogact_drawer_visual_matching.sh  $ckpt_path 1
wait $pid8

echo "计算最终结果...final_result.log"
python /mnt/petrelfs/yangshuai1/rep/InstructVLA_official/SimplerEnv/calc_metrics_evaluation_easy.py --ckpt-mapping $(basename "$ckpt_path") --log-dir-root $result_path  > "${log_path}/final_result.log" 2>&1