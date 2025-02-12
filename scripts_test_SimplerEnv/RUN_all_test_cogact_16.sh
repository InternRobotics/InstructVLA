# you only need to change ckpt_path
ckpt_path="/mnt/petrelfs/lihao3/EmbodiedMLLM/CogACTx_DIT_Atten_HisF_MultiF_R_Silence/outputs/HIS_Sil_ditb_8_lr2e-5_b16_fa15_pa0_shuffle_bridge_rt_1/HIS_Sil_prism-dinosiglip-224px+bridge_rt_1+diffusion+n8+b16+x42--image_aug/checkpoints/step-050000-epoch-04-loss=0.0249.pt"

cd ./SimplerEnv
base_dir=$(dirname "$ckpt_path")
base_dir=$(dirname "$base_dir")
file_name=$(basename "$ckpt_path" .pt)
result_path="${base_dir}/results_${file_name}"

if [ ! -d "$result_path" ]; then
    mkdir -p "$result_path"
fi
log_path="${result_path}/log/"
if [ ! -d "$log_path" ]; then
    mkdir -p "$log_path"
fi

bash ./scripts_self/cogact_put_in_drawer_visual_matching.sh $ckpt_path  > "${log_path}/log1.log" 2>&1 &
pid1=$!
echo "测试cogact_put_in_drawer_visual_matching: $pid1"

bash ./scripts_self/cogact_put_in_drawer_variant_agg.sh $ckpt_path > "${log_path}/log2.log" 2>&1 &
pid2=$!
echo "测试cogact_put_in_drawer_variant_agg: $pid2"

bash ./scripts_self/cogact_pick_coke_can_visual_matching.sh $ckpt_path  > "${log_path}/log3.log" 2>&1 &
pid3=$!
echo "测试cogact_pick_coke_can_visual_matching: $pid3"

bash ./scripts_self/cogact_pick_coke_can_variant_agg.sh $ckpt_path  > "${log_path}/log4.log" 2>&1 &
pid4=$!
echo "测试cogact_pick_coke_can_variant_agg: $pid4"

bash ./scripts_self/cogact_move_near_visual_matching.sh $ckpt_path  > "${log_path}/log5.log" 2>&1 &
pid5=$!
echo "测试cogact_move_near_visual_matching: $pid5"

bash ./scripts_self/cogact_move_near_variant_agg.sh $ckpt_path  > "${log_path}/log6.log" 2>&1 &
pid6=$!
echo "测试cogact_move_near_variant_agg: $pid6"

bash ./scripts_self/cogact_drawer_variant_agg.sh $ckpt_path  > "${log_path}/log7.log" 2>&1 &
pid7=$!
echo "测试cogact_drawer_variant_agg: $pid7"

bash ./scripts_self/cogact_bridge.sh $ckpt_path  > "${log_path}/log8.log" 2>&1 &
pid8=$!
echo "测试cogact_bridge: $pid8"


wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8

echo "计算最终结果...final_result.log"
python /mnt/petrelfs/lihao3/EmbodiedMLLM/CogACTv1/SimplerEnv/calc_metrics_evaluation_easy.py --ckpt-mapping $(basename "$ckpt_path") --log-dir-root $result_path  > "${log_path}/final_result.log" 2>&1