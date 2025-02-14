# you only need to change ckpt_path
ckpt_path="/mnt/petrelfs/lihao3/EmbodiedMLLM/CogACTx_DIT_Atten_HisF_MultiF_R_Silence/outputs/HIS_Sil_ditb_8_lr2e-5_b16_fa15_pa0_shuffle_bridge_rt_1/HIS_Sil_prism-dinosiglip-224px+bridge_rt_1+diffusion+n8+b16+x42--image_aug/checkpoints/step-012500-epoch-01-loss=0.0364.pt"

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

bash ./scripts_self/balance_scripts/1_cogact_bridge-4A1_cogact_put_in_drawer_visual_matching-8.sh $ckpt_path 0 > "${log_path}/log1.log" 2>&1 &
pid1=$!
echo "测试1_cogact_bridge-4A1_cogact_put_in_drawer_visual_matching-8: $pid1"

bash ./scripts_self/balance_scripts/1_cogact_drawer_variant_agg-12.sh $ckpt_path 1 > "${log_path}/log2.log" 2>&1 &
pid2=$!
echo "测试1_cogact_drawer_variant_agg-12: $pid2"

bash ./scripts_self/balance_scripts/1_cogact_move_near_visual_matching-4A1_cogact_move_near_variant_agg-7.sh $ckpt_path 2 > "${log_path}/log3.log" 2>&1 &
pid3=$!
echo "测试1_cogact_move_near_visual_matching-4A1_cogact_move_near_variant_agg-7: $pid3"

bash ./scripts_self/balance_scripts/1_cogact_pick_coke_can_variant_agg-12.sh $ckpt_path 3 > "${log_path}/log4.log" 2>&1 &
pid4=$!
echo "测试1_cogact_pick_coke_can_variant_agg-12: $pid4"

bash ./scripts_self/balance_scripts/1_cogact_put_in_drawer_variant_agg-13.sh $ckpt_path 4 > "${log_path}/log5.log" 2>&1 &
pid5=$!
echo "测试1_cogact_put_in_drawer_variant_agg-13: $pid5"

bash ./scripts_self/balance_scripts/2_cogact_drawer_variant_agg-4A1_cogact_pick_coke_can_visual_matching-8.sh $ckpt_path 5 > "${log_path}/log6.log" 2>&1 &
pid6=$!
echo "测试2_cogact_drawer_variant_agg-4A1_cogact_pick_coke_can_visual_matching-8: $pid6"

bash ./scripts_self/balance_scripts/2_cogact_pick_coke_can_variant_agg-2A1_cogact_drawer_visual_matching-10.sh $ckpt_path 6 > "${log_path}/log7.log" 2>&1 &
pid7=$!
echo "测试2_cogact_pick_coke_can_variant_agg-2A1_cogact_drawer_visual_matching-10: $pid7"

bash ./scripts_self/balance_scripts/2_cogact_put_in_drawer_variant_agg-2A2_cogact_drawer_visual_matching-10.sh $ckpt_path 7 > "${log_path}/log8.log" 2>&1 &
pid8=$!
echo "测试2_cogact_put_in_drawer_variant_agg-2A2_cogact_drawer_visual_matching-10: $pid8"


wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8

echo "计算最终结果...final_result.log"
python /mnt/petrelfs/lihao3/EmbodiedMLLM/CogACTv1/SimplerEnv/calc_metrics_evaluation_easy.py --ckpt-mapping $(basename "$ckpt_path") --log-dir-root $result_path  > "${log_path}/final_result.log" 2>&1