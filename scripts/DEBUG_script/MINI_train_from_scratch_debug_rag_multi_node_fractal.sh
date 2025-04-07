export S3_ENDPOINT=http://10.140.14.204:80
export AWS_ACCESS_KEY_ID=REMOVED_TOKEN
export AWS_SECRET_ACCESS_KEY=REMOVED_TOKEN

# to test if you can access ceph, you are expected to see:
#                            PRE open_x_embodiment_origin/

# to fix: libcudnn_ops_infer.so.8 with link time referencesymbol _ZN15TracebackLoggerC1EPKc
export LD_LIBRARY_PATH=~/miniconda3/envs/openvla-simpler/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=~/miniconda3/envs/openvla-simpler/lib/python3.10/site-packages/nvidia/cudnn/lib/libcudnn_ops_infer.so.8

export hf_token=REMOVED_TOKEN
export MASTER_PORT=$((RANDOM % 101 + 20000))

# --master_addr SH-IDC1-10-140-0-212:15555  CUDA_VISIBLE_DEVICES=0 
torchrun --nnodes 1 --master_port 21451 --node_rank 0 --nproc-per-node 8 scripts/train_rag.py \
  --vla.base_vlm /mnt/hwfile/OpenRobotLab/lihao3/MLLM/store_ckpt_data/ckpt/openvla/prism-qwen25-extra-dinosiglip-224px-0_5b \
  --pretrained_checkpoint /mnt/petrelfs/yangshuai1/rep/TMP_CogACTmini_x_DIT_Atten_HisF_MultiF_R_Silence/ckpt/fractal_mini/checkpoints/step-016500-epoch-06-loss=0.0444.pt \
  --vla.type prism-qwen25-dinosiglip-224px+0_5b \
  --vla.data_mix RT_only \
  --vla.expected_world_size 8 \
  --vla.global_batch_size 384 \
  --vla.per_device_batch_size 48 \
  --vla.learning_rate 2e-5 \
  --data_root_dir "s3://real_data_raw/open_x_embodiment_origin" \
  --run_root_dir ./outputs/HIS_ditb_8_lr2e-5_b16_fa15_pa0_shuffle_bridge_rt_1 \
  --run_id debug_rag_top_k=8_fractal_mlp_seg_from_step_25000_multi_layer_features \
  --image_aug True \
  --wandb_project "cogact" \
  --wandb_entity "shuaiyang2003" \
  --save_interval 2500 \
  --repeated_diffusion_steps 4 \
  --future_action_window_size 15 \
  --past_action_window_size 1 \
  --action_model_type DiT-B \
  --hf_token hf_token \
  --is_resume True \
  --resume_step 16500 \
  --resume_epoch 6 \
  --rag_source /mnt/petrelfs/yangshuai1/rep/TMP_CogACTmini_x_DIT_Atten_HisF_MultiF_R_Silence/fractal_mini_step-016500-epoch-06.pt-segment.pt \
  --top_k 8 \
  --drop_p 0.3