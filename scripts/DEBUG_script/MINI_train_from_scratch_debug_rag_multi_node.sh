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


torchrun --nnodes 4 --master_addr SH-IDC1-10-140-1-41 --master_port 21451 --node_rank 0 --nproc-per-node 8 scripts/train_rag.py \
  --vla.base_vlm /mnt/hwfile/OpenRobotLab/lihao3/MLLM/store_ckpt_data/ckpt/openvla/prism-qwen25-extra-dinosiglip-224px-0_5b \
  --pretrained_checkpoint /mnt/petrelfs/yangshuai1/rep/TMP_CogACTmini_x_DIT_Atten_HisF_MultiF_R_Silence/ckpt/bridge_mini/checkpoints/step-017500-epoch-12-loss=0.0558.pt \
  --vla.type prism-qwen25-dinosiglip-224px+0_5b \
  --vla.data_mix RT_only \
  --vla.epochs 1 \
  --vla.expected_world_size 32 \
  --vla.global_batch_size 1536 \
  --vla.per_device_batch_size 48 \
  --vla.learning_rate 2e-5 \
  --data_root_dir "s3://real_data_raw/open_x_embodiment_origin" \
  --run_root_dir ./outputs/HIS_ditb_8_lr2e-5_b16_fa15_pa0_shuffle_bridge_rt_1 \
  --run_id debug_rag_top_k=3_round_2 \
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
  --resume_step 17500 \
  --resume_epoch 12 \
  --rag_source /mnt/petrelfs/yangshuai1/rep/TMP_CogACTmini_x_DIT_Atten_HisF_MultiF_R_Silence/step-017500-epoch-12_rag.pt \
  --top_k 8 \
  --drop_p 0.3