export S3_ENDPOINT=http://10.140.14.204:80
export AWS_ACCESS_KEY_ID=REMOVED_TOKEN
export AWS_SECRET_ACCESS_KEY=REMOVED_TOKEN
aws s3 ls s3://real_data_raw --endpoint-url http://10.140.14.204:80

# to test if you can access ceph, you are expected to see:
#                            PRE open_x_embodiment_origin/

# to fix: libcudnn_ops_infer.so.8 with link time referencesymbol _ZN15TracebackLoggerC1EPKc
export LD_LIBRARY_PATH=~/anaconda3/envs/cogact/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=~/anaconda3/envs/cogact/lib/python3.10/site-packages/nvidia/cudnn/lib/libcudnn_ops_infer.so.8

export hf_token=REMOVED_TOKEN
export MASTER_PORT=$((RANDOM % 101 + 20000))

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes 1 --nproc-per-node 1 scripts/train.py \
  --vla.base_vlm /mnt/hwfile/OpenRobotLab/lihao3/MLLM/store_ckpt_data/ckpt/openvla/prism-qwen25-extra-dinosiglip-224px-0_5b \
  --vla.type prism-qwen25-dinosiglip-224px+0_5b \
  --vla.data_mix bridge_rt_1 \
  --vla.expected_world_size 1 \
  --vla.global_batch_size 8 \
  --vla.per_device_batch_size 8 \
  --vla.learning_rate 2e-5 \
  --data_root_dir "s3://real_data_raw/open_x_embodiment_origin" \
  --run_root_dir ./outputs/HIS_ditb_8_lr2e-5_b16_fa15_pa0_shuffle_bridge_rt_1 \
  --run_id HIS_Sil_prism-qwen25-dinosiglip-224px+bridge_rt_1+diffusion+n8+b16+x42 \
  --image_aug True \
  --wandb_project cogact \
  --wandb_entity lihaohn \
  --save_interval 10000 \
  --repeated_diffusion_steps 4 \
  --future_action_window_size 15 \
  --past_action_window_size 6 \
  --action_model_type DiT-B \
  --hf_token hf_token \
  --is_resume False