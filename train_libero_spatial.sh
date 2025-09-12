#!/bin/bash

export GPUS_PER_NODE=8
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29323

# to fix: libcudnn_ops_infer.so.8 with link time referencesymbol _ZN15TracebackLoggerC1EPKc
export LD_LIBRARY_PATH=~/miniconda3/envs/openvla/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=~/miniconda3/envs/openvla/lib/python3.10/site-packages/nvidia/cudnn/lib/libcudnn_ops_infer.so.8

python -m torch.distributed.run \
 --nproc_per_node $GPUS_PER_NODE --nnodes 1 --node_rank 0 \
 --master_port $MASTER_PORT \
 scripts/train_eagle_dual_v2_action_only_meta_query_v2_libero_wrist.py \
  --vla.base_vlm "ckpt/Eagle2-2B" \
  --vla.type prism-qwen25-dinosiglip-224px+0_5b \
  --vla.data_mix libero_spatial_no_noops \
  --vla.expected_world_size 8 \
  --vla.global_batch_size 256 \
  --vla.per_device_batch_size 32 \
  --vla.train_strategy 'fsdp-full-shard' \
  --vla.learning_rate 5e-5 \
  --data_root_dir "/mnt/inspurfs/efm_t/robot_data/cache/LIBERO/dataset" \
  --run_root_dir ./outputs/libero_wrist \
  --run_id sys12_meta_query_action_only_sync_pretraining_v2_query_64_mlp_lora_libero_spatial_wrist \
  --image_aug True \
  --wandb_project "dual_sys_libero" \
  --wandb_entity "shuaiyang2003" \
  --save_interval 1500 \
  --repeated_diffusion_steps 4 \
  --future_action_window_size 7 \
  --past_action_window_size 0 \
  --is_resume False \
  --stage stage1 \
  --use_mm False \
  --fix_system1 False