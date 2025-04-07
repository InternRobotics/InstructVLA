#!/bin/bash
#SBATCH --job-name=🤗ablation_mlp        # name
#SBATCH -p efm
#SBATCH -N 8                    # nodes
#SBATCH --time 3600                    # nodes
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=16          # number of cores per tasks
#SBATCH --gres=gpu:8                 # number of gpus
#SBATCH --output=trash/%x-%j.out           # output file name
#SBATCH -e trash/%x-%j.err

export GPUS_PER_NODE=8
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$((RANDOM % 101 + 20000))

export S3_ENDPOINT=http://10.140.14.204:80
export AWS_ACCESS_KEY_ID=REMOVED_TOKEN
export AWS_SECRET_ACCESS_KEY=REMOVED_TOKEN
aws s3 ls s3://real_data_raw --endpoint-url http://10.140.14.204:80

export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=mlx5_0
export NCCL_TIMEOUT=3600

# to test if you can access ceph, you are expected to see:
#                            PRE open_x_embodiment_origin/

# to fix: libcudnn_ops_infer.so.8 with link time referencesymbol _ZN15TracebackLoggerC1EPKc
export LD_LIBRARY_PATH=~/miniconda3/envs/openvla-simpler/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=~/miniconda3/envs/openvla-simpler/lib/python3.10/site-packages/nvidia/cudnn/lib/libcudnn_ops_infer.so.8

export hf_token=REMOVED_TOKEN

srun --jobid $SLURM_JOBID bash -c 'python -m torch.distributed.run \
 --nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
 --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
 scripts/train_eagle.py \
  --vla.base_vlm "/mnt/petrelfs/yangshuai1/yangshuai1/share_mllm/Eagle2-2B" \
  --vla.type prism-qwen25-dinosiglip-224px+0_5b \
  --vla.data_mix bridge_rt_1 \
  --vla.expected_world_size 64 \
  --vla.global_batch_size 1024 \
  --vla.per_device_batch_size 16 \
  --vla.train_strategy 'fsdp-shard-grad-op' \
  --vla.learning_rate 2e-5 \
  --data_root_dir "s3://real_data_raw/open_x_embodiment_origin" \
  --run_root_dir ./outputs/head_balation \
  --run_id eagle_mlp_64_fixbug_modify \
  --image_aug True \
  --wandb_project "dual_sys" \
  --wandb_entity "shuaiyang2003" \
  --save_interval 1500 \
  --repeated_diffusion_steps 4 \
  --future_action_window_size 15 \
  --past_action_window_size 1 \
  --action_model_type DiT-B \
  --hf_token hf_token \
  --is_resume False'