#!/bin/bash
#SBATCH --job-name=super_rag        # name
#SBATCH -p mozi-S1
#SBATCH -N 1                    # nodes
#SBATCH --time 240
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

# to test if you can access ceph, you are expected to see:
#                            PRE open_x_embodiment_origin/

# to fix: libcudnn_ops_infer.so.8 with link time referencesymbol _ZN15TracebackLoggerC1EPKc
export LD_LIBRARY_PATH=~/miniconda3/envs/openvla-simpler/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=~/miniconda3/envs/openvla-simpler/lib/python3.10/site-packages/nvidia/cudnn/lib/libcudnn_ops_infer.so.8

export hf_token=REMOVED_TOKEN
export MASTER_PORT=$((RANDOM % 101 + 20000))


srun --jobid $SLURM_JOBID bash -c  'torchrun  --nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
 --master_addr $MASTER_ADDR --master_port $MASTER_PORT scripts/train_rag.py \
  --vla.base_vlm /mnt/hwfile/OpenRobotLab/lihao3/MLLM/store_ckpt_data/ckpt/openvla/prism-qwen25-extra-dinosiglip-224px-0_5b \
  --pretrained_checkpoint /mnt/petrelfs/yangshuai1/rep/TMP_CogACTmini_x_DIT_Atten_HisF_MultiF_R_Silence/ckpt/bridge_mini/checkpoints/step-017500-epoch-12-loss=0.0558.pt \
  --vla.type prism-qwen25-dinosiglip-224px+0_5b \
  --vla.data_mix bridge \
  --vla.epochs 1 \
  --vla.expected_world_size 8 \
  --vla.global_batch_size 384 \
  --vla.per_device_batch_size 48 \
  --vla.learning_rate 2e-5 \
  --data_root_dir "s3://real_data_raw/open_x_embodiment_origin" \
  --run_root_dir ./outputs/HIS_ditb_8_lr2e-5_b16_fa15_pa0_shuffle_bridge_rt_1 \
  --run_id debug_rag_top_k=25 \
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
  --resume_step 20000 \
  --resume_epoch 14 \
  --rag_source /mnt/petrelfs/yangshuai1/rep/TMP_CogACTmini_x_DIT_Atten_HisF_MultiF_R_Silence/test_rag.pt \
  --top_k 25 \
  --drop_p 0.3'