#!/bin/bash
#SBATCH --job-name=robovlm-cosmos      # name
#SBATCH -p efm
#SBATCH -N 2                    # nodes
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=10          # number of cores per tasks
#SBATCH --gres=gpu:8                 # number of gpus
#SBATCH --output=job-logs/%x-%j.out           # output file name
#SBATCH -e job-logs/%x-%j.err


CUR_DIR=$(cd $(dirname $0); pwd)
# sudo chmod 777 -R ./


echo "master port: ${port}"

set -x
# export PYTHONUNBUFFERED=1
export GPUS_PER_NODE=8
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=mlx5_0
WORKER_NUM=2
# MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
# MASTER_PORT=$((RANDOM % 101 + 20000))
export MASTER_PORT=$((RANDOM % 101 + 20000))
# METIS_WORKER_0_HOST=127.0.0.1
# export METIS_WORKER_0_HOST=127.0.0.1

# convert deepspeed checkpoint first
if [ $NODE_ID == "0" ]; then
  echo "---------- Converting deepspeed checkpoint to fp32. ----------"
  python3 tools/convert_deepspeed_to_fp32.py ${@:1}
fi

subfix=`date "+%H-%M"`

# echo "RUNNING:"
# echo python -m torch.distributed.run \
#     --nnodes $WORKER_NUM \
#     --node_rank $NODE_ID \
#     --nproc_per_node $GPUS_PER_NODE \
#     --master_addr $MASTER_ADDR \
#     --master_port $MASTER_PORT \
#     main.py \
#     --exp_name ${subfix} \
#     ${@:1} \
#     --gpus $GPUS_PER_NODE \
#     --num_nodes $WORKER_NUM

srun --jobid $SLURM_JOBID bash -c 'python -m torch.distributed.run \
    --nnodes $SLURM_NNODES \
    --node_rank $SLURM_PROCID \
    --nproc_per_node $GPUS_PER_NODE \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    main.py \
    --exp_name ${subfix} \
    ${@:1} \
    --gpus $GPUS_PER_NODE \
    --num_nodes $WORKER_NUM \'