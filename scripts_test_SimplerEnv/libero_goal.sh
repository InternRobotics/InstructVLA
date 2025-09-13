#!/bin/bash

# to fix: libcudnn_ops_infer.so.8 with link time referencesymbol _ZN15TracebackLoggerC1EPKc
export LD_LIBRARY_PATH=~/miniconda3/envs/openvla-simpler/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=~/miniconda3/envs/openvla-simpler/lib/python3.10/site-packages/nvidia/cudnn/lib/libcudnn_ops_infer.so.8

CKPT_LIST=(
"path/to/checkpoint_1"
"path/to/checkpoint_2"
"..."
)

# Loop over the checkpoint list and GPUs
for i in "${!CKPT_LIST[@]}"; do
  GPU_ID=$((i % 8))  # Cycle through GPUs 0-7
  CHECKPOINT="${CKPT_LIST[$i]}"
  
  # Run the evaluation script for each checkpoint and GPU
  CUDA_VISIBLE_DEVICES=$GPU_ID python deploy/libero/run_libero_eval.py \
    --model_family instruct_vla \
    --pretrained_checkpoint "$CHECKPOINT" \
    --task_suite_name libero_goal \
    --local_log_dir Libero/release_ensemble \
    --use_length -1 \
    --center_crop True &

    # --use_length == -1: ensemble
    # --use_length >= 1 : execute action_chunk[0:use_length]
  
  sleep 5
done

# Wait for all background jobs to finish
wait
