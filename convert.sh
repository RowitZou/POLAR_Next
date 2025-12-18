#!/bin/bash
set -ex

CHECKPOINT_DIR="/mnt/shared-storage-user/ailab-hs/zouyicheng/POLAR_Next/outputs/verl_opd_policy_Qwen3-8B_reward_ZERO_ref_Qwen3-30B-A3B_data_General_lr_1e-6"
GLOBAL_STEP="global_step_1000"  # 选择要转换的 checkpoint
OUTPUT_BASE_DIR="/mnt/shared-storage-user/ailab-hs/zouyicheng/POLAR_Next/outputs/verl_opd_policy_Qwen3-8B_reward_ZERO_ref_Qwen3-30B-A3B_data_General_lr_1e-6/hf_models"

cmd="source /mnt/shared-storage-user/ailab-hs/zouyicheng/.bashrc && conda activate verl-061 && \
cd /mnt/shared-storage-user/ailab-hs/zouyicheng/POLAR_Next/verl && python -m verl.model_merger merge \
--backend fsdp \
--local_dir "$CHECKPOINT_DIR/$GLOBAL_STEP/actor" \
--target_dir "$OUTPUT_BASE_DIR/actor_${GLOBAL_STEP}" \
--use_cpu_initialization"

REPLICAS=1
name=CONVERT_CKPT
rjob submit -e DISTRIBUTED_JOB=true \
    --image=registry.h.pjlab.org.cn/library/ml-base:22.04-pjlab \
    --host-network=true --name $name -P $REPLICAS --gpu 8 --cpu 96  --memory 1600000 --namespace ailab-hs --charged-group hs_gpu \
    --private-machine='group' \
    --gang-start=true \
    --mount=gpfs://gpfs1/songdemin:/mnt/shared-storage-user/songdemin \
    --mount=gpfs://gpfs1/ailab-hs:/mnt/shared-storage-user/ailab-hs \
    --mount=gpfs://gpfs1/large-model-center-share-weights:/mnt/shared-storage-user/large-model-center-share-weights \
    --mount=gpfs://gpfs2/intern-pretrain-shared02:/mnt/shared-storage-gpfs2/intern-pretrain-shared02 \
    --custom-resources rdma/mlnx_shared=8 \
    --custom-resources mellanox.com/mlnx_rdma=1 \
    --host-network=True \
    -- bash -c "$cmd"
