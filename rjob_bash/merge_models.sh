#!/bin/bash
set -ex

cmd='eval "$(/mnt/shared-storage-user/ailab-hs/yangyuming/miniconda3/bin/conda shell.bash hook 2> /dev/null)" && \
conda env list && \
source activate /mnt/shared-storage-user/songdemin/user/zouyicheng/miniconda3/envs/verl-061 && \
cd /mnt/shared-storage-user/ailab-hs/chenjiayi/POLAR_Next/verl && \
python -m verl.model_merger merge --backend fsdp --local_dir /mnt/shared-storage-user/ailab-hs/chenjiayi/POLAR_Next/outputs/sft_model_Qwen3-8B-base_data_openthoughts3-400k/global_step_1562 --target_dir /mnt/shared-storage-user/ailab-hs/chenjiayi/models/polar-next/sft/qwen3-8b-base-sft-400k'


REPLICAS=1
name=merge-model

rjob submit -e DISTRIBUTED_JOB=false \
    --image=registry.h.pjlab.org.cn/ailab/zouyicheng:verl050-vllm083-dev \
    --host-network=true --name $name -P $REPLICAS --gpu 1 --cpu 64  --memory 500000 --charged-group llmfudan_gpu \
    --private-machine='group' \
    --mount=gpfs://gpfs1/shared-storage-ailab-llmfudan:/mnt/shared-storage-user/shared-storage-ailab-llmfudan \
    --mount=gpfs://gpfs1/ailab-hs:/mnt/shared-storage-user/ailab-hs \
    --mount=gpfs://gpfs1/songdemin:/mnt/shared-storage-user/songdemin \
    --host-network=True \
    --namespace=ailab-llmfudan \
    --priority 9 \
    -- bash -c "$cmd"