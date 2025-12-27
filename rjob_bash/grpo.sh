#!/bin/bash
set -ex

# cmd='eval "$(/mnt/shared-storage-user/ailab-hs/yangyuming/miniconda3/bin/conda shell.bash hook 2> /dev/null)" && \
# conda env list && \
# source activate /mnt/shared-storage-user/songdemin/user/zouyicheng/miniconda3/envs/verl-061 && \
# export http_proxy=http://100.100.67.157:1081 && \
# export https_proxy=http://100.100.67.157:1081 && \
# cd /mnt/shared-storage-user/ailab-hs/chenjiayi/POLAR_Next/verl && \
# bash ../examples/grpo/qwen3-8b-32b.sh'

cmd='eval "$(/mnt/shared-storage-user/ailab-hs/yangyuming/miniconda3/bin/conda shell.bash hook 2> /dev/null)" && \
conda env list && \
source activate /mnt/shared-storage-user/songdemin/user/zouyicheng/miniconda3/envs/verl-061 && \
cd /mnt/shared-storage-user/ailab-hs/chenjiayi/POLAR_Next/verl && \
bash ../examples/grpo/qwen3-8b-32b.sh'



REPLICAS=2
name=opd-kl-1-qwen3-8b-base-sft-400k
rjob submit -e DISTRIBUTED_JOB=false \
    --image=registry.h.pjlab.org.cn/ailab/zouyicheng:verl050-vllm083-dev \
    --host-network=true --name $name -P $REPLICAS --gpu 8 --cpu 110  --memory 1600000 --charged-group hs_gpu \
    --private-machine='group' \
    --mount=gpfs://gpfs1/shared-storage-ailab-llmfudan:/mnt/shared-storage-user/shared-storage-ailab-llmfudan \
    --mount=gpfs://gpfs1/ailab-hs:/mnt/shared-storage-user/ailab-hs \
    --mount=gpfs://gpfs1/songdemin:/mnt/shared-storage-user/songdemin \
    --host-network=True \
    --namespace=ailab-hs \
    --priority 9 \
    -- bash -c "$cmd"
