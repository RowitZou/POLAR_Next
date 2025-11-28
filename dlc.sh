#!/bin/bash
set -ex

cmd="source /mnt/shared-storage-user/ailab-hs/zouyicheng/.bashrc && conda activate verl-061 && \
cd /mnt/shared-storage-user/ailab-hs/zouyicheng/POLAR_Next/verl && \
bash ../examples/opd/qwen3-8b_cmphysbench.sh"

data=cmphysbench
policy=qwen3-8b
reward=OPD

REPLICAS=1
name="verl-polar-${policy}-${reward}-${data}"
rjob submit -e DISTRIBUTED_JOB=true \
    --image=registry.h.pjlab.org.cn/library/ml-base:22.04-pjlab \
    --host-network=true --name $name -P $REPLICAS --gpu 8 --cpu 128  --memory 1600000 --namespace ailab-hs --charged-group hs_gpu \
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
