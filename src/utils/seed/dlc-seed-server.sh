#!/bin/bash
set -ex

cmd="source /mnt/shared-storage-user/ailab-hs/zouyicheng/.bashrc && conda activate verl-061 && \
cd /mnt/shared-storage-user/ailab-hs/zouyicheng/POLAR_Next && \
bash src/utils/start_seed_server.sh -w 32 -p 30030 -t 300.0"

REPLICAS=1
name="seed-score-server"
rjob submit -e DISTRIBUTED_JOB=false \
    --image=registry.h.pjlab.org.cn/library/ml-base:22.04-pjlab \
    --host-network=true --name $name -P $REPLICAS --gpu 0 --cpu 32 --memory 182000 --namespace ailab-hs --charged-group hs_gpu \
    --private-machine='group' \
    --gang-start=true \
    --mount=gpfs://gpfs1/songdemin:/mnt/shared-storage-user/songdemin \
    --mount=gpfs://gpfs1/ailab-hs:/mnt/shared-storage-user/ailab-hs \
    --mount=gpfs://gpfs1/large-model-center-share-weights:/mnt/shared-storage-user/large-model-center-share-weights \
    --custom-resources rdma/mlnx_shared=8 \
    --custom-resources mellanox.com/mlnx_rdma=1 \
    --host-network=True \
    -- bash -c "$cmd"
