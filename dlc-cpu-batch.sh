#!/bin/bash
set -e

# 使用方法: ./dlc-cpu-batch.sh <start_idx> <end_idx> [batch_size]
# 例如: ./dlc-cpu-batch.sh 0 100       # 将 0-100 分成每5个一组
# 例如: ./dlc-cpu-batch.sh 0 100 10    # 将 0-100 分成每10个一组

if [ $# -lt 2 ]; then
    echo "用法: $0 <start_idx> <end_idx> [batch_size]"
    echo "  start_idx:  起始编号"
    echo "  end_idx:    结束编号"
    echo "  batch_size: 每组文件数量 (默认: 5)"
    exit 1
fi

START=$1
END=$2
BATCH_SIZE=${3:-5}  # 默认每5个文件一组

echo "=========================================="
echo "任务分发配置:"
echo "  起始编号: $START"
echo "  结束编号: $END"
echo "  每组大小: $BATCH_SIZE"
echo "=========================================="

# 计算总共需要提交多少个任务
total_files=$((END - START))
num_batches=$(( (total_files + BATCH_SIZE - 1) / BATCH_SIZE ))

echo "总文件数: $total_files"
echo "将分成 $num_batches 个批次任务"
echo "=========================================="

# 遍历每个批次并提交任务
i=0
while [ $i -lt $num_batches ]; do
    batch_start=$((START + i * BATCH_SIZE))
    batch_end=$((batch_start + BATCH_SIZE - 1))
    
    # 确保最后一个批次不超过 END
    if [ $batch_end -gt $END ]; then
        batch_end=$END
    fi
    
    echo "提交任务 $((i+1))/$num_batches: start=$batch_start, end=$batch_end"
    
    cmd="source /mnt/shared-storage-user/ailab-hs/zouyicheng/.bashrc && conda activate verl-061 && \
cd /mnt/shared-storage-user/ailab-hs/zouyicheng/POLAR_Next && \
PYTHONPATH=. python ./src/process/compute_score.py --start ${batch_start} --end ${batch_end} --mode gpt"

    REPLICAS=1
    name="compute-score-${batch_start}-${batch_end}"
    
    rjob submit -e DISTRIBUTED_JOB=false \
        --image=registry.h.pjlab.org.cn/library/ml-base:22.04-pjlab \
        --host-network=true --name $name -P $REPLICAS --gpu 0 --cpu 16 --memory 182000 --namespace ailab-hs --charged-group hs_gpu \
        --private-machine='group' \
        --gang-start=true \
        --mount=gpfs://gpfs1/songdemin:/mnt/shared-storage-user/songdemin \
        --mount=gpfs://gpfs1/ailab-hs:/mnt/shared-storage-user/ailab-hs \
        --mount=gpfs://gpfs1/large-model-center-share-weights:/mnt/shared-storage-user/large-model-center-share-weights \
        --custom-resources rdma/mlnx_shared=8 \
        --custom-resources mellanox.com/mlnx_rdma=1 \
        --host-network=True \
        -- bash -c "$cmd"
    
    echo "任务 $name 已提交"
    echo "------------------------------------------"
    
    # 稍微等待一下，避免提交过快
    sleep 1
    
    i=$((i + 1))
done

echo "=========================================="
echo "所有 $num_batches 个任务已提交完成!"
echo "=========================================="
