#!/bin/bash
# 监控checkpoint目录，每小时检查一次，发现新的checkpoint就执行convert
# 用法: ./monitor_ckpt.sh <checkpoint_dir_name>
# 例如: ./monitor_ckpt.sh verl_grpo_policy_Qwen3-30B-A3B-ML_reward_RULE_data_chemistry_moi_half_part

set -e

# 配置
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_OUTPUT_DIR="/mnt/shared-storage-user/ailab-hs/zouyicheng/POLAR_Next/outputs"

# 检查参数
if [ -z "$1" ]; then
    echo "用法: $0 <checkpoint_dir_name>"
    echo "例如: $0 verl_grpo_policy_Qwen3-30B-A3B-ML_reward_RULE_data_chemistry_moi_half_part"
    exit 1
fi

CHECKPOINT_NAME="$1"
CHECKPOINT_DIR="${BASE_OUTPUT_DIR}/${CHECKPOINT_NAME}"

# 检查目录是否存在
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "错误: 目录不存在: $CHECKPOINT_DIR"
    exit 1
fi

OUTPUT_BASE_DIR="${CHECKPOINT_DIR}/hf_models"
mkdir -p "${SCRIPT_DIR}/tmp" "${SCRIPT_DIR}/logs"
PROCESSED_FILE="${SCRIPT_DIR}/tmp/.processed_checkpoints_${CHECKPOINT_NAME}"
LOG_FILE="${SCRIPT_DIR}/logs/monitor_${CHECKPOINT_NAME}.log"
CHECK_INTERVAL=3600  # 1小时 = 3600秒

# 日志函数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# 如果processed文件不存在，创建它
if [ ! -f "$PROCESSED_FILE" ]; then
    touch "$PROCESSED_FILE"
    log "创建已处理checkpoint记录文件: $PROCESSED_FILE"
fi

# 转换单个checkpoint的函数
convert_checkpoint() {
    local step_name=$1
    log "开始转换checkpoint: $step_name"
    
    local cmd="source /mnt/shared-storage-user/ailab-hs/zouyicheng/.bashrc && conda activate verl-061 && \
cd /mnt/shared-storage-user/ailab-hs/zouyicheng/POLAR_Next/verl && python -m verl.model_merger merge \
--backend fsdp \
--local_dir \"$CHECKPOINT_DIR/$step_name/actor\" \
--target_dir \"$OUTPUT_BASE_DIR/actor_${step_name}\" \
--use_cpu_initialization"

    local REPLICAS=1
    # 截取CHECKPOINT_NAME后缀作为任务名前缀，避免过长
    local name_prefix=$(echo "$CHECKPOINT_NAME" | rev | cut -d'_' -f1-3 | rev)
    local name="${name_prefix}_${step_name}"
    
    rjob submit -e DISTRIBUTED_JOB=true \
        --image=registry.h.pjlab.org.cn/library/ml-base:22.04-pjlab \
        --host-network=true --name "$name" -P $REPLICAS --gpu 1 --cpu 8 --memory 256000 --namespace ailab-llmbr --charged-group llmbr_gpu \
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
    
    if [ $? -eq 0 ]; then
        log "成功提交转换任务: $step_name"
        echo "$step_name" >> "$PROCESSED_FILE"
    else
        log "提交转换任务失败: $step_name"
    fi
}

# 主循环
log "============================================"
log "开始监控checkpoint目录: $CHECKPOINT_DIR"
log "检查间隔: ${CHECK_INTERVAL}秒 (1小时)"
log "============================================"

while true; do
    log "开始检查新的checkpoint..."
    
    # 查找所有global_step_*目录
    for ckpt_dir in "$CHECKPOINT_DIR"/global_step_*; do
        if [ -d "$ckpt_dir" ]; then
            step_name=$(basename "$ckpt_dir")
            
            # 检查是否已处理过
            if ! grep -qx "$step_name" "$PROCESSED_FILE" 2>/dev/null; then
                # 检查actor目录是否存在（确保checkpoint完整）
                if [ -d "$ckpt_dir/actor" ]; then
                    log "发现新的checkpoint: $step_name"
                    convert_checkpoint "$step_name"
                else
                    log "发现checkpoint $step_name 但actor目录不存在，跳过"
                fi
            fi
        fi
    done
    
    log "检查完成，等待下一次检查..."
    sleep $CHECK_INTERVAL
done
