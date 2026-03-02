#!/bin/bash
# 监控hf_models目录，每小时检查一次，发现新的模型就提交评测任务
# 用法: ./monitor_eval.sh <checkpoint_dir_name>
# 例如: ./monitor_eval.sh verl_grpo_policy_Qwen3-30B-A3B-ML_reward_RULE_data_chemistry_moi_half_part

set -e

# 关闭代理
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

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
HF_MODELS_DIR="${CHECKPOINT_DIR}/hf_models"

# 检查目录是否存在
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "错误: 目录不存在: $CHECKPOINT_DIR"
    exit 1
fi

mkdir -p "${SCRIPT_DIR}/tmp" "${SCRIPT_DIR}/logs"
PROCESSED_FILE="${SCRIPT_DIR}/tmp/.evaluated_checkpoints_${CHECKPOINT_NAME}"
LOG_FILE="${SCRIPT_DIR}/logs/eval_${CHECKPOINT_NAME}.log"
CHECK_INTERVAL=3600  # 1小时 = 3600秒

# 日志函数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# 如果processed文件不存在，创建它
if [ ! -f "$PROCESSED_FILE" ]; then
    touch "$PROCESSED_FILE"
    log "创建已评测checkpoint记录文件: $PROCESSED_FILE"
fi

# 提交评测任务的函数
submit_eval() {
    local step_num=$1
    log "开始提交评测任务: step ${step_num}"
    
    # 调用评测脚本
    if "${SCRIPT_DIR}/submit_eval_task.sh" "$CHECKPOINT_NAME" "$step_num" --submit; then
        log "成功提交评测任务: step ${step_num}"
        echo "$step_num" >> "$PROCESSED_FILE"
    else
        log "提交评测任务失败: step ${step_num}"
    fi
}

# 主循环
log "============================================"
log "开始监控hf_models目录: $HF_MODELS_DIR"
log "检查间隔: ${CHECK_INTERVAL}秒 (1小时)"
log "============================================"

while true; do
    log "开始检查新的模型..."
    
    # 检查hf_models目录是否存在
    if [ ! -d "$HF_MODELS_DIR" ]; then
        log "hf_models目录不存在，等待创建..."
        sleep $CHECK_INTERVAL
        continue
    fi
    
    # 查找所有actor_global_step_*目录
    for model_dir in "$HF_MODELS_DIR"/actor_global_step_*; do
        if [ -d "$model_dir" ]; then
            # 提取step数字，如 actor_global_step_20 -> 20
            dir_name=$(basename "$model_dir")
            step_num=$(echo "$dir_name" | sed 's/actor_global_step_//')
            
            # 检查是否已评测过
            if ! grep -qx "$step_num" "$PROCESSED_FILE" 2>/dev/null; then
                # 检查模型文件是否存在（确保转换完成）
                if [ -f "$model_dir/config.json" ]; then
                    log "发现新的模型: $dir_name (step ${step_num})"
                    submit_eval "$step_num"
                else
                    log "发现模型 $dir_name 但config.json不存在，可能还在转换中，跳过"
                fi
            fi
        fi
    done
    
    log "检查完成，等待下一次检查..."
    sleep $CHECK_INTERVAL
done
