#!/bin/bash
# 自动提交Jenkins评测任务脚本
# 用法: ./submit_eval_task.sh <model_path> <step_name> [subdataset]
# 例如: ./submit_eval_task.sh verl_grpo_policy_Qwen3-30B-A3B-ML_reward_RULE_LLM_JUDGE_data_chemistry_moi_half_part global_step_20

set -e

# ==================== 配置区域 ====================
JENKINS_URL="http://10.140.52.82:8080"
JOB_NAME="api_eval_v3"

# Jenkins 认证配置
# 方式1: 直接设置用户名和API Token (不推荐,有安全风险)
JENKINS_USER="rowitzou"
JENKINS_TOKEN="1103ee8a1bc5b80a774625ae32a7ed9f34"

# 基础配置 (按任务3837的参数设置)
CLUSTER="yidian"
WORKSPACE_ID="hs_gpu"
USER="zouyicheng"
OUTPUT_DIR="zouyicheng"

# 评测配置
EVAL_TYPE="chat_objective"
AUTO_EVAL_VERSION="ld_0110_oc_286a6eb_v3"
OCP_VERSION="fullbench_v1_8"
INFER_WORKER_NUMS="16"
EVAL_NUMS="15"
DATASET_MAX_OUT_LEN="32768"
DEFAULT_SUBDATASET="[*mol_gen_selfies_datasets]"

# 推理后端配置
INFER_IMAGE="registry.h.pjlab.org.cn/ailab-puyu-puyu_gpu/lmdeploy:v0.11.0-cu12.8"
INFER_ENGINE="lmdeploy"
GPU_NUM="1"
MEMORY="80000"
CPU="8"
OC_CPU="1"
OC_MEM="4000"
END_NUM="7"
NODE_NUM="1"
DPEP="false"
DELETE="false"
START_INFER="true"
INFER_EXTRA_PARAMS="base64://LS10cD0x"  # base64 encoded "--tp=1"

# 模型推理配置 (OpenCompass)
TOKENIZER_PATH="/mnt/shared-storage-user/songdemin/user/caomaosong/ckpts/235b/base01_d005004_fix_s1_s2merge_lr2e_5_512gpus_2/20250625201507/hf-2427"

# 飞书通知
FEISHU_TOKEN="https://open.feishu.cn/open-apis/bot/v2/hook/e6de9b43-4bc4-4f64-987d-7b04031249c7"

# 基础路径
BASE_OUTPUT_DIR="/mnt/shared-storage-user/ailab-hs/zouyicheng/POLAR_Next/outputs"

# ==================== 参数处理 ====================

# 解析参数，支持 --submit 在任意位置
SUBMIT_MODE=false
POSITIONAL_ARGS=()

for arg in "$@"; do
    if [ "$arg" = "--submit" ]; then
        SUBMIT_MODE=true
    else
        POSITIONAL_ARGS+=("$arg")
    fi
done

# 重新赋值位置参数
set -- "${POSITIONAL_ARGS[@]}"

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "用法: $0 <checkpoint_dir_name> <step_name> [subdataset] [--submit]"
    echo "例如: $0 verl_grpo_policy_Qwen3-30B-A3B-ML_reward_RULE_LLM_JUDGE_data_chemistry_moi_half_part 20"
    echo "      $0 verl_grpo_policy_Qwen3-30B-A3B-ML_reward_RULE_LLM_JUDGE_data_chemistry_moi_half_part 20 --submit"
    echo ""
    echo "参数说明:"
    echo "  checkpoint_dir_name: outputs目录下的checkpoint文件夹名"
    echo "  step_name: 训练步数,如 20 (会自动拼接为 global_step_20)"
    echo "  subdataset: (可选) 评测数据集子集,默认为 ${DEFAULT_SUBDATASET}"
    echo "  --submit: (可选) 添加此参数直接提交任务,否则只预览"
    exit 1
fi

CHECKPOINT_NAME="$1"
STEP_NAME="$2"
SUBDATASET="${3:-$DEFAULT_SUBDATASET}"

# 构建模型路径
MODEL_PATH="${BASE_OUTPUT_DIR}/${CHECKPOINT_NAME}/hf_models/actor_global_step_${STEP_NAME}"

# 检查模型路径是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo "警告: 模型路径不存在: $MODEL_PATH"
    echo "请确认模型已转换完成。"
    read -p "是否继续提交任务? (y/n): " confirm
    if [ "$confirm" != "y" ]; then
        echo "已取消"
        exit 1
    fi
fi

# 构建model_abbr (实验名称)
MODEL_ABBR="${CHECKPOINT_NAME}_step_${STEP_NAME}"

echo "=========================================="
echo "准备提交评测任务"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT_NAME"
echo "Step: $STEP_NAME"
echo "Model Path: $MODEL_PATH"
echo "Model Abbr: $MODEL_ABBR"
echo "Subdataset: $SUBDATASET"
echo "=========================================="

# ==================== 构建参数 ====================

# 构建 infer_backend_config JSON
INFER_BACKEND_CONFIG=$(cat <<EOF
{
    "end_num": ${END_NUM},
    "gpu_num": ${GPU_NUM},
    "memory": "${MEMORY}",
    "cpu": ${CPU},
    "oc_cpu": ${OC_CPU},
    "oc_mem": ${OC_MEM},
    "model": "${MODEL_ABBR}",
    "model_path": "${MODEL_PATH}",
    "image": "${INFER_IMAGE}",
    "infer_engine": "${INFER_ENGINE}",
    "extra_envs": "",
    "infer_extra_params": "${INFER_EXTRA_PARAMS}",
    "delete": "${DELETE}",
    "start_infer": "${START_INFER}",
    "node_num": ${NODE_NUM},
    "dpep": "${DPEP}"
}
EOF
)

# 构建 model_infer_config (OpenCompass格式)
# 注意: openai_api_base 会在任务启动时由Jenkins自动生成
MODEL_INFER_CONFIG=$(cat <<'EOF'
dict(
    type=OpenAISDK,
    key='sk-admin',
    openai_api_base=[
        'http://s-20260104203038-22bhb-decode.ailab-evalservice.svc:4000/v1',
    ],
    meta_template=dict(
        round=[
            dict(role='SYSTEM', api_role='SYSTEM'),
            dict(role='HUMAN', api_role='HUMAN'),
            dict(role='BOT', api_role='BOT', generate=True),
        ]
    ),
    query_per_second=8,
    batch_size=32,
    temperature=0.6,
    tokenizer_path='TOKENIZER_PATH_PLACEHOLDER',
    retry=10,
    max_out_len=65536,
    max_seq_len=65536,
    max_workers=32,
    pred_postprocessor=dict(
        type=extract_non_reasoning_content,
    ),
)
EOF
)

# 替换tokenizer路径
MODEL_INFER_CONFIG="${MODEL_INFER_CONFIG/TOKENIZER_PATH_PLACEHOLDER/$TOKENIZER_PATH}"

# ==================== 提交任务 ====================

submit_task() {
    echo ""
    
    # 检查认证信息
    if [ -z "$JENKINS_USER" ] || [ -z "$JENKINS_TOKEN" ]; then
        echo "❌ 错误: 缺少Jenkins认证信息!"
        echo ""
        echo "请设置环境变量后重试:"
        echo "  export JENKINS_USER='your_username'"
        echo "  export JENKINS_TOKEN='your_api_token'"
        echo ""
        echo "API Token获取方式: Jenkins -> 用户设置 -> API Token -> 添加新Token"
        exit 1
    fi
    
    echo "正在提交任务 (用户: ${JENKINS_USER})..."
    
    # 使用 curl 提交到 Jenkins (需要绕过代理)
    # Jenkins buildWithParameters API，使用 Basic Auth 认证
    response=$(curl --noproxy '*' -s -w "\n%{http_code}" -X POST \
        -u "${JENKINS_USER}:${JENKINS_TOKEN}" \
        "${JENKINS_URL}/job/${JOB_NAME}/buildWithParameters" \
        --data-urlencode "cluster=${CLUSTER}" \
        --data-urlencode "workspace_id=${WORKSPACE_ID}" \
        --data-urlencode "model_abbr=${MODEL_ABBR}" \
        --data-urlencode "infer_backend_config=${INFER_BACKEND_CONFIG}" \
        --data-urlencode "model_infer_config=${MODEL_INFER_CONFIG}" \
        --data-urlencode "llm_judger_config=" \
        --data-urlencode "infer_worker_nums=${INFER_WORKER_NUMS}" \
        --data-urlencode "eval_nums=${EVAL_NUMS}" \
        --data-urlencode "eval_type=${EVAL_TYPE}" \
        --data-urlencode "auto_eval_version=${AUTO_EVAL_VERSION}" \
        --data-urlencode "ocp_version=${OCP_VERSION}" \
        --data-urlencode "subdataset=${SUBDATASET}" \
        --data-urlencode "output_dir=${OUTPUT_DIR}" \
        --data-urlencode "cli_extra=" \
        --data-urlencode "dataset_max_out_len=${DATASET_MAX_OUT_LEN}" \
        --data-urlencode "feishu_token=${FEISHU_TOKEN}" \
        --data-urlencode "user=${USER}" \
        2>&1)
    
    # 提取HTTP状态码 (最后一行)
    http_code=$(echo "$response" | tail -1)
    body=$(echo "$response" | sed '$d')
    
    if [ "$http_code" = "201" ] || [ "$http_code" = "200" ] || [ "$http_code" = "302" ]; then
        echo "✅ 任务提交成功! HTTP状态码: $http_code"
        echo "请访问 ${JENKINS_URL}/job/${JOB_NAME}/ 查看任务状态"
    else
        echo "❌ 任务提交失败! HTTP状态码: $http_code"
        if [ "$http_code" = "401" ] || [ "$http_code" = "403" ]; then
            echo "认证失败! 请检查 JENKINS_USER 和 JENKINS_TOKEN 是否正确"
        fi
        echo "响应内容: $body"
        exit 1
    fi
}

# ==================== 预览模式 ====================

preview_task() {
    echo ""
    echo "==================== 任务参数预览 ===================="
    echo "Jenkins URL: ${JENKINS_URL}/job/${JOB_NAME}/buildWithParameters"
    echo ""
    echo "--- 基础参数 ---"
    echo "cluster: ${CLUSTER}"
    echo "workspace_id: ${WORKSPACE_ID}"
    echo "user: ${USER}"
    echo "output_dir: ${OUTPUT_DIR}"
    echo ""
    echo "--- 模型参数 ---"
    echo "model_abbr: ${MODEL_ABBR}"
    echo "model_path: ${MODEL_PATH}"
    echo ""
    echo "--- 评测参数 ---"
    echo "eval_type: ${EVAL_TYPE}"
    echo "auto_eval_version: ${AUTO_EVAL_VERSION}"
    echo "ocp_version: ${OCP_VERSION}"
    echo "subdataset: ${SUBDATASET}"
    echo "dataset_max_out_len: ${DATASET_MAX_OUT_LEN}"
    echo ""
    echo "--- 推理参数 ---"
    echo "infer_worker_nums: ${INFER_WORKER_NUMS}"
    echo "eval_nums: ${EVAL_NUMS}"
    echo ""
    echo "--- infer_backend_config ---"
    echo "$INFER_BACKEND_CONFIG"
    echo ""
    echo "--- model_infer_config ---"
    echo "$MODEL_INFER_CONFIG"
    echo "======================================================="
}

# ==================== 主逻辑 ====================

# 默认为预览模式,不自动提交
preview_task

echo ""
echo "提示: 当前为预览模式,任务未提交。"

# 检查认证状态
if [ -z "$JENKINS_USER" ] || [ -z "$JENKINS_TOKEN" ]; then
    echo ""
    echo "⚠️  警告: 未设置Jenkins认证信息"
    echo "提交任务前请先设置环境变量:"
    echo "  export JENKINS_USER='your_username'"
    echo "  export JENKINS_TOKEN='your_api_token'"
else
    echo "✓ 认证信息已配置 (用户: ${JENKINS_USER})"
fi

echo ""
echo "如需提交任务,请运行: $0 $1 $2 ${3:+\"$3\"} --submit"

# 检查是否有 --submit 参数
if [ "$SUBMIT_MODE" = true ]; then
    submit_task
fi
