#!/bin/bash
# Verl OPD (On-Policy Distillation) training script using the new recipe implementation
# This script uses the opd recipe which only uses KL loss (no advantage)
# Key differences from original:
# 1. n can be 1 since we don't need group sampling
# 2. Uses built-in zero reward function (no need for external reward_zero.py)
set -x

# Parameters from original script
nodes=2
train_batch_size=64
actor_lr=2e-6
data_name=cmphysbench
policy_model_name=Qwen3-8B
ref_model_name=Qwen3-30B-A3B
reward_model_name=ZERO

# Model paths
actor_path=/mnt/shared-storage-user/ailab-hs/zouyicheng/POLAR/models/Qwen3-8B
ref_path=/mnt/shared-storage-user/large-model-center-share-weights/hf_hub/models--Qwen--Qwen3-30B-A3B/snapshots/ae659febe817e4b3ebd7355f47792725801204c9

# Data paths
train_data_path=/mnt/shared-storage-user/ailab-hs/zouyicheng/POLAR_Next/data/CMPhysBench/train_raw.parquet
test_data_path=/mnt/shared-storage-user/ailab-hs/zouyicheng/POLAR_Next/data/CMPhysBench/test.parquet

# The OPD recipe provides built-in zero reward functions in reward/ directory:
# - recipe.opd.reward.zero_reward:compute_score_zero (for naive reward manager)
# - recipe.opd.reward.zero_reward:compute_score_zero_batch (for batch reward manager)
# Note: Zero reward is automatically loaded by load_reward_manager() if no custom reward is specified

# Experiment name - add "_recipe" suffix to distinguish from original
name="verl_opd_recipe_policy_${policy_model_name}_reward_${reward_model_name}_ref_${ref_model_name}_data_${data_name}_lr_${actor_lr}"
output_dir="../outputs/${name}"

# Create output directory if it doesn't exist
mkdir -p $output_dir

# Disable NCCL monitoring if still having issues (uncomment if needed)
export TORCH_NCCL_ENABLE_MONITORING=0

# ============ Other Configuration ============
export WANDB_API_KEY=c89518a9cc46b986f6f2ad122a952229a76d1445
export http_proxy=http://100.100.67.192:1081
export https_proxy=http://100.100.67.192:1081

# Set wandb to offline mode to prevent online sync
# export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

TARGET_FILE="$output_dir/addr_${name}.txt"
RANK=${RANK:-${NODE_RANK:-0}}
MASTER_PORT=6379
MASTER_ADDR=${MASTER_ADDR}
echo "MASTER_ADDR: $MASTER_ADDR"
echo "Rank $RANK is running on $MASTER_ADDR"

if [ "$RANK" -eq 0 ]; then 
    echo "Starting head node (RANK=${RANK}) on port $MASTER_PORT..."
    
    MASTER_ADDR=${MASTER_ADDR}
    echo "$MASTER_ADDR" > "$TARGET_FILE"

    ray start --head --num-gpus 8 --dashboard-host=0.0.0.0 --dashboard-port=8265 --disable-usage-stats --block &
    sleep 60
    
    echo "Executing main program on head node..."

    # Use the new OPD recipe instead of verl.trainer.main_ppo
    # Key changes:
    # 1. Use recipe.opd.main_opd as entry point
    # 2. Use algorithm.adv_estimator=opd (zero advantage)
    # 3. n=1 is now possible since we don't need group sampling
    python3 -m recipe.opd.main_opd \
    algorithm.adv_estimator=opd \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0 \
    \
    data.train_files="$train_data_path" \
    data.val_files="$test_data_path" \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=2048 \
    data.max_response_length=32768 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.prompt_key='prompt' \
    \
    actor_rollout_ref.model.path="$actor_path" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.nccl_timeout=3600 \
    \
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$train_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=1.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.data_parallel_size=1 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.max_num_seqs=64 \
    actor_rollout_ref.rollout.max_num_batched_tokens=557056 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    \
    +actor_rollout_ref.ref.model.path="$ref_path" \
    +actor_rollout_ref.ref.model.use_remove_padding=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.forward_only=True \
    actor_rollout_ref.ref.fsdp_config.optimizer_offload=True \
    \
    reward_model.enable=False \
    reward_model.reward_manager=batch \
    \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$nodes \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_opd_cmphysbench' \
    trainer.val_before_train=True \
    trainer.experiment_name="$name" \
    trainer.save_freq=10 \
    trainer.total_epochs=20 \
    trainer.test_freq=5 \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.max_critic_ckpt_to_keep=2 \
    trainer.default_local_dir=$output_dir \
    \
    trainer.rollout_data_dir="${output_dir}/trajectory_data/rollout" \
    trainer.validation_data_dir="${output_dir}/trajectory_data/validation"
    $@

else 
    sleep 30
    MASTER_ADDR=$(cat "$TARGET_FILE")

    echo "Starting worker node (RANK=${RANK}), connecting to ${MASTER_ADDR}:${MASTER_PORT}..."
    ray start --address ${MASTER_ADDR}:${MASTER_PORT}  --num-gpus 8 --block &
    
    sleep 120
    while true; do
        status=$(ray status 2>&1)

        if echo "$status" | grep -q "Active:"; then
            echo "Active nodes found. Sleeping for 10 min..."
            sleep 600
        else
            echo "No active nodes found. Exiting..."
            exit 0
        fi
    done
fi
