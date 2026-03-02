#!/bin/bash
# Verl PPO training script for Qwen2.5-7B
set -x

# Parameters from original script
nodes=4
train_batch_size=512
actor_lr=1e-6
data_name=chemistry_moi
policy_model_name=Qwen3-30B-A3B-MM
reward_model_name=RULE_LLM_JUDGE

# Model paths
actor_path=/mnt/shared-storage-user/ailab-hs/zouyicheng/POLAR/outputs/sft/Qwen3_30B_A3_instruct-general-continue-mix-remove-cot/20260217005852/hf-latest

# Data paths
train_data_path=/mnt/shared-storage-user/ailab-hs/zouyicheng/POLAR_Next/data/chemistry_moi/train/train.parquet
test_data_path=/mnt/shared-storage-user/ailab-hs/zouyicheng/POLAR_Next/data/chemistry_moi/train/train.parquet

# Reward Configuration
reward_func_path="../src/reward/mix_env/reward_chemistry.py"

# Experiment name
name="verl_grpo_policy_${policy_model_name}_reward_${reward_model_name}_data_${data_name}"
output_dir="../outputs/${name}"

# Create output directory if it doesn't exist
mkdir -p $output_dir

# Disable NCCL monitoring if still having issues (uncomment if needed)
export TORCH_NCCL_ENABLE_MONITORING=0

export WANDB_API_KEY=c89518a9cc46b986f6f2ad122a952229a76d1445
export http_proxy=http://100.100.67.192:1082
export https_proxy=http://100.100.67.192:1082

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

    python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
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
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$train_batch_size \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$(((2048 + 32768)*2)) \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.data_parallel_size=1 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$(((2048 + 32768)*4)) \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.max_num_seqs=512 \
    actor_rollout_ref.rollout.max_num_batched_tokens=557056 \
    \
    reward_model.enable=False \
    reward_model.reward_manager=batch \
    custom_reward_function.path=$reward_func_path \
    custom_reward_function.name=compute_score_batch \
    \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$nodes \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_chemistry_moi' \
    trainer.val_before_train=False \
    trainer.experiment_name="$name" \
    trainer.save_freq=20 \
    trainer.total_epochs=1 \
    trainer.test_freq=-1 \
    trainer.max_actor_ckpt_to_keep=5 \
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
