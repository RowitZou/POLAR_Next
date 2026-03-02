#!/usr/bin/env bash
set -xeuo pipefail

# Parameters from original script
nodes=8
train_batch_size=64
actor_lr=1e-6
data_name=cmphysbench
policy_model_name=Qwen3-30B-A3B
reward_model_name=SEED
max_prompt_length=2048
max_response_length=32768
n_resp_per_prompt=8

# Model paths
actor_path=/mnt/shared-storage-user/large-model-center-share-weights/hf_hub/models--Qwen--Qwen3-30B-A3B/snapshots/ae659febe817e4b3ebd7355f47792725801204c9

# Data paths
train_data_path=/mnt/shared-storage-user/ailab-hs/zouyicheng/POLAR/data/CMPhysBench/train.parquet
test_data_path=/mnt/shared-storage-user/ailab-hs/zouyicheng/POLAR/data/CMPhysBench/test.parquet

# Reward Configuration
reward_func_path="../src/reward/rule/reward_seed.py"

# Experiment name
name="verl_dapo_policy_${policy_model_name}_reward_${reward_model_name}_data_${data_name}"
output_dir="../outputs/${name}"

# DAPO settings
clip_ratio_low=0.2
clip_ratio_high=0.3

loss_agg_mode="token-mean"

enable_filter_groups=True
filter_groups_metric=score
max_num_gen_batches=10

use_dynamic_bsz=True

# Create output directory if it doesn't exist
mkdir -p $output_dir

export WANDB_API_KEY=c89518a9cc46b986f6f2ad122a952229a76d1445
export http_proxy=http://100.100.67.213:1081
export https_proxy=http://100.100.67.213:1081

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

    python3 -m recipe.dapo.main_dapo \
    data.train_files="$train_data_path" \
    data.val_files="$test_data_path" \
    data.prompt_key=prompt \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.gen_batch_size=$((train_batch_size * 3)) \
    data.train_batch_size=${train_batch_size} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$(((max_prompt_length + max_response_length)*8)) \
    actor_rollout_ref.model.path="${actor_path}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=${actor_lr} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_batch_size} \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.90 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$(((max_prompt_length + max_response_length)*8)) \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    reward_model.enable=False \
    reward_model.reward_manager=dapo \
    custom_reward_function.path=$reward_func_path \
    custom_reward_function.name=compute_score \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_dapo_cmphysbench' \
    trainer.experiment_name="${name}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${nodes} \
    trainer.val_before_train=True \
    trainer.test_freq=5 \
    trainer.save_freq=10 \
    trainer.total_epochs=20 \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.max_critic_ckpt_to_keep=2 \
    trainer.default_local_dir=$output_dir \
    trainer.rollout_data_dir="${output_dir}/trajectory_data/rollout" \
    trainer.validation_data_dir="${output_dir}/trajectory_data/validation" \
    trainer.resume_mode=auto

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
