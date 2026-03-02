#!/bin/bash
# On-Policy Distillation (OPD) Training Script
#
# This script trains a student model to match a teacher model using OPD.
# Key settings:
# - adv_estimator=opd: Uses KL-based advantage (works with n=1)
# - No critic needed
# - No KL in reward or KL loss (KL is computed as advantage directly)

set -e

# Model paths
STUDENT_MODEL_PATH=${STUDENT_MODEL_PATH:-"/path/to/student/model"}
TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-"/path/to/teacher/model"}

# Data paths
TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-"/path/to/train/data.parquet"}
VAL_DATA_PATH=${VAL_DATA_PATH:-"/path/to/val/data.parquet"}

# Training settings
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
LEARNING_RATE=${LEARNING_RATE:-1e-6}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
NNODES=${NNODES:-1}

# Experiment settings
PROJECT_NAME=${PROJECT_NAME:-"verl-opd"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"opd_experiment"}

# Run OPD training
python3 -m recipe.opd.main_opd \
    algorithm.adv_estimator=opd \
    algorithm.opd_estimator=token_level \
    algorithm.use_kl_in_reward=False \
    algorithm.normalize_advantages=False \
    \
    data.train_files="$TRAIN_DATA_PATH" \
    data.val_files="$VAL_DATA_PATH" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=2048 \
    data.max_response_length=32768 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.prompt_key='prompt' \
    \
    actor_rollout_ref.model.path="$STUDENT_MODEL_PATH" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.nccl_timeout=3600 \
    \
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.data_parallel_size=1 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.max_num_seqs=64 \
    \
    +actor_rollout_ref.ref.model.path="$TEACHER_MODEL_PATH" \
    +actor_rollout_ref.ref.model.use_remove_padding=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.forward_only=True \
    actor_rollout_ref.ref.fsdp_config.optimizer_offload=True \
    \
    reward_model.enable=False \
    \
    critic.enable=False \
    \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.save_freq=100 \
    trainer.val_freq=50 \
    trainer.total_epochs=20 \
    trainer.total_training_steps=1000
