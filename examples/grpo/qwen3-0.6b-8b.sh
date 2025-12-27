# Tested successfully on the hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0 image.
# It outperforms the Qwen2 7B base model by two percentage points on the test set of GSM8K.

set -x

# conda 挂载
# config配置
export WANDB_MODE=offline
# vLLM 不支持 PyTorch 的 expandable segments 内存分配机制（2024 起默认开启
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
# 关闭 NCCL DEBUG 输出
export NCCL_DEBUG=NONE
export NCCL_DEBUG_SUBSYS=ALL


export HYDRA_FULL_ERROR=1
export NCCL_CUMEM_ENABLE=0 
export VLLM_WORKER_MULTIPROC_METHOD=spawn
ulimit -l unlimited

exec > >(tee -a "/mnt/shared-storage-user/ailab-hs/chenjiayi/POLAR_Next/logs/test_small.log") 2>&1

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    \
    data.train_files=/mnt/shared-storage-user/ailab-hs/zhaojun/luyuyang/data/super_gpqa/random/boxed_train_90.parquet \
    data.val_files=/mnt/shared-storage-user/ailab-hs/zhaojun/luyuyang/data/super_gpqa/random/boxed_test_10.parquet \
    data.train_batch_size=64 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    \
    actor_rollout_ref.model.path=/mnt/shared-storage-user/shared-storage-ailab-llmfudan/models/Qwen3/Qwen3-0.6B \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.05 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=1 \
    \
    ++actor_rollout_ref.ref.model.path=/mnt/shared-storage-user/ailab-hs/chenjiayi/models/opensource/Qwen3-8B\
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    ++actor_rollout_ref.ref.tensor_model_parallel_size=2 \
    actor_rollout_ref.ref.fsdp_config.fsdp_size=-1 \
    \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_example_gpqa' \
    trainer.experiment_name='qwen3_8b_function_rm' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=5 \
    trainer.total_epochs=3 \
    trainer.default_local_dir="../outputs/1125_small" \
    \
    custom_reward_function.path=recipe/char_count/reward_function.py \
    custom_reward_function.name=char_count_reward_function \
    \
    trainer.rollout_data_dir="../outputs/1125_small/trajectory_data/rollout" $@