# Tested successfully on the hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0 image.
# It outperforms the Qwen2 7B base model by two percentage points on the test set of GSM8K.

set -x

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

nodes=2
train_batch_size= 1024 # 512
actor_lr=1e-6
kl_loss_coef=1 # 0.001
data_name=supergpqa_kl_1
policy_model_name=Qwen3-8B
ref_model_name=Qwen3-32B
train_epoch=5 #10 

# Model paths
actor_path=/mnt/shared-storage-user/ailab-hs/chenjiayi/models/opensource/Qwen3-8B #/mnt/shared-storage-user/ailab-hs/chenjiayi/models/polar_next/kl_1_continue_kl_0_3epoch
ref_path=/mnt/shared-storage-user/ailab-hs/chenjiayi/models/opensource/Qwen3-32B

# Data paths
train_data_path=/mnt/shared-storage-user/ailab-hs/zhaojun/luyuyang/data/super_gpqa/random/boxed_train_90.parquet
test_data_path=/mnt/shared-storage-user/ailab-hs/zhaojun/luyuyang/data/super_gpqa/random/boxed_test_10.parquet

# Reward Configuration
reward_func_path="../src/polar/reward_func_on_policy.py"

# Experiment name
name="verl_grpo_policy_${policy_model_name}_ref_${ref_model_name}_data_${data_name}"
output_dir="../outputs/${name}"

# Create output directory if it doesn't exist
mkdir -p $output_dir

mkdir -p "$output_dir/logs"

timestamp=$(date +"%Y%m%d_%H%M%S")
logfile="${output_dir}/logs/train_${timestamp}_rank${RANK:-0}.log"

echo "Logging to $logfile"
exec > >(tee -a "$logfile") 2>&1

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
    sleep 30
    
    echo "Executing main program on head node..."

    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        algorithm.use_kl_in_reward=False \
        \
        data.train_files="$train_data_path" \
        data.val_files="$test_data_path" \
        data.train_batch_size=$train_batch_size \
        data.max_prompt_length=1000 \
        data.max_response_length=20000 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        \
        actor_rollout_ref.model.path="$actor_path" \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        \
        actor_rollout_ref.actor.optim.lr=$actor_lr \
        actor_rollout_ref.actor.ppo_mini_batch_size=$train_batch_size \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
        actor_rollout_ref.rollout.n=1 \
        actor_rollout_ref.rollout.max_num_batched_tokens=30000 \
        \
        ++actor_rollout_ref.ref.model.path="$ref_path" \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        \
        trainer.critic_warmup=0 \
        trainer.logger='["console","wandb"]' \
        trainer.project_name='verl_grpo_example_gpqa' \
        trainer.experiment_name="$name" \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=$nodes \
        trainer.save_freq=100 \
        trainer.test_freq=5 \
        trainer.total_epochs=$train_epoch \
        trainer.default_local_dir=$output_dir \
        \
        custom_reward_function.path=$reward_func_path \
        custom_reward_function.name=compute_score \
        \
        trainer.rollout_data_dir="${output_dir}/trajectory_data/rollout" \
        $@

else 
    # sleep 10
    # MASTER_ADDR=$(cat "$TARGET_FILE")

    # echo "Starting worker node (RANK=${RANK}), connecting to ${MASTER_ADDR}:${MASTER_PORT}..."
    # ray start --address ${MASTER_ADDR}:${MASTER_PORT} --num-gpus 8 --block &
    
    # sleep 60
    # while true; do
    #     status=$(ray status 2>&1)

    #     if echo "$status" | grep -q "Active:"; then
    #         echo "Active nodes found. Sleeping for 10 min..."
    #         sleep 600
    #     else
    #         echo "No active nodes found. Exiting..."
    #         exit 0
    #     fi
    # done
    sleep 10
    MASTER_ADDR=$(cat "$TARGET_FILE")

    echo "Starting worker node (RANK=${RANK}), connecting to ${MASTER_ADDR}:${MASTER_PORT}..."
    
    # 使用 --block 选项会让进程一直运行,直到 Ray 集群关闭
    ray start --address ${MASTER_ADDR}:${MASTER_PORT} --num-gpus 8 --block
    
    # 当 ray start --block 退出时,说明集群已关闭
    echo "Ray cluster shut down. Worker node exiting..."
    exit 0
fi



