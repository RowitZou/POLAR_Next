set -x

# if [ "$#" -lt 2 ]; then
#     echo "Usage: run_deepseek_6b7.sh <nproc_per_node> <save_path> [other_configs...]"
#     exit 1
# fi

# Shift the arguments so $@ refers to the rest
# shift 2

export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HYDRA_FULL_ERROR=1

nodes=1
train_batch_size=256 # 512
data_name=openthoughts3-400k
model_name=Qwen3-8B-base
train_epoch=1 

# lr=1e-6

# Model paths
model_path=/mnt/shared-storage-user/ailab-hs/chenjiayi/models/opensource/Qwen3-8B-Base

# Data paths
train_data_path=/mnt/shared-storage-user/ailab-hs/chenjiayi/data/OpenThoughts3-1.2M/process_data/train_400k.parquet
test_data_path=/mnt/shared-storage-user/ailab-hs/chenjiayi/data/OpenThoughts3-1.2M/process_data/train_50k.parquet # nouse

# Experiment name
name="sft_model_${model_name}_data_${data_name}"
output_dir="../outputs/${name}"

# Create output directory if it doesn't exist
mkdir -p $output_dir

mkdir -p "$output_dir/logs"

timestamp=$(date +"%Y%m%d_%H%M%S")
logfile="${output_dir}/logs/train_${timestamp}_rank${RANK:-0}.log"

echo "Logging to $logfile"
exec > >(tee -a "$logfile") 2>&1

torchrun --standalone --nnodes=$nodes --nproc_per_node=8 \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$train_data_path \
    data.val_files=$test_data_path \
    data.prompt_key=question \
    data.response_key=answer \
    data.micro_batch_size_per_gpu=1 \
    data.train_batch_size=$train_batch_size \
    data.max_length=34816 \
    data.truncation='error' \
    \
    model.partial_pretrain=$model_path \
    model.fsdp_config.model_dtype="bf16" \
    +model.gradient_checkpointing=True \
    +model.use_flash_attention_2=True \
    \
    trainer.default_local_dir=$output_dir \
    trainer.project_name=openthoughts-sft \
    trainer.experiment_name="$name" \
    trainer.total_epochs=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.logger='["console","wandb"]' $@