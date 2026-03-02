# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 POLAR Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Main entry point for On-Policy Distillation (OPD) training.

OPD distills knowledge from a teacher model to a student model by:
1. Student generates trajectories (on-policy sampling)
2. Teacher computes log probabilities on trajectories
3. Reverse KL divergence is used as the training signal (in compute_advantage)
4. Student is updated to minimize KL divergence from teacher

This implementation uses RayOPDTrainer which extends RayPPOTrainer with:
- adv_estimator = "opd" (registered in core_algos.py)
- No critic (use_critic = False)
- No KL in reward (use_kl_in_reward = False)
- No KL loss (use_kl_loss = False)  
- KL computation happens in fit() before compute_advantage()
- Zero reward function (placeholder)
"""

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.ppo.reward import load_reward_manager
from verl.utils.device import is_cuda_available

# Import core_algos to register the "opd" advantage estimator
from . import core_algos  # noqa: F401
from .opd_ray_trainer import RayOPDTrainer


@hydra.main(config_path="config", config_name="opd_trainer", version_base=None)
def main(config):
    """Main entry point for OPD training with Hydra configuration."""
    run_opd(config)


def run_opd(config) -> None:
    """Initialize Ray cluster and run OPD training process."""
    if not ray.is_initialized():
        from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    try:
        if (
            is_cuda_available
            and config.global_profiler.tool == "nsys"
            and OmegaConf.select(config.global_profiler, "steps") is not None
            and len(OmegaConf.select(config.global_profiler, "steps")) > 0
        ):
            nsight_options = OmegaConf.to_container(
                config.global_profiler.global_tool_config.nsys.controller_nsight_options
            )
            runner = TaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
        else:
            runner = TaskRunner.remote()
        ray.get(runner.run.remote(config))
    finally:
        if ray.is_initialized():
            ray.shutdown()


@ray.remote(num_cpus=1)
class TaskRunner:
    """Ray remote class for executing OPD training tasks."""

    def run(self, config):
        """Execute the OPD training workflow using RayPPOTrainer."""
        from pprint import pprint
        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        # Download checkpoint from remote storage if needed
        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        # Instantiate tokenizer
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        # Import OPD trainer and register "opd" estimator
        from . import core_algos  # noqa: F401
        from .opd_ray_trainer import RayOPDTrainer

        # Define worker classes based on strategy
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            from verl.workers.fsdp_workers import ActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup
        elif config.actor_rollout_ref.actor.strategy == "megatron":
            from verl.workers.megatron_workers import ActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup
        else:
            raise NotImplementedError(
                f"Unsupported actor strategy: {config.actor_rollout_ref.actor.strategy}"
            )

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        }

        # Setup resource pools
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
        }

        # Add reference (teacher) model - required for OPD
        # The teacher model provides log probabilities for KL computation
        role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
        mapping[Role.RefPolicy] = global_pool_id

        # Add reward model worker if enabled (optional for OPD)
        if config.reward_model.enable:
            if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError(
                    f"Unsupported reward model strategy: {config.reward_model.strategy}"
                )
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # Load reward functions
        # For OPD, we use ZeroRewardManager as placeholder
        from .core_algos import ZeroRewardManager

        # Use zero reward by default, or load from config if specified
        if config.get("reward_model", {}).get("reward_fn_path"):
            reward_fn = load_reward_manager(
                config,
                tokenizer,
                num_examine=0,
                max_resp_len=config.data.max_response_length,
                overlong_buffer_cfg=config.reward_model.get("overlong_buffer", {}),
            )
            val_reward_fn = load_reward_manager(
                config,
                tokenizer,
                num_examine=1,
                max_resp_len=config.data.max_response_length,
                overlong_buffer_cfg=config.reward_model.get("overlong_buffer", {}),
            )
        else:
            # Use ZeroRewardManager for OPD (compatible with AbstractRewardManager interface)
            reward_fn = ZeroRewardManager(tokenizer=tokenizer, num_examine=0)
            val_reward_fn = ZeroRewardManager(tokenizer=tokenizer, num_examine=1)

        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, 
            mapping=mapping
        )

        # Create OPD trainer
        # OPD-specific logic is handled in RayOPDTrainer.fit():
        # 1. adv_estimator = "opd" in config
        # 2. use_critic = False (no critic needed)
        # 3. use_kl_in_reward = False (KL computed directly in fit())
        # 4. use_kl_loss = False (no separate KL loss term)
        # 5. token_level_rewards = ref_log_prob - old_log_probs (set before compute_advantage)
        trainer = RayOPDTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )

        # Initialize workers and start training
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
