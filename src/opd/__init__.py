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
On-Policy Distillation (OPD) Recipe.

This recipe implements knowledge distillation from a teacher model to a student model
using on-policy sampling with reverse KL divergence as the training signal.

Usage:
    python -m recipe.opd.main_opd \\
        actor_rollout_ref.model.path=/path/to/student \\
        actor_rollout_ref.ref.model.path=/path/to/teacher \\
        ...

Key design choices:
- Extends RayPPOTrainer with RayOPDTrainer for OPD-specific fit() logic
- Registers "opd" advantage estimator via core_algos.py
- KL computation copied from verl for easy customization
- Works with single sample (n=1), unlike GRPO which requires n>=2
- No critic, no KL in reward, no KL loss - KL is used directly as advantage
"""

from .core_algos import (
    compute_opd_outcome_advantage,
    kl_penalty,
    kl_penalty_forward,
    zero_reward_function,
)
from .opd_ray_trainer import RayOPDTrainer
from .main_opd import main, run_opd

__all__ = [
    "main",
    "run_opd",
    "RayOPDTrainer",
    "compute_opd_outcome_advantage",
    "kl_penalty",
    "kl_penalty_forward",
    "zero_reward_function",
]
