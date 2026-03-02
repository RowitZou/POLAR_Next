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
Core algorithms for On-Policy Distillation (OPD).

This module registers the "opd" advantage estimator that can be used with
the standard verl compute_advantage function. Unlike GRPO which requires
multiple samples per prompt (n>=2) for group-wise normalization, OPD works
with single samples (n=1) by using the teacher log probability as direct signal.

Key difference from GRPO:
- GRPO: Advantage = reward - mean(reward in group), requires n >= 2 samples per prompt
- OPD: Advantage = log π_teacher - log π_student = -KL(student || teacher), works with n = 1

The KL penalty computation is copied from verl/trainer/ppo/core_algos.py
for easy customization without modifying the original codebase.

IMPORTANT: For OPD to work correctly, token_level_rewards should be set to:
    token_level_rewards = ref_log_prob - old_log_probs
This is done in the RayOPDTrainer.fit() method before calling compute_advantage().
"""

from typing import Optional, Tuple

import numpy as np
import torch

from verl.trainer.ppo.core_algos import register_adv_est
from verl.trainer.config import AlgoConfig
import verl.utils.torch_functional as verl_F


# ============================================================================
# KL penalty functions - copied from verl/trainer/ppo/core_algos.py
# ============================================================================

def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty_type: str) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob. Optionally using straight through to bind k2 on other
    kl penalty compute method for unbiased KL gradient estimation.
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob: Log probabilities from current policy
        ref_logprob: Log probabilities from reference policy
        kl_penalty_type: Type of KL penalty ("kl", "abs", "mse", "low_var_kl", "full", etc.)

    Returns:
        kl_estimate: Token-level KL estimates
    """
    forward_score = kl_penalty_forward(logprob, ref_logprob, kl_penalty_type)
    if not kl_penalty_type.endswith("+") or kl_penalty_type in ("mse", "k2"):
        return forward_score

    """
    The expectation of k1 and k3 estimator is the expected value of KL, but the expected gradient of k1 and k3
    estimator is not the expected gradient of KL. On the other hand k2 estimator gives right gradient estimator, 
    so we use a straight through trick here if the kl_penalty method ends with '+', e.g., k3+. 
    """
    backward_score = 0.5 * (logprob - ref_logprob).square()

    return backward_score - backward_score.detach() + forward_score.detach()


def kl_penalty_forward(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty_type: str) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob: Log probabilities from current policy
        ref_logprob: Log probabilities from reference policy
        kl_penalty_type: Type of KL penalty

    Returns:
        kl_estimate: Token-level KL estimates
    """
    # Standard KL(current || ref) = E[log p - log ref] 
    # k1: Simple difference
    if kl_penalty_type in ("kl", "k1"):
        return logprob - ref_logprob

    # Absolute difference
    if kl_penalty_type == "abs":
        return (logprob - ref_logprob).abs()

    # k2: MSE (squared difference), used for gradient correction
    if kl_penalty_type in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    # k3: Low variance KL estimator from Schulman
    # J. Schulman. Approximating kl divergence, 2020.
    # URL http://joschu.net/blog/kl-approx.html
    if kl_penalty_type in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        # For numerical stability
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty_type == "full":
        # Full KL requires logits for entire vocabulary - not implemented
        raise NotImplementedError("Full KL divergence requires vocabulary logits")

    raise NotImplementedError(f"Unknown kl_penalty_type: {kl_penalty_type}")


# ============================================================================
# OPD Advantage Estimator - registered to work with verl's compute_advantage
# ============================================================================

@register_adv_est("opd")
def compute_opd_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: Optional[np.ndarray] = None,
    config: Optional[AlgoConfig] = None,
    # OPD-specific parameters (passed directly, not through compute_advantage)
    old_log_probs: Optional[torch.Tensor] = None,
    ref_log_prob: Optional[torch.Tensor] = None,
    kl_penalty_type: str = "kl",
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Compute advantage estimates for On-Policy Distillation (OPD).
    
    Unlike GRPO which computes advantage as reward relative to group mean,
    OPD uses the reverse KL divergence as the advantage signal:
    
        Advantage = -KL(student || teacher) = log π_teacher - log π_student
    
    This function computes KL using the kl_penalty function, allowing easy
    customization of KL computation methods (k1, k2, k3, etc.).
    
    The advantage encourages the student to match the teacher's distribution:
    - When teacher assigns higher prob than student -> positive advantage -> encourage
    - When student assigns higher prob than teacher -> negative advantage -> discourage
    
    This works with ANY number of samples (including n=1) because we don't need
    group-wise comparison like GRPO.
    
    Args:
        token_level_rewards: External rewards (usually zeros for OPD),
                            shape (batch_size, response_length)
        response_mask: Mask for valid response tokens, shape (batch_size, response_length)
        index: Optional group index (not used in OPD, included for API compatibility)
        config: Algorithm configuration containing OPD-specific settings
        old_log_probs: Log probabilities from student policy, shape (batch_size, response_length)
        ref_log_prob: Log probabilities from teacher policy, shape (batch_size, response_length)
        kl_penalty_type: Type of KL penalty to use ("kl", "k1", "k2", "k3", "low_var_kl", etc.)
        **kwargs: Additional arguments for API compatibility
    
    Returns:
        advantages: Advantage estimates, shape (batch_size, response_length)
        returns: Returns (same as advantages for OPD), shape (batch_size, response_length)
        metrics: Dictionary containing KL statistics and other metrics
    """
    metrics = {}
    
    # ========================================================================
    # Step 1: Compute KL divergence using kl_penalty function
    # ========================================================================
    if old_log_probs is not None and ref_log_prob is not None:
        # Compute KL(student || teacher) using the configurable kl_penalty function
        # kl_penalty returns: log π_student - log π_teacher (for "kl" type)
        # We negate it to get: log π_teacher - log π_student = -KL
        kl_divergence = kl_penalty(old_log_probs, ref_log_prob, kl_penalty_type)
        
        # For OPD, advantage = -KL = log π_teacher - log π_student
        # When using "kl" type: kl_penalty returns (old - ref), so we negate
        # This encourages student to match teacher
        kl_based_reward = -kl_divergence
        
        # Apply response mask
        kl_based_reward = kl_based_reward * response_mask
        
        # Log KL statistics
        valid_tokens = response_mask.sum()
        if valid_tokens > 0:
            # Mean KL per token (KL = student - teacher = -kl_based_reward)
            mean_kl = kl_divergence.sum() / valid_tokens
            metrics["opd/kl_mean"] = mean_kl.item()
            
            # Per-sequence KL statistics
            seq_lengths = response_mask.sum(dim=-1).clamp(min=1)
            seq_kl = (kl_divergence * response_mask).sum(dim=-1) / seq_lengths
            metrics["opd/seq_kl_mean"] = seq_kl.mean().item()
            metrics["opd/seq_kl_std"] = seq_kl.std().item() if len(seq_kl) > 1 else 0.0
            metrics["opd/seq_kl_max"] = seq_kl.max().item()
            metrics["opd/seq_kl_min"] = seq_kl.min().item()
            
            # Log advantage statistics
            metrics["opd/advantage_mean"] = (kl_based_reward.sum() / valid_tokens).item()
        
        advantages = kl_based_reward
    else:
        # Fallback: use token_level_rewards directly (for compatibility)
        # This path is used when called through standard compute_advantage()
        advantages = token_level_rewards.clone() * response_mask
    
    # ========================================================================
    # Step 2: Apply OPD-specific transformations
    # ========================================================================
    if config is not None:
        # Option 1: Token-level (default) - use advantages directly
        opd_estimator = config.get("opd_estimator", "token_level")
        
        if opd_estimator == "sequence_level":
            # Option 2: Sequence-level - broadcast sequence mean to all tokens
            seq_lengths = response_mask.sum(dim=-1, keepdim=True).clamp(min=1)
            seq_advantages = (advantages.sum(dim=-1, keepdim=True) / seq_lengths)
            advantages = seq_advantages.expand_as(advantages) * response_mask
            metrics["opd/estimator"] = "sequence_level"
        else:
            metrics["opd/estimator"] = "token_level"
        
        # Optional: Normalize advantages
        normalize_adv = config.get("normalize_advantages", False)
        if normalize_adv:
            advantages = verl_F.masked_whiten(advantages, response_mask)
            metrics["opd/normalized"] = True
        
        # Optional: Clip advantages
        clip_adv = config.get("clip_advantage", None)
        if clip_adv is not None:
            advantages = torch.clamp(advantages, -clip_adv, clip_adv)
            metrics["opd/clip_advantage"] = clip_adv
    
    # ========================================================================
    # Step 3: Compute returns (same as advantages for OPD, no value function)
    # ========================================================================
    returns = advantages.clone()
    
    return advantages, returns, metrics


# ============================================================================
# Zero reward function/manager for OPD (placeholder)
# ============================================================================

def zero_reward_function(data, return_dict=False, **kwargs):
    """
    Zero reward function for OPD.
    
    OPD doesn't use external rewards - the KL divergence from teacher serves
    as the training signal. This function returns all zeros as placeholder.
    
    Args:
        data: DataProto containing batch information
        return_dict: If True, return a dict with "reward_tensor" key; 
                    If False, return tuple (reward_tensor, info_dict)
        **kwargs: Additional arguments (ignored)
    
    Returns:
        If return_dict: {"reward_tensor": tensor, "reward_extra_info": {}}
        Else: Tuple of (reward_tensor, {})
    """
    responses = data.batch["responses"]
    batch_size, response_length = responses.shape
    
    # Return zero rewards
    reward_tensor = torch.zeros(
        batch_size, response_length,
        dtype=torch.float32,
        device=responses.device
    )
    
    if return_dict:
        return {"reward_tensor": reward_tensor, "reward_extra_info": {}}
    return reward_tensor, {}


class ZeroRewardManager:
    """
    Zero reward manager for OPD.
    
    This class is compatible with verl's AbstractRewardManager interface.
    It returns zero rewards for all inputs, as OPD uses KL divergence from
    the teacher model as the training signal instead of external rewards.
    
    Usage:
        reward_fn = ZeroRewardManager(tokenizer)
        result = reward_fn(data, return_dict=True)
    """
    
    def __init__(
        self,
        tokenizer=None,
        num_examine: int = 0,
        compute_score=None,
        reward_fn_key: str = "data_source",
        **kwargs,
    ):
        """
        Initialize ZeroRewardManager.
        
        Args:
            tokenizer: Tokenizer (not used, for API compatibility)
            num_examine: Number of samples to examine (not used)
            compute_score: Custom compute score function (not used)
            reward_fn_key: Key for reward function (not used)
            **kwargs: Additional arguments (ignored)
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
    
    def __call__(
        self,
        data,
        return_dict: bool = False,
    ):
        """
        Compute zero rewards for the given data.
        
        Args:
            data: DataProto containing batch information
            return_dict: If True, return dict; if False, return tensor
        
        Returns:
            If return_dict: {"reward_tensor": tensor, "reward_extra_info": {}}
            Else: reward_tensor
        """
        responses = data.batch["responses"]
        batch_size, response_length = responses.shape
        
        reward_tensor = torch.zeros(
            batch_size, response_length,
            dtype=torch.float32,
            device=responses.device
        )
        
        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": {}}
        return reward_tensor

