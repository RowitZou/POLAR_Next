# On-Policy Distillation (OPD)

## Overview

On-Policy Distillation (OPD) is a knowledge distillation method that trains a student model to match a teacher model's output distribution using on-policy samples. Unlike traditional offline distillation, OPD generates samples from the student model and uses the teacher's log probabilities to compute a training signal.

## Key Concepts

### Reverse KL Divergence

The core training signal in OPD is the reverse KL divergence:

$$D_{KL}(\pi_{student} \| \pi_{teacher}) = \mathbb{E}_{x \sim \pi_{student}}[\log \pi_{student}(x) - \log \pi_{teacher}(x)]$$

By minimizing this divergence, the student learns to match the teacher's distribution on samples it generates itself.

### Key Difference from GRPO

| Feature | OPD | GRPO |
|---------|-----|------|
| **Samples per prompt** | n=1 (works with single sample) | n≥2 (requires multiple samples) |
| **Advantage signal** | KL divergence from teacher | Reward relative to group mean |
| **Baseline** | Teacher log probability | Group average reward |
| **Critic** | Not needed | Not needed |

### Algorithm Flow

```
1. Student generates response (on-policy sampling)
2. Compute student log probabilities (old_log_probs)
3. Compute teacher log probabilities (ref_log_prob)  
4. token_level_rewards = ref_log_prob - old_log_probs
5. compute_advantage() with "opd" estimator
6. Update student using PPO-style policy gradient
```

## Usage

### Basic Training Script

```bash
python -m recipe.opd.main_opd \
    actor_rollout_ref.model.path=/path/to/student/model \
    +actor_rollout_ref.ref.model.path=/path/to/teacher/model \
    data.train_files=/path/to/train/data.parquet \
    data.val_files=/path/to/val/data.parquet \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1
```

### Key Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `actor_rollout_ref.model.path` | Path to student model | Required |
| `actor_rollout_ref.ref.model.path` | Path to teacher model | Required |
| `algorithm.adv_estimator` | Advantage estimator | `opd` |
| `algorithm.opd_estimator` | OPD variant | `token_level` |
| `algorithm.normalize_advantages` | Normalize advantages | `False` |
| `algorithm.clip_advantage` | Clip advantages | `null` |
| `actor_rollout_ref.rollout.n` | Samples per prompt | `1` |
| `actor_rollout_ref.actor.use_kl_loss` | Use KL loss | `False` |
| `critic.enable` | Enable critic | `False` |

### OPD Estimator Variants

| Estimator | Description |
|-----------|-------------|
| `token_level` | Advantage at each token position (recommended) |
| `sequence_level` | Same advantage for all tokens in a sequence |

## Architecture

```
verl/recipe/opd/
├── __init__.py           # Module initialization
├── config/
│   └── opd_trainer.yaml  # Default configuration
├── core_algos.py         # OPD algorithms (KL computation, advantage estimator)
├── main_opd.py           # Entry point
├── opd_ray_trainer.py    # RayOPDTrainer (extends RayPPOTrainer)
├── run_opd.sh            # Example training script
└── README.md             # This file
```

## Implementation Details

### core_algos.py

Contains the core algorithmic components:
- `kl_penalty()`: Computes KL divergence (copied from verl for customization)
- `kl_penalty_forward()`: Forward KL computation variants
- `compute_opd_outcome_advantage()`: Registered "opd" advantage estimator
- `zero_reward_function()`: Placeholder reward function

### opd_ray_trainer.py

Extends `RayPPOTrainer` with OPD-specific logic:
- Overrides `fit()` to set `token_level_rewards = ref_log_prob - old_log_probs`
- All other training logic inherited from parent class

### main_opd.py

Entry point that:
- Initializes Ray cluster
- Sets up workers (actor, reference/teacher)
- Creates and runs RayOPDTrainer

## Key Design Choices

1. **Minimal modification**: Extends RayPPOTrainer with one modification in `fit()`
2. **Registered estimator**: "opd" estimator registered via `@register_adv_est`
3. **Copied KL functions**: KL computation copied for easy customization
4. **No critic**: Critic disabled since advantage comes from teacher
5. **Single sample**: Works with n=1, unlike GRPO

## Best Practices

1. **Teacher Model Size**: Use a larger/more capable teacher model for effective distillation
2. **Learning Rate**: Start with a smaller learning rate (1e-6 to 2e-6) for stable training
3. **Clipping**: Use `clip_ratio=0.2` for stability
4. **Monitoring**: Watch `opd/kl_mean` metric to track distillation progress
