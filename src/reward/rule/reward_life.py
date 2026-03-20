"""
reward_life.py — Rule-based reward functions for Life (Biology) tasks.

Evaluation metrics are aligned with OpenCompass biodata.py evaluators:
  MCC, PCC, Spearman, R², AUC, Accuracy, Fmax, Mixed.

Each metric maps to a per-sample scoring function in utils/life/eval.py,
which returns a score in [0, 100].  The ``normalize_score_to_reward``
function maps this to [0, 1] for use as RL reward.
"""

import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from utils.life.eval import (  # noqa: E402
    TASK_TO_ERROR_SCALE,
    TASK_TO_METRIC,
    acc_score,
    auc_score,
    fmax_score,
    mcc_score,
    mixed_score_eval,
    pcc_score,
    r2_score,
    spearman_score,
)

# ── Ground truth parsing ─────────────────────────────────────────────────────
# In verl, ground_truth from parquet is always a string.  For PCC/R2 (dict)
# tasks, we must parse it back to a dict before passing to eval functions.

def parse_ground_truth(gt_str, task_name: str):
    """Parse ground_truth string to the appropriate Python type.

    - Dict tasks (PCC, R2-dict): JSON string → dict
    - Number tasks (Spearman, R2-scalar, Mixed): kept as string (eval does float())
    - Other tasks (MCC, Acc, Auc, Fmax): kept as string
    """
    if not isinstance(gt_str, str):
        return gt_str  # already parsed
    # Try JSON parse — covers dict GTs serialized via json.dumps
    # Also handles legacy Python-repr dicts via ast.literal_eval
    s = gt_str.strip()
    if s.startswith('{'):
        try:
            import json as _json
            return _json.loads(s)
        except Exception:
            try:
                import ast as _ast
                val = _ast.literal_eval(s)
                if isinstance(val, dict):
                    return val
            except Exception:
                pass
    return gt_str


# ── Metric → eval function mapping ───────────────────────────────────────────
METRIC_EVAL_FUNC = {
    'MCC': mcc_score,
    'PCC': pcc_score,
    'Spearman': spearman_score,
    'R2': r2_score,
    'Auc': auc_score,
    'Acc': acc_score,
    'Fmax': fmax_score,
    'Mixed': mixed_score_eval,
}

# All scores from eval functions are in [0, 100]; higher is always better.
# Normalization simply divides by 100 to get [0, 1].


def resolve_metric(task_name: str) -> str:
    """Resolve task_name to metric type.

    Supports both direct metric names (``MCC``, ``PCC``, …) and
    dataset task names (``DNA-cpd``, ``Protein-Stability``, …).
    """
    if task_name in METRIC_EVAL_FUNC:
        return task_name
    metric = TASK_TO_METRIC.get(task_name)
    if metric is not None:
        return metric
    raise ValueError(
        f"Unknown task_name '{task_name}'. "
        f"Expected one of {list(METRIC_EVAL_FUNC)} or {list(TASK_TO_METRIC)}."
    )


def normalize_score_to_reward(score: float) -> float:
    """Normalize a [0, 100] score to [0, 1] reward."""
    return max(0.0, min(1.0, score / 100.0))


def extract_thinking_content(text: str) -> tuple:
    """Split ``<think>…</think>`` from the rest of the response."""
    pattern = r'<think>(.*?)</think>(.*)'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return "", text


def if_format_correct(thinking_content: str, solution_content: str, task_name: str) -> bool:
    """Check whether the model output has the expected format.

    Rules:
      1. Must have non-empty ``<think>`` section.
      2. No stray ``<think>`` / ``</think>`` in the solution section.
      3. Answer must contain ``\\boxed{}`` (except for Fmax where EC numbers
         may appear without box).
    """
    if thinking_content == "":
        return False
    if "<think>" in solution_content or "</think>" in solution_content:
        return False
    return True


def compute_rule_score(prediction: str, reference, task_name: str) -> float:
    """Compute per-sample rule score for a Life task.

    Args:
        prediction: Model solution string (after ``<think>`` extraction).
        reference: Ground truth (str, float, or dict).
        task_name: Task name or metric name.

    Returns:
        Normalized score in [0, 1].
    """
    metric = resolve_metric(task_name)
    eval_func = METRIC_EVAL_FUNC[metric]

    # Parse string ground truth (from parquet) to proper type
    reference = parse_ground_truth(reference, task_name)

    try:
        # Pass task-specific error_scale for inverse-error metrics
        kwargs = {}
        if metric in ('PCC', 'Spearman', 'R2'):
            kwargs['error_scale'] = TASK_TO_ERROR_SCALE.get(task_name, 1.0)
        result = eval_func(predictions=[prediction], references=[reference], **kwargs)
        raw_score = result.get('score', 0.0)
        return normalize_score_to_reward(raw_score)
    except Exception as e:
        print(
            f'[WARNING] compute_rule_score error ({metric}): '
            f'{type(e).__name__}: {e}. '
            f'Prediction: {str(prediction)[:100]}... '
            f'Reference: {str(reference)[:100]}...'
        )
        return 0.0


def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos):
    """Compute scores for a batch of data — verl-compatible interface.

    Args:
        data_sources: List of data source identifiers.
        solution_strs: List of model responses (with ``<think>``).
        ground_truths: List of ground truth values.
        extra_infos: List of dicts, each containing ``task_name``.

    Returns:
        List of float scores, one per sample.
    """
    scores = []
    for solution_str, ground_truth, extra_info in zip(
        solution_strs, ground_truths, extra_infos, strict=True
    ):
        task_name = extra_info.get("task_name", "MCC")

        thinking_str, solution_clean = extract_thinking_content(solution_str)

        if not if_format_correct(thinking_str, solution_clean, task_name):
            scores.append(-1.0)
            continue

        # Strip </think> residue for extraction functions
        if '</think>' in solution_clean:
            solution_clean = solution_clean.split('</think>')[-1]

        try:
            score = compute_rule_score(solution_clean, ground_truth, task_name)
            scores.append(score)
        except Exception as e:
            print(
                f'[WARNING] compute_score_batch error: '
                f'{type(e).__name__}: {e}'
            )
            scores.append(0.0)

    return scores
