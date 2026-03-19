"""
LifeRewardManager (Rule-Only) — verl 0.7.0 外挂 reward manager

设计思路：
  - 继承官方 RateLimitedRewardManager，利用其三层并发控制
  - 使用 reward_life.py 中的规则评分逻辑（MCC / PCC / Spearman / R² / AUC / Acc / Fmax / Mixed）
  - 评分方法与 OpenCompass biodata.py 评测完全对齐

加载方式（shell script 中配置）：
    reward_model.reward_loop_source=importlib
    reward_model.reward_loop_module_path="../src/reward/rule/life_reward_manager.py"
    reward_model.reward_loop_class_name=LifeRewardManager
    reward_model.max_concurrent=4096
    reward_model.timeout=600
"""

import importlib.util
import os
import sys

# ── 确保 utils 所在目录（src/）在 sys.path 上 ──
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# ── 显式加载同目录下的 reward_life.py，避免与其他同名文件冲突 ──
_RL_PATH = os.path.join(_THIS_DIR, "reward_life.py")
_rl_spec = importlib.util.spec_from_file_location("reward_life_rule", _RL_PATH)
_rl_mod = importlib.util.module_from_spec(_rl_spec)
_rl_spec.loader.exec_module(_rl_mod)

extract_thinking_content = _rl_mod.extract_thinking_content
if_format_correct = _rl_mod.if_format_correct
compute_rule_score = _rl_mod.compute_rule_score

from verl.experimental.reward_loop.reward_manager.limited import RateLimitedRewardManager  # noqa: E402


def _life_compute_score(
    data_source: str,
    solution_str: str,
    ground_truth,
    extra_info: dict,
) -> dict:
    """单条样本评分函数（纯规则，无 LLM Judge），符合 verl RateLimitedRewardManager 的 compute_score 签名。

    Args:
        data_source:  数据来源标识（兼容参数，此处不使用）。
        solution_str: 模型完整回复，含 <think>…</think> 标签。
        ground_truth: 参考答案（str / float / dict）。
        extra_info:   需包含 ``task_name``（任务名称或度量类型）。

    Returns:
        dict::

            {
                "score"      : float,        # 最终 reward
                "rule_score" : float | None,  # 规则分
            }
    """
    task_name = extra_info.get("task_name", "MCC")

    # Step 1: 提取 thinking / solution，检查格式
    thinking_str, solution_clean = extract_thinking_content(solution_str)
    if not if_format_correct(thinking_str, solution_clean, task_name):
        return {
            "score": -1.0,
            "rule_score": None,
        }

    # Strip residual </think> from solution
    if '</think>' in solution_clean:
        solution_clean = solution_clean.split('</think>')[-1]

    # Step 2: 纯规则评分
    try:
        rule_score = compute_rule_score(solution_clean, ground_truth, task_name)
        return {
            "score": rule_score,
            "rule_score": rule_score,
        }
    except Exception as e:
        print(
            f"[WARNING] _life_compute_score exception: "
            f"{type(e).__name__}: {e}"
        )
        return {
            "score": -1.0,
            "rule_score": None,
        }


class LifeRewardManager(RateLimitedRewardManager):
    """生命科学任务奖励管理器（纯规则 baseline，无 LLM Judge）。

    继承自官方 ``RateLimitedRewardManager``，获得三层并发控制。
    评分逻辑使用 ``reward_life.py`` 中的规则评分，覆盖 MCC / PCC /
    Spearman / R² / AUC / Acc / Fmax / Mixed 全部 8 类度量。
    """

    def __init__(
        self,
        config,
        tokenizer,
        compute_score=None,
        reward_router_address=None,
        reward_model_tokenizer=None,
    ):
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            compute_score=_life_compute_score,
            reward_router_address=reward_router_address,
            reward_model_tokenizer=reward_model_tokenizer,
        )
