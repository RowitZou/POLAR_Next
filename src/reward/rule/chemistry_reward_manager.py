"""
ChemistryRewardManager (Rule-Only) — verl 0.7.0 外挂 reward manager（无需修改 verl 源码）

设计思路：
  - 继承官方 RateLimitedRewardManager，利用其三层并发控制
  - 仅使用 reward_chemistry.py 中的规则评分逻辑，不含 LLM Judge
  - 作为 baseline 实验的 reward 函数

加载方式（shell script 中配置）：
    reward_model.reward_loop_source=importlib
    reward_model.reward_loop_module_path="../src/reward/rule/chemistry_reward_manager.py"
    reward_model.reward_loop_class_name=ChemistryRewardManager
    reward_model.max_concurrent=4096
    reward_model.timeout=600
"""

import importlib.util
import os
import sys

# ── 确保 utils 所在目录（src/）在 sys.path 上，供 reward_chemistry.py 内部导入 ──
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# ── 显式加载同目录下的 reward_chemistry.py，避免与 mix_env 下同名文件冲突 ────
_RC_PATH = os.path.join(_THIS_DIR, "reward_chemistry.py")
_rc_spec = importlib.util.spec_from_file_location("reward_chemistry_rule", _RC_PATH)
_rc_mod = importlib.util.module_from_spec(_rc_spec)
_rc_spec.loader.exec_module(_rc_mod)

extract_thinking_content = _rc_mod.extract_thinking_content
if_format_correct = _rc_mod.if_format_correct
compute_rule_score = _rc_mod.compute_rule_score

from verl.experimental.reward_loop.reward_manager.limited import RateLimitedRewardManager  # noqa: E402


def _chemistry_compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
) -> dict:
    """单条样本评分函数（纯规则，无 LLM Judge），符合 verl RateLimitedRewardManager 的 compute_score 签名。

    Args:
        data_source:  数据来源标识（兼容参数，此处不使用）。
        solution_str: 模型完整回复，含 <think>…</think> 标签。
        ground_truth: 参考答案。
        extra_info:   需包含 ``task_name``（任务类型）。

    Returns:
        dict::

            {
                "score"        : float,        # 最终 reward（= rule_score）
                "rule_score"   : float | None,  # 规则分
            }
    """
    task_name = extra_info.get("task_name", "RS")

    # Step 1: 提取 thinking / solution，检查格式
    thinking_str, solution_clean = extract_thinking_content(solution_str)
    if not if_format_correct(thinking_str, solution_clean, task_name):
        return {
            "score": -1.0,
            "rule_score": None,
        }

    # Step 2: 纯规则评分，无 LLM Judge
    try:
        rule_score = compute_rule_score(solution_clean, ground_truth, task_name)
        return {
            "score": rule_score,
            "rule_score": rule_score,
        }
    except Exception as e:
        print(
            f"[WARNING] _chemistry_compute_score exception: "
            f"{type(e).__name__}: {e}"
        )
        return {
            "score": -1.0,
            "rule_score": None,
        }


class ChemistryRewardManager(RateLimitedRewardManager):
    """化学任务奖励管理器（纯规则 baseline，无 LLM Judge）。

    继承自官方 ``RateLimitedRewardManager``，获得三层并发控制。
    评分逻辑仅使用 ``reward_chemistry.py`` 中的规则评分，
    不调用任何 LLM Judge，适合作为 baseline 实验。
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
            compute_score=_chemistry_compute_score,
            reward_router_address=reward_router_address,
            reward_model_tokenizer=reward_model_tokenizer,
        )
