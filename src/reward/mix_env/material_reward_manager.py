"""
MaterialRewardManager — verl 0.7.0 外挂 reward manager（无需修改 verl 源码）

设计思路：
  - 继承官方 RateLimitedRewardManager，利用其三层并发控制
    (max_concurrent / max_rpm / max_tpm / timeout)
  - 复用 reward_material.py 中的 LLM Judge 评分逻辑
  - 分类任务 (matbench_expt_is_metal / matbench_glass):
      LLM 直接对比模型输出和 ground truth → reward 1.0 / 0.0
  - 回归任务 (matbench_expt_gap / matbench_steels):
      LLM 从模型输出中提取数值 → MAE → reward

加载方式（shell script 中配置）：
    reward_model.reward_loop_source=importlib
    reward_model.reward_loop_module_path="../src/reward/mix_env/material_reward_manager.py"
    reward_model.reward_loop_class_name=MaterialRewardManager
    reward_model.max_concurrent=4096
    reward_model.timeout=600
"""

import itertools
import json
import os
import re
import sys

import httpx
from openai import OpenAI  # 同步客户端

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LLM Judge 配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OPENAI_API_BASES = [
    "http://10.102.243.60:8000/v1",
    "http://10.102.209.49:8000/v1",
]
OPENAI_API_KEY = ""
MODEL_NAME = "/mnt/shared-storage-user/ailab-hs/yangyuming/models/models--openai--gpt-oss-120b/snapshots/eabf0c518da7584a2e7dab4ab272709785a72126"
MAX_TOKENS = 16384
TIMEOUT_SECONDS = 600

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in [_THIS_DIR, os.path.dirname(_THIS_DIR), os.path.dirname(os.path.dirname(_THIS_DIR))]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from reward_material import (  # noqa: E402
    CLASSIFICATION_JUDGE_TEMPLATE,
    REGRESSION_EXTRACT_TEMPLATE,
    CLASSIFICATION_TASKS,
    REGRESSION_TASKS,
    MAE_SCALE,
    extract_thinking_content,
    if_format_correct,
    mae_to_reward,
    _parse_json_from_response,
)

from verl.experimental.reward_loop.reward_manager.limited import RateLimitedRewardManager  # noqa: E402


# ── 同步 OpenAI 客户端（每服务器一个）──────────────────────────────────────────
_MAX_POOL_PER_SERVER = 1024


def _build_sync_client(base_url: str) -> OpenAI:
    hc = httpx.Client(
        trust_env=False,
        limits=httpx.Limits(
            max_connections=_MAX_POOL_PER_SERVER + 32,
            max_keepalive_connections=_MAX_POOL_PER_SERVER,
            keepalive_expiry=30.0,
        ),
    )
    return OpenAI(base_url=base_url, api_key=OPENAI_API_KEY, http_client=hc)


_sync_clients: list[OpenAI] = [_build_sync_client(url) for url in OPENAI_API_BASES]
_server_selector = itertools.cycle(range(len(_sync_clients)))


def _llm_call_sync(prompt: str, client: OpenAI) -> str:
    """Make a single synchronous LLM call."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=MAX_TOKENS,
        timeout=TIMEOUT_SECONDS,
        reasoning_effort="high",
    )
    return response.choices[0].message.content.strip()


def _judge_classification_sync(
    question: str, prediction: str, ground_truth: str, client: OpenAI,
) -> dict:
    """Synchronous classification judge: reward 1.0 (correct) or 0.0 (incorrect)."""
    prompt = CLASSIFICATION_JUDGE_TEMPLATE.format(
        question=question,
        ground_truth=ground_truth,
        prediction=prediction,
    )
    try:
        content = _llm_call_sync(prompt, client)
        parsed = _parse_json_from_response(content)

        if parsed is not None:
            result = bool(parsed.get("result", False))
            explain = parsed.get("explain", "")
            score = 1.0 if result else 0.0
            return {"score": score, "judge_explain": explain,
                    "predicted_value": None, "gold_value": None, "mae": None}

        # Fallback: try to parse from raw text
        content_lower = content.lower()
        if '"result": true' in content_lower or '"result":true' in content_lower:
            return {"score": 1.0, "judge_explain": f"Parsed from raw: {content[:200]}",
                    "predicted_value": None, "gold_value": None, "mae": None}
        elif '"result": false' in content_lower or '"result":false' in content_lower:
            return {"score": 0.0, "judge_explain": f"Parsed from raw: {content[:200]}",
                    "predicted_value": None, "gold_value": None, "mae": None}
        else:
            print(f"[WARNING] Classification judge parse failed: {content[:200]}")
            return {"score": 0.0, "judge_explain": f"Parse failed: {content[:200]}",
                    "predicted_value": None, "gold_value": None, "mae": None}
    except Exception as e:
        print(f"[WARNING] Classification judge error: {type(e).__name__}: {e}")
        return {"score": 0.0, "judge_explain": f"Exception: {type(e).__name__}: {e}",
                "predicted_value": None, "gold_value": None, "mae": None}


def _judge_regression_sync(
    question: str, prediction: str, ground_truth: str,
    task_type: str, client: OpenAI,
) -> dict:
    """Synchronous regression judge: extract float values, compute MAE-based reward."""
    prompt = REGRESSION_EXTRACT_TEMPLATE.format(
        question=question,
        ground_truth=ground_truth,
        prediction=prediction,
    )
    try:
        content = _llm_call_sync(prompt, client)
        parsed = _parse_json_from_response(content)

        if parsed is None:
            numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', content)
            if len(numbers) >= 2:
                parsed = {"predicted_value": float(numbers[0]), "gold_value": float(numbers[1]),
                          "explain": f"Fallback parsed from raw: {content[:200]}"}
            else:
                print(f"[WARNING] Regression judge parse failed: {content[:200]}")
                return {"score": 0.0, "predicted_value": None, "gold_value": None,
                        "mae": None, "judge_explain": f"Parse failed: {content[:200]}"}

        pred_val = parsed.get("predicted_value")
        gold_val = parsed.get("gold_value")
        explain = parsed.get("explain", "")

        if pred_val is None or gold_val is None:
            return {"score": 0.0, "predicted_value": pred_val, "gold_value": gold_val,
                    "mae": None, "judge_explain": f"Null value. {explain}"}

        pred_val = float(pred_val)
        gold_val = float(gold_val)
        mae = abs(pred_val - gold_val)
        scale = MAE_SCALE.get(task_type, 1.0)
        reward = mae_to_reward(mae, mae_scale=scale)

        return {"score": reward, "predicted_value": pred_val, "gold_value": gold_val,
                "mae": mae, "judge_explain": f"{explain} | pred={pred_val}, gold={gold_val}, mae={mae}, reward={reward}"}

    except Exception as e:
        print(f"[WARNING] Regression judge error: {type(e).__name__}: {e}")
        return {"score": 0.0, "predicted_value": None, "gold_value": None,
                "mae": None, "judge_explain": f"Exception: {type(e).__name__}: {e}"}


def _material_compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
) -> dict:
    """单条样本评分函数，符合 verl RateLimitedRewardManager 的 compute_score 签名。

    同步函数，由 RateLimitedRewardManager 通过 run_in_executor 在线程池中调用。

    Args:
        data_source:  数据来源标识（兼容参数，此处不使用）。
        solution_str: 模型完整回复，含 <think>…</think> 标签。
        ground_truth: 参考答案。
        extra_info:   需包含 ``prompt``（原始问题）和 ``task_name``（任务类型）。

    Returns:
        dict with at least "score" key.
    """
    task_name = extra_info.get("task_name", "")
    prompt = extra_info.get("prompt", "")

    # Step 1: 提取 thinking / solution，检查格式
    thinking_str, solution_clean = extract_thinking_content(solution_str)
    if not if_format_correct(thinking_str, solution_clean, task_name):
        return {
            "score": -1.0,
            "judge_explain": "Format check failed",
            "task_type": task_name,
            "predicted_value": None, "gold_value": None, "mae": None,
        }

    # Step 2: LLM Judge（轮询分配到各服务器）
    server_idx = next(_server_selector)
    client = _sync_clients[server_idx]
    server_url = str(client.base_url)

    try:
        if task_name in CLASSIFICATION_TASKS:
            result = _judge_classification_sync(
                prompt, solution_clean, ground_truth, client,
            )
        elif task_name in REGRESSION_TASKS:
            result = _judge_regression_sync(
                prompt, solution_clean, ground_truth, task_name, client,
            )
        else:
            print(f"[WARNING] Unknown task_name: {task_name}, assigning score=0.0")
            result = {"score": 0.0, "judge_explain": f"Unknown task_name: {task_name}",
                     "predicted_value": None, "gold_value": None, "mae": None}

        # Canonical key order — must be identical across ALL return paths
        # to avoid AssertionError in DataProto.concat (reward_extra_keys mismatch)
        return {
            "score": result["score"],
            "judge_explain": result.get("judge_explain", ""),
            "task_type": task_name,
            "predicted_value": result.get("predicted_value"),
            "gold_value": result.get("gold_value"),
            "mae": result.get("mae"),
        }

    except Exception as e:
        print(
            f"[WARNING] _material_compute_score exception (server={server_url}): "
            f"{type(e).__name__}: {e}"
        )
        return {
            "score": -1.0,
            "judge_explain": f"Exception@{server_url}: {type(e).__name__}: {e}",
            "task_type": task_name,
            "predicted_value": None, "gold_value": None, "mae": None,
        }


class MaterialRewardManager(RateLimitedRewardManager):
    """材料科学任务奖励管理器（verl 0.7.0，importlib 外挂，无需修改 verl 源码）。

    继承自官方 ``RateLimitedRewardManager``，获得三层并发控制：

    * ``max_concurrent``  — asyncio.Semaphore，全局最大并发请求数
    * ``max_rpm``         — 令牌桶，每分钟最大请求数（可选）
    * ``max_tpm``         — 令牌桶，每分钟最大 token 数（可选）
    * ``timeout``         — 单条请求超时（默认 600s）

    评分逻辑：
    * 分类任务 → LLM Judge 判对错 → reward 1.0 / 0.0
    * 回归任务 → LLM 提取数值 → MAE → reward ∈ [0, 1]
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
            compute_score=_material_compute_score,
            reward_router_address=reward_router_address,
            reward_model_tokenizer=reward_model_tokenizer,
        )
