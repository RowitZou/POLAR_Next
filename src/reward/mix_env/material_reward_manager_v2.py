"""
MaterialRewardManager V2 — verl 0.7.0 外挂 reward manager（无需修改 verl 源码）

设计思路：
  - 继承官方 RateLimitedRewardManager，利用其三层并发控制
    (max_concurrent / max_rpm / max_tpm / timeout)
  - 复用 reward_material_v2.py 中的所有评分逻辑（LLM Judge + Anti-Hack Rubrics），不重复实现
  - 仅需向父类注入一个符合 verl 签名的单条同步评分函数

V2 变更：
  - 新增 anti-hack rubric 评判，每条 rubric 独立评判，全部通过才算通过
  - 支持 thinking_required 字段：True 显示 CoT，False 隐藏 CoT 防止 hack
  - 针对 5 种观测到的 hack 模式设计专门 rubric：
    1. 伪造文献引用（Fabricated Citations）
    2. 伪造计算证据（Fabricated DFT/Computational Evidence）
    3. 伪造实验证据（Fabricated Experimental Evidence）
    4. 权威性语言包装（Authority Language Overlay）
    5. 零偏差 band gap 预测（Zero-Bias Prediction）
  - 返回 per_rubric 详细信息

加载方式（shell script 中配置）：
    reward_model.reward_loop_source=importlib
    reward_model.reward_loop_module_path="../src/reward/mix_env/material_reward_manager_v2.py"
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

from reward_material_v2 import (  # noqa: E402
    CLASSIFICATION_JUDGE_TEMPLATE,
    REGRESSION_EXTRACT_TEMPLATE,
    judge_template,
    judge_template_no_cot,
    rubrics,
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


# ── Classification / Regression LLM Judge (同步版) ────────────────────────────

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


# ── Anti-Hack Rubric Judging (同步版) ─────────────────────────────────────────

def _judge_single_rubric(
    query: str, thinking: str, prediction: str,
    rubric_text: str, client: OpenAI,
    thinking_required: bool = True,
) -> dict:
    """同步：评判单条 rubric criterion。

    Args:
        thinking_required: True 时使用 judge_template（显示 CoT），
                           False 时使用 judge_template_no_cot（隐藏 CoT 防 hack）。

    Returns:
        dict: {"result": bool, "explain": str}
    """
    if thinking_required:
        prompt = judge_template.format(
            query=query,
            thinking_content=thinking,
            answer_content=prediction,
            rubric=rubric_text,
        )
    else:
        prompt = judge_template_no_cot.format(
            query=query,
            answer_content=prediction,
            rubric=rubric_text,
        )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=MAX_TOKENS,
        timeout=TIMEOUT_SECONDS,
        reasoning_effort="high",
    )

    content = response.choices[0].message.content.strip()

    # Try to extract JSON from markdown code blocks
    if "```json" in content:
        json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if json_match:
            content = json_match.group(1)
    elif "```" in content:
        json_match = re.search(r'```\s*(.*?)\s*```', content, re.DOTALL)
        if json_match:
            content = json_match.group(1)

    try:
        result = json.loads(content)
        return {"result": bool(result.get("result", False)),
                "explain": result.get("explain", "")}
    except json.JSONDecodeError:
        content_lower = content.lower()
        if '"result": true' in content_lower or '"result":true' in content_lower:
            return {"result": True, "explain": f"Parsed from raw: {content[:200]}"}
        elif '"result": false' in content_lower or '"result":false' in content_lower:
            return {"result": False, "explain": f"Parsed from raw: {content[:200]}"}
        else:
            print(f"[WARNING] Cannot parse judger response: {content[:200]}")
            return {"result": False, "explain": f"Parse failed: {content[:200]}"}


def _judge_process(query: str, thinking: str, prediction: str, task_type: str, client: OpenAI) -> dict:
    """同步 LLM Judge：逐条评判 anti-hack rubric，全部通过才算通过。

    Returns:
        dict: {"process_valid": bool, "punishment_score": float,
               "judge_explain": str,
               "rubric_contents": str, "rubric_thinking_required": str,
               "rubric_punishment_scores": str, "rubric_results": str}
    """
    rubric_list = rubrics.get(task_type)
    if not rubric_list:
        raise ValueError(f"Unknown task type for rubrics: {task_type}")

    per_rubric = []
    all_pass = True
    explains = []
    failed_punishment_scores = []

    for rubric_item in rubric_list:
        rubric_text = rubric_item["content"]
        thinking_req = rubric_item.get("thinking_required", True)
        punishment = rubric_item.get("punishment_score", -0.5)

        res = _judge_single_rubric(
            query, thinking, prediction,
            rubric_text, client,
            thinking_required=thinking_req,
        )

        passed = res["result"]
        per_rubric.append({
            "rubric": rubric_text,
            "thinking_required": thinking_req,
            "punishment_score": punishment,
            "result": passed,
            "explain": res["explain"],
        })
        if not passed:
            all_pass = False
            failed_punishment_scores.append(punishment)
            explains.append(f"[FAIL] {res['explain']}")
        else:
            explains.append(f"[PASS] {res['explain']}")

    worst_punishment = min(failed_punishment_scores) if failed_punishment_scores else 0.0

    rubric_texts = [item["rubric"] for item in per_rubric]
    rubric_thinking_reqs = [str(item["thinking_required"]) for item in per_rubric]
    rubric_punishments = [str(item["punishment_score"]) for item in per_rubric]
    rubric_results = [str(item["result"]) for item in per_rubric]

    return {
        "process_valid": all_pass,
        "punishment_score": worst_punishment,
        "judge_explain": " | ".join(explains),
        "rubric_contents": " | ".join(rubric_texts),
        "rubric_thinking_required": " | ".join(rubric_thinking_reqs),
        "rubric_punishment_scores": " | ".join(rubric_punishments),
        "rubric_results": " | ".join(rubric_results),
    }


# ── 单条样本评分函数 ──────────────────────────────────────────────────────────

def _material_compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
) -> dict:
    """单条样本评分函数，符合 verl RateLimitedRewardManager 的 compute_score 签名。

    同步函数，由 RateLimitedRewardManager 通过 run_in_executor 在线程池中调用。

    评分流程：
    1. 提取 thinking / solution，检查格式
    2. LLM Classification/Regression Judge → llm_score
    3. Anti-Hack Rubric Judge → process_valid
    4. 合并：final_score = llm_score if process_valid else punishment

    Args:
        data_source:  数据来源标识（兼容参数，此处不使用）。
        solution_str: 模型完整回复，含 <think>…</think> 标签。
        ground_truth: 参考答案。
        extra_info:   需包含 ``prompt``（原始问题）和 ``task_name``（任务类型）。

    Returns:
        dict with keys: score, llm_score, process_valid, punishment_score,
                        judge_explain, task_type, predicted_value, gold_value, mae,
                        rubric_contents, rubric_thinking_required,
                        rubric_punishment_scores, rubric_results.
    """
    task_name = extra_info.get("task_name", "")
    prompt = extra_info.get("prompt", "")

    # Step 1: 提取 thinking / solution，检查格式
    thinking_str, solution_clean = extract_thinking_content(solution_str)
    if not if_format_correct(thinking_str, solution_clean, task_name):
        return {
            "score": -1.0,
            "llm_score": None,
            "process_valid": None,
            "punishment_score": 0.0,
            "judge_explain": "Format check failed",
            "task_type": task_name,
            "predicted_value": None, "gold_value": None, "mae": None,
            "rubric_contents": "",
            "rubric_thinking_required": "",
            "rubric_punishment_scores": "",
            "rubric_results": "",
        }

    # Step 2: LLM Judge + Rubric Check（轮询分配到各服务器）
    server_idx = next(_server_selector)
    client = _sync_clients[server_idx]
    server_url = str(client.base_url)

    try:
        # 2a: Classification/Regression LLM score
        if task_name in CLASSIFICATION_TASKS:
            llm_result = _judge_classification_sync(
                prompt, solution_clean, ground_truth, client,
            )
        elif task_name in REGRESSION_TASKS:
            llm_result = _judge_regression_sync(
                prompt, solution_clean, ground_truth, task_name, client,
            )
        else:
            print(f"[WARNING] Unknown task_name: {task_name}, assigning score=0.0")
            llm_result = {"score": 0.0, "judge_explain": f"Unknown task_name: {task_name}",
                         "predicted_value": None, "gold_value": None, "mae": None}

        llm_score = llm_result["score"]

        # 2b: Anti-hack rubric evaluation
        rubric_result = _judge_process(prompt, thinking_str, solution_clean, task_name, client)
        process_valid = rubric_result["process_valid"]
        rubric_explain = rubric_result["judge_explain"]
        punishment = rubric_result.get("punishment_score", 0.0)

        # Step 3: Combine — use llm_score if all rubrics pass, else punishment
        final_score = llm_score if process_valid else punishment

        return {
            "score": final_score,
            "llm_score": llm_score,
            "process_valid": process_valid,
            "punishment_score": punishment,
            "judge_explain": f"LLM: {llm_result.get('judge_explain', '')} | Rubric: {rubric_explain}",
            "task_type": task_name,
            "predicted_value": llm_result.get("predicted_value"),
            "gold_value": llm_result.get("gold_value"),
            "mae": llm_result.get("mae"),
            "rubric_contents": rubric_result["rubric_contents"],
            "rubric_thinking_required": rubric_result["rubric_thinking_required"],
            "rubric_punishment_scores": rubric_result.get("rubric_punishment_scores", ""),
            "rubric_results": rubric_result["rubric_results"],
        }

    except Exception as e:
        print(
            f"[WARNING] _material_compute_score exception (server={server_url}): "
            f"{type(e).__name__}: {e}"
        )
        return {
            "score": -1.0,
            "llm_score": None,
            "process_valid": None,
            "punishment_score": 0.0,
            "judge_explain": f"Exception@{server_url}: {type(e).__name__}: {e}",
            "task_type": task_name,
            "predicted_value": None, "gold_value": None, "mae": None,
            "rubric_contents": "",
            "rubric_thinking_required": "",
            "rubric_punishment_scores": "",
            "rubric_results": "",
        }


class MaterialRewardManager(RateLimitedRewardManager):
    """材料科学任务奖励管理器 V2（verl 0.7.0，importlib 外挂，无需修改 verl 源码）。

    继承自官方 ``RateLimitedRewardManager``，获得三层并发控制：

    * ``max_concurrent``  — asyncio.Semaphore，全局最大并发请求数
    * ``max_rpm``         — 令牌桶，每分钟最大请求数（可选）
    * ``max_tpm``         — 令牌桶，每分钟最大 token 数（可选）
    * ``timeout``         — 单条请求超时（默认 600s）

    评分逻辑：
    * 分类任务 → LLM Judge 判对错 → reward 1.0 / 0.0
    * 回归任务 → LLM 提取数值 → MAE → reward ∈ [0, 1]
    * Anti-Hack Rubric → 独立评判伪造引用/证据等 → 全部通过才保留 llm_score

    V2 变更：
    * 每条 anti-hack rubric 独立评判，全部通过才算通过
    * 支持 thinking_required 控制 CoT 可见性
    * 返回 per_rubric 详细评判信息
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
