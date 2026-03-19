"""
ChemistryRewardManager V3 — verl 0.7.0 外挂 reward manager（无需修改 verl 源码）

设计思路：
  - 继承官方 RateLimitedRewardManager，利用其三层并发控制
    (max_concurrent / max_rpm / max_tpm / timeout)
  - 复用 reward_chemistry_v3.py 中的所有评分逻辑（Rule + LLM Judge），不重复实现
  - 仅需向父类注入一个符合 verl 签名的单条异步评分函数

V3 变更（基于 V2）：
  - 每条 rubric 独立评判，全部通过才算通过
  - 支持 thinking_required 字段：True 显示 CoT，False 隐藏 CoT 防止 hack
  - 支持 answer_required 字段：True 显示 Answer，False 隐藏 Answer 仅评判 CoT
  - 三种 Judge 模板：CoT+Answer / Answer-only / CoT-only
  - 返回 per_rubric 详细信息

加载方式（shell script 中配置）：
    reward_model.reward_loop_source=importlib
    reward_model.reward_loop_module_path="../src/reward/mix_env/chemistry_reward_manager_v3.py"
    reward_model.reward_loop_class_name=ChemistryRewardManager
    reward_model.max_concurrent=4096       # 总并发上限，建议 = num_servers * per_server_limit
    reward_model.timeout=600               # 单条请求最大等待秒数

    # 不再需要：
    #   reward_model.reward_manager=batch
    #   custom_reward_function.path=...
    #   custom_reward_function.name=...
"""

import itertools
import json
import os
import re
import sys

import httpx
from openai import OpenAI  # 同步客户端

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LLM Judge 配置（独立于 reward_chemistry_v2.py，按需修改）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OPENAI_API_BASES = [
    "http://10.102.200.9:8000/v1",
    "http://10.102.214.44:8000/v1",
    "http://10.102.97.29:8000/v1",
    "http://10.102.243.44:8000/v1",
]
OPENAI_API_KEY = ""
MODEL_NAME = "/mnt/shared-storage-user/ailab-hs/yangyuming/models/models--openai--gpt-oss-120b/snapshots/eabf0c518da7584a2e7dab4ab272709785a72126"
MAX_TOKENS = 16384

# HTTP 请求超时（秒）。注意：这与 reward_model.timeout（asyncio 层）不冗余——
# _chemistry_compute_score 在线程池中运行，asyncio 超时无法中断阻塞的 HTTP 调用，
# 只有这里的 HTTP 超时才能真正打断 client.create() 的阻塞。
# 建议：TIMEOUT_SECONDS <= reward_model.timeout（shell 中配置）
TIMEOUT_SECONDS = 600

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── 确保 reward_chemistry_v2.py 及其依赖所在目录在 sys.path 上 ────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in [_THIS_DIR, os.path.dirname(_THIS_DIR), os.path.dirname(os.path.dirname(_THIS_DIR))]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 从 reward_chemistry_v3.py 导入纯逻辑构件（不含任何配置常量）
from reward_chemistry_v3 import (  # noqa: E402
    judge_template,            # LLM Judge 提示词模板（CoT + Answer）
    judge_template_no_cot,     # LLM Judge 提示词模板（Answer only）
    judge_template_no_answer,  # LLM Judge 提示词模板（CoT only）
    rubrics,                   # 各任务类型评分标准（list[dict] 格式）
    extract_thinking_content,  # (text) -> (thinking_str, solution_str)
    if_format_correct,         # (thinking, solution, task_type) -> bool
    compute_rule_score,        # (prediction, reference, task_type) -> float
)

from verl.experimental.reward_loop.reward_manager.limited import RateLimitedRewardManager  # noqa: E402


# ── 同步 OpenAI 客户端（每服务器一个）──────────────────────────────────────────
# 连接池大小设为固定值，实际并发由 verl 的 max_concurrent Semaphore 统一控制。
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

# ── 轮询选择器：均匀分发请求到各推理服务器 ──────────────────────────────────────
_server_selector = itertools.cycle(range(len(_sync_clients)))


def _judge_single_rubric(
    query: str, thinking: str, prediction: str,
    rubric_text: str, client: OpenAI,
    thinking_required: bool = True,
    answer_required: bool = True,
) -> dict:
    """同步：评判单条 rubric criterion。

    Args:
        thinking_required: True 时向 judger 展示 CoT，False 时隐藏 CoT 防 hack。
        answer_required:   True 时向 judger 展示 Answer，False 时隐藏 Answer
                           让 judger 仅基于 CoT 评判。

    Template selection:
        thinking=T, answer=T → judge_template          (CoT + Answer)
        thinking=F, answer=T → judge_template_no_cot    (Answer only)
        thinking=T, answer=F → judge_template_no_answer (CoT only)

    Returns:
        dict: {"result": bool, "explain": str}
    """
    if thinking_required and answer_required:
        prompt = judge_template.format(
            query=query,
            thinking_content=thinking,
            answer_content=prediction,
            rubric=rubric_text,
        )
    elif not thinking_required and answer_required:
        prompt = judge_template_no_cot.format(
            query=query,
            answer_content=prediction,
            rubric=rubric_text,
        )
    else:  # thinking_required=True, answer_required=False
        prompt = judge_template_no_answer.format(
            query=query,
            thinking_content=thinking,
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

    # Parse JSON result
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
    """同步 LLM Judge：逐条评判 rubric，全部通过才算通过。

    与 reward_chemistry_v3.judge_process_async 逻辑完全相同，
    使用同步 openai.OpenAI 客户端，可安全在线程池中调用。

    Returns:
        dict: {"process_valid": bool, "judge_explain": str,
               "per_rubric": [{"rubric": str, "thinking_required": bool,
                               "answer_required": bool,
                               "result": bool, "explain": str}, ...]}
    """
    rubric_list = rubrics.get(task_type)
    if not rubric_list:
        raise ValueError(f"Unknown task type: {task_type}")

    # 逐条评判（同步串行；并发由 verl RateLimitedRewardManager 在外层控制）
    per_rubric = []
    all_pass = True
    explains = []
    failed_punishment_scores = []

    for rubric_item in rubric_list:
        rubric_text = rubric_item["content"]
        thinking_req = rubric_item.get("thinking_required", True)
        answer_req = rubric_item.get("answer_required", True)
        punishment = rubric_item.get("punishment_score", -0.5)

        res = _judge_single_rubric(
            query, thinking, prediction,
            rubric_text, client,
            thinking_required=thinking_req,
            answer_required=answer_req,
        )

        passed = res["result"]
        per_rubric.append({
            "rubric": rubric_text,
            "thinking_required": thinking_req,
            "answer_required": answer_req,
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

    # Worst (minimum) punishment score among all failed rubrics
    worst_punishment = min(failed_punishment_scores) if failed_punishment_scores else 0.0

    rubric_texts = [item["rubric"] for item in per_rubric]
    rubric_thinking_reqs = [str(item["thinking_required"]) for item in per_rubric]
    rubric_answer_reqs = [str(item["answer_required"]) for item in per_rubric]
    rubric_punishments = [str(item["punishment_score"]) for item in per_rubric]
    rubric_results = [str(item["result"]) for item in per_rubric]

    return {
        "process_valid": all_pass,
        "punishment_score": worst_punishment,
        "judge_explain": " | ".join(explains),
        "rubric_contents": " | ".join(rubric_texts),
        "rubric_thinking_required": " | ".join(rubric_thinking_reqs),
        "rubric_answer_required": " | ".join(rubric_answer_reqs),
        "rubric_punishment_scores": " | ".join(rubric_punishments),
        "rubric_results": " | ".join(rubric_results),
    }


def _chemistry_compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
) -> dict:
    """单条样本评分函数，符合 verl RateLimitedRewardManager 的 compute_score 签名。

    同步函数，由 RateLimitedRewardManager 通过 run_in_executor 在线程池中调用。
    并发控制完全委托给 RateLimitedRewardManager（max_concurrent / max_rpm / max_tpm）。

    Args:
        data_source:  数据来源标识（兼容参数，此处不使用）。
        solution_str: 模型完整回复，含 <think>…</think> 标签。
        ground_truth: 参考答案。
        extra_info:   需包含 ``prompt``（原始问题）和 ``task_name``（任务类型）。

    Returns:
        dict::

            {
                "score"              : float,         # 最终 reward
                "rule_score"         : float | None,  # 规则分
                "process_valid"      : bool  | None,  # LLM Judge 结果
                "judge_explain"      : str,           # Judge 说明
                "rubric_contents"    : str,           # 各 rubric 内容（| 分隔）
                "rubric_thinking_required": str,      # 各 rubric 是否需要 CoT（| 分隔）
                "rubric_answer_required": str,        # 各 rubric 是否需要 Answer（| 分隔）
                "rubric_results"     : str,           # 各 rubric 评判结果（| 分隔）
            }
    """
    task_name = extra_info.get("task_name", "RS")
    prompt = extra_info.get("prompt", "")

    # Step 1: 提取 thinking / solution，检查格式
    thinking_str, solution_clean = extract_thinking_content(solution_str)
    if not if_format_correct(thinking_str, solution_clean, task_name):
        return {
            "score": -1.0,
            "rule_score": None,
            "process_valid": None,
            "punishment_score": 0.0,
            "judge_explain": "Format check failed",
            "rubric_contents": "",
            "rubric_thinking_required": "",
            "rubric_answer_required": "",
            "rubric_punishment_scores": "",
            "rubric_results": "",
        }

    # Step 2: 规则评分 + LLM Judge（轮询分配到各服务器）
    server_idx = next(_server_selector)
    client = _sync_clients[server_idx]
    server_url = str(client.base_url)

    try:
        rule_score = compute_rule_score(solution_clean, ground_truth, task_name)
        judge_result = _judge_process(prompt, thinking_str, solution_clean, task_name, client=client)
        process_valid = judge_result["process_valid"]
        judge_explain = judge_result["judge_explain"]
        punishment = judge_result.get("punishment_score", -0.5)
        final_score = rule_score if process_valid else punishment

        return {
            "score": final_score,
            "rule_score": rule_score,
            "process_valid": process_valid,
            "punishment_score": punishment,
            "judge_explain": judge_explain,
            "rubric_contents": judge_result["rubric_contents"],
            "rubric_thinking_required": judge_result["rubric_thinking_required"],
            "rubric_answer_required": judge_result.get("rubric_answer_required", ""),
            "rubric_punishment_scores": judge_result.get("rubric_punishment_scores", ""),
            "rubric_results": judge_result["rubric_results"],
        }

    except Exception as e:
        print(
            f"[WARNING] _chemistry_compute_score exception (server={server_url}): "
            f"{type(e).__name__}: {e}"
        )
        return {
            "score": -1.0,
            "rule_score": None,
            "process_valid": None,
            "punishment_score": 0.0,
            "judge_explain": f"Exception@{server_url}: {type(e).__name__}: {e}",
            "rubric_contents": "",
            "rubric_thinking_required": "",
            "rubric_answer_required": "",
            "rubric_punishment_scores": "",
            "rubric_results": "",
        }


class ChemistryRewardManager(RateLimitedRewardManager):
    """化学任务奖励管理器 V2（verl 0.7.0，importlib 外挂，无需修改 verl 源码）。

    继承自官方 ``RateLimitedRewardManager``，获得三层并发控制：

    * ``max_concurrent``  — asyncio.Semaphore，全局最大并发请求数
    * ``max_rpm``         — 令牌桶，每分钟最大请求数（可选）
    * ``max_tpm``         — 令牌桶，每分钟最大 token 数（可选）
    * ``timeout``         — 单条请求超时（默认 600s）

    评分逻辑委托给 ``_chemistry_compute_score``，
    该函数复用 ``reward_chemistry_v3.py`` 中的所有原子构件。

    V3 变更（基于 V2）：
    * 每条 rubric 独立评判，全部通过才算通过
    * 支持 thinking_required 控制 CoT 可见性
    * 支持 answer_required 控制 Answer 可见性
    * 三种 Judge 模板：CoT+Answer / Answer-only / CoT-only
    * 返回 per_rubric 详细评判信息
    """

    def __init__(
        self,
        config,
        tokenizer,
        compute_score=None,           # verl 框架传入，本类忽略，始终使用内置函数
        reward_router_address=None,
        reward_model_tokenizer=None,
    ):
        # 强制注入 _chemistry_compute_score，忽略框架传进来的 compute_score
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            compute_score=_chemistry_compute_score,
            reward_router_address=reward_router_address,
            reward_model_tokenizer=reward_model_tokenizer,
        )
