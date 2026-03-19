"""
Reward function for Material Science (MatBench) RL tasks.

Tasks:
  Classification:
    - matbench_expt_is_metal: Is the composition metallic? (True/False)
    - matbench_glass: Does the compound have glass formation ability? (True/False)
  Regression:
    - matbench_expt_gap: Predict the band gap (eV)
    - matbench_steels: Predict the yield strength (MPa)

Scoring strategy (all via LLM Judge, no rule-based parsing):
  - Classification: LLM compares model output with ground truth → reward 1.0 (correct) / 0.0 (incorrect)
  - Regression: LLM extracts a float from model output, then MAE against ground truth → reward
"""

import os
import re
import sys
import json
import asyncio
import httpx
from openai import AsyncOpenAI

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))


# ===== LLM Configuration =====
OPENAI_API_BASES = [
    "http://10.102.243.64:8000/v1"
]
OPENAI_API_KEY = ""
MODEL_NAME = "/mnt/shared-storage-user/ailab-hs/yangyuming/models/models--openai--gpt-oss-120b/snapshots/eabf0c518da7584a2e7dab4ab272709785a72126"

MAX_CONCURRENT_PER_SERVER = 1024
TIMEOUT_SECONDS = 600
MAX_TOKENS = 16384
MAX_CONCURRENT_REQUESTS = MAX_CONCURRENT_PER_SERVER


def _build_server_client(base_url: str) -> AsyncOpenAI:
    hc = httpx.AsyncClient(
        trust_env=False,
        limits=httpx.Limits(
            max_connections=MAX_CONCURRENT_PER_SERVER + 128,
            max_keepalive_connections=MAX_CONCURRENT_PER_SERVER,
            keepalive_expiry=30.0,
        ),
    )
    return AsyncOpenAI(base_url=base_url, api_key=OPENAI_API_KEY, http_client=hc)


_server_clients: list[AsyncOpenAI] = [_build_server_client(url) for url in OPENAI_API_BASES]
async_client = _server_clients[0]

_server_semaphores = None


def _get_server_semaphores() -> list[asyncio.Semaphore]:
    global _server_semaphores
    if _server_semaphores is None:
        _server_semaphores = [asyncio.Semaphore(MAX_CONCURRENT_PER_SERVER) for _ in OPENAI_API_BASES]
    return _server_semaphores


OPENAI_API_BASE = OPENAI_API_BASES[0]


# ===== Task type classification =====
CLASSIFICATION_TASKS = {'matbench_expt_is_metal', 'matbench_glass'}
REGRESSION_TASKS = {'matbench_expt_gap', 'matbench_steels'}


# ===== LLM Judge Templates =====

# Classification: LLM directly judges whether model's answer matches ground truth
CLASSIFICATION_JUDGE_TEMPLATE = """Please as a grading expert, judge whether the final answer given by the candidate below is consistent with the standard answer, that is, whether the candidate answered correctly.

Here are some evaluation criteria:
1. Please refer to the given standard answer. You don't need to re-generate the answer to the question because the standard answer has been given. You only need to judge whether the candidate's answer is consistent with the standard answer according to the form of the question. Don't try to answer the original question. You can assume that the standard answer is definitely correct.
2. Because the candidate's answer may be different from the standard answer in the form of expression, before making a judgment, please understand the question and the standard answer first, and then judge whether the candidate's answer is correct, but be careful not to try to answer the original question.
3. Some answers may be expressed in different ways, such as "True", "Yes", "It is metallic", "The substance is metal", etc. As long as the meaning expressed is the same as the standard answer, it should be considered correct.
4. If the prediction is given with \\boxed{{}}, please ignore the \\boxed{{}} and only judge whether the candidate's answer is consistent with the standard answer.

<Original Question Begin>:
{question}
<Original Question End>

<Gold Target Begin>:
{ground_truth}
<Gold Target End>

<Predicted Answer Begin>:
{prediction}
<Predicted Answer End>

**Output Format:**
Return a single JSON object:
{{"result": true, "explain": "Brief explanation..."}} if the candidate's answer is CORRECT.
{{"result": false, "explain": "Brief explanation..."}} if the candidate's answer is INCORRECT.
"""

# Regression: LLM extracts numerical values from both model output and ground truth
REGRESSION_EXTRACT_TEMPLATE = """Please function as a precise data extraction engine. Your task is to extract numerical values from the provided information.

Follow these strict rules:
1. Extract the numerical prediction from the candidate's answer and the gold value from the standard answer.
2. If the predicted answer is incomplete or does not contain a determinable numerical result, set "predicted_value" to null.
3. Extract the gold value from the standard answer provided.
4. Provide a brief explanation of how you extracted the values.

<Original Question Begin>:
{question}
<Original Question End>

<Gold Target Begin>:
{ground_truth}
<Gold Target End>

<Predicted Answer Begin>:
{prediction}
<Predicted Answer End>

**Output Format:**
Return a single JSON object:
{{"predicted_value": <float or null>, "gold_value": <float>, "explain": "Brief explanation of extraction..."}}
"""


# ===== Utility Functions =====

def extract_thinking_content(text: str) -> tuple[str, str]:
    """Extract thinking content from <think>...</think> tags."""
    pattern = r'<think>(.*?)</think>(.*)'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        thinking_content = match.group(1).strip()
        remaining_content = match.group(2).strip()
        return thinking_content, remaining_content
    return "", text


def if_format_correct(thinking_content: str, solution_content: str, task_type: str) -> bool:
    """Check if the format of the response is correct.

    For material science tasks, we only require:
    1. Thinking content exists (model used <think> tags)
    2. No nested <think> tags in solution
    """
    if thinking_content == "":
        return False
    if "<think>" in solution_content or "</think>" in solution_content:
        return False
    if solution_content.strip() == "":
        return False
    return True


def mae_to_reward(mae: float, mae_scale: float = 1.0) -> float:
    """Convert MAE to reward in [0, 1] using inverse transformation.

    reward = 1 / (1 + mae / mae_scale)

    Args:
        mae: Mean Absolute Error value (>= 0)
        mae_scale: Scale factor controlling the "half-life" point where reward=0.5
                   - matbench_expt_gap: band gap values typically 0-10 eV, use mae_scale=1.0
                   - matbench_steels: yield strength typically 0-2000 MPa, use mae_scale=100.0

    Returns:
        Reward in [0, 1], higher means smaller MAE (better prediction)
    """
    return 1.0 / (1.0 + mae / mae_scale)


# TODO: mae_scale 需要根据实际训练中各任务的数值范围进行调参
# matbench_expt_gap (band gap): 值域约 0~10 eV，建议 mae_scale=1.0
# matbench_steels (yield strength): 值域约 0~2000 MPa，建议 mae_scale=100.0
MAE_SCALE = {
    'matbench_expt_gap': 1.0,
    'matbench_steels': 300.0,
}


# ===== LLM Judge Functions =====

def _parse_json_from_response(content: str) -> dict | None:
    """Try to parse JSON from LLM response, handling markdown code blocks."""
    if "```json" in content:
        json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if json_match:
            content = json_match.group(1)
    elif "```" in content:
        json_match = re.search(r'```\s*(.*?)\s*```', content, re.DOTALL)
        if json_match:
            content = json_match.group(1)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None


async def _llm_call_async(prompt: str, client: AsyncOpenAI) -> str:
    """Make a single LLM call and return the response content."""
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=MAX_TOKENS,
        timeout=TIMEOUT_SECONDS,
    )
    return response.choices[0].message.content.strip()


async def judge_classification_async(
    question: str, prediction: str, ground_truth: str,
    client: AsyncOpenAI,
) -> dict:
    """LLM Judge for classification tasks: returns reward 1.0 or 0.0.

    Returns:
        dict: {"score": float, "judge_explain": str}
    """
    prompt = CLASSIFICATION_JUDGE_TEMPLATE.format(
        question=question,
        ground_truth=ground_truth,
        prediction=prediction,
    )
    try:
        content = await _llm_call_async(prompt, client)
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


async def judge_regression_async(
    question: str, prediction: str, ground_truth: str,
    task_type: str, client: AsyncOpenAI,
) -> dict:
    """LLM Judge for regression tasks: extracts float values and computes MAE-based reward.

    Returns:
        dict: {"score": float, "predicted_value": float|None, "gold_value": float|None,
               "mae": float|None, "judge_explain": str}
    """
    prompt = REGRESSION_EXTRACT_TEMPLATE.format(
        question=question,
        ground_truth=ground_truth,
        prediction=prediction,
    )
    try:
        content = await _llm_call_async(prompt, client)
        parsed = _parse_json_from_response(content)

        if parsed is None:
            # Fallback: try to find numbers in the response
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


# ===== Combined Scoring =====

async def _score_single_item_async(
    idx: int,
    item: dict,
    semaphore: asyncio.Semaphore,
    client: AsyncOpenAI,
) -> tuple[int, dict]:
    """Score a single item using LLM-based evaluation.

    Args:
        idx: Index of the item in the batch.
        item: Dict with keys: prompt, prediction, thinking, ground_truth, task_type.
        semaphore: Per-server semaphore limiting concurrent LLM calls.
        client: AsyncOpenAI client for the assigned server.

    Returns:
        tuple: (index, result_dict)
    """
    task_type = item["task_type"]

    # Step 1: Check format
    if not if_format_correct(item["thinking"], item["prediction"], task_type):
        return idx, {
            "score": -1.0,
            "judge_explain": "Format check failed",
            "task_type": task_type,
            "predicted_value": None, "gold_value": None, "mae": None,
        }

    # Step 2: LLM Judge
    async with semaphore:
        try:
            if task_type in CLASSIFICATION_TASKS:
                result = await judge_classification_async(
                    item["prompt"], item["prediction"], item["ground_truth"], client,
                )
            elif task_type in REGRESSION_TASKS:
                result = await judge_regression_async(
                    item["prompt"], item["prediction"], item["ground_truth"],
                    task_type, client,
                )
            else:
                print(f"[WARNING] Unknown task_type: {task_type}, assigning score=0.0")
                result = {"score": 0.0, "judge_explain": f"Unknown task_type: {task_type}",
                         "predicted_value": None, "gold_value": None, "mae": None}

            result["task_type"] = task_type
            return idx, result

        except Exception as e:
            server_url = str(client.base_url)
            print(f"[WARNING] Scoring error (server={server_url}): {type(e).__name__}: {e}")
            return idx, {
                "score": -1.0,
                "judge_explain": f"Exception@{server_url}: {type(e).__name__}: {e}",
                "task_type": task_type,
                "predicted_value": None, "gold_value": None, "mae": None,
            }


async def score_batch_async(items: list[dict]) -> list[dict]:
    """Score a batch of items concurrently across multiple LLM servers.

    Args:
        items: List of dicts, each with keys:
               prompt, prediction, thinking, ground_truth, task_type.

    Returns:
        List of result dicts (same order as input).
    """
    semaphores = _get_server_semaphores()
    n_servers = len(_server_clients)

    coros = []
    for i, item in enumerate(items):
        server_idx = i % n_servers
        coros.append(_score_single_item_async(
            i, item, semaphores[server_idx], _server_clients[server_idx],
        ))

    raw_results = await asyncio.gather(*coros)
    results = [None] * len(items)
    for idx, result in raw_results:
        results[idx] = result
    return results


def score_batch(items: list[dict]) -> list[dict]:
    """Sync wrapper for score_batch_async."""
    return asyncio.run(score_batch_async(items))
