"""
Mixed Environment Reward for Chemistry Tasks.

Combines:
- Rule-based scoring for final answer correctness (FTS, MAE, METEOR)
- LLM-based judging for reasoning process quality (CoT evaluation)

Final reward = rule_score if LLM judges the process as reasonable, else -1.0 (format/process error)
"""

import os
import re
import sys
import json
import asyncio
import httpx
from openai import AsyncOpenAI

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from utils import fts_score, mae_score, meteor_score  # noqa: E402


# ===== LLM Configuration =====
# Multiple inference servers for distributed dispatch (data-parallel style).
# Each entry is one server URL.  Single-server usage: keep only one entry.
OPENAI_API_BASES = [
    "http://10.102.97.47:8000/v1",
    "http://10.102.241.51:8000/v1",
    "http://10.102.241.23:8000/v1",
    "http://10.102.243.27:8000/v1",
]
OPENAI_API_KEY = ""
MODEL_NAME = "/mnt/shared-storage-user/ailab-hs/yangyuming/models/models--openai--gpt-oss-120b/snapshots/eabf0c518da7584a2e7dab4ab272709785a72126"

# Per-server concurrency limit.  Each server receives at most this many parallel requests.
# Total effective concurrency = MAX_CONCURRENT_PER_SERVER * len(OPENAI_API_BASES)
MAX_CONCURRENT_PER_SERVER = 1024
TIMEOUT_SECONDS = 600
MAX_TOKENS = 16384

# ---- keep legacy name as alias so verl call-sites that read it still work ----
MAX_CONCURRENT_REQUESTS = MAX_CONCURRENT_PER_SERVER


def _build_server_client(base_url: str) -> AsyncOpenAI:
    """Create one AsyncOpenAI client per server with a properly-sized connection pool."""
    hc = httpx.AsyncClient(
        trust_env=False,
        limits=httpx.Limits(
            max_connections=MAX_CONCURRENT_PER_SERVER + 128,
            max_keepalive_connections=MAX_CONCURRENT_PER_SERVER,
            keepalive_expiry=30.0,
        ),
    )
    return AsyncOpenAI(base_url=base_url, api_key=OPENAI_API_KEY, http_client=hc)


# One client per server entry
_server_clients: list[AsyncOpenAI] = [_build_server_client(url) for url in OPENAI_API_BASES]

# Legacy single-client alias (used by judge_process_async default path)
async_client = _server_clients[0]

# One semaphore per server, created lazily inside the async context
# (asyncio.Semaphore must be created insider the running event loop on Python <3.10)
_server_semaphores = None  # type: list[asyncio.Semaphore] | None


def _get_server_semaphores() -> list[asyncio.Semaphore]:
    """Return (or lazily create) one Semaphore per server."""
    global _server_semaphores
    if _server_semaphores is None:
        _server_semaphores = [asyncio.Semaphore(MAX_CONCURRENT_PER_SERVER) for _ in OPENAI_API_BASES]
    return _server_semaphores


# Backwards-compat: single api-base string pointing at the first server
OPENAI_API_BASE = OPENAI_API_BASES[0]


# ===== Rule-based Score Info =====
# Score ranges and normalization info for each task type
# Format: {'range': (min, max), 'higher_is_better': bool}
SCORE_INFO = {
    'RP': {'range': (0, 100), 'higher_is_better': True, 'eval_func': 'fts_score'},   # Tanimoto similarity * 100
    'MG': {'range': (0, 100), 'higher_is_better': True, 'eval_func': 'fts_score'},   # Tanimoto similarity * 100
    'FS': {'range': (0, 100), 'higher_is_better': True, 'eval_func': 'fts_score'},   # Tanimoto similarity * 100
    'RS': {'range': (0, 100), 'higher_is_better': True, 'eval_func': 'fts_score'},   # Tanimoto similarity * 100
    'PP': {'range': (0, None), 'higher_is_better': False, 'eval_func': 'mae_score'},  # MAE, lower is better
    'MC': {'range': (0, 1), 'higher_is_better': True, 'eval_func': 'meteor_score'},  # METEOR score
}


# ===== LLM Judge Template and Rubrics =====
judge_template = """You are an expert chemistry process evaluator.

**Sample to be Evaluated:**
- **User Question:** {query}
- **Model Chain of Thought:** {thinking_content}
- **Model Final Answer:** {answer_content}

**Evaluation Rubric:**
{rubric}

**CRITICAL INSTRUCTION:**
- **DO NOT** evaluate whether the final answer is factually correct or accurate.
- Accuracy verification is handled by external rule-based systems.
- Your ONLY job is to evaluate: (1) CoT-Answer Consistency, (2) Output Integrity (anti-cheating), (3) CoT Completeness & Reasonableness.

Analyze the model's performance based strictly on the provided Rubric.
If the model MEETS ALL the criteria described in the rubric, the result must be True.
If the model FAILS ANY of the criterion, the result must be False.

**Boundary Case Handling:**
If ANY of the following boundary cases occur, the result must be **False**:
- The Model Final Answer is absent, empty, or consists only of generic filler text.
- The Chain of Thought is absent, empty, or consists only of generic filler text.

**Output Format:**
Return a single JSON object:
{{"result": true, "explain": "Brief explanation..."}} or {{"result": false, "explain": "Brief explanation..."}}"""

# Rubrics focus on: Consistency, Anti-cheating, CoT Completeness & Reasonableness
# NOTE: Accuracy is NOT evaluated here - handled by external rule-based systems
rubrics = {
    "FS": """1. **CoT-Answer Consistency**: The final SELFIES must logically follow from the CoT conclusion. If CoT concludes X but answer gives Y, auto-fail. (Do NOT verify chemical correctness)
2. **Anti-Cheating**: Output must NOT be copied/appended input reactants or reagents. Must show genuine transformation attempt.
3. **CoT Completeness**: CoT must contain substantive reasoning (reaction type identification, mechanism discussion, bond changes)—not vague/generic filler text.""",

    "RP": """1. **CoT-Answer Consistency**: The final SELFIES reagents must logically follow from the CoT conclusion. Contradictions auto-fail. (Do NOT verify if reagents are correct)
2. **Anti-Cheating**: Output must NOT be copied/appended input reactants or target product. Must show genuine prediction attempt.
3. **CoT Completeness**: CoT must analyze the transformation and provide reasoning for reagent selection—not vague/generic filler text.""",

    "RS": """1. **CoT-Answer Consistency**: Final reactants/reagents must logically follow from CoT conclusion. Contradictions auto-fail. (Do NOT verify if retrosynthesis is correct)
2. **Anti-Cheating**: Output must NOT be copied/appended input product. Must show genuine retrosynthesis attempt.
3. **CoT Completeness**: CoT must contain retrosynthetic reasoning (disconnection analysis, synthon discussion)—not vague/generic filler text.""",

    "MG": """1. **CoT-Answer Consistency**: Final SELFIES must match the structure described in CoT conclusion. Contradictions auto-fail. (Do NOT verify if SELFIES is chemically correct)
2. **Anti-Cheating**: Output must be a SELFIES string—not copied input natural language text or generic filler.
3. **CoT Completeness**: CoT must systematically analyze input description (identifying rings, groups, heteroatoms)—not vague/generic filler text.""",

    "PP": """1. **CoT-Answer Consistency**: Final \\boxed{{}} value must align with CoT's estimation/trend. Contradictions auto-fail. (Do NOT verify if the value is accurate)
2. **Anti-Cheating**: CoT must reference input molecule's structural features—not arbitrary number without any reasoning attempt.
3. **CoT Completeness**: CoT must contain Structure-Property Relationship analysis (discussing relevant structural factors)—not vague/generic filler text.""",

    "MC": """1. **CoT-Answer Consistency**: Final description must align with structural features identified in CoT. Contradictions auto-fail. (Do NOT verify factual accuracy)
2. **Anti-Cheating**: Output must contain specific useful information—not generic boilerplate text or repetitive padding.
3. **CoT Completeness**: CoT must parse the input SELFIES and identify substructures (rings, groups, heteroatoms)—not vague/generic filler text."""
}


# ===== Utility Functions =====

def normalize_score_to_reward(score: float, task_type: str, mae_scale: float = 1.0) -> float:
    """
    Normalize raw score to reward in range [0, 1], where higher is better.

    Args:
        score: Raw score from the evaluation function
        task_type: Task type (RP, MG, FS, RS, PP, MC)
        mae_scale: Scale factor for MAE normalization (default 1.0)
                   Controls the "half-life" point where reward=0.5
                   - Good models: MAE typically 0-0.5, use mae_scale=0.5~1.0
                   - Poor models: MAE can reach 10+, use mae_scale=5.0~10.0

    Returns:
        Normalized reward in [0, 1], higher means better performance

    Score ranges:
        - RP, MG, FS, RS (fts_score): 0-100, higher is better
          Normalization: reward = score / 100

        - PP (mae_score): 0-inf, lower is better
          Normalization: reward = 1 / (1 + score / mae_scale)
          Examples with mae_scale=1.0:
            MAE=0.0 → reward=1.0
            MAE=0.17 → reward≈0.85
            MAE=0.5 → reward≈0.67
            MAE=1.0 → reward=0.5
            MAE=10.0 → reward≈0.09

        - MC (meteor_score): 0-1, higher is better
          Normalization: reward = score (already in [0,1])
    """
    if task_type not in SCORE_INFO:
        raise ValueError(f"Unknown task type: {task_type}")

    if task_type == 'PP':
        # MAE: lower is better, use inverse transformation
        # reward = 1 / (1 + score / mae_scale)
        # When score=0, reward=1; when score=mae_scale, reward=0.5; when score->inf, reward->0
        reward = 1.0 / (1.0 + score / mae_scale)
    elif task_type == 'MC':
        # METEOR: already in [0, 1], higher is better
        reward = max(0.0, min(1.0, score))
    else:
        # FTS (RP, MG, FS, RS): 0-100, higher is better
        reward = max(0.0, min(1.0, score / 100.0))

    return reward


def extract_thinking_content(text: str) -> tuple[str, str]:
    """Extract thinking content from <think>...</think> tags."""
    pattern = r'<think>(.*?)</think>(.*)'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        thinking_content = match.group(1).strip()
        remaining_content = match.group(2).strip()
        return thinking_content, remaining_content
    return "", text


def MC_task_format(pred, gold):
    """
    清洗预测值和真值，去除固定的模板废话，为 METEOR 分数计算做准备。
    """

    def clean_text(text):
        if not text or not isinstance(text, str):
            return ""

        # --- 步骤 1: 提取型清洗 (针对 Natural Product) ---
        natural_prod_pattern = r"The molecule is a natural product found in\s+(.*?)\s+with data available\.?"
        text = re.sub(natural_prod_pattern, r"\1", text, flags=re.IGNORECASE)

        # 2. 去除句首的引导词 (保留后面的核心内容)
        text = re.sub(
            r"^(The molecule|It) (is|appears as|appears|exists as|shows)\s+",
            "",
            text,
            flags=re.IGNORECASE
        )

        # 3. 细节清理
        text = text.strip().rstrip(".")

        return text

    formated_pred = clean_text(pred)
    formated_gold = clean_text(gold)

    return formated_pred, formated_gold


def if_format_correct(thinking_content, solution_content, task_type):
    """Check if the format of the response is correct."""
    if thinking_content == "":
        return False
    if "<think>" in solution_content or "</think>" in solution_content:
        return False
    if task_type != "PP" and task_type != "MC":
        # For RP, MG, FS, RS tasks, the solution should be a valid <SELFIES>***</SELFIES> string
        solution_content = solution_content.strip()
        # 安全检查：确保标签成对出现且唯一，防止 "Answer 1: ... Answer 2: ..." 的情况
        if solution_content.count("<SELFIES>") != 1 or solution_content.count("</SELFIES>") != 1:
            return False
        selfies_pattern = r'.*<SELFIES>.*</SELFIES>$'
        if not re.match(selfies_pattern, solution_content, re.DOTALL):
            return False
    return True


# ===== Rule-based Scoring =====

def compute_rule_score(prediction: str, reference: str, task_type: str, mae_scale: float = 1.0) -> float:
    """
    Compute rule-based score for a single sample.
    
    Args:
        prediction: Model's prediction string.
        reference: Ground truth reference string.
        task_type: Task type (RP, MG, FS, RS, PP, MC).
        mae_scale: Scale factor for MAE normalization.
    
    Returns:
        Normalized score in [0, 1].
    """
    pred, ref = prediction, reference
    try:
        if task_type == 'PP':
            eval_func = mae_score
        elif task_type == 'MC':
            eval_func = meteor_score
            pred, ref = MC_task_format(pred, ref)
        else:
            eval_func = fts_score
        
        result = eval_func(predictions=[pred], references=[ref])
        
        if task_type == "PP":
            score = result.get('score', float('inf'))
        else:
            score = result.get('score', 0.0)

        normalized_score = normalize_score_to_reward(score, task_type, mae_scale=mae_scale)
        return normalized_score
        
    except RecursionError:
        print(f'[WARNING] RecursionError in {task_type} evaluation. '
              f'Assigning score=0.0. '
              f'Prediction: {prediction[:100]}... Reference: {reference[:100]}...')
        return 0.0
    except Exception as e:
        print(f'[WARNING] Unexpected error in {task_type} evaluation: {type(e).__name__}: {str(e)}. '
              f'Assigning score=0.0. '
              f'Prediction: {prediction[:100]}... Reference: {reference[:100]}...')
        return 0.0


# ===== LLM-based Process Judging =====

async def judge_process_async(query: str, thinking: str, prediction: str, task_type: str, client: AsyncOpenAI | None = None) -> dict:
    """
    Async: Call external LLM judger to evaluate if the reasoning process is reasonable.
    
    Args:
        query: The original user question/prompt.
        thinking: The model's chain of thought reasoning.
        prediction: The model's final answer/prediction.
        task_type: The task type (FS, RP, RS, MG, PP, MC) to select appropriate rubric.
        client: AsyncOpenAI client to use.  Defaults to the first server client.
    
    Returns:
        dict: {"process_valid": bool, "judge_explain": str}
    """
    if client is None:
        client = _server_clients[0]
    rubric = rubrics.get(task_type, "")
    if not rubric:
        raise ValueError(f"Unknown task type: {task_type}")
    
    prompt = judge_template.format(
        query=query,
        thinking_content=thinking,
        answer_content=prediction,
        rubric=rubric
    )
    
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=MAX_TOKENS,
        timeout=TIMEOUT_SECONDS,
        reasoning_effort="high"
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
        process_valid = result.get("result", False)
        judge_explain = result.get("explain", "")
        return {"process_valid": process_valid, "judge_explain": judge_explain}
    except json.JSONDecodeError:
        # Fallback: try to find true/false pattern in the response
        content_lower = content.lower()
        if '"result": true' in content_lower or '"result":true' in content_lower:
            return {"process_valid": True, "judge_explain": f"Parsed from raw: {content[:200]}"}
        elif '"result": false' in content_lower or '"result":false' in content_lower:
            return {"process_valid": False, "judge_explain": f"Parsed from raw: {content[:200]}"}
        else:
            print(f"[WARNING] Cannot parse judger response: {content[:200]}")
            return {"process_valid": False, "judge_explain": f"Parse failed: {content[:200]}"}


def judge_process(query: str, thinking: str, prediction: str, task_type: str) -> dict:
    """Sync wrapper for judge_process_async."""
    return asyncio.run(judge_process_async(query, thinking, prediction, task_type))


# ===== Combined Scoring (Rule + LLM) =====

async def _score_single_item_async(
    idx: int,
    item: dict,
    semaphore: asyncio.Semaphore,
    client: AsyncOpenAI,
) -> tuple[int, dict]:
    """
    Async: Score a single item using both rule-based and LLM-based evaluation.

    Args:
        idx: Index of the item in the batch (used to maintain order).
        item: Dict with keys: prompt, reference, prediction, thinking, task_type.
        semaphore: Per-server semaphore limiting concurrent LLM calls.
        client: AsyncOpenAI client for the assigned server.

    Returns:
        tuple: (index, result_dict)
            result_dict contains:
            - "score": float (final combined reward)
            - "rule_score": float (rule-based correctness score)
            - "process_valid": bool/None (LLM judge result)
            - "judge_explain": str (explanation from LLM judge)
    """
    # Step 1: Check format
    if not if_format_correct(item["thinking"], item["prediction"], item["task_type"]):
        return idx, {
            "score": -1.0,
            "rule_score": None,
            "process_valid": None,
            "judge_explain": "Format check failed"
        }

    # Step 2: Compute rule-based score and LLM judge together inside semaphore.
    async with semaphore:
        try:
            # Compute rule-based score (CPU-bound, runs synchronously in event loop)
            rule_score = compute_rule_score(
                item["prediction"],
                item["reference"],
                item["task_type"],
            )

            # LLM judge process reasonableness (routed to the assigned server)
            judge_result = await judge_process_async(
                item["prompt"],
                item["thinking"],
                item["prediction"],
                item["task_type"],
                client=client,
            )

            process_valid = judge_result["process_valid"]
            judge_explain = judge_result["judge_explain"]

            # Step 3: Combine scores
            # If process is not valid, return -0.5 (process reward penalty)
            # If process is valid, return the rule-based score
            if process_valid:
                final_score = rule_score
            else:
                final_score = -0.5

            return idx, {
                "score": final_score,
                "rule_score": rule_score,
                "process_valid": process_valid,
                "judge_explain": judge_explain
            }

        except Exception as e:
            server_url = str(client.base_url)
            print(f'[WARNING] Unexpected error in {item["task_type"]} evaluation '
                  f'(server={server_url}): '
                  f'{type(e).__name__}: {str(e)}. Assigning score=-1.0. '
                  f'Prediction: {item["prediction"][:100]}... Reference: {item["reference"][:100]}...')
            return idx, {
                "score": -1.0,
                "rule_score": None,
                "process_valid": None,
                "judge_explain": f"Exception@{server_url}: {type(e).__name__}: {str(e)}"
            }


async def compute_score_batch_async(data_sources, solution_strs, ground_truths, extra_infos, max_workers=MAX_CONCURRENT_REQUESTS):
    """
    Compute combined scores for a batch of data using both rule-based and LLM-based evaluation.
    
    This function:
    1. Uses rule-based scoring (FTS, MAE, METEOR) to evaluate final answer correctness
    2. Uses LLM judging to evaluate reasoning process quality
    3. Combines both: only valid processes receive the rule-based score; invalid processes get -0.5
    
    Args:
        data_sources: List of data sources.
        solution_strs: List of solution strings.
        ground_truths: List of ground truth strings or {"role": xxx, "content": xxx} messages.
        extra_infos: List of extra information dictionaries containing:
            - "prompt": the original user prompt
            - "task_name": the task type (FS, RP, RS, MG, PP, MC)
        max_workers: Maximum number of concurrent LLM judge calls.

    Returns:
        List[dict]: A list of dicts for each sample, each containing:
            - "score": float (final combined reward: rule_score if process valid, else -0.5)
            - "rule_score": float or None (rule-based correctness score)
            - "process_valid": bool or None (LLM judge result: True if reasoning is reasonable)
            - "judge_explain": str (explanation from LLM judge or error message)
        
        Note: verl's BatchRewardManager will automatically extract "score" as the reward,
        and save other fields to the rollout jsonl files.
    """

    batch_data = []
    for data_source, solution_str, ground_truth, extra_info in zip(
        data_sources, solution_strs, ground_truths, extra_infos, strict=True
    ):
        thinking_str, solution_str = extract_thinking_content(solution_str)

        data = {
            "prompt": extra_info["prompt"],
            "reference": ground_truth,
            "prediction": solution_str,
            "thinking": thinking_str,
            "task_type": extra_info["task_name"],
        }
        batch_data.append(data)

    # ---- Multi-server distributed dispatch ----
    # Partition items round-robin across all servers so each server gets an
    # equal share of work (data-parallel style).
    num_servers = len(_server_clients)
    semaphores = _get_server_semaphores()

    tasks = [
        _score_single_item_async(
            idx,
            item,
            semaphore=semaphores[idx % num_servers],
            client=_server_clients[idx % num_servers],
        )
        for idx, item in enumerate(batch_data)
    ]
    results = await asyncio.gather(*tasks)

    # Sort by index to maintain order
    scores = [None] * len(batch_data)
    for idx, result in results:
        scores[idx] = result

    return scores


def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos):
    """
    Sync wrapper for compute_score_batch_async.

    For verl backward compatibility and easier usage.
    """
    return asyncio.run(compute_score_batch_async(
        data_sources, solution_strs, ground_truths, extra_infos, MAX_CONCURRENT_REQUESTS
    ))


if __name__ == "__main__":
    import os
    import argparse
    import time

    # ===== Helper Functions for Main =====

    async def test_connection() -> bool:
        """Test if all configured API endpoints are reachable."""
        all_ok = True
        for i, (url, client) in enumerate(zip(OPENAI_API_BASES, _server_clients)):
            print(f"Testing connection to server[{i}]: {url} ...")
            try:
                response = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": "Hello, respond with 'OK' only."}],
                    temperature=1.0,
                    max_tokens=32768,
                    timeout=30
                )
                content = response.choices[0].message.content
                content_str = content.strip() if content is not None else "(no content, reasoning-only response)"
                print(f"  server[{i}] OK — response: {content_str[:80]}")
            except Exception as e:
                print(f"  server[{i}] ERROR: {type(e).__name__}: {e}")
                all_ok = False
        return all_ok

    def load_processed_ids(output_path: str, skip_errors: bool = False) -> set:
        """Load already processed IDs from output file."""
        processed_ids = set()
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf8") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        if skip_errors and item.get("status") == "error":
                            continue
                        processed_ids.add(item["id"])
                    except json.JSONDecodeError:
                        continue
        return processed_ids

    def remove_errors_from_output(output_path: str) -> int:
        """Remove error records from output file. Returns number removed."""
        if not os.path.exists(output_path):
            return 0
        success_records = []
        error_count = 0
        with open(output_path, "r", encoding="utf8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if item.get("status") == "error":
                        error_count += 1
                    else:
                        success_records.append(line)
                except json.JSONDecodeError:
                    continue
        if error_count == 0:
            return 0
        temp_path = output_path + ".tmp"
        with open(temp_path, "w", encoding="utf8") as f:
            for line in success_records:
                f.write(line)
        os.replace(temp_path, output_path)
        return error_count

    async def process_batch(all_samples: list) -> list:
        """
        Process a batch of samples by directly calling compute_score_batch_async.

        Adapts jsonl fields → verl-style inputs, calls the canonical scoring function,
        then re-attaches id/status/original_score/task_type for output bookkeeping.
        """
        sample_ids  = [s[0] for s in all_samples]
        samples     = [s[1] for s in all_samples]

        # Build verl-style input lists
        data_sources  = [s.get("source_file", "") for s in samples]
        solution_strs = [s["output"] for s in samples]
        ground_truths = [s.get("gts", "") for s in samples]
        extra_infos   = [
            {"prompt": s["input"], "task_name": s.get("task_type", "RS")}
            for s in samples
        ]

        start_time = time.time()
        # Delegate entirely to the canonical batch function
        score_results = await compute_score_batch_async(
            data_sources, solution_strs, ground_truths, extra_infos,
            max_workers=args.concurrent
        )
        elapsed_total = time.time() - start_time

        # Re-attach bookkeeping fields
        results = []
        for sample_id, sample, sr in zip(sample_ids, samples, score_results):
            result = {
                "id": sample_id,
                "status": "success",
                **sr,  # score, rule_score, process_valid, judge_explain
                "original_score": sample.get("score", 0.0),
                "task_type": sample.get("task_type", "RS"),
                "elapsed_seconds": round(elapsed_total / len(samples), 2),
            }
            results.append(result)
        return results

    def print_summary(results: list):
        """Print summary statistics for mix_env (rule score + process valid)."""
        success_count = sum(1 for r in results if r.get("status") == "success")
        error_count = sum(1 for r in results if r.get("status") == "error")

        print("\n" + "=" * 60)
        print("Processing Summary (Mix Env: Rule + LLM Judge)")
        print("=" * 60)
        print(f"Total processed: {len(results)}")
        print(f"  Success: {success_count}")
        print(f"  Errors:  {error_count}")

        if error_count > 0:
            print("\nError breakdown:")
            error_types = {}
            for r in results:
                if r.get("status") == "error":
                    code = r.get("error_code", "unknown")
                    error_types[code] = error_types.get(code, 0) + 1
            for code, count in error_types.items():
                print(f"  {code}: {count}")

        successful = [r for r in results if r.get("status") == "success"]

        # score=-1.0 covers two distinct cases: genuine format errors vs LLM call exceptions.
        # Distinguish them via judge_explain content (exceptions start with "Exception@...").
        minus1_samples = [r for r in successful if r.get("score") == -1.0]
        exception_samples = [r for r in minus1_samples
                             if str(r.get("judge_explain", "")).startswith("Exception")]
        format_errors  = len(minus1_samples) - len(exception_samples)
        llm_exceptions = len(exception_samples)
        process_failed = sum(1 for r in successful if r.get("score") == -0.5)
        # rule_scored = reached rule scoring stage (process_valid True or False, but not format/exception error)
        rule_scored    = [r for r in successful if r.get("rule_score") is not None]
        truly_valid    = [r for r in rule_scored if r.get("process_valid") is True]

        total = len(successful)
        print(f"\nScore Summary ({total} samples):")
        print(f"  Format errors  (score=-1.0):  {format_errors}")
        print(f"  LLM exceptions (score=-1.0):  {llm_exceptions}")
        print(f"  Process invalid (score=-0.5): {process_failed}")
        print(f"  Process valid  (rule scored): {len(truly_valid)}")
        print(f"  [Check] {format_errors}+{llm_exceptions}+{process_failed}+{len(truly_valid)} = "
              f"{format_errors+llm_exceptions+process_failed+len(truly_valid)} (expect {total})")

        # Per-server exception breakdown (useful for diagnosing unstable servers)
        if llm_exceptions > 0:
            server_exc = {}
            for r in exception_samples:
                explain = str(r.get("judge_explain", ""))
                # Format: "Exception@<url>: TypeName: msg"  or legacy "Exception: TypeName: msg"
                if explain.startswith("Exception@"):
                    rest = explain[len("Exception@"):]   # "<url>: TypeName: msg"
                    sep = rest.find(": ")
                    url = rest[:sep] if sep != -1 else rest
                else:
                    url = "unknown"
                server_exc[url] = server_exc.get(url, 0) + 1
            print(f"\n  Exception breakdown by server:")
            for url, cnt in sorted(server_exc.items(), key=lambda x: -x[1]):
                print(f"    {url}: {cnt}")

        if rule_scored:
            rule_scores = [r["rule_score"] for r in rule_scored]
            final_scores = [r["score"] for r in rule_scored]
            process_pass = sum(1 for r in rule_scored if r.get("process_valid"))
            print(f"\n  Rule Score  — avg: {sum(rule_scores)/len(rule_scores):.4f}, "
                  f"max: {max(rule_scores):.4f}, min: {min(rule_scores):.4f}")
            print(f"  Final Score — avg: {sum(final_scores)/len(final_scores):.4f}")
            print(f"  Process valid rate: {process_pass}/{len(rule_scored)} "
                  f"= {process_pass/len(rule_scored):.2%}")

        # Per-task-type breakdown
        task_stats = {}
        for r in rule_scored:
            tt = r.get("task_type", "UNKNOWN")
            if tt not in task_stats:
                task_stats[tt] = {"rule_scores": [], "process_valid": 0, "total": 0}
            task_stats[tt]["rule_scores"].append(r["rule_score"])
            task_stats[tt]["total"] += 1
            if r.get("process_valid"):
                task_stats[tt]["process_valid"] += 1

        if task_stats:
            print("\nPer-task breakdown (valid samples):")
            for tt, stats in sorted(task_stats.items()):
                avg_rule = sum(stats["rule_scores"]) / len(stats["rule_scores"])
                pv_rate = stats["process_valid"] / stats["total"]
                print(f"  {tt}: n={stats['total']}, "
                      f"rule_score_avg={avg_rule:.4f}, "
                      f"process_valid={pv_rate:.2%}")

    # ===== Argument Parser =====
    parser = argparse.ArgumentParser(description="Mix Env (Rule + LLM Judge) for Chemistry Tasks")
    parser.add_argument(
        "--input",
        type=str,
        default="data/chemistry_test/test_samples.jsonl",
        help="Input file path (default: data/chemistry_test/test_samples.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/chemistry_test/mix_env_output.jsonl",
        help="Output file path (default: data/chemistry_test/mix_env_output.jsonl)",
    )
    parser.add_argument(
        "--servers",
        type=str,
        nargs="+",
        metavar="URL",
        default=None,
        help="Override OPENAI_API_BASES with one or more server URLs. "
             "Example: --servers http://host1:8000/v1 http://host2:8000/v1",
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=MAX_CONCURRENT_PER_SERVER,
        help=f"Max concurrent requests PER SERVER (default: {MAX_CONCURRENT_PER_SERVER})",
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompt",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show verbose output for each request",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Retry failed/error samples (removes error records from output file)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Use debug_samples.jsonl instead of test_samples.jsonl",
    )
    args = parser.parse_args()

    # Apply --servers override: replace global server list and rebuild clients/semaphores
    if args.servers:
        OPENAI_API_BASES.clear()
        OPENAI_API_BASES.extend(args.servers)
        _server_clients.clear()
        _server_clients.extend(_build_server_client(url) for url in OPENAI_API_BASES)
        _server_semaphores = None  # will be recreated lazily inside the event loop

    if args.debug:
        args.input = "data/chemistry_test/debug_samples.jsonl"
        args.output = "data/chemistry_test/debug_mix_env_output.jsonl"

    # Resolve paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.join(base_dir, "../..")
    input_path = os.path.join(project_dir, args.input) if not os.path.isabs(args.input) else args.input
    output_path = os.path.join(project_dir, args.output) if not os.path.isabs(args.output) else args.output
    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)

    print("=" * 60)
    print("Mix Env (Rule + LLM Judge) for Chemistry Tasks")
    print("=" * 60)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Model:  {MODEL_NAME}")
    print(f"Servers ({len(OPENAI_API_BASES)}):")
    for i, url in enumerate(OPENAI_API_BASES):
        print(f"  [{i}] {url}")
    print(f"Concurrent per server: {args.concurrent}")
    print(f"Effective total concurrency: {args.concurrent * len(OPENAI_API_BASES)}")
    print("=" * 60)

    async def main():
        # Test connection first
        if not await test_connection():
            print("\nOne or more connection tests failed. Please check:")
            print(f"  1. Server URLs: {OPENAI_API_BASES}")
            print(f"  2. Network connectivity")
            print(f"  3. API key validity")
            return

        # Handle retry-errors
        if args.retry_errors:
            error_count = remove_errors_from_output(output_path)
            if error_count > 0:
                print(f"\nRemoved {error_count} error records from output file for retry.")

        # Load already processed IDs
        print(f"\nLoading processed IDs from {output_path}...")
        processed_ids = load_processed_ids(output_path, skip_errors=args.retry_errors)
        print(f"Found {len(processed_ids)} already processed items.")

        # Load input samples
        print(f"Loading input data from {input_path}...")
        all_samples = []
        with open(input_path, 'r', encoding='utf8') as f:
            for i, line in enumerate(f):
                sample = json.loads(line)
                sample_id = f"sample_{i}"
                if sample_id not in processed_ids:
                    all_samples.append((sample_id, sample))

        print(f"Found {len(all_samples)} items to process.")

        if len(all_samples) == 0:
            print("All items have been processed. Nothing to do.")
            return

        # Show task type distribution
        task_counts = {}
        for _, sample in all_samples:
            task_type = sample.get("task_type", "UNKNOWN")
            task_counts[task_type] = task_counts.get(task_type, 0) + 1
        print("\nTask type distribution (pending):")
        for task, count in sorted(task_counts.items()):
            print(f"  {task}: {count}")

        # Confirm before proceeding
        if not args.yes:
            response = input(f"\nProceed with processing {len(all_samples)} items? [y/N]: ")
            if response.lower() != "y":
                print("Aborted.")
                return

        # Process items with asyncio — directly via compute_score_batch_async
        print(f"\nStarting processing with {args.concurrent} concurrent requests (asyncio)...")
        print(f"(Calling compute_score_batch_async on {len(all_samples)} samples...)")

        from tqdm import tqdm
        t_start = time.time()
        with tqdm(total=len(all_samples), desc="Processing") as pbar:
            results = await process_batch(all_samples)
            pbar.update(len(results))
        t_end = time.time()

        wall_time = t_end - t_start
        throughput = len(results) / wall_time if wall_time > 0 else float('inf')

        # Save all results to output file
        with open(output_path, 'a', encoding='utf8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        # Write per-sample verbose info to a separate file (always, not just --verbose)
        verbose_path = output_path.rsplit(".", 1)[0] + ".verbose.jsonl"
        if args.verbose:
            with open(verbose_path, 'a', encoding='utf8') as vf:
                for result in results:
                    status = result.get("status", "unknown")
                    if status == "success":
                        score = result.get("score", None)
                        vf.write(json.dumps({
                            "id": result["id"],
                            "task_type": result.get("task_type"),
                            "score": score,
                            "rule_score": result.get("rule_score"),
                            "process_valid": result.get("process_valid"),
                            "judge_explain": result.get("judge_explain"),
                            "elapsed_seconds": result.get("elapsed_seconds"),
                        }, ensure_ascii=False) + "\n")
                    else:
                        vf.write(json.dumps({
                            "id": result["id"],
                            "status": "error",
                            "error_code": result.get("error_code"),
                            "error_message": result.get("error_message"),
                        }, ensure_ascii=False) + "\n")
            print(f"\nVerbose per-sample log written to: {verbose_path}")

        print_summary(results)

        print(f"\n[Benchmark] Total wall-clock time : {wall_time:.2f}s")
        print(f"[Benchmark] Samples processed     : {len(results)}")
        print(f"[Benchmark] Throughput            : {throughput:.2f} samples/s")
        print(f"[Benchmark] Avg time per sample   : {wall_time/len(results)*1000:.1f} ms")

    asyncio.run(main())
