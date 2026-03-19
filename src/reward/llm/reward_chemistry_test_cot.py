import re
import json
import asyncio
from openai import AsyncOpenAI


# ===== Configuration =====
# OPENAI_API_BASE = "http://35.220.164.252:3888/v1"  # Replace with your API base URL
# OPENAI_API_KEY = "sk-IXpq3OfXdzZtrcISoP4ahtmzHesQyQtQnAPp0VHwOP3HJ09F"  # Replace with your API key
# MODEL_NAME = "gemini-3-pro-preview-thinking"  # Replace with your model name or ID in the OpenAI-compatible server

OPENAI_API_BASE = "http://10.102.243.64:8000/v1"
OPENAI_API_KEY = ""
MODEL_NAME = "/mnt/shared-storage-user/ailab-hs/yangyuming/models/models--openai--gpt-oss-120b/snapshots/eabf0c518da7584a2e7dab4ab272709785a72126"  # Replace with your model name

MAX_CONCURRENT_REQUESTS = 1024
TIMEOUT_SECONDS = 600
MAX_TOKENS = 32768  # Increased for thinking models (reasoning + output tokens)

# Initialize async OpenAI client with config
async_client = AsyncOpenAI(base_url=OPENAI_API_BASE, api_key=OPENAI_API_KEY)

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
- Your ONLY job is to evaluate: Output Integrity (anti-cheating).

Analyze the model's performance based strictly on the provided Rubric.
If the model MEETS ALL the criteria described in the rubric, the result must be True.
If the model FAILS ANY of the criterion, the result must be False.

**Boundary Case Handling:**
If ANY of the following boundary cases occur, the result must be **False**:
- The Model Final Answer is absent, empty, or consists only of generic filler text.

**Output Format:**
Return a single JSON object:
{{"result": true, "explain": "Brief explanation..."}} or {{"result": false, "explain": "Brief explanation..."}}"""

# Rubrics focus on: Consistency, Anti-cheating, CoT Completeness & Reasonableness
# NOTE: Accuracy is NOT evaluated here - handled by external rule-based systems
rubrics = {
    "FS": """Output must NOT be copied/appended input reactants or reagents. Must show genuine transformation attempt.""",

    "RP": """Output must NOT be copied/appended input reactants or target product. Must show genuine prediction attempt.""",

    "RS": """Output must NOT be copied/appended input product. Must show genuine retrosynthesis attempt.""",

    "MG": """Output must be a SELFIES string—not copied input natural language text or generic filler.""",

    "PP": """CoT must reference input molecule's structural features—not arbitrary number without any reasoning attempt.""",

    "MC": """Output must contain specific useful information—not generic boilerplate text or repetitive padding."""
}


def extract_thinking_content(text: str) -> tuple[str, str]:
    pattern = r'<think>(.*?)</think>(.*)'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        thinking_content = match.group(1).strip()
        remaining_content = match.group(2).strip()
        return thinking_content, remaining_content
    return "", text


def if_format_correct(thinking_content, solution_content, task_type):
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


async def judge_ans_async(query, thinking, prediction, task_type):
    """Async version: Call external LLM judger to evaluate if the model output is reasonable.
    
    Args:
        query: The original user question/prompt.
        thinking: The model's chain of thought reasoning.
        prediction: The model's final answer/prediction.
        task_type: The task type (FS, RP, RS, MG, PP, MC) to select appropriate rubric.
    
    Returns:
        dict: {"score": float, "judge_result": bool, "judge_explain": str}
    """
    # Get the rubric for this task type
    rubric = rubrics.get(task_type, "")
    if not rubric:
        raise ValueError(f"Unknown task type: {task_type}")
    
    # Format the judge prompt
    prompt = judge_template.format(
        query=query,
        thinking_content=thinking,
        answer_content=prediction,
        rubric=rubric
    )
    
    # Call OpenAI API asynchronously
    response = await async_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=MAX_TOKENS,
        timeout=TIMEOUT_SECONDS,
        reasoning_effort="high"
    )
    
    # Parse the response
    content = response.choices[0].message.content.strip()
    
    # Try to extract JSON from the response
    # Handle cases where the response might be wrapped in markdown code blocks
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
        judge_result = result.get("result", False)
        judge_explain = result.get("explain", "")
        score = 1.0 if judge_result else 0.0
        return {"score": score, "judge_result": judge_result, "judge_explain": judge_explain}
    except json.JSONDecodeError:
        # Fallback: try to find true/false pattern in the response
        content_lower = content.lower()
        if '"result": true' in content_lower or '"result":true' in content_lower:
            return {"score": 1.0, "judge_result": True, "judge_explain": f"Parsed from raw: {content[:200]}"}
        elif '"result": false' in content_lower or '"result":false' in content_lower:
            return {"score": 0.0, "judge_result": False, "judge_explain": f"Parsed from raw: {content[:200]}"}
        else:
            # Cannot parse result, return 0.0 as safe default
            print(f"[WARNING] Cannot parse judger response: {content[:200]}")
            return {"score": 0.0, "judge_result": None, "judge_explain": f"Parse failed: {content[:200]}"}


def judge_ans(query, thinking, prediction, task_type):
    """Sync wrapper for judge_ans_async. For backward compatibility."""
    return asyncio.run(judge_ans_async(query, thinking, prediction, task_type))


async def _score_single_item_async(idx: int, item: dict, semaphore: asyncio.Semaphore) -> tuple[int, dict]:
    """Async version: Score a single item. Returns (index, result_dict) to maintain order.

    Returns:
        tuple: (index, {"score": float, "judge_result": bool/None, "judge_explain": str})
    """
    if not if_format_correct(item["thinking"], item["prediction"], item["task_type"]):
        return idx, {"score": -1.0, "judge_result": None, "judge_explain": "Format check failed"}

    async with semaphore:
        try:
            result = await judge_ans_async(item["prompt"], item["thinking"], item["prediction"], item["task_type"])
            return idx, result
        except Exception as e:
            print(f'[WARNING] Unexpected error in {item["task_type"]} evaluation: {type(e).__name__}: {str(e)}. '
                  f'Assigning score=0.0. '
                  f'Prediction: {item["prediction"][:100]}... Reference: {item["reference"][:100]}...')
            return idx, {"score": 0.0, "judge_result": None, "judge_explain": f"Exception: {type(e).__name__}: {str(e)}"}


async def compute_score_batch_async(data_sources, solution_strs, ground_truths, extra_infos, max_workers=16):
    """Compute scores for a batch of data using the POLAR reward model for VERL.

    Args:
        data_sources: List of data sources.
        solution_strs: List of solution strings.
        ground_truths: List of ground truth strings or {"role": xxx, "content": xxx} messages.
        extra_infos: List of extra information dictionaries containing prompt_key,
            which is the  dictionary-style input prompt for policy model and POLAR.
        max_workers: Maximum number of concurrent threads for LLM judge calls.

    Returns:
        List[dict]: A list of dicts for each sample, each containing:
            - "score": float (the reward score, -1.0 for format error, 0.0/1.0 for judge result)
            - "judge_result": bool or None (True/False from judge, None if format error or exception)
            - "judge_explain": str (explanation from judge or error message)
        
        Note: verl's BatchRewardManager will automatically extract "score" as the reward,
        and save "judge_result" and "judge_explain" to the rollout jsonl files.
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

    # Concurrent execution with asyncio
    semaphore = asyncio.Semaphore(max_workers)
    tasks = [
        _score_single_item_async(idx, item, semaphore)
        for idx, item in enumerate(batch_data)
    ]
    results = await asyncio.gather(*tasks)
    
    # Sort by index to maintain order
    scores = [None] * len(batch_data)
    for idx, result in results:
        scores[idx] = result

    return scores


def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos, max_workers=16):
    """Sync wrapper for compute_score_batch_async. For verl backward compatibility."""
    return asyncio.run(compute_score_batch_async(data_sources, solution_strs, ground_truths, extra_infos, max_workers))


if __name__ == "__main__":
    import os
    import argparse
    import time
    from typing import Set
    import threading
    
    # ===== Helper Functions for Main =====
    
    async def test_connection() -> bool:
        """Test if the API endpoint is reachable."""
        print(f"Testing connection to {OPENAI_API_BASE}...")
        try:
            # Simple test request
            response = await async_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": "Hello, respond with 'OK' only."}],
                temperature=1.0,
                max_tokens=32768,
                timeout=30
            )
            print(response)
            content = response.choices[0].message.content.strip()
            print(f"  Connection successful! Response: {content[:50]}")
            return True
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            return False
    
    def load_processed_ids(output_path: str, skip_errors: bool = False) -> Set[str]:
        """Load already processed IDs from output file.
        
        Args:
            output_path: Path to the output file.
            skip_errors: If True, error samples will NOT be added to processed_ids,
                         allowing them to be retried.
        """
        processed_ids = set()
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf8") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        # Skip error samples if retry_errors is enabled
                        if skip_errors and item.get("status") == "error":
                            continue
                        processed_ids.add(item["id"])
                    except json.JSONDecodeError:
                        continue
        return processed_ids
    
    def remove_errors_from_output(output_path: str) -> int:
        """Remove error records from output file.
        
        Returns:
            Number of error records removed.
        """
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
        
        # Write to a temporary file first
        temp_path = output_path + ".tmp"
        with open(temp_path, "w", encoding="utf8") as f:
            for line in success_records:
                f.write(line)
        
        # Atomically replace the original file
        os.replace(temp_path, output_path)
        
        return error_count
    
    async def process_single_sample(sample_id: str, sample: dict, semaphore: asyncio.Semaphore) -> dict:
        """Process a single sample asynchronously.
        
        Returns:
            dict with id, status, and result fields
        """
        start_time = time.time()
        
        try:
            # Extract thinking and prediction
            thinking_str, prediction_str = extract_thinking_content(sample["output"])
            task_type = sample.get("task_type", "RS")
            
            # Format check
            if not if_format_correct(thinking_str, prediction_str, task_type):
                result = {
                    "id": sample_id,
                    "status": "success",
                    "score": -1.0,
                    "judge_result": None,
                    "judge_explain": "Format check failed",
                    "original_score": sample.get("score", 0.0),
                    "task_type": task_type,
                }
            else:
                # Call judge with semaphore for concurrency control
                async with semaphore:
                    judge_result = await judge_ans_async(sample["input"], thinking_str, prediction_str, task_type)
                result = {
                    "id": sample_id,
                    "status": "success",
                    "score": judge_result["score"],
                    "judge_result": judge_result["judge_result"],
                    "judge_explain": judge_result["judge_explain"],
                    "original_score": sample.get("score", 0.0),
                    "task_type": task_type,
                }
        except Exception as e:
            result = {
                "id": sample_id,
                "status": "error",
                "error_code": type(e).__name__,
                "error_message": str(e)[:500],
                "original_score": sample.get("score", 0.0),
                "task_type": sample.get("task_type", "RS"),
            }
        
        elapsed = time.time() - start_time
        result["elapsed_seconds"] = round(elapsed, 2)
        
        return result
    
    def print_summary(results: list):
        """Print summary statistics."""
        success_count = sum(1 for r in results if r.get("status") == "success")
        error_count = sum(1 for r in results if r.get("status") == "error")
        
        print("\n" + "=" * 60)
        print("Processing Summary")
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
        
        # Score summary for successful samples
        valid_scores = [r["score"] for r in results if r.get("status") == "success" and r.get("score", -2) >= 0]
        format_errors = sum(1 for r in results if r.get("status") == "success" and r.get("score", 0) < 0)
        
        print(f"\nScore Summary (successful samples):")
        print(f"  Valid judge scores: {len(valid_scores)}")
        print(f"  Format errors: {format_errors}")
        if valid_scores:
            print(f"  Average LLM Judge score: {sum(valid_scores) / len(valid_scores):.4f}")
            print(f"  Pass rate (score=1.0): {sum(1 for s in valid_scores if s == 1.0) / len(valid_scores):.2%}")
    
    # ===== Argument Parser =====
    parser = argparse.ArgumentParser(description="LLM Judge for Chemistry Tasks")
    parser.add_argument(
        "--input",
        type=str,
        default="data/chemistry_test/test_samples.jsonl",
        help="Input file path (default: data/chemistry_test/test_samples.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/chemistry_test/judge_output.jsonl",
        help="Output file path (default: data/chemistry_test/judge_output.jsonl)",
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=MAX_CONCURRENT_REQUESTS,
        help=f"Max concurrent requests (default: {MAX_CONCURRENT_REQUESTS})",
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
    
    # Handle debug mode
    if args.debug:
        args.input = "data/chemistry_test/debug_samples.jsonl"
        args.output = "data/chemistry_test/debug_judge_output.jsonl"
    
    # Resolve paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.join(base_dir, "../..")
    input_path = os.path.join(project_dir, args.input) if not os.path.isabs(args.input) else args.input
    output_path = os.path.join(project_dir, args.output) if not os.path.isabs(args.output) else args.output
    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)
    
    print("=" * 60)
    print("LLM Judge for Chemistry Tasks")
    print("=" * 60)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Model:  {MODEL_NAME}")
    print(f"Concurrent requests: {args.concurrent}")
    print("=" * 60)
    
    async def main():
        # Test connection first
        if not await test_connection():
            print("\nConnection test failed. Please check:")
            print(f"  1. API endpoint: {OPENAI_API_BASE}")
            print(f"  2. Network connectivity")
            print(f"  3. API key validity")
            return
        
        # Handle retry-errors: remove error records from output file
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
        
        # Process items with asyncio
        print(f"\nStarting processing with {args.concurrent} concurrent requests (asyncio)...")
        
        semaphore = asyncio.Semaphore(args.concurrent)
        results = []
        file_lock = threading.Lock()
        
        # Create all tasks
        async def process_with_progress(sample_id, sample, pbar):
            result = await process_single_sample(sample_id, sample, semaphore)
            results.append(result)
            # Save immediately with lock (sync write is fine for small data)
            with file_lock:
                with open(output_path, 'a', encoding='utf8') as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
            pbar.update(1)
            if args.verbose:
                status = result.get("status", "unknown")
                if status == "success":
                    print(f"  [{sample_id}] score={result.get('score', 'N/A')}, "
                          f"elapsed={result.get('elapsed_seconds', 0):.1f}s")
                else:
                    print(f"  [{sample_id}] ERROR: {result.get('error_code', 'unknown')}")
            return result
        
        # Use tqdm for progress bar
        from tqdm import tqdm
        with tqdm(total=len(all_samples), desc="Processing") as pbar:
            tasks = [
                process_with_progress(sample_id, sample, pbar)
                for sample_id, sample in all_samples
            ]
            await asyncio.gather(*tasks)
        
        # Print summary
        print_summary(results)
    
    # Run the async main
    asyncio.run(main())
