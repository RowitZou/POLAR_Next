#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SEED Server Benchmark Script

This script benchmarks the SEED server by:
1. Loading trajectory data from rollout files
2. Sending requests to the SEED server using the same flow as reward_seed_remote
3. Comparing computed scores with the ground truth scores from files
4. Reporting accuracy and performance metrics

Usage:
    python benchmark_seed_server.py --data-dir /path/to/rollout --batch-size 512 --num-batches 10
"""

import os
import sys
import json
import argparse
import time
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import statistics

# Add project root to path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_current_dir)
_project_root = os.path.dirname(_src_dir)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from utils.seed_client import SEEDClient


# ============================================================================
# Data parsing functions (same as reward_seed_remote.py)
# ============================================================================

def remove_boxed(s):
    if s is None:
        return None
    if "\\boxed " in s:
        left = "\\boxed "
        try:
            assert s[: len(left)] == left
        except:
            return None
        return s[len(left):]

    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
    except:
        return None

    return s[len(left): -1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    retval = None if right_brace_idx is None else string[idx: right_brace_idx + 1]
    return retval


def extract_solution(solution_str):
    ans = remove_boxed(last_boxed_only_string(solution_str))
    return ans.strip() if ans is not None else None


def extract_thinking_content(text: str) -> Tuple[str, str]:
    pattern = r'<think>(.*?)</think>(.*)'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        thinking_content = match.group(1).strip()
        remaining_content = match.group(2).strip()
        return thinking_content, remaining_content
    return "", text


@dataclass
class TestSample:
    """A single test sample from trajectory data."""
    file_path: str
    reference: str  # gts field
    raw_output: str  # Full model output
    extracted_output: Optional[str]  # Extracted from boxed
    expected_score: float  # score field from file
    step: int


def load_samples_from_file(file_path: str) -> List[TestSample]:
    """Load test samples from a jsonl file."""
    samples = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                
                # Get reference (gts field)
                reference = data.get('gts', '')
                
                # Get the raw output from 'output' field
                raw_output = data.get('output', '')
                
                # Extract solution using the same logic as reward_seed_remote
                _, remaining = extract_thinking_content(raw_output)
                extracted = extract_solution(remaining)
                
                expected_score = data.get('score', 0.0)
                step = data.get('step', 0)
                
                samples.append(TestSample(
                    file_path=file_path,
                    reference=reference,
                    raw_output=raw_output,
                    extracted_output=extracted,
                    expected_score=expected_score,
                    step=step
                ))
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
    
    return samples


def load_all_samples(
    data_dir: str, 
    max_files: Optional[int] = None,
    max_samples: Optional[int] = None
) -> List[TestSample]:
    """Load samples from jsonl files in directory.
    
    Args:
        data_dir: Directory containing jsonl files
        max_files: Maximum number of files to load (default: all)
        max_samples: Maximum total samples to load (default: all)
    """
    data_path = Path(data_dir)
    files = sorted(data_path.glob("*.jsonl"), key=lambda x: int(x.stem) if x.stem.isdigit() else 0)
    
    if max_files:
        files = files[:max_files]
    
    all_samples = []
    for f in files:
        samples = load_samples_from_file(str(f))
        all_samples.extend(samples)
        
        # Early stop if we have enough samples
        if max_samples and len(all_samples) >= max_samples:
            all_samples = all_samples[:max_samples]
            break
    
    print(f"Loaded {len(all_samples)} samples from {min(len(files), max_files or len(files))} files")
    return all_samples


def benchmark_batch(
    client: SEEDClient,
    samples: List[TestSample],
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Benchmark a batch of samples.
    
    Returns metrics including:
    - accuracy: How many scores match expected
    - timing: Request latency
    - errors: Any errors encountered
    """
    references = []
    outputs = []
    expected_scores = []
    valid_indices = []
    
    for i, sample in enumerate(samples):
        if sample.extracted_output is not None:
            references.append(sample.reference)
            outputs.append(sample.extracted_output)
            expected_scores.append(sample.expected_score)
            valid_indices.append(i)
        else:
            # Invalid output, should get score 0
            expected_scores.append(sample.expected_score)
    
    if not references:
        return {
            'num_samples': len(samples),
            'num_valid': 0,
            'latency_ms': 0,
            'exact_matches': 0,
            'close_matches': 0,
            'errors': ['No valid samples in batch']
        }
    
    # Call SEED server
    start_time = time.time()
    result = client.compute_batch(
        references=references,
        outputs=outputs,
        types=["Expression"] * len(references)
    )
    latency_ms = (time.time() - start_time) * 1000
    
    if result is None:
        return {
            'num_samples': len(samples),
            'num_valid': len(references),
            'latency_ms': latency_ms,
            'exact_matches': 0,
            'close_matches': 0,
            'errors': ['Server request failed']
        }
    
    # Compare scores
    server_scores = result.get('scores', [])
    exact_matches = 0
    close_matches = 0  # Within 0.1 tolerance
    score_diffs = []
    mismatches = []
    
    for i, (idx, expected) in enumerate(zip(valid_indices, [expected_scores[j] for j in valid_indices])):
        if i >= len(server_scores):
            continue
        
        computed = server_scores[i]
        diff = abs(computed - expected)
        score_diffs.append(diff)
        
        if diff < 0.001:  # Exact match (floating point tolerance)
            exact_matches += 1
        if diff < 0.1:
            close_matches += 1
        else:
            mismatches.append({
                'sample_idx': idx,
                'expected': expected,
                'computed': computed,
                'diff': diff,
                'reference': references[i][:100] if i < len(references) else 'N/A',
                'output': outputs[i][:100] if i < len(outputs) else 'N/A'
            })
    
    return {
        'num_samples': len(samples),
        'num_valid': len(references),
        'latency_ms': latency_ms,
        'exact_matches': exact_matches,
        'close_matches': close_matches,
        'score_diffs': score_diffs,
        'mismatches': mismatches[:5] if verbose else [],  # Only show first 5 mismatches
        'errors': result.get('errors', [])
    }


def run_benchmark(
    data_dir: str,
    server_address: str,
    batch_size: int = 512,
    num_batches: Optional[int] = None,
    max_files: Optional[int] = None,
    timeout: float = 600.0,
    verbose: bool = False
):
    """Run the full benchmark."""
    print(f"=" * 60)
    print(f"SEED Server Benchmark")
    print(f"=" * 60)
    print(f"Server: {server_address}")
    print(f"Data directory: {data_dir}")
    print(f"Batch size: {batch_size}")
    print(f"Max files: {max_files or 'all'}")
    print(f"Timeout: {timeout}s")
    print()
    
    # Create client
    client = SEEDClient(server_address=server_address, timeout=timeout)
    
    # Health check
    print("Checking server health...")
    if not client.health_check():
        print("ERROR: Server health check failed!")
        return
    print("Server is healthy")
    print()
    
    # Load samples - limit based on max_files or num_batches
    print("Loading samples...")
    max_samples = num_batches * batch_size if num_batches else None
    samples = load_all_samples(data_dir, max_files=max_files, max_samples=max_samples)
    
    if not samples:
        print("ERROR: No samples loaded!")
        return
    
    # Split into batches
    batches = []
    for i in range(0, len(samples), batch_size):
        batches.append(samples[i:i + batch_size])
    
    if num_batches:
        batches = batches[:num_batches]
    
    print(f"Running {len(batches)} batches...")
    print()
    
    # Run benchmark
    all_results = []
    total_exact = 0
    total_close = 0
    total_valid = 0
    total_samples = 0
    latencies = []
    all_diffs = []
    
    for batch_idx, batch in enumerate(batches):
        print(f"Batch {batch_idx + 1}/{len(batches)} ({len(batch)} samples)...", end=" ")
        sys.stdout.flush()
        
        result = benchmark_batch(client, batch, verbose=verbose)
        all_results.append(result)
        
        total_exact += result['exact_matches']
        total_close += result['close_matches']
        total_valid += result['num_valid']
        total_samples += result['num_samples']
        latencies.append(result['latency_ms'])
        all_diffs.extend(result.get('score_diffs', []))
        
        print(f"Latency: {result['latency_ms']:.0f}ms, "
              f"Exact: {result['exact_matches']}/{result['num_valid']}, "
              f"Close: {result['close_matches']}/{result['num_valid']}")
        
        if verbose and result.get('mismatches'):
            print("  Mismatches:")
            for m in result['mismatches']:
                print(f"    Expected: {m['expected']:.4f}, Got: {m['computed']:.4f}, Diff: {m['diff']:.4f}")
                print(f"    Ref: {m['reference'][:50]}...")
                print(f"    Out: {m['output'][:50]}...")
    
    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total samples:     {total_samples}")
    print(f"Valid samples:     {total_valid}")
    print(f"Exact matches:     {total_exact} ({100.0 * total_exact / max(total_valid, 1):.2f}%)")
    print(f"Close matches:     {total_close} ({100.0 * total_close / max(total_valid, 1):.2f}%)")
    print()
    print("Latency Statistics:")
    print(f"  Min:    {min(latencies):.0f}ms")
    print(f"  Max:    {max(latencies):.0f}ms")
    print(f"  Mean:   {statistics.mean(latencies):.0f}ms")
    print(f"  Median: {statistics.median(latencies):.0f}ms")
    if len(latencies) > 1:
        print(f"  Stdev:  {statistics.stdev(latencies):.0f}ms")
    print(f"  Total:  {sum(latencies) / 1000:.1f}s")
    print()
    
    if all_diffs:
        print("Score Difference Statistics:")
        print(f"  Min diff:    {min(all_diffs):.6f}")
        print(f"  Max diff:    {max(all_diffs):.6f}")
        print(f"  Mean diff:   {statistics.mean(all_diffs):.6f}")
        print(f"  Median diff: {statistics.median(all_diffs):.6f}")
    
    # Check server stats
    print()
    print("Server Stats:")
    stats = client.get_stats()
    if stats:
        for k, v in stats.items():
            print(f"  {k}: {v}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark SEED Server")
    parser.add_argument(
        "--data-dir", "-d",
        type=str,
        default="/mnt/shared-storage-user/ailab-hs/zouyicheng/POLAR_Next/outputs/verl_grpo_policy_Qwen3-8B_reward_SEED_data_cmphysbench/trajectory_data/rollout",
        help="Directory containing trajectory jsonl files"
    )
    parser.add_argument(
        "--server", "-s",
        type=str,
        default=os.environ.get('SEED_SERVER_ADDRESS', '10.102.243.64:30030'),
        help="SEED server address (host:port)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=512,
        help="Batch size for requests"
    )
    parser.add_argument(
        "--num-batches", "-n",
        type=int,
        default=None,
        help="Number of batches to run (default: all)"
    )
    parser.add_argument(
        "--max-files", "-f",
        type=int,
        default=None,
        help="Maximum number of files to load (default: all). Use -f 1 for quick testing."
    )
    parser.add_argument(
        "--timeout", "-t",
        type=float,
        default=600.0,
        help="Timeout for each request in seconds"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed mismatch information"
    )
    
    args = parser.parse_args()
    
    run_benchmark(
        data_dir=args.data_dir,
        server_address=args.server,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        max_files=args.max_files,
        timeout=args.timeout,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
