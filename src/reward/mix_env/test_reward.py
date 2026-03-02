"""
Test script for mix_env/reward_chemistry.py
Validates results against debug_judge_output.jsonl

Usage:
    python test_reward.py              # Full test (Rule + LLM judge)
    python test_reward.py --rule-only  # Only test rule-based scoring
"""
import os
import json
import sys
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def load_jsonl(path):
    """Load JSONL file."""
    data = []
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def fmt_score(val):
    """Format score nicely."""
    if val is None:
        return "N/A"
    return f"{val:.4f}"


def test_rule_only(samples, expected_dict):
    """Only test rule-based scoring (no LLM judge)."""
    from mix_env.reward_chemistry import compute_rule_score, extract_thinking_content, if_format_correct
    
    print("\n" + "=" * 60)
    print("Testing Rule-based Scoring ONLY")
    print("=" * 60)
    
    for i, sample in enumerate(samples):
        sample_id = f"sample_{i}"
        expected_item = expected_dict.get(sample_id, {})
        
        # Extract thinking and prediction
        thinking_str, solution_str = extract_thinking_content(sample["output"])
        task_type = sample["task_type"]
        
        # Check format
        format_ok = if_format_correct(thinking_str, solution_str, task_type)
        
        if not format_ok:
            rule_score = None
            print(f"\n[{sample_id}] task={task_type}")
            print(f"  Format: FAILED")
            print(f"  Rule score: N/A (format error)")
            continue
        
        # Compute rule score
        rule_score = compute_rule_score(solution_str, sample["gts"], task_type)
        old_original_score = expected_item.get("original_score")
        
        # Compare
        diff = abs(rule_score - old_original_score) if old_original_score else float('inf')
        match = "✓" if diff < 0.01 else "✗"
        
        print(f"\n[{sample_id}] task={task_type}")
        print(f"  Format: OK")
        print(f"  Rule score: new={fmt_score(rule_score)}  | old={fmt_score(old_original_score)}  {match}")


def test_full(samples, expected_dict):
    """Full test with Rule + LLM judge."""
    from mix_env.reward_chemistry import compute_score_batch
    
    # Prepare inputs for compute_score_batch
    data_sources = []
    solution_strs = []
    ground_truths = []
    extra_infos = []
    
    for i, sample in enumerate(samples):
        data_sources.append(f"sample_{i}")
        solution_strs.append(sample["output"])
        ground_truths.append(sample["gts"])
        extra_infos.append({
            "prompt": sample["input"],
            "task_name": sample["task_type"]
        })
    
    print("\n" + "=" * 60)
    print("Running compute_score_batch (Rule + LLM Judge)...")
    print("=" * 60)
    
    # Run the function (with fewer workers for debugging)
    results = compute_score_batch(
        data_sources, solution_strs, ground_truths, extra_infos,
    )
    
    print("\n" + "=" * 60)
    print("Comparing results...")
    print("=" * 60)
    
    # Compare results
    match_count = 0
    mismatch_count = 0
    
    for i, (result, sample) in enumerate(zip(results, samples)):
        sample_id = f"sample_{i}"
        expected_item = expected_dict.get(sample_id, {})
        
        # Get values
        new_score = result["score"]
        new_rule_score = result.get("rule_score")
        new_process_valid = result.get("process_valid")
        new_judge_explain = result.get("judge_explain", "")[:100]
        
        old_score = expected_item.get("score")
        old_judge_result = expected_item.get("judge_result")
        old_original_score = expected_item.get("original_score")
        
        task_type = sample["task_type"]
        
        # Check if process_valid matches judge_result from old output
        process_match = (new_process_valid == old_judge_result)
        
        print(f"\n[{sample_id}] task={task_type}")
        print(f"  Rule score:       new={fmt_score(new_rule_score):>8}  | old={fmt_score(old_original_score):>8}")
        print(f"  Process valid:    new={str(new_process_valid):>8}  | old={str(old_judge_result):>8}  {'✓' if process_match else '✗'}")
        print(f"  Final score:      new={fmt_score(new_score):>8}  | old={fmt_score(old_score):>8}")
        print(f"  Explain: {new_judge_explain}...")
        
        if process_match:
            match_count += 1
        else:
            mismatch_count += 1
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total samples: {len(samples)}")
    print(f"Process valid match: {match_count}")
    print(f"Process valid mismatch: {mismatch_count}")
    print(f"Match rate: {match_count / len(samples) * 100:.1f}%")


def compare_two_files(file_a_path, file_b_path, verbose=False):
    """Compare scores between two v2-format judge output JSONL files."""
    data_a = {item["id"]: item for item in load_jsonl(file_a_path)}
    data_b = {item["id"]: item for item in load_jsonl(file_b_path)}

    common_ids = sorted(set(data_a) & set(data_b))
    only_a = set(data_a) - set(data_b)
    only_b = set(data_b) - set(data_a)

    print(f"\n{'='*70}")
    print(f"File A: {file_a_path}")
    print(f"File B: {file_b_path}")
    print(f"{'='*70}")
    print(f"File A total: {len(data_a)}   File B total: {len(data_b)}")
    print(f"Common samples: {len(common_ids)}   Only-A: {len(only_a)}   Only-B: {len(only_b)}")

    # Per-task accumulators
    from collections import defaultdict
    task_stats = defaultdict(lambda: {
        "scores_a": [], "scores_b": [],
        "rule_a": [], "rule_b": [],
        "pv_agree": 0, "pv_total": 0,
    })

    score_diffs = []
    rule_diffs = []
    pv_agree_total = 0

    if verbose:
        print(f"\n{'='*70}")
        print(f"{'ID':<12} {'Task':<4} {'Score-A':>8} {'Score-B':>8} {'Diff':>8} {'Rule-A':>8} {'Rule-B':>8} {'PV-A':>6} {'PV-B':>6} {'PV=?':>5}")
        print("-"*70)

    for sid in common_ids:
        a = data_a[sid]
        b = data_b[sid]
        task = a.get("task_type", b.get("task_type", "?"))

        sa = a.get("score")
        sb = b.get("score")
        ra = a.get("rule_score")
        rb = b.get("rule_score")
        pva = a.get("process_valid")
        pvb = b.get("process_valid")

        diff = (sb - sa) if (sa is not None and sb is not None) else None
        rdiff = (rb - ra) if (ra is not None and rb is not None) else None
        pv_match = (pva == pvb)

        ts = task_stats[task]
        if sa is not None: ts["scores_a"].append(sa)
        if sb is not None: ts["scores_b"].append(sb)
        if ra is not None: ts["rule_a"].append(ra)
        if rb is not None: ts["rule_b"].append(rb)
        ts["pv_total"] += 1
        if pv_match: ts["pv_agree"] += 1

        if diff is not None: score_diffs.append(diff)
        if rdiff is not None: rule_diffs.append(rdiff)
        if pv_match: pv_agree_total += 1

        if verbose:
            d_str = f"{diff:+.4f}" if diff is not None else "   N/A"
            print(f"{sid:<12} {task:<4} {fmt_score(sa):>8} {fmt_score(sb):>8} {d_str:>8} "
                  f"{fmt_score(ra):>8} {fmt_score(rb):>8} {str(pva):>6} {str(pvb):>6} {'✓' if pv_match else '✗':>5}")

    # Per-task summary
    print(f"\n{'='*70}")
    print(f"{'Task':<6} {'N':>5} {'AvgScore-A':>11} {'AvgScore-B':>11} {'AvgDiff':>9} {'AvgRule-A':>10} {'AvgRule-B':>10} {'PV-Match%':>10}")
    print("-"*70)

    all_tasks = sorted(task_stats.keys())
    for task in all_tasks:
        ts = task_stats[task]
        n = ts["pv_total"]
        avg_sa = sum(ts["scores_a"]) / len(ts["scores_a"]) if ts["scores_a"] else float('nan')
        avg_sb = sum(ts["scores_b"]) / len(ts["scores_b"]) if ts["scores_b"] else float('nan')
        avg_diff = avg_sb - avg_sa if (ts["scores_a"] and ts["scores_b"]) else float('nan')
        avg_ra = sum(ts["rule_a"]) / len(ts["rule_a"]) if ts["rule_a"] else float('nan')
        avg_rb = sum(ts["rule_b"]) / len(ts["rule_b"]) if ts["rule_b"] else float('nan')
        pv_pct = ts["pv_agree"] / n * 100 if n else float('nan')
        print(f"{task:<6} {n:>5} {avg_sa:>11.4f} {avg_sb:>11.4f} {avg_diff:>+9.4f} {avg_ra:>10.4f} {avg_rb:>10.4f} {pv_pct:>9.1f}%")

    # Overall summary
    n = len(common_ids)
    overall_avg_a = sum(data_a[sid]["score"] for sid in common_ids if data_a[sid].get("score") is not None) / n
    overall_avg_b = sum(data_b[sid]["score"] for sid in common_ids if data_b[sid].get("score") is not None) / n
    avg_diff = sum(score_diffs) / len(score_diffs) if score_diffs else float('nan')
    pv_pct = pv_agree_total / n * 100 if n else float('nan')

    print(f"\n{'='*70}")
    print(f"Overall ({n} samples)")
    print(f"  Avg Score A:          {overall_avg_a:.4f}")
    print(f"  Avg Score B:          {overall_avg_b:.4f}")
    print(f"  Avg Score diff (B-A): {avg_diff:+.4f}")
    print(f"  Process-valid agree:  {pv_agree_total}/{n} = {pv_pct:.1f}%")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Test mix_env reward_chemistry.py")
    parser.add_argument("--rule-only", action="store_true", help="Only test rule-based scoring")
    parser.add_argument("--compare", action="store_true", help="Compare two v2 judge output files")
    parser.add_argument("--file-a", type=str, default="data/chemistry_test/MH_LLM_JUDGE_60_v2_samples.jsonl", help="Path to file A (v2 format)")
    parser.add_argument("--file-b", type=str, default="data/chemistry_test/v2_judge_output_gpt-oss_server_v7.jsonl", help="Path to file B (v2 format)")
    parser.add_argument("--verbose", action="store_true", help="Print per-sample comparison rows")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.join(base_dir, "../..")

    if args.compare:
        if not args.file_a or not args.file_b:
            parser.error("--compare requires --file-a and --file-b")
        fa = args.file_a if os.path.isabs(args.file_a) else os.path.join(project_dir, args.file_a)
        fb = args.file_b if os.path.isabs(args.file_b) else os.path.join(project_dir, args.file_b)
        compare_two_files(fa, fb, verbose=args.verbose)
        return

    # Paths
    samples_path = os.path.join(project_dir, "data/chemistry_test/debug_samples.jsonl")
    expected_path = os.path.join(project_dir, "data/chemistry_test/debug_judge_output.jsonl")
    
    print(f"Loading samples from: {samples_path}")
    print(f"Loading expected from: {expected_path}")
    
    # Load data
    samples = load_jsonl(samples_path)
    expected = load_jsonl(expected_path)
    
    print(f"Loaded {len(samples)} samples, {len(expected)} expected results")
    
    # Build expected dict by id
    expected_dict = {item["id"]: item for item in expected}
    
    if args.rule_only:
        test_rule_only(samples, expected_dict)
    else:
        test_full(samples, expected_dict)


if __name__ == "__main__":
    main()
