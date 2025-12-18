#!/usr/bin/env python3
"""
Trajectory Score Visualization Script

This script analyzes the evaluation results from trajectory data files.
Each file represents model evaluation at a specific training step.
Each file contains 100 test cases, each with 8 samples.

The script computes:
- Mean and std of average pass-8 scores
- Mean and std of max pass-8 scores  
- Mean and std of min pass-8 scores

And generates 6 plots showing the trends across training steps.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
import argparse


def load_jsonl_file(file_path: str) -> list:
    """Load a JSONL file and return list of records."""
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def group_by_input(records: list) -> dict:
    """
    Group records by their 'input' field.
    Returns a dict: {input_text: [reward1, reward2, ...]}
    """
    groups = defaultdict(list)
    for record in records:
        input_text = record.get('input', '')
        reward = record.get('reward', 0)
        groups[input_text].append(reward)
    return groups


def compute_pass_k_stats(groups: dict) -> dict:
    """
    Compute pass-k statistics for each test case.
    
    For each test case (group of samples with same input):
    - average: mean of all sample scores
    - max: maximum of all sample scores
    - min: minimum of all sample scores
    - inner_std: std within the samples (measures sample consistency)
    
    Note: The number of samples per test case is automatically detected
    from the data, no need to specify k.
    
    Returns dict with 'avg', 'max', 'min', 'inner_std' lists and 'samples_per_case' info.
    """
    avg_scores = []
    max_scores = []
    min_scores = []
    inner_stds = []
    sample_counts = []  # 记录每个测试例的样本数
    
    for input_text, rewards in groups.items():
        if rewards:
            avg_scores.append(np.mean(rewards))
            max_scores.append(np.max(rewards))
            min_scores.append(np.min(rewards))
            inner_stds.append(np.std(rewards))
            sample_counts.append(len(rewards))
    
    return {
        'avg': avg_scores,
        'max': max_scores,
        'min': min_scores,
        'inner_std': inner_stds,
        'sample_counts': sample_counts
    }


def analyze_score_directory(score_dir: str) -> dict:
    """
    Analyze all JSONL files in the score directory.
    
    Returns dict: {step: {metric: {'mean': float, 'std': float}}}
    """
    results = {}
    
    # Get all jsonl files
    score_path = Path(score_dir)
    jsonl_files = list(score_path.glob("*.jsonl"))
    
    for jsonl_file in jsonl_files:
        # Extract step number from filename (e.g., "400.jsonl" -> 400)
        try:
            step = int(jsonl_file.stem)
        except ValueError:
            print(f"Warning: Could not parse step from filename: {jsonl_file.name}")
            continue
        
        # Load records
        records = load_jsonl_file(str(jsonl_file))
        
        # Group by input
        groups = group_by_input(records)
        
        # Compute pass-k stats (k is auto-detected from data)
        stats = compute_pass_k_stats(groups)
        
        # Get samples per case info
        avg_samples = np.mean(stats['sample_counts']) if stats['sample_counts'] else 0
        
        # Compute mean and std for each metric
        results[step] = {
            'avg_mean': np.mean(stats['avg']) if stats['avg'] else 0,
            'max_mean': np.mean(stats['max']) if stats['max'] else 0,
            'min_mean': np.mean(stats['min']) if stats['min'] else 0,
            'inner_std_mean': np.mean(stats['inner_std']) if stats['inner_std'] else 0,
            'num_test_cases': len(stats['avg']),
            'avg_samples_per_case': avg_samples
        }
    
    return results


def plot_metrics(results: dict, output_dir: str, title_prefix: str = ""):
    """
    Generate 6 plots for the metrics.
    
    Plots:
    1. Average pass-8 score mean
    2. Average pass-8 score std
    3. Max pass-8 score mean
    4. Max pass-8 score std
    5. Min pass-8 score mean
    6. Min pass-8 score std
    """
    # Sort steps
    steps = sorted(results.keys())
    
    # Extract metrics
    avg_means = [results[s]['avg_mean'] for s in steps]
    max_means = [results[s]['max_mean'] for s in steps]
    min_means = [results[s]['min_mean'] for s in steps]
    inner_std_means = [results[s]['inner_std_mean'] for s in steps]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot configurations - 4个核心指标
    plot_configs = [
        ('avg_mean', avg_means, 'Average Pass-8 Score', 'tab:blue'),
        ('max_mean', max_means, 'Max Pass-8 Score', 'tab:green'),
        ('min_mean', min_means, 'Min Pass-8 Score', 'tab:red'),
        ('inner_std_mean', inner_std_means, 'Inner Std (Sample Consistency)', 'tab:purple'),
    ]
    
    # Generate individual plots
    for metric_name, values, title, color in plot_configs:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(steps, values, marker='o', color=color, linewidth=2, markersize=4)
        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'{title_prefix}{title}', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(axis='both', labelsize=10)
        
        # Add trend line
        if len(steps) > 2:
            z = np.polyfit(steps, values, 1)
            p = np.poly1d(z)
            ax.plot(steps, p(steps), "--", color='gray', alpha=0.5, label='Trend')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{metric_name}.png'), dpi=150)
        plt.close()
    
    # Generate combined plot (2x2 grid for 4 metrics)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()  # Flatten to 1D array for easier indexing
    
    for idx, (metric_name, values, title, color) in enumerate(plot_configs):
        ax = axes[idx]
        
        ax.plot(steps, values, marker='o', color=color, linewidth=2, markersize=4)
        ax.set_xlabel('Training Step', fontsize=10)
        ax.set_ylabel('Score', fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(axis='both', labelsize=9)
    
    plt.suptitle(f'{title_prefix}Trajectory Score Analysis', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_metrics.png'), dpi=150)
    plt.close()
    

    
    print(f"Plots saved to: {output_dir}")


def print_summary(results: dict):
    """Print a summary table of the results."""
    print("\n" + "="*90)
    print("Summary of Trajectory Scores")
    print("="*90)
    print(f"{'Step':<8} {'#Cases':<8} {'#Samples':<10} {'Avg Mean':<12} {'Max Mean':<12} {'Min Mean':<12} {'Inner Std':<12}")
    print("-"*90)
    
    for step in sorted(results.keys()):
        r = results[step]
        print(f"{step:<8} {r['num_test_cases']:<8} {r['avg_samples_per_case']:<10.1f} "
              f"{r['avg_mean']:<12.2f} {r['max_mean']:<12.2f} "
              f"{r['min_mean']:<12.2f} {r['inner_std_mean']:<12.2f}")
    
    print("="*90)
    print("\nNote:")
    print("  - #Cases: Number of unique test cases (grouped by 'input' field)")
    print("  - #Samples: Average number of samples per test case (auto-detected)")
    print("  - Avg/Max/Min Mean: Mean of average/max/min scores across all test cases")
    print("  - Inner Std: Mean standard deviation within each test case's samples")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize trajectory scores from evaluation results.')
    parser.add_argument(
        '--score_dir', 
        type=str,
        default='outputs/verl_opd_policy_Qwen3-8B_Genral_OPD_reward_ZERO_ref_Qwen3-30B-A3B_data_cmphysbench_lr_1e-6/trajectory_data/score',
        help='Path to the score directory containing JSONL files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/verl_opd_policy_Qwen3-8B_Genral_OPD_reward_ZERO_ref_Qwen3-30B-A3B_data_cmphysbench_lr_1e-6/trajectory_data/score_visualization',
        help='Path to save the output plots'
    )
    parser.add_argument(
        '--title_prefix',
        type=str,
        default='GRPO Qwen3-30B-A3B: ',
        help='Prefix for plot titles'
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = Path(__file__).parent.parent.parent  # Go up to POLAR_Next root
    score_dir = script_dir / args.score_dir
    output_dir = script_dir / args.output_dir
    
    # Check if score directory exists
    if not score_dir.exists():
        print(f"Error: Score directory not found: {score_dir}")
        return
    
    print(f"Analyzing scores from: {score_dir}")
    print(f"Output will be saved to: {output_dir}")
    
    # Analyze
    results = analyze_score_directory(str(score_dir))
    
    if not results:
        print("No valid JSONL files found in the score directory.")
        return
    
    # Print summary
    print_summary(results)
    
    # Generate plots
    plot_metrics(results, str(output_dir), args.title_prefix)
    
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
