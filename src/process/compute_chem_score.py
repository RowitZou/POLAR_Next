import argparse
import os
import json
import re
import math
from glob import glob

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils import fts_score, mae_score, meteor_score


name_dict = {
    'RP': 'reagent_prediction',
    'MG': 'description_guided_molecule_design',
    'FS': 'forward_reaction_prediction',
    'RS': 'retrosynthesis',
    'PP': 'property_prediction',
    'MC': 'molecular_description_generation',
}

# Score ranges and normalization info for each task type
# Format: {'range': (min, max), 'higher_is_better': bool}
SCORE_INFO = {
    'RP': {'range': (0, 100), 'higher_is_better': True, 'eval_func': 'fts_score'},   # Tanimoto similarity * 100
    'MG': {'range': (0, 100), 'higher_is_better': True, 'eval_func': 'fts_score'},   # Tanimoto similarity * 100
    'FS': {'range': (0, 100), 'higher_is_better': True, 'eval_func': 'fts_score'},   # Tanimoto similarity * 100
    'RS': {'range': (0, 100), 'higher_is_better': True, 'eval_func': 'fts_score'},   # Tanimoto similarity * 100
    'PP': {'range': (0, None), 'higher_is_better': False, 'eval_func': 'mae_score'}, # MAE, lower is better, no upper bound
    'MC': {'range': (0, 1), 'higher_is_better': True, 'eval_func': 'meteor_score'},  # METEOR score
}


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
    
    info = SCORE_INFO[task_type]
    
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


def normalize_score_to_reward_batch(scores: list, task_type: str, mae_scale: float = 1.0) -> list:
    """
    Normalize a batch of raw scores to rewards.
    
    Args:
        scores: List of raw scores
        task_type: Task type (RP, MG, FS, RS, PP, MC)
        mae_scale: Scale factor for MAE normalization
    
    Returns:
        List of normalized rewards in [0, 1]
    """
    return [normalize_score_to_reward(s, task_type, mae_scale) for s in scores]


def extract_thinking_content(text: str, mode="qwen") -> tuple[str, str]:
    if mode == "qwen":
        if "<think>" not in text:
            text = "<think>\n" + text
        pattern = r'<think>(.*?)</think>(.*)'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            thinking_content = match.group(1).strip()
            remaining_content = match.group(2).strip()
            return thinking_content, remaining_content
        return "", text
    elif mode == "gpt":
        if "assistantfinal" not in text:
            return "", text
        thinking_content, remaining_content = text.split("assistantfinal")[0][8:], text.split("assistantfinal")[1]
        return thinking_content.strip(), remaining_content.strip()
    else:
        raise ValueError("Unsupported mode: {}".format(mode))


# compute chemical scores for a batch of samples
def compute_chem_score_batch(predictions, references, task_type):
    predictions = [extract_thinking_content(prediction, mode="qwen")[1] for prediction in predictions]
    if task_type == 'PP':
        eval_func = mae_score
    elif task_type == 'MC':
        eval_func = meteor_score
    else:
        eval_func = fts_score
    score = eval_func(predictions=predictions, references=references)
    return score


# compute chemical scores for each sample individually and aggregate
def compute_chem_score(predictions, references, task_type):
    predictions = [extract_thinking_content(prediction, mode="qwen")[1] for prediction in predictions]

    if task_type == 'PP':
        eval_func = mae_score
    elif task_type == 'MC':
        eval_func = meteor_score
    else:
        eval_func = fts_score

    all_scores = []
    all_valid = []
    details = []

    for pred, ref in zip(predictions, references):
        # Call eval function for single sample (as list of 1)
        result = eval_func(predictions=[pred], references=[ref])
        score = result.get('score', 0)
        all_scores.append(score)

        # Track valid score if available
        if 'valid_score' in result:
            all_valid.append(result['valid_score'])

        if 'details' in result and result['details']:
            details.append(result['details'][0])

    # Aggregate results
    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
    result = {
        'score': avg_score,
        'details': details,
        'individual_scores': all_scores,
    }

    if all_valid:
        result['valid_score'] = sum(all_valid) / len(all_valid)

    return result


def load_json_files_by_prefix(directory: str, prefix: str) -> dict:
    """
    Load all JSON files with the given prefix from the directory.
    Returns a combined dictionary with all predictions and references.
    """
    pattern = os.path.join(directory, f"{prefix}*.json")
    files = glob(pattern)

    all_predictions = []
    all_references = []

    for file_path in sorted(files):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Each file contains a dict with numeric string keys
            for key in sorted(data.keys(), key=lambda x: int(x)):
                item = data[key]
                if 'prediction' in item and 'gold' in item:
                    all_predictions.append(item['prediction'])
                    all_references.append(item['gold'])
        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}")
            continue

    return all_predictions, all_references


def process_directory(directory: str, output_file: str = None, mode: str = "qwen", 
                      score_method: str = "batch", compare: bool = False):
    """
    Process all files in the directory and compute scores for each task type.

    Args:
        directory: Directory containing prediction JSON files
        output_file: Output JSON file to save results (optional)
        mode: Mode for extracting thinking content ('qwen' or 'gpt')
        score_method: 'batch' for batch scoring, 'single' for individual scoring
        compare: If True, compute both methods and compare
    """
    results = {}
    results_compare = {} if compare else None

    # Get all files in the directory
    all_files = os.listdir(directory)

    # Find which task types are present
    present_task_types = set()
    for filename in all_files:
        for task_type in name_dict.keys():
            if filename.startswith(task_type + "-") or filename.startswith(task_type + "_"):
                present_task_types.add(task_type)
                break

    print(f"Found task types: {present_task_types}")
    print("-" * 60)

    for task_type in sorted(present_task_types):
        task_name = name_dict.get(task_type, task_type)
        print(f"\nProcessing {task_type} ({task_name})...")

        # Load all files for this task type
        # Try both "-" and "_" separators
        predictions, references = [], []
        for sep in ["-", "_"]:
            prefix = f"{task_type}{sep}"
            preds, refs = load_json_files_by_prefix(directory, prefix)
            predictions.extend(preds)
            references.extend(refs)

        if not predictions:
            print(f"  No data found for {task_type}")
            continue

        print(f"  Loaded {len(predictions)} samples")

        # Compute score based on selected method
        if score_method == "batch" or compare:
            score_result_batch = compute_chem_score_batch(predictions, references, task_type)
            results[task_type] = {
                'task_name': task_name,
                'num_samples': len(predictions),
                'score': score_result_batch.get('score', 0),
                'valid_score': score_result_batch.get('valid_score', None),
                'method': 'batch',
            }

        if score_method == "single" or compare:
            score_result_single = compute_chem_score(predictions, references, task_type)
            if compare:
                results_compare[task_type] = {
                    'task_name': task_name,
                    'num_samples': len(predictions),
                    'score': score_result_single.get('score', 0),
                    'valid_score': score_result_single.get('valid_score', None),
                    'method': 'single',
                }
            else:
                results[task_type] = {
                    'task_name': task_name,
                    'num_samples': len(predictions),
                    'score': score_result_single.get('score', 0),
                    'valid_score': score_result_single.get('valid_score', None),
                    'method': 'single',
                }

        # Print results
        if compare:
            print(f"  Batch Score:  {results[task_type]['score']:.4f}")
            print(f"  Single Score: {results_compare[task_type]['score']:.4f}")
            diff = results_compare[task_type]['score'] - results[task_type]['score']
            print(f"  Difference:   {diff:+.4f}")
            if results[task_type]['valid_score'] is not None:
                print(f"  Valid Score (Batch):  {results[task_type]['valid_score']:.4f}")
            if results_compare[task_type]['valid_score'] is not None:
                print(f"  Valid Score (Single): {results_compare[task_type]['valid_score']:.4f}")
        else:
            print(f"  Score: {results[task_type]['score']:.4f}")
            if results[task_type]['valid_score'] is not None:
                print(f"  Valid Score: {results[task_type]['valid_score']:.4f}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if compare:
        print(f"{'Task Type':<10} {'Task Name':<35} {'Samples':<8} {'Batch':<10} {'Single':<10} {'Diff':<10}")
        print("-" * 93)

        total_batch = 0
        total_single = 0
        total_samples = 0

        for task_type in sorted(results.keys()):
            result = results[task_type]
            result_cmp = results_compare[task_type]
            diff = result_cmp['score'] - result['score']
            print(f"{task_type:<10} {result['task_name']:<35} {result['num_samples']:<8} {result['score']:<10.4f} {result_cmp['score']:<10.4f} {diff:+.4f}")
            total_batch += result['score'] * result['num_samples']
            total_single += result_cmp['score'] * result_cmp['num_samples']
            total_samples += result['num_samples']

        if total_samples > 0:
            avg_batch = total_batch / total_samples
            avg_single = total_single / total_samples
            print("-" * 93)
            print(f"{'Average':<10} {'':<35} {total_samples:<8} {avg_batch:<10.4f} {avg_single:<10.4f} {avg_single-avg_batch:+.4f}")

        if results:
            avg_task_batch = sum(r['score'] for r in results.values()) / len(results)
            avg_task_single = sum(r['score'] for r in results_compare.values()) / len(results_compare)
            print(f"{'Avg(Task)':<10} {'':<35} {len(results)} tasks  {avg_task_batch:<10.4f} {avg_task_single:<10.4f} {avg_task_single-avg_task_batch:+.4f}")
    else:
        print(f"{'Task Type':<10} {'Task Name':<40} {'Samples':<10} {'Score':<10}")
        print("-" * 60)

        total_score = 0
        total_samples = 0

        for task_type, result in sorted(results.items()):
            print(f"{task_type:<10} {result['task_name']:<40} {result['num_samples']:<10} {result['score']:.4f}")
            total_score += result['score'] * result['num_samples']
            total_samples += result['num_samples']

        if total_samples > 0:
            avg_score = total_score / total_samples
            print("-" * 60)
            print(f"{'Average':<10} {'':<40} {total_samples:<10} {avg_score:.4f}")

        # Calculate average score across all task types (equal weight)
        if results:
            avg_task_score = sum(r['score'] for r in results.values()) / len(results)
            print(f"{'Avg(Task)':<10} {'':<40} {len(results)} tasks    {avg_task_score:.4f}")

    # Save results if output file is specified
    if output_file:
        if compare:
            output_data = {
                'directory': directory,
                'score_method': 'comparison',
                'results_batch': results,
                'results_single': results_compare,
                'summary': {
                    'total_samples': total_samples,
                    'weighted_avg_batch': total_batch / total_samples if total_samples > 0 else 0,
                    'weighted_avg_single': total_single / total_samples if total_samples > 0 else 0,
                    'task_avg_batch': avg_task_batch if results else 0,
                    'task_avg_single': avg_task_single if results_compare else 0,
                }
            }
        else:
            output_data = {
                'directory': directory,
                'score_method': score_method,
                'results': results,
                'summary': {
                    'total_samples': total_samples,
                    'weighted_avg_score': total_score / total_samples if total_samples > 0 else 0,
                    'task_avg_score': avg_task_score if results else 0,
                }
            }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_file}")

    return results, results_compare if compare else results


def main():
    parser = argparse.ArgumentParser(description='Compute chemical task scores from prediction files')
    parser.add_argument('--input_dir', '-i', type=str, required=True,
                        help='Directory containing prediction JSON files')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output JSON file to save results (optional)')
    parser.add_argument('--mode', '-m', type=str, default='qwen', choices=['qwen', 'gpt'],
                        help='Mode for extracting thinking content (default: qwen)')
    parser.add_argument('--score_method', '-s', type=str, default='batch', choices=['batch', 'single'],
                        help='Scoring method: batch (compute all at once) or single (compute each sample individually)')
    parser.add_argument('--compare', '-c', action='store_true',
                        help='Compare both batch and single scoring methods')

    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: {args.input_dir} is not a valid directory")
        return

    print(f"Processing directory: {args.input_dir}")
    if args.compare:
        print(f"Mode: Comparing batch vs single scoring")
    else:
        print(f"Scoring method: {args.score_method}")
    print("=" * 60)

    process_directory(args.input_dir, args.output, args.mode, args.score_method, args.compare)


if __name__ == '__main__':
    main()
