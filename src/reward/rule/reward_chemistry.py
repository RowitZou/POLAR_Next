import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from utils import fts_score, mae_score, meteor_score  # noqa: E402

# Score ranges and normalization info for each task type
# Format: {'range': (min, max), 'higher_is_better': bool}
SCORE_INFO = {
    'RP': {'range': (0, 100), 'higher_is_better': True, 'eval_func': 'fts_score'},   # Tanimoto similarity * 100
    'MG': {'range': (0, 100), 'higher_is_better': True, 'eval_func': 'fts_score'},   # Tanimoto similarity * 100
    'FS': {'range': (0, 100), 'higher_is_better': True, 'eval_func': 'fts_score'},   # Tanimoto similarity * 100
    'RS': {'range': (0, 100), 'higher_is_better': True, 'eval_func': 'fts_score'},   # Tanimoto similarity * 100
    'PP': {'range': (0, None), 'higher_is_better': False, 'eval_func': 'mae_score'},  # MAE, lower is better, no upper bound
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


def extract_thinking_content(text: str) -> tuple[str, str]:
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

    # 内部清洗函数，复用逻辑
    def clean_text(text):
        if not text or not isinstance(text, str):
            return ""

        # --- 步骤 1: 提取型清洗 (针对 Natural Product) ---
        # 目标：把 "The molecule ... found in X with data available." 变成 "X"
        # 逻辑：
        #   Prefix: "The molecule is a natural product found in"
        #   Capture Group (.*?): 捕获中间的所有内容 (即 Piper aequale)
        #   Suffix: "with data available."
        natural_prod_pattern = r"The molecule is a natural product found in\s+(.*?)\s+with data available\.?"

        # re.sub 的第二个参数用 r"\1" 表示：用第一个捕获组的内容替换整个匹配到的句子
        text = re.sub(natural_prod_pattern, r"\1", text, flags=re.IGNORECASE)

        # 2. 去除句首的引导词 (保留后面的核心内容)
        # 覆盖: is, appears as, appears, exists as, shows
        text = re.sub(
            r"^(The molecule|It) (is|appears as|appears|exists as|shows)\s+",
            "",
            text,
            flags=re.IGNORECASE
        )

        # 3. 细节清理
        # 去除首尾空格，去除句尾的句号 (METEOR 对标点不敏感，去掉更纯粹)
        text = text.strip().rstrip(".")

        return text

    # 分别处理 pred 和 gold
    formated_pred = clean_text(pred)
    formated_gold = clean_text(gold)

    return formated_pred, formated_gold


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


def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos):
    """Compute scores for a batch of data using the POLAR reward model for VERL.

    Args:
        data_sources: List of data sources.
        solution_strs: List of solution strings.
        ground_truths: List of ground truth strings or {"role": xxx, "content": xxx} messages.
        extra_infos: List of extra information dictionaries containing prompt_key,
            which is the  dictionary-style input prompt for policy model and POLAR.

    Returns:
        scores: A list of computed scores for each data source.
    """

    batch_data = []
    for data_source, solution_str, ground_truth, extra_info in zip(
        data_sources, solution_strs, ground_truths, extra_infos, strict=True
    ):

        thinking_str, solution_str = extract_thinking_content(solution_str)

        data = {
            "reference": ground_truth,
            "prediction": solution_str,
            "thinking": thinking_str,
            "task_type": extra_info["task_name"],
        }
        batch_data.append(data)

    scores = []
    for item in batch_data:
        if not if_format_correct(item["thinking"], item["prediction"], item["task_type"]):
            scores.append(-1.0)
            continue

        pred, ref = item["prediction"], item["reference"]
        try:
            if item["task_type"] == 'PP':
                eval_func = mae_score
            elif item["task_type"] == 'MC':
                eval_func = meteor_score
                pred, ref = MC_task_format(pred, ref)

            else:
                eval_func = fts_score
            result = eval_func(predictions=[pred], references=[ref])
            if item["task_type"] == "PP":
                score = result.get('score', float('inf'))
            else:
                score = result.get('score', 0.0)

            normalize_score = normalize_score_to_reward(
                score, item["task_type"], mae_scale=1.0)

            scores.append(normalize_score)
        except RecursionError:
            print(f'[WARNING] RecursionError in {item["task_type"]} evaluation. '
                  f'Assigning score=0.0. '
                  f'Prediction: {item["prediction"][:100]}... Reference: {item["reference"][:100]}...')
            scores.append(0.0)
        except Exception as e:
            print(f'[WARNING] Unexpected error in {item["task_type"]} evaluation: {type(e).__name__}: {str(e)}. '
                  f'Assigning score=0.0. '
                  f'Prediction: {item["prediction"][:100]}... Reference: {item["reference"][:100]}...')
            scores.append(0.0)

    return scores
