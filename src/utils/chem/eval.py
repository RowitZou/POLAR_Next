# flake8: noqa: W605
import re

from nltk.translate.meteor_score import meteor_score as nltk_meteor_score


def extract_chem_tag(text, tag):
    pattern = re.compile(rf'<({tag})>(.*?)</\1>', re.DOTALL)
    matches = pattern.findall(text)
    if not matches:
        return None, None
    # 返回最后一个匹配的类型和内容
    last_match = matches[-1]
    return last_match[0], last_match[1].strip()  # (类型, 内容)


def fts_score(predictions, references, tag='SELFIES'):
    if len(predictions) != len(references):
        return {
            'error': 'predictions and references have different '
            'length'
        }

    valid_cnt = 0
    details = []
    for ori_pred, ori_ans in zip(predictions, references):
        pred = extract_chem_tag(ori_pred, tag)
        ans = extract_chem_tag(ori_ans, tag)
        if not pred[1]:
            pred = ('SMILES', ori_pred)
        if not ans[1]:
            ans = ('SMILES', ori_ans)
        detail = {'pred': pred[1], 'answer': ans[1]}
        # 将 SMILES 转换为 RDKit 分子对象
        if pred[0] == 'SELFIES':
            try:
                import selfies as sf
                pred = sf.decoder(pred[1])
                ans = sf.decoder(ans[1])
            except:
                detail['score'] = 0
                details.append(detail)
                continue
        else:
            pred = pred[1]
            ans = ans[1]
        from rdkit import Chem
        mol1 = Chem.MolFromSmiles(pred)
        mol2 = Chem.MolFromSmiles(ans)
        if mol1 is None or mol2 is None:
            detail['score'] = 0
            details.append(detail)
            continue
        valid_cnt += 1
        # 生成 Morgan 指纹（等同于 ECFP4）
        from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
        generator = GetMorganGenerator(radius=2, fpSize=2048)
        fp1 = generator.GetFingerprint(mol1)
        fp2 = generator.GetFingerprint(mol2)
        from rdkit.Chem import DataStructs
        similarity = DataStructs.TanimotoSimilarity(fp1, fp2) * 100
        detail['score'] = similarity
        details.append(detail)

    score = sum(detail['score'] for detail in details) / len(predictions)
    valid_score = valid_cnt / len(predictions) * 100

    return {'score': score, 'valid_score': valid_score, 'details': details}


def extract_number(text):
    pattern = re.compile(
        r'(?:<NUMBER>\s*|\\boxed\{)\s*(-?\d*\.?\d+)\s*(?:</NUMBER>|\})')
    matches = pattern.findall(text)
    if not matches:
        return None
    return [float(match) for match in matches][-1]


def mae_score(predictions, references, default_mae_on_failure=float('inf')):
    """
    Compute Mean Absolute Error (MAE) for predictions vs references.
    
    Args:
        predictions: List of prediction strings containing numbers in <NUMBER> or \\boxed{} format
        references: List of ground truth values
        default_mae_on_failure: MAE value to assign when number extraction fails.
                               Default is inf (worst possible score).
                               Set to None to use |reference| as fallback (legacy behavior).
    
    Returns:
        Dictionary with 'score' (average MAE), 'details' (per-sample info)
    """
    if len(predictions) != len(references):
        return {
            'error': 'predictions and references have different '
            'length'
        }

    extracted_preds = [
        extract_number(prediction) for prediction in predictions
    ]

    details = []
    for pred, ans in zip(extracted_preds, references):
        ans_float = float(ans)
        detail = {'answer': ans_float}
        
        if pred is None:
            # Extraction failed - assign penalty score
            detail['pred'] = None
            detail['extracted'] = False
            if default_mae_on_failure is None:
                # Legacy behavior: treat as pred=0, MAE=|ans|
                detail['score'] = abs(ans_float)
            else:
                detail['score'] = default_mae_on_failure
        else:
            detail['pred'] = pred
            detail['extracted'] = True
            detail['score'] = abs(pred - ans_float)
        
        details.append(detail)

    # For averaging, treat inf as a large number or handle separately
    finite_scores = [d['score'] for d in details if d['score'] != float('inf')]
    inf_count = sum(1 for d in details if d['score'] == float('inf'))
    
    if inf_count == len(details):
        # All failed extraction
        score = float('inf')
    elif inf_count > 0:
        # Some failed: average finite scores, but mark overall as degraded
        score = sum(finite_scores) / len(finite_scores) if finite_scores else float('inf')
        # Alternatively, could use: score = float('inf') to be strict
    else:
        score = sum(d['score'] for d in details) / len(details)

    return {'score': score, 'details': details}


def _compute_length_penalty(pred_len, ref_len, threshold=10.0, steepness=2.0):
    """
    Compute length penalty for overly long predictions.
    
    The penalty follows a soft-step function:
    - When pred_len/ref_len < threshold: penalty ≈ 0 (almost no penalty)
    - When pred_len/ref_len > threshold: penalty increases rapidly
    - When pred_len/ref_len >> threshold: penalty → 1.0
    
    Args:
        pred_len: Length of prediction (word count)
        ref_len: Length of reference (word count)
        threshold: Length ratio threshold before penalty kicks in (default: 10.0)
        steepness: How sharply the penalty increases after threshold (default: 2.0)
    
    Returns:
        penalty: Value in [0, 1], where 0 means no penalty, 1 means full penalty
    
    Examples (threshold=10.0, steepness=2.0):
        ratio=1.0 → penalty≈0.000045 (negligible)
        ratio=5.0 → penalty≈0.0067 (very small)
        ratio=10.0 → penalty=0.5 (moderate)
        ratio=15.0 → penalty≈0.9933 (high)
        ratio=20.0 → penalty≈0.99995 (almost full)
    """
    import math
    if ref_len == 0:
        return 1.0 if pred_len > 0 else 0.0
    
    ratio = pred_len / ref_len
    # Sigmoid-based soft-step: 1 / (1 + exp(-steepness * (ratio - threshold)))
    penalty = 1.0 / (1.0 + math.exp(-steepness * (ratio - threshold)))
    return penalty


def meteor_score(predictions, references, length_penalty_threshold=10.0, length_penalty_steepness=2.0):
    """
    Compute METEOR score with length penalty to prevent reward hacking.
    
    Args:
        predictions: List of predicted strings
        references: List of reference strings
        length_penalty_threshold: Length ratio above which penalty starts (default: 10.0)
        length_penalty_steepness: How sharply penalty increases (default: 2.0)
    
    Returns:
        dict with 'score' and 'details'
    """
    if len(predictions) != len(references):
        return {
            'error': 'predictions and references have different '
            'length'
        }
    avg_score = 0
    details = []
    for pred, ans in zip(predictions, references):
        try:
            raw_score = (nltk_meteor_score([ans.split()], pred.split())
                        if ans and pred else 0.0)
            
            # Apply length penalty
            pred_len = len(pred.split()) if pred else 0
            ref_len = len(ans.split()) if ans else 1
            penalty = _compute_length_penalty(
                pred_len, ref_len, 
                threshold=length_penalty_threshold, 
                steepness=length_penalty_steepness
            )
            score = raw_score * (1.0 - penalty)
            
        except AttributeError:
            print(f'Failed to compute METEOR'
                    f"score:\npred='{pred}'\nans='{ans}'")
            score = 0.0
        except RecursionError:
            print(f'RecursionError in METEOR score (likely caused by special characters or extremely long tokens). '
                    f'Assigning score=0.0.\npred=\'{pred[:100]}...\' (truncated)\nans=\'{ans[:100]}...\' (truncated)')
            score = 0.0
        except Exception as e:
            print(f'Unexpected error in METEOR score: {type(e).__name__}: {str(e)}. '
                    f'Assigning score=0.0.\npred=\'{pred[:100]}...\' (truncated)\nans=\'{ans[:100]}...\' (truncated)')
            score = 0.0
        avg_score += score
        detail = {'pred': pred, 'answer': ans, 'score': score, 'length_ratio': len(pred.split()) / max(len(ans.split()), 1) if pred and ans else 0}
        details.append(detail)

    score = avg_score / len(predictions)

    return {'score': score, 'details': details}
