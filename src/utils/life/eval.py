# flake8: noqa: E501
"""
Life (Biology) task evaluation functions for RL reward computation.

Adapted from OpenCompass biodata.py evaluators.
Each scoring function works on lists (can be length 1 for per-sample RL reward).
No OpenCompass dependencies required.

Evaluation metrics and their per-sample scoring strategies:
  - MCC  → per-sample accuracy (1 if correct, 0 if wrong)
  - PCC  → per-sample inverse error across dict keys
  - Spearman → per-sample inverse error
  - R2   → per-sample inverse error
  - AUC  → per-sample label-set F1
  - Acc  → per-sample exact match accuracy
  - Fmax → per-sample F1 via count_f1_max
  - Mixed → per-sample mixed_score (MAE + Range-MAE + F1)
"""

import ast
import json
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    matthews_corrcoef,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ═════════════════════════════════════════════════════════════════════════════
# Text extraction helpers (identical to OpenCompass biodata.py)
# ═════════════════════════════════════════════════════════════════════════════


def extract_boxed_text(text):
    """Extract content from ``\\boxed{...}``.  Returns cleaned last match."""
    pattern = re.compile(r'\\boxed\{((?:[^{}]|{[^{}]*})*)\}', re.DOTALL)
    matches = pattern.findall(text)
    if not matches:
        return None
    boxed_content = matches[-1].strip()
    clean_content = re.sub(r'\\text\{([^}]*)\}', r'\1', boxed_content)
    clean_content = re.sub(r'\\(.)', r'\1', clean_content)
    return clean_content


def extract_number(text):
    """Extract number from ``\\boxed{}`` or ``<NUMBER>`` tags."""
    pattern = re.compile(
        r'(?:<NUMBER>\s*|\\boxed\{)\s*(-?\d*\.?\d+)\s*(?:</NUMBER>|\})')
    matches = pattern.findall(text)
    if not matches:
        return None
    return [float(match) for match in matches][-1]


def extract_dict_text(text):
    """Extract the last JSON dict string from *text*."""
    pattern = re.compile(r'\{[^{}]*\}', re.DOTALL)
    matches = pattern.findall(text)
    if not matches:
        return None
    last = matches[-1]
    if last == '{...}':
        return None
    return last


# ═════════════════════════════════════════════════════════════════════════════
# EC number helpers
# ═════════════════════════════════════════════════════════════════════════════

EC_PATTERN = re.compile(r'^ec(\d+|-)\.(\d+|-)\.(\d+|-)\.(\d+|-)$')


def dedup_ec_codes(ec_number_list):
    """Remove specific EC codes when a more general version is present."""
    normalized = [c.strip().lower() for c in ec_number_list]
    remaining = set(normalized)
    for code in normalized:
        if code not in remaining:
            continue
        m = EC_PATTERN.match(code)
        if not m:
            continue
        parts = list(m.groups())
        for i in range(3, 0, -1):
            generalized = parts[:i] + ['-'] * (4 - i)
            gen_code = 'ec' + '.'.join(generalized)
            if gen_code in remaining and gen_code != code:
                remaining.discard(code)
                break
    return [c for c in normalized if c in remaining]


# ═════════════════════════════════════════════════════════════════════════════
# Predefined constants
# ═════════════════════════════════════════════════════════════════════════════

AUC_PREDEFINED_LABELS = [
    'atoi', 'm6a', 'none', 'm1a', 'm5c', 'm5u', 'm6am', 'm7g',
    'cm', 'am', 'gm', 'um', 'psi',
]

EC_LABELS = [
    '1.4.3.-', '4.2.1.1', '2.7.6.-', '3.5.1.88', '5.4.99.-',
    '3.1.21.4', '2.3.1.48', '7.2.1.-', '1.16.3.1', '3.4.19.12',
    '1.3.8.-', '2.7.7.19', '2.4.2.-', '1.1.1.169', '2.4.2.10',
    '3.5.3.1', '3.3.2.-', '3.1.26.4', '7.1.1.9', '3.4.13.9',
    '1.1.1.100', '5.1.1.-', '3.1.22.-', '3.4.21.-', '1.11.1.-',
    '3.4.22.28', '4.6.1.18', '3.4.21.4', '6.1.1.4', '1.15.1.1',
    '3.4.19.-', '3.4.11.1', '3.4.25.1', '1.3.1.9', '2.4.99.-',
    '1.1.99.-', '1.97.1.12', '2.7.4.6', '1.17.1.-', '1.7.2.-',
    '2.7.1.71', '4.2.1.11', '3.4.21.90', '4.1.99.-', '2.8.1.1',
    '3.6.4.13', '6.3.4.4', '2.7.11.1', '2.7.7.23', '1.9.3.-',
    '2.3.1.39', '4.1.1.11', '3.2.1.26', '3.1.13.-', '1.8.99.-',
    '1.11.1.6', '4.2.1.2', '5.3.4.1', '3.2.1.4', '3.4.21.5',
    '1.14.16.-', '6.3.4.2', '2.1.1.37', '1.12.99.-', '2.1.1.361',
    '2.1.1.-', '3.1.8.1', '2.7.11.-', '2.3.2.24', '2.7.7.48',
    '3.5.1.98', '3.1.1.31', '2.7.11.22', '1.18.1.2', '3.1.13.4',
    '4.1.1.23', '3.2.2.27', '2.5.1.7', '1.14.11.-', '3.5.1.2',
    '6.3.1.2', '4.3.2.-', '4.1.2.25', '5.3.2.-', '2.7.1.1', '3.1.11.-',
    '2.7.7.4', '3.6.3.14', '4.2.1.20', '2.3.1.41', '1.18.6.1',
    '2.3.1.74', '6.3.2.19', '3.2.2.22', '2.8.4.1', '2.4.2.17',
    '2.1.1.56', '1.10.3.-', '2.5.1.54', '1.2.1.11', '4.2.1.8',
    '3.1.3.16', '1.12.7.2', '2.7.1.-', '1.17.4.1', '2.7.10.-',
    '3.1.2.14', '5.3.1.6', '3.4.21.91', '2.2.1.-', '2.7.1.2',
    '2.7.3.-', '3.1.22.4', '2.3.1.16', '4.99.1.-', '2.7.1.35',
    '3.4.22.69', '3.1.27.-', '1.12.7.-', '5.1.3.-', '2.7.7.65',
    '1.17.4.-', '5.3.1.24', '4.2.1.59', '2.5.1.10', '1.8.4.11',
    '3.4.25.-', '2.7.10.2', '5.2.1.-', '6.1.1.3', '2.7.7.49',
    '2.8.4.-', '4.1.3.-', '2.3.1.31', '1.1.1.205', '3.1.30.-',
    '3.4.23.-', '6.5.1.2', '6.1.1.15', '3.6.4.-', '6.2.1.-', '4.1.3.3',
    '2.7.7.60', '6.3.2.6', '5.1.1.3', '2.4.2.9', '1.14.13.-',
    '1.1.2.-', '1.1.1.1', '5.1.99.-', '2.8.2.-', '2.5.1.47',
    '2.7.11.24', '3.4.22.15', '2.6.1.42', '2.1.3.2', '3.2.2.9',
    '4.2.3.3', '2.6.1.-', '1.5.1.-', '2.7.7.24', '2.1.1.57',
    '6.1.1.20', '5.3.1.5', '2.7.1.25', '2.2.1.1', '3.6.1.1',
    '2.3.3.16', '6.3.4.-', '2.7.7.9', '1.18.1.-', '4.2.99.-',
    '4.1.2.4', '3.1.3.1', '5.3.3.8', '3.2.1.1', '2.7.10.1',
    '4.2.1.113', '4.2.2.2', '6.1.1.1', '3.1.3.-', '1.2.4.-',
    '1.6.99.-', '2.5.1.18', '3.4.22.29', '3.1.3.2', '1.1.1.27',
    '2.3.1.286', '1.14.15.-', '2.7.7.3', '3.1.13.2', '2.7.7.6',
    '5.4.99.18', '4.1.1.39', '2.7.4.8', '5.6.2.1', '2.3.1.-',
    '1.7.1.-', '1.6.5.2', '1.18.6.-', '4.6.1.2', '2.6.1.52', '3.1.6.-',
    '1.6.5.-', '2.3.2.31', '3.6.5.5', '2.8.3.-', '2.3.2.-', '3.2.1.18',
    '3.5.99.-', '3.1.4.35', '3.1.8.-', '2.3.1.12', '1.6.2.-',
    '2.1.1.72', '2.3.3.-', '3.4.21.92', '1.14.11.27', '2.7.11.17',
    '2.1.1.359', '2.7.13.-', '2.5.1.15', '6.3.3.-', '3.2.1.22',
    '6.1.1.6', '4.3.3.7', '3.4.11.18', '4.2.3.4', '3.1.2.-', '2.4.2.8',
    '4.1.1.48', '3.7.1.-', '3.1.4.53', '7.2.2.-', '2.7.6.5', '3.6.5.3',
    '4.3.3.-', '2.7.1.21', '3.1.4.-', '1.11.1.24', '3.6.4.10',
    '1.14.14.1', '3.5.1.60', '3.2.1.52', '1.16.3.-', '3.1.26.3',
    '3.4.24.69', '3.5.1.11', '2.1.1.193', '1.7.2.1', '1.14.99.-',
    '3.6.5.2', '2.7.7.n1', '4.1.1.-', '2.3.2.26', '2.4.1.15',
    '2.5.1.-', '3.1.3.11', '1.14.11.67', '2.3.1.180', '2.4.2.7',
    '3.6.4.12', '2.5.1.19', '1.1.1.-', '1.8.1.9', '1.9.3.1',
    '2.7.1.15', '2.7.1.11', '3.4.11.-', '2.1.2.-', '7.6.2.-',
    '3.5.1.28', '3.8.1.5', '6.3.4.13', '1.11.1.9', '1.13.11.-',
    '4.1.2.13', '3.4.16.4', '6.1.1.7', '1.1.3.-', '2.4.1.129',
    '3.1.1.29', '3.4.21.53', '3.6.5.-', '3.1.1.4', '1.1.1.86',
    '3.2.2.-', '2.6.1.1', '4.2.99.18', '5.5.1.-', '2.7.11.30',
    '3.1.26.-', '1.8.1.4', '1.14.14.-', '3.2.1.14', '3.5.4.-',
    '2.1.2.1', '3.2.1.3', '1.97.1.-', '6.3.4.14', '2.7.12.1',
    '2.7.11.25', '2.5.1.17', '6.3.5.2', '6.3.2.1', '3.2.1.78',
    '3.1.11.2', '1.6.99.3', '2.1.2.2', '1.1.1.42', '7.1.1.8',
    '7.1.1.2', '3.1.2.2', '3.4.22.46', '3.1.3.36', '2.4.1.-',
    '3.1.3.33', '4.3.2.2', '1.14.12.-', '1.13.12.-', '3.4.22.-',
    '5.3.1.1', '4.2.1.-', '3.2.1.169', '3.2.1.17', '6.5.1.1',
    '1.1.1.35', '1.3.5.-', '1.2.1.3', '2.5.1.1', '7.2.2.8', '6.3.1.-',
    '2.5.1.78', '2.7.7.50', '2.1.3.-', '2.7.7.-', '5.4.3.8', '2.7.2.4',
    '1.10.3.2', '3.1.1.3', '3.6.1.15', '5.6.2.2', '3.1.3.3',
    '3.2.1.20', '3.6.1.9', '2.3.2.27', '3.6.1.23', '6.1.1.-',
    '4.2.1.10', '3.4.13.-', '6.3.2.4', '1.1.1.2', '3.4.21.107',
    '1.6.99.1', '2.7.4.9', '1.15.1.-', '3.6.1.34', '1.3.1.-',
    '3.6.1.55', '3.4.24.-', '3.6.1.-', '4.1.1.50', '4.2.2.-',
    '3.3.1.1', '3.4.22.1', '3.1.4.11', '3.5.1.1', '3.3.1.-',
    '1.1.1.267', '3.2.1.55', '1.1.1.25', '3.6.1.7', '2.7.13.3',
    '2.7.1.40', '2.3.1.9', '1.7.3.-', '5.4.2.-', '1.7.1.17', '3.2.2.6',
    '4.1.1.33', '1.8.5.-', '5.3.3.-', '3.2.1.31', '6.3.5.-',
    '1.14.19.-', '6.1.1.11', '1.12.99.6', '1.4.1.-', '4.6.1.1',
    '3.1.3.86', '3.2.1.91', '4.3.2.10', '3.4.16.-', '3.1.3.5',
    '3.5.4.4', '6.4.1.-', '1.17.1.8', '2.5.1.16', '4.3.1.-',
    '3.4.23.16', '6.3.3.1', '3.2.1.73', '5.1.3.13', '1.2.1.12',
    '1.6.2.4', '6.1.1.17', '2.3.1.1', '3.5.3.-', '3.2.1.8', '2.1.1.45',
    '3.2.1.21', '5.1.3.2', '2.3.1.129', '2.7.2.-', '5.3.1.-',
    '2.7.2.8', '2.4.2.1', '1.14.14.18', '3.5.1.5', '3.5.4.38',
    '5.4.3.-', '6.5.1.-', '2.7.12.2', '2.5.1.55', '1.8.1.7',
    '3.1.21.-', '1.8.4.12', '1.11.1.15', '1.1.1.85', '3.6.1.3',
    '2.7.1.33', '2.7.8.7', '3.1.3.25', '3.2.1.96', '7.1.1.-',
    '3.2.1.39', '2.4.2.3', '3.5.4.9', '2.2.1.2', '3.6.3.-', '3.5.1.-',
    '1.11.1.7', '1.5.1.5', '2.4.2.19', '1.8.3.2', '1.3.99.-',
    '1.5.3.-', '5.6.1.-', '1.8.1.-', '2.7.3.9', '2.5.1.6', '3.4.21.62',
    '1.8.4.-', '5.3.1.16', '2.3.2.2', '3.4.17.-', '2.7.4.-', '3.2.1.-',
    '6.3.1.5', '4.3.3.6', '2.1.3.3', '1.5.1.3', '3.5.2.3', '5.4.2.11',
    '2.7.7.8', '2.8.1.7', '2.7.7.7', '3.2.1.37', '2.7.12.-', '5.3.1.9',
    '1.2.7.-', '6.1.1.2', '7.1.2.-', '2.3.1.5', '3.4.14.-', '6.1.1.10',
    '1.16.1.-', '2.1.1.228', '3.5.2.6', '2.1.1.354', '2.7.4.3',
    '4.1.2.-', '4.4.1.5', '5.3.4.-', '4.6.1.-', '5.1.1.1', '3.4.24.3',
    '4.4.1.-', '1.3.7.-', '2.3.1.117', '5.4.99.5', '1.2.1.-',
    '2.4.2.30', '1.14.18.-', '3.1.1.1', '3.1.3.48', '3.4.21.98',
    '5.6.2.-', '3.1.26.5', '7.2.1.1', '2.7.11.12', '1.3.3.-',
    '2.7.7.18', '3.1.1.-', '5.2.1.8', '2.7.1.69', '1.1.1.37',
    '3.6.4.6', '3.1.4.17', '2.7.11.13', '3.5.2.-', '4.2.1.17',
    '2.7.2.3', '4.2.3.-', '5.3.99.-', '3.1.1.72', '2.3.1.179',
    '3.2.1.23', '1.14.13.25', '3.1.26.13', '3.8.1.-', '4.1.1.20',
    '2.7.11.26', '6.1.1.21', '2.7.11.21', '2.7.4.22', '1.8.3.-',
    '2.3.1.57', '1.3.5.1', '3.1.1.53', '7.1.2.2', '6.4.1.2', '2.7.8.-',
    '6.3.2.-', '2.8.1.-', '3.5.4.5', '4.6.1.12', '2.3.2.23',
]

# Task name → metric type mapping (from biodata_task_gen.py)
TASK_TO_METRIC = {
    'DNA-cpd': 'MCC',
    'DNA-emp': 'MCC',
    'DNA-pd': 'MCC',
    'DNA-tf-h': 'MCC',
    'DNA-tf-m': 'MCC',
    'Multi_sequence-antibody_antigen': 'MCC',
    'Multi_sequence-promoter_enhancer_interaction': 'MCC',
    'Multi_sequence-rna_protein_interaction': 'MCC',
    'DNA-enhancer_activity': 'PCC',
    'RNA-CRISPROnTarget': 'Spearman',
    'Protein-Fluorescence': 'Spearman',
    'Protein-Stability': 'Spearman',
    'Protein-Thermostability': 'Spearman',
    'RNA-Isoform': 'R2',
    'RNA-MeanRibosomeLoading': 'R2',
    'RNA-ProgrammableRNASwitches': 'R2',
    'RNA-Modification': 'Auc',
    'Protein-Solubility': 'Acc',
    'RNA-NoncodingRNAFamily': 'Acc',
    'Protein-FunctionEC': 'Fmax',
    'Multi_sequence-sirnaEfficiency': 'Mixed',
}


# ═════════════════════════════════════════════════════════════════════════════
# Statistical / metric computation helpers (same as OpenCompass biodata.py)
# ═════════════════════════════════════════════════════════════════════════════


def pearson_correlation_coefficient(y_true, y_pred):
    """Pearson correlation coefficient, handling inf values."""
    result_values = np.array(y_pred).flatten()
    label_values = np.array(y_true).flatten()

    near_infinity_mask = np.isinf(result_values)
    valid_mask = (~near_infinity_mask
                  & np.isfinite(result_values)
                  & np.isfinite(label_values))
    valid_result_values = result_values[valid_mask]
    valid_label_values = label_values[valid_mask]

    if len(valid_result_values) > 0:
        correlation, _ = pearsonr(valid_label_values, valid_result_values)
    else:
        correlation = 0

    total_data_points = len(result_values)
    num_infinity_values = near_infinity_mask.sum()

    if num_infinity_values > 0:
        total_valid_points = valid_mask.sum()
        final_score = (correlation * total_valid_points) / total_data_points
    else:
        final_score = correlation
    return final_score


def spearman_correlation_coefficient(y_true, y_pred):
    """Spearman rank correlation coefficient, handling inf values."""
    result_values = np.array(y_pred).flatten()
    label_values = np.array(y_true).flatten()

    near_infinity_mask = np.isinf(result_values)
    valid_mask = (~near_infinity_mask
                  & np.isfinite(result_values)
                  & np.isfinite(label_values))
    valid_result_values = result_values[valid_mask]
    valid_label_values = label_values[valid_mask]

    if len(valid_result_values) > 0:
        spearman, _ = spearmanr(valid_label_values, valid_result_values)
    else:
        spearman = 0

    total_data_points = len(result_values)
    num_infinity_values = near_infinity_mask.sum()

    if num_infinity_values > 0:
        total_valid_points = valid_mask.sum()
        final_spearman_score = (spearman * total_valid_points) / total_data_points
    else:
        final_spearman_score = spearman
    return final_spearman_score


def r_squared(y_true, y_pred):
    """R² = PCC², handling inf values."""
    result_values = np.array(y_pred).flatten()
    label_values = np.array(y_true).flatten()

    near_infinity_mask = np.isinf(result_values)
    valid_mask = (~near_infinity_mask
                  & np.isfinite(result_values)
                  & np.isfinite(label_values))
    valid_result_values = result_values[valid_mask]
    valid_label_values = label_values[valid_mask]

    if len(valid_result_values) > 0:
        try:
            pcc, _ = pearsonr(valid_label_values, valid_result_values)
            R2 = pcc ** 2
        except Exception:
            R2 = np.inf
    else:
        R2 = 0

    total_data_points = len(result_values)
    num_infinity_values = near_infinity_mask.sum()

    if num_infinity_values > 0:
        total_valid_points = valid_mask.sum()
        final_R2_score = (R2 * total_valid_points) / total_data_points
    else:
        final_R2_score = R2
    return final_R2_score


def multiple_label_auc(y_true, y_pred):
    """Multi-label macro AUC.  y_true, y_pred: (N, C)."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_pred)
    assert y_true.shape == y_score.shape
    assert y_true.ndim == 2
    _N, C = y_true.shape

    per_class = []
    for c in range(C):
        yt = y_true[:, c]
        ys = y_score[:, c]
        if yt.min() == yt.max():
            per_class.append(np.nan)
            continue
        per_class.append(roc_auc_score(yt, ys))
    per_class = np.array(per_class, dtype=float)
    macro_auc = np.nanmean(per_class)
    return float(macro_auc)


def compute_mixed_score(y_true, y_pred, low_range=(30, 1e3)):
    """Mixed score combining MAE, Range-MAE and F1 (identical to biodata.py)."""
    result_values = pd.to_numeric(y_pred, errors='coerce').flatten()
    label_values = pd.to_numeric(y_true, errors='coerce').flatten()

    near_infinity_mask = np.abs(result_values) > low_range[1]
    valid_mask = (~near_infinity_mask
                  & np.isfinite(result_values)
                  & np.isfinite(label_values))
    valid_result_values = result_values[valid_mask]
    valid_label_values = label_values[valid_mask]

    num_infinity_values = near_infinity_mask.sum()
    mixed_score_infinity = 0  # score for inf pairs

    # Binary classification: low / high based on threshold
    label_binary = (valid_label_values < low_range[0]).astype(int)
    result_binary = (valid_result_values < low_range[0]).astype(int)

    if len(label_binary) > 0 and len(np.unique(label_binary)) > 0:
        prec = precision_score(label_binary, result_binary, average='binary', zero_division=0)
        rec = recall_score(label_binary, result_binary, average='binary', zero_division=0)
    else:
        prec, rec = 0, 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) != 0 else 0

    try:
        mae = mean_absolute_error(valid_label_values, valid_result_values)
    except ValueError:
        mae = np.inf

    mask = (valid_result_values >= 0) & (valid_result_values <= low_range[0])
    if mask.sum() > 0:
        range_mae = mean_absolute_error(valid_label_values[mask],
                                        valid_result_values[mask])
    else:
        range_mae = 100

    mae = min(mae, 100)
    range_mae = min(range_mae, 100)

    mixed_score_valid = (1 - mae / 100) * 0.5 + (1 - range_mae / 100) * f1 * 0.5

    total_data_points = len(result_values)
    total_valid_points = valid_mask.sum()

    if num_infinity_values > 0:
        final_mixed_score = (
            mixed_score_valid * total_valid_points
            + mixed_score_infinity * num_infinity_values
        ) / total_data_points
    else:
        final_mixed_score = mixed_score_valid

    return final_mixed_score


def count_f1_max(pred, target):
    """F1 score with the optimal threshold (from biodata.py).

    Parameters:
        pred (Tensor): predictions of shape (B, N)
        target (Tensor): binary targets of shape (B, N)

    Returns:
        Tensor or float: The maximum F1 score or 0.0 if inputs are empty.
    """
    if pred.numel() == 0 or target.numel() == 0:
        return 0.0

    order = pred.argsort(descending=True, dim=1, stable=True)
    target = target.gather(1, order)
    precision = target.cumsum(1) / torch.ones_like(target).cumsum(1)
    recall = target.cumsum(1) / (target.sum(1, keepdim=True) + 1e-10)

    is_start = torch.zeros_like(target).bool()
    is_start[:, 0] = 1
    is_start = torch.scatter(is_start, 1, order, is_start)
    all_order = pred.flatten().argsort(descending=True, stable=True)
    order = order + torch.arange(
        order.shape[0], device=order.device).unsqueeze(1) * order.shape[1]
    order = order.flatten()
    inv_order = torch.zeros_like(order)
    inv_order[order] = torch.arange(order.shape[0], device=order.device)
    is_start = is_start.flatten()[all_order]
    all_order = inv_order[all_order]

    precision = precision.flatten()
    recall = recall.flatten()

    all_precision = precision[all_order] - torch.where(
        is_start, torch.zeros_like(precision), precision[all_order - 1])
    all_precision = all_precision.cumsum(0) / is_start.cumsum(0)
    all_recall = recall[all_order] - torch.where(
        is_start, torch.zeros_like(recall), recall[all_order - 1])
    all_recall = all_recall.cumsum(0) / pred.shape[0]
    all_f1 = 2 * all_precision * all_recall / (all_precision + all_recall + 1e-10)

    if torch.isnan(all_f1).any():
        return 0.0

    return all_f1.max()


def ec_to_multihot(ec_list, ec_labels):
    """Convert list of EC code strings to a multi-hot torch tensor."""
    multihot = torch.zeros(len(ec_labels))
    if not ec_list:
        return multihot
    for ec in ec_list:
        if ec in ec_labels:
            idx = ec_labels.index(ec)
            multihot[idx] = 1
    return multihot


# ═════════════════════════════════════════════════════════════════════════════
# Per-sample scoring functions
# Each takes lists of prediction strings and reference values.
# Returns dict with 'score' (average) and 'details'.
# ═════════════════════════════════════════════════════════════════════════════


def mcc_score(predictions, references):
    """MCC tasks — per-sample accuracy (aligned with batch MCC).

    Extraction: ``\\boxed{yes/no/positive/negative}``
    Per-sample: 100 if correct, 0 if wrong.
    Returns: ``{'score': avg_accuracy, 'details': [...]}``.
    """
    ans_dict = {'positive': 1, 'negative': 0, 'yes': 1, 'no': 0}
    details = []
    for pred, ans in zip(predictions, references):
        pred_text = extract_boxed_text(pred)
        ans_label = ans_dict.get(str(ans).lower(), 0)
        if not pred_text or pred_text.lower() not in ans_dict:
            pred_label = 1 - ans_label
        else:
            pred_label = ans_dict[pred_text.lower()]
        sample_score = 100.0 if pred_label == ans_label else 0.0
        details.append({'pred': pred_text, 'answer': ans, 'score': sample_score})

    avg = sum(d['score'] for d in details) / len(details) if details else 0.0
    return {'score': avg, 'details': details}


def pcc_score(predictions, references, error_scale=1.0):
    """PCC tasks (dict-valued regression) — per-sample inverse-error proxy.

    Extraction: JSON dict from prediction text.
    Per-sample: for each key, ``100 / (1 + |pred_k - ref_k| / scale)``,
                averaged across keys.  Range [0, 100].
    """
    details = []
    for pred, ans in zip(predictions, references):
        pred_text = extract_dict_text(pred)
        parsed = None
        if pred_text:
            try:
                parsed = json.loads(pred_text)
            except Exception:
                try:
                    parsed = ast.literal_eval(pred_text)
                except Exception:
                    pass

        if not parsed or not isinstance(parsed, dict) or set(parsed.keys()) != set(ans.keys()):
            parsed = {key: np.inf for key in ans.keys()}

        key_scores = []
        for key in ans.keys():
            try:
                pred_num = float(parsed[key])
            except Exception:
                pred_num = np.inf
            ans_num = float(ans[key])
            if np.isinf(pred_num):
                key_scores.append(0.0)
            else:
                key_scores.append(100.0 / (1.0 + abs(pred_num - ans_num) / error_scale))
        sample_score = sum(key_scores) / len(key_scores) if key_scores else 0.0
        details.append({'pred': parsed, 'answer': ans, 'score': sample_score})

    avg = sum(d['score'] for d in details) / len(details) if details else 0.0
    return {'score': avg, 'details': details}


def spearman_score(predictions, references, error_scale=1.0):
    """Spearman tasks (number regression) — per-sample inverse-error proxy.

    Extraction: number from ``\\boxed{}`` or ``<NUMBER>``
    Per-sample: ``100 / (1 + |pred - ref| / scale)``.  Range [0, 100].
    """
    details = []
    for pred, ans in zip(predictions, references):
        pred_num = extract_number(pred)
        ans_num = float(ans)
        if pred_num is None or np.isinf(pred_num):
            sample_score = 0.0
        else:
            sample_score = 100.0 / (1.0 + abs(pred_num - ans_num) / error_scale)
        details.append({'pred': pred_num, 'answer': ans, 'score': sample_score})

    avg = sum(d['score'] for d in details) / len(details) if details else 0.0
    return {'score': avg, 'details': details}


def r2_score(predictions, references, error_scale=1.0):
    """R² tasks — per-sample inverse-error proxy.

    Supports both dict references and scalar references.
    Per-sample: ``100 / (1 + |pred - ref| / scale)``  (scalar)
          or average over keys (dict).  Range [0, 100].
    """
    details = []
    for pred, ans in zip(predictions, references):
        if isinstance(ans, dict):
            # dict regression
            pred_text = extract_dict_text(pred)
            parsed = None
            if pred_text:
                try:
                    parsed = json.loads(pred_text)
                except Exception:
                    try:
                        parsed = ast.literal_eval(pred_text)
                    except Exception:
                        pass
            if not parsed or not isinstance(parsed, dict) or set(parsed.keys()) != set(ans.keys()):
                parsed = {key: np.inf for key in ans.keys()}
            else:
                for key in parsed:
                    try:
                        parsed[key] = float(parsed[key])
                    except Exception:
                        parsed[key] = np.inf

            key_scores = []
            for key in ans.keys():
                pred_num = parsed.get(key, np.inf)
                ans_num = float(ans[key])
                if np.isinf(pred_num):
                    key_scores.append(0.0)
                else:
                    key_scores.append(100.0 / (1.0 + abs(pred_num - ans_num) / error_scale))
            sample_score = sum(key_scores) / len(key_scores) if key_scores else 0.0
            details.append({'pred': parsed, 'answer': ans, 'score': sample_score})
        else:
            # scalar regression
            pred_num = extract_number(pred)
            ans_num = float(ans)
            if pred_num is None or np.isinf(pred_num):
                sample_score = 0.0
            else:
                sample_score = 100.0 / (1.0 + abs(pred_num - ans_num) / error_scale)
            details.append({'pred': pred_num, 'answer': ans, 'score': sample_score})

    avg = sum(d['score'] for d in details) / len(details) if details else 0.0
    return {'score': avg, 'details': details}


def auc_score(predictions, references, predefined_labels=None):
    """AUC tasks (multi-label classification) — per-sample label-set F1 proxy.

    Since per-sample AUC is undefined (needs ≥2 samples), we use
    F1 between predicted and true label sets as alignment proxy.
    Extraction: ``\\boxed{comma-separated labels}``
    Per-sample: set F1 × 100.  Range [0, 100].
    """
    if predefined_labels is None:
        predefined_labels = AUC_PREDEFINED_LABELS

    details = []
    for pred, ans in zip(predictions, references):
        pred_text = extract_boxed_text(pred)

        ans_set = set(a.lower().strip() for a in str(ans).split(',') if a.strip())

        if not pred_text:
            pred_set = set()
        else:
            pred_set = set(p.lower().strip() for p in pred_text.split(',') if p.strip())

        # Per-sample F1 between label sets
        intersection = pred_set & ans_set
        if len(pred_set) == 0 and len(ans_set) == 0:
            sample_score = 100.0
        elif len(pred_set) == 0 or len(ans_set) == 0:
            sample_score = 0.0
        else:
            prec = len(intersection) / len(pred_set)
            rec = len(intersection) / len(ans_set)
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            sample_score = f1 * 100.0

        details.append({'pred': pred_text, 'answer': ans, 'score': sample_score})

    avg = sum(d['score'] for d in details) / len(details) if details else 0.0
    return {'score': avg, 'details': details}


def acc_score(predictions, references):
    """Accuracy tasks — per-sample exact match.

    Extraction: ``\\boxed{text}``
    Per-sample: 100 if matches, 0 if not.
    """
    ans_dict = {'positive': 'yes', 'negative': 'no'}

    details = []
    for pred, ans in zip(predictions, references):
        pred_text = extract_boxed_text(pred)
        if not pred_text:
            details.append({'pred': pred_text, 'answer': ans, 'score': 0.0})
            continue
        pred_lower = pred_text.lower()
        ans_lower = str(ans).lower()
        if ans_lower in ans_dict:
            ans_lower = ans_dict[ans_lower]
        if pred_lower in ans_dict:
            pred_lower = ans_dict[pred_lower]
        sample_score = 100.0 if pred_lower == ans_lower else 0.0
        details.append({'pred': pred_text, 'answer': ans, 'score': sample_score})

    avg = sum(d['score'] for d in details) / len(details) if details else 0.0
    return {'score': avg, 'details': details}


def fmax_score(predictions, references, ec_labels=None):
    """Fmax tasks (EC number prediction) — per-sample F1 via count_f1_max.

    Extraction: ``\\boxed{EC numbers}``
    Per-sample: F1-max × 100.  Range [0, 100].
    """
    if ec_labels is None:
        ec_labels = EC_LABELS

    details = []
    for pred, ans in zip(predictions, references):
        pred_text = extract_boxed_text(pred)
        if not pred_text:
            result_ec = []
        else:
            result_ec = re.findall(r'\d+\.\d+\.\d+\.\-?\d*', pred_text)
        label_ec = re.findall(r'\d+\.\d+\.\d+\.\-?\d*', str(ans))

        pred_multihot = ec_to_multihot(result_ec, ec_labels)
        label_multihot = ec_to_multihot(label_ec, ec_labels)

        cur_f1 = count_f1_max(
            torch.stack([pred_multihot]),
            torch.stack([label_multihot]),
        )
        if isinstance(cur_f1, torch.Tensor):
            cur_f1 = cur_f1.item()
        sample_score = cur_f1 * 100.0
        details.append({'pred': pred_text, 'answer': ans, 'score': sample_score})

    avg = sum(d['score'] for d in details) / len(details) if details else 0.0
    return {'score': avg, 'details': details}


def mixed_score_eval(predictions, references):
    """Mixed tasks — per-sample mixed score (MAE + Range-MAE + F1).

    Extraction: number from ``\\boxed{}`` or ``<NUMBER>``
    Per-sample: ``compute_mixed_score([ans], [pred]) * 100``.  Range [0, 100].
    """
    details = []
    for pred, ans in zip(predictions, references):
        pred_num = extract_number(pred)
        if pred_num is None:
            pred_num = 0
        ans_num = float(ans)
        sample_score = compute_mixed_score([ans_num], [pred_num]) * 100.0
        details.append({'pred': pred_num, 'answer': ans, 'score': sample_score})

    avg = sum(d['score'] for d in details) / len(details) if details else 0.0
    return {'score': avg, 'details': details}
