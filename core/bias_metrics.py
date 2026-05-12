"""
core/bias_metrics.py
─────────────────────
Comprehensive fairness / bias metrics:

  • Per-group accuracy
  • Bias Gap (max_acc − min_acc)
  • Demographic Parity Difference
  • Equalized Odds Difference  (TPR gap + FPR gap)
  • Predictive Parity
  • Disparate Impact Ratio
  • Dataset Distribution Imbalance (chi-square)
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report
)
from scipy.stats import chi2_contingency
from typing import Dict, List, Tuple


# ══════════════════════════════════════════════════════════════
#  Helper
# ══════════════════════════════════════════════════════════════

def _binary_rates(y_true: np.ndarray, y_pred: np.ndarray,
                  positive_class: int = 1) -> Tuple[float, float]:
    """Returns (TPR, FPR) for a binary scenario."""
    tp = np.sum((y_pred == positive_class) & (y_true == positive_class))
    fn = np.sum((y_pred != positive_class) & (y_true == positive_class))
    fp = np.sum((y_pred == positive_class) & (y_true != positive_class))
    tn = np.sum((y_pred != positive_class) & (y_true != positive_class))

    tpr = tp / (tp + fn + 1e-9)
    fpr = fp / (fp + tn + 1e-9)
    return tpr, fpr


# ══════════════════════════════════════════════════════════════
#  Per-group metrics
# ══════════════════════════════════════════════════════════════

def group_accuracy(y_true: np.ndarray, y_pred: np.ndarray,
                   group_indices: Dict[str, List[int]]) -> Dict[str, float]:
    """Per-group accuracy dictionary."""
    result = {}
    for grp, idx in group_indices.items():
        yt = y_true[idx]
        yp = y_pred[idx]
        result[grp] = float(accuracy_score(yt, yp)) if len(yt) > 0 else 0.0
    return result


def bias_gap(group_acc: Dict[str, float]) -> float:
    """max_acc − min_acc across all groups."""
    vals = list(group_acc.values())
    return float(max(vals) - min(vals)) if vals else 0.0


# ══════════════════════════════════════════════════════════════
#  Fairness metrics (binary-focused, extended for multi-class)
# ══════════════════════════════════════════════════════════════

def demographic_parity_difference(y_pred: np.ndarray,
                                   group_indices: Dict[str, List[int]],
                                   positive_class: int = 1) -> float:
    """
    DPD = max P(Ŷ=1 | G=g) − min P(Ŷ=1 | G=g)
    Ideal: 0.0
    """
    rates = {}
    for grp, idx in group_indices.items():
        yp = y_pred[idx]
        rates[grp] = float(np.mean(yp == positive_class))
    vals = list(rates.values())
    return float(max(vals) - min(vals)) if vals else 0.0, rates


def equalized_odds_difference(y_true: np.ndarray, y_pred: np.ndarray,
                               group_indices: Dict[str, List[int]],
                               positive_class: int = 1) -> Dict:
    """
    EOD = max |TPR_g − TPR_g'| + max |FPR_g − FPR_g'|
    Ideal: 0.0
    """
    tprs, fprs = {}, {}
    for grp, idx in group_indices.items():
        yt = y_true[idx]
        yp = y_pred[idx]
        tpr, fpr = _binary_rates(yt, yp, positive_class)
        tprs[grp] = tpr
        fprs[grp] = fpr

    tpr_vals = list(tprs.values())
    fpr_vals = list(fprs.values())

    tpr_gap = float(max(tpr_vals) - min(tpr_vals)) if tpr_vals else 0.0
    fpr_gap = float(max(fpr_vals) - min(fpr_vals)) if fpr_vals else 0.0
    eod     = tpr_gap + fpr_gap

    return {
        "equalized_odds_diff": eod,
        "tpr_gap": tpr_gap,
        "fpr_gap": fpr_gap,
        "tprs":    tprs,
        "fprs":    fprs,
    }


def disparate_impact_ratio(y_pred: np.ndarray,
                            group_indices: Dict[str, List[int]],
                            positive_class: int = 1) -> float:
    """
    DIR = min_g P(Ŷ=1|G=g) / max_g P(Ŷ=1|G=g)
    Ideal: 1.0  (values < 0.8 indicate adverse impact)
    """
    rates = {}
    for grp, idx in group_indices.items():
        yp = y_pred[idx]
        rates[grp] = float(np.mean(yp == positive_class)) + 1e-9
    vals = list(rates.values())
    return float(min(vals) / max(vals)) if vals else 1.0


def predictive_parity(y_true: np.ndarray, y_pred: np.ndarray,
                       group_indices: Dict[str, List[int]],
                       positive_class: int = 1) -> Dict:
    """
    Precision gap across groups.
    Ideal: 0.0
    """
    precisions = {}
    for grp, idx in group_indices.items():
        yt = y_true[idx]
        yp = y_pred[idx]
        tp = np.sum((yp == positive_class) & (yt == positive_class))
        fp = np.sum((yp == positive_class) & (yt != positive_class))
        precisions[grp] = float(tp / (tp + fp + 1e-9))

    vals = list(precisions.values())
    gap  = float(max(vals) - min(vals)) if vals else 0.0
    return {"predictive_parity_gap": gap, "precisions": precisions}


# ══════════════════════════════════════════════════════════════
#  Dataset distribution imbalance
# ══════════════════════════════════════════════════════════════

def dataset_imbalance(class_distribution: Dict[str, int]) -> Dict:
    """
    Chi-square test for uniform distribution.
    Returns chi2 stat, p-value, and imbalance ratio.
    """
    counts = np.array(list(class_distribution.values()), dtype=float)
    total  = counts.sum()

    if len(counts) < 2:
        return {"chi2": 0.0, "p_value": 1.0, "imbalance_ratio": 1.0}

    expected = np.full_like(counts, total / len(counts))
    chi2     = float(np.sum((counts - expected) ** 2 / expected))

    # Degrees of freedom
    from scipy.stats import chi2 as chi2_dist
    df      = len(counts) - 1
    p_value = float(1 - chi2_dist.cdf(chi2, df))

    imbalance_ratio = float(counts.max() / (counts.min() + 1e-9))

    return {
        "chi2":             chi2,
        "p_value":          p_value,
        "imbalance_ratio":  imbalance_ratio,
        "is_imbalanced":    p_value < 0.05,
    }


# ══════════════════════════════════════════════════════════════
#  Full report
# ══════════════════════════════════════════════════════════════

def full_bias_report(y_true: np.ndarray, y_pred: np.ndarray,
                     group_indices: Dict[str, List[int]],
                     class_distribution: Dict[str, int],
                     positive_class: int = 1) -> Dict:
    """
    Runs all metrics and returns a single dictionary.
    """
    grp_acc = group_accuracy(y_true, y_pred, group_indices)
    bg      = bias_gap(grp_acc)
    dpd, dp_rates = demographic_parity_difference(
                        y_pred, group_indices, positive_class)
    eod     = equalized_odds_difference(
                        y_true, y_pred, group_indices, positive_class)
    dir_val = disparate_impact_ratio(y_pred, group_indices, positive_class)
    pp      = predictive_parity(y_true, y_pred, group_indices, positive_class)
    imb     = dataset_imbalance(class_distribution)
    overall_acc = float(accuracy_score(y_true, y_pred))

    return {
        "overall_accuracy":        overall_acc,
        "group_accuracy":          grp_acc,
        "bias_gap":                bg,
        "demographic_parity_diff": dpd,
        "demographic_parity_rates": dp_rates,
        "equalized_odds":          eod,
        "disparate_impact_ratio":  dir_val,
        "predictive_parity":       pp,
        "dataset_imbalance":       imb,
    }
