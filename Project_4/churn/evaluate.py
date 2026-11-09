"""Метрики, вибір порога за вартістю помилок, довірчі інтервали, калібрування."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def threshold_grid() -> np.ndarray:
    return np.round(np.arange(0.02, 0.98, 0.01), 2)


def confusion(y_true: np.ndarray, flagged: np.ndarray) -> tuple[int, int, int, int]:
    """Повертає (tn, fp, fn, tp) для бінарного рішення «позначити клієнта як ризиковий»."""
    y_true = np.asarray(y_true).astype(bool)
    flagged = np.asarray(flagged).astype(bool)
    tp = int(np.sum(flagged & y_true))
    fp = int(np.sum(flagged & ~y_true))
    fn = int(np.sum(~flagged & y_true))
    tn = int(np.sum(~flagged & ~y_true))
    return tn, fp, fn, tp


def cost_curve(y_true, proba, thresholds, cost_fn: float, cost_fp: float) -> pd.DataFrame:
    """Сумарна вартість помилок для кожного порога (клієнт позначається, якщо p >= поріг)."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    rows = []
    for t in thresholds:
        tn, fp, fn, tp = confusion(y_true, proba >= t)
        rows.append({
            "threshold": float(t), "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "cost": fn * cost_fn + fp * cost_fp,
            "precision": tp / (tp + fp) if (tp + fp) else 0.0,
            "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        })
    return pd.DataFrame(rows)


def choose_threshold(y_true, proba, cost_fn: float, cost_fp: float) -> tuple[float, pd.DataFrame]:
    """Поріг із мінімальною вартістю помилок. Викликати лише на out-of-fold прогнозах навчальної вибірки."""
    curve = cost_curve(y_true, proba, threshold_grid(), cost_fn, cost_fp)
    best = curve.loc[curve["cost"].idxmin(), "threshold"]
    return float(best), curve


def threshold_report(y_true, proba, threshold: float, cost_fn: float, cost_fp: float) -> dict:
    """Метрики рішення за порогом, включно з вартістю та тривіальними стратегіями."""
    y_true = np.asarray(y_true)
    tn, fp, fn, tp = confusion(y_true, np.asarray(proba) >= threshold)
    n = len(y_true)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    cost = fn * cost_fn + fp * cost_fp
    cost_flag_nobody = (tp + fn) * cost_fn          # нікого не втримуємо
    cost_flag_everybody = (tn + fp) * cost_fp       # пропонуємо втримання всім
    return {
        "threshold": float(threshold),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "precision": precision, "recall": recall, "f1": f1,
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "flagged_share": (tp + fp) / n,
        "cost": cost,
        "cost_per_customer": cost / n,
        "cost_flag_nobody": cost_flag_nobody,
        "cost_flag_everybody": cost_flag_everybody,
        "cost_reduction_vs_best_trivial": 1 - cost / min(cost_flag_nobody, cost_flag_everybody),
    }


def ranking_metrics(y_true, proba) -> dict:
    """Метрики, що не залежать від порога."""
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
        "log_loss": float(log_loss(y_true, proba)),
        "base_rate": float(np.mean(y_true)),
    }


def bootstrap_ci(y_true, proba, seed: int, n_boot: int = 1000, alpha: float = 0.05) -> dict:
    """Довірчі інтервали ROC-AUC та PR-AUC (bootstrap за клієнтами тестової вибірки)."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    roc, pr = [], []
    while len(roc) < n_boot:
        idx = rng.integers(0, n, n)
        if y_true[idx].min() == y_true[idx].max():
            continue  # у вибірці лише один клас
        roc.append(roc_auc_score(y_true[idx], proba[idx]))
        pr.append(average_precision_score(y_true[idx], proba[idx]))
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        "roc_auc": [float(np.percentile(roc, lo)), float(np.percentile(roc, hi))],
        "pr_auc": [float(np.percentile(pr, lo)), float(np.percentile(pr, hi))],
        "n_boot": n_boot,
        "confidence": 1 - alpha,
    }


def calibration_table(y_true, proba, bins: int = 10) -> list[dict]:
    """Середня передбачена ймовірність проти фактичної частки відтоку по децилях."""
    frame = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(proba)})
    frame["bin"] = pd.qcut(frame["p"], bins, duplicates="drop")
    grouped = frame.groupby("bin", observed=True).agg(n=("y", "size"), mean_predicted=("p", "mean"), observed_rate=("y", "mean"))
    return [
        {"n": int(r.n), "mean_predicted": float(r.mean_predicted), "observed_rate": float(r.observed_rate)}
        for r in grouped.itertuples()
    ]
