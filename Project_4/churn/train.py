"""Навчання, порівняння моделей, вибір порога, оцінювання та збереження артефакту.

Запуск:  python -m churn.train

Методологія (без витоку даних):
  1. Стратифікований поділ на train/test (test не використовується до самого кінця).
  2. Для кожної моделі-кандидата — підбір гіперпараметрів RandomizedSearchCV зі
     стратифікованою крос-валідацією ТІЛЬКИ на train. Уся обробка ознак (медіани,
     масштаб, one-hot) вчиться всередині folds, бо входить у Pipeline.
  3. Модель обирається за середнім CV ROC-AUC.
  4. Поріг рішення обирається за вартістю помилок на out-of-fold прогнозах train.
  5. Test використовується один раз: фінальні метрики, довірчі інтервали,
     калібрування, важливість ознак (permutation) — лише для звіту, не для вибору.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict, cross_validate, train_test_split

from churn import __version__, config
from churn.data import DataError, file_sha256, load_dataset
from churn.evaluate import (
    bootstrap_ci,
    calibration_table,
    choose_threshold,
    ranking_metrics,
    threshold_report,
)
from churn.features import build_pipeline, prepare_features
from churn.recommendations import build_context
from churn.schema import FEATURE_COLUMNS, ID_COLUMN

log = logging.getLogger("churn.train")

BUNDLE_FORMAT_VERSION = 1
ALL_MODELS = ("logreg", "random_forest", "lightgbm")


@dataclass
class TrainConfig:
    data_path: Path = field(default_factory=config.get_data_path)
    model_path: Path = field(default_factory=config.get_model_path)
    reports_dir: Path = field(default_factory=config.get_reports_dir)
    seed: int = config.SEED
    test_size: float = config.TEST_SIZE
    cv_folds: int = config.CV_FOLDS
    n_iter: int = config.N_ITER
    cost_fn: float = config.COST_FALSE_NEGATIVE
    cost_fp: float = config.COST_FALSE_POSITIVE
    n_jobs: int = -1
    models: tuple[str, ...] = ALL_MODELS
    n_bootstrap: int = 1000


def candidate_models(seed: int, names: tuple[str, ...]) -> dict:
    """Кандидати та простори пошуку. Ім'я параметра має префікс model__ (крок Pipeline)."""
    # Моделі однопотокові: паралелізм забезпечує RandomizedSearchCV (за fit-ами).
    spaces = {
        "logreg": (
            LogisticRegression(max_iter=5000, random_state=seed),
            {"model__C": loguniform(1e-3, 1e2)},
        ),
        "random_forest": (
            RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1),
            {
                "model__max_depth": randint(3, 14),
                "model__min_samples_leaf": randint(1, 31),
                "model__max_features": uniform(0.1, 0.6),
            },
        ),
        "lightgbm": (
            lightgbm.LGBMClassifier(
                random_state=seed, n_jobs=1, verbose=-1,
                deterministic=True, force_row_wise=True, subsample_freq=1,
            ),
            {
                "model__n_estimators": randint(100, 501),
                "model__learning_rate": loguniform(0.01, 0.15),
                "model__num_leaves": randint(4, 33),
                "model__min_child_samples": randint(10, 101),
                "model__subsample": uniform(0.6, 0.4),
                "model__colsample_bytree": uniform(0.5, 0.5),
                "model__reg_lambda": loguniform(1e-2, 20),
            },
        ),
    }
    unknown = set(names) - set(spaces)
    if unknown:
        raise ValueError(f"Невідомі моделі: {sorted(unknown)}; доступні: {list(spaces)}")
    return {name: spaces[name] for name in names}


def _jsonable(obj):
    """Рекурсивно перетворює numpy/pandas/Path у типи, зрозумілі JSON."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _atomic_write(path: Path, writer) -> None:
    """Пише у тимчасовий файл і підміняє цільовий, щоб API не побачив напівзаписаний файл."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        writer(Path(tmp_name))
        os.chmod(tmp_name, 0o644)  # mkstemp створює файл лише для власника; контейнеру потрібне читання
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def run_training(cfg: TrainConfig) -> dict:
    """Повний цикл навчання. Повертає метадані (те саме, що зберігається в артефакті)."""
    cfg.data_path, cfg.model_path, cfg.reports_dir = Path(cfg.data_path), Path(cfg.model_path), Path(cfg.reports_dir)
    data, y = load_dataset(cfg.data_path)
    log.info("Дані: %d клієнтів, частка відтоку %.1f%%", len(data), 100 * y.mean())

    # ---- 1. Поділ. Тест лишається недоторканим до кроку 5. ----
    X_train, X_test, y_train, y_test = train_test_split(
        data, y, test_size=cfg.test_size, stratify=y, random_state=cfg.seed
    )
    assert set(X_train[ID_COLUMN]).isdisjoint(X_test[ID_COLUMN])
    ids_train, ids_test = X_train[ID_COLUMN].copy(), X_test[ID_COLUMN].copy()
    X_train, X_test = X_train[list(FEATURE_COLUMNS)], X_test[list(FEATURE_COLUMNS)]
    log.info("Train: %d, test: %d", len(X_train), len(X_test))

    cv = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.seed)
    scoring = {"roc_auc": "roc_auc", "pr_auc": "average_precision", "brier": "neg_brier_score"}

    # ---- 2–3. Базова модель і підбір гіперпараметрів лише на train ----
    comparison: list[dict] = []
    fitted: dict = {}
    baseline = cross_validate(
        build_pipeline(DummyClassifier(strategy="prior")), X_train, y_train,
        cv=cv, scoring=scoring, n_jobs=cfg.n_jobs,
    )
    comparison.append({
        "model": "baseline_prior", "cv_roc_auc_mean": float(baseline["test_roc_auc"].mean()),
        "cv_roc_auc_std": float(baseline["test_roc_auc"].std()),
        "cv_pr_auc_mean": float(baseline["test_pr_auc"].mean()),
        "cv_brier_mean": float(-baseline["test_brier"].mean()), "best_params": {},
    })
    for name, (estimator, space) in candidate_models(cfg.seed, cfg.models).items():
        log.info("Підбір гіперпараметрів: %s (n_iter=%d, cv=%d)", name, cfg.n_iter, cfg.cv_folds)
        search = RandomizedSearchCV(
            build_pipeline(estimator), space, n_iter=cfg.n_iter, scoring=scoring, refit="roc_auc",
            cv=cv, random_state=cfg.seed, n_jobs=cfg.n_jobs, error_score="raise",
        )
        search.fit(X_train, y_train)
        i = search.best_index_
        res = search.cv_results_
        comparison.append({
            "model": name,
            "cv_roc_auc_mean": float(res["mean_test_roc_auc"][i]),
            "cv_roc_auc_std": float(res["std_test_roc_auc"][i]),
            "cv_pr_auc_mean": float(res["mean_test_pr_auc"][i]),
            "cv_brier_mean": float(-res["mean_test_brier"][i]),
            "best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
        })
        fitted[name] = search.best_estimator_
        log.info("  %s: CV ROC-AUC %.4f ± %.4f", name, comparison[-1]["cv_roc_auc_mean"], comparison[-1]["cv_roc_auc_std"])

    best_name = max(fitted, key=lambda n: next(c for c in comparison if c["model"] == n)["cv_roc_auc_mean"])
    pipeline = fitted[best_name]  # уже донавчений на всьому train (refit)
    log.info("Обрано модель: %s", best_name)

    # ---- 4. Поріг за вартістю помилок — лише out-of-fold прогнози train ----
    oof = cross_val_predict(clone(pipeline), X_train, y_train, cv=cv, method="predict_proba", n_jobs=cfg.n_jobs)[:, 1]
    threshold, curve = choose_threshold(y_train, oof, cfg.cost_fn, cfg.cost_fp)
    oof_report = threshold_report(y_train, oof, threshold, cfg.cost_fn, cfg.cost_fp)
    log.info("Поріг (за OOF train): %.2f", threshold)

    # ---- 5. Єдине оцінювання на test ----
    proba_test = pipeline.predict_proba(X_test)[:, 1]
    test_metrics = {
        **ranking_metrics(y_test, proba_test),
        "confidence_intervals": bootstrap_ci(y_test, proba_test, cfg.seed, cfg.n_bootstrap),
        "at_threshold": threshold_report(y_test, proba_test, threshold, cfg.cost_fn, cfg.cost_fp),
        "at_threshold_0_5": threshold_report(y_test, proba_test, 0.5, cfg.cost_fn, cfg.cost_fp),
        "calibration": calibration_table(y_test, proba_test),
    }
    importance = permutation_importance(
        pipeline, X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=cfg.seed, n_jobs=1
    )
    feature_importance = sorted(
        ({"feature": f, "importance_mean": float(m), "importance_std": float(s)}
         for f, m, s in zip(FEATURE_COLUMNS, importance.importances_mean, importance.importances_std)),
        key=lambda r: -r["importance_mean"],
    )

    prepared_train = prepare_features(X_train)
    rule_context = build_context(prepared_train, y_train)

    metadata = {
        "format_version": BUNDLE_FORMAT_VERSION,
        "project_version": __version__,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_name": best_name,
        "best_params": next(c for c in comparison if c["model"] == best_name)["best_params"],
        "threshold": threshold,
        "cost_fn": cfg.cost_fn,
        "cost_fp": cfg.cost_fp,
        "settings": {"seed": cfg.seed, "test_size": cfg.test_size, "cv_folds": cfg.cv_folds,
                     "n_iter": cfg.n_iter, "models": list(cfg.models)},
        "data": {
            "file": cfg.data_path.name, "sha256": file_sha256(cfg.data_path), "n_rows": int(len(data)),
            "churn_rate": float(y.mean()), "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        },
        "cv_comparison": comparison,
        "threshold_selection": {"source": "out-of-fold predictions on train", "oof_report": oof_report,
                                "cost_curve": curve[["threshold", "cost", "precision", "recall"]].to_dict("records")},
        "test_metrics": test_metrics,
        "feature_importance": feature_importance,
        "rule_context": {"monthly_charges_q75": rule_context.monthly_charges_q75, "evidence": rule_context.evidence,
                         "overall_train_churn_rate": float(y_train.mean())},
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__, "lightgbm": lightgbm.__version__},
    }
    metadata = _jsonable(metadata)
    bundle = {"pipeline": pipeline, "metadata": metadata}

    # ---- Збереження артефакту й звітів ----
    _atomic_write(cfg.model_path, lambda p: joblib.dump(bundle, p))
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    (cfg.reports_dir / "training_report.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(comparison).drop(columns="best_params").to_csv(cfg.reports_dir / "cv_comparison.csv", index=False)
    pd.DataFrame(feature_importance).to_csv(cfg.reports_dir / "feature_importance.csv", index=False)
    pd.DataFrame({ID_COLUMN: ids_test.to_numpy(), "churn_probability": proba_test, "churn": y_test.to_numpy()}
                 ).to_csv(cfg.reports_dir / "test_predictions.csv", index=False)
    log.info("Модель збережено: %s", cfg.model_path)
    return metadata


def format_summary(meta: dict) -> str:
    t = meta["test_metrics"]
    ci = t["confidence_intervals"]
    thr = t["at_threshold"]
    lines = ["", "=== Порівняння моделей (CV на train) ==="]
    for c in meta["cv_comparison"]:
        lines.append(f"  {c['model']:<15} ROC-AUC {c['cv_roc_auc_mean']:.4f} ± {c['cv_roc_auc_std']:.4f}   "
                     f"PR-AUC {c['cv_pr_auc_mean']:.4f}   Brier {c['cv_brier_mean']:.4f}")
    lines += [
        f"Обрано: {meta['model_name']}  | поріг {meta['threshold']:.2f} (FN:FP = {meta['cost_fn']:g}:{meta['cost_fp']:g})",
        "", "=== Тестова вибірка (одноразове оцінювання) ===",
        f"  ROC-AUC {t['roc_auc']:.4f}  (95% CI {ci['roc_auc'][0]:.4f}–{ci['roc_auc'][1]:.4f})",
        f"  PR-AUC  {t['pr_auc']:.4f}  (95% CI {ci['pr_auc'][0]:.4f}–{ci['pr_auc'][1]:.4f})   базова частка відтоку {t['base_rate']:.4f}",
        f"  Brier {t['brier']:.4f}, log-loss {t['log_loss']:.4f}",
        f"  За порогом: precision {thr['precision']:.3f}, recall {thr['recall']:.3f}, F1 {thr['f1']:.3f}, "
        f"позначено {thr['flagged_share']:.1%} клієнтів",
        f"  Вартість помилок: {thr['cost']:g} проти {thr['cost_flag_nobody']:g} (нікого не втримувати) "
        f"і {thr['cost_flag_everybody']:g} (втримувати всіх)",
        "", "Топ-5 ознак за permutation importance (test):",
    ]
    lines += [f"  {r['feature']:<18} {r['importance_mean']:+.4f}" for r in meta["feature_importance"][:5]]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Навчання моделі відтоку клієнтів")
    parser.add_argument("--data", type=Path, default=None, help="шлях до CSV (за замовчуванням data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv)")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--reports-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--n-iter", type=int, default=config.N_ITER, help="кількість випадкових комбінацій гіперпараметрів на модель")
    parser.add_argument("--cv-folds", type=int, default=config.CV_FOLDS)
    parser.add_argument("--cost-fn", type=float, default=config.COST_FALSE_NEGATIVE, help="вартість пропущеного клієнта, що піде (відносні одиниці)")
    parser.add_argument("--cost-fp", type=float, default=config.COST_FALSE_POSITIVE, help="вартість зайвої пропозиції утримання")
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)

    cfg = TrainConfig(seed=args.seed, n_iter=args.n_iter, cv_folds=args.cv_folds,
                      cost_fn=args.cost_fn, cost_fp=args.cost_fp, n_jobs=args.n_jobs)
    if args.data:
        cfg.data_path = args.data
    if args.model_path:
        cfg.model_path = args.model_path
    if args.reports_dir:
        cfg.reports_dir = args.reports_dir
    try:
        meta = run_training(cfg)
    except DataError as exc:
        print(f"ПОМИЛКА ДАНИХ: {exc}", file=sys.stderr)
        return 2
    print(format_summary(meta))
    return 0


if __name__ == "__main__":
    sys.exit(main())
