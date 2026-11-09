"""Спільні фікстури. Для швидких тестів використовується синтетичний датасет тієї ж схеми;
реальний датасет перевіряється окремими тестами (схема) та CLI-навчанням."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from churn.schema import ADDON_COLUMNS, CATEGORICAL_ALLOWED, NO_INTERNET, NO_PHONE  # noqa: E402
from churn.train import TrainConfig, run_training  # noqa: E402


def make_synthetic_telco(n: int = 800, seed: int = 0) -> pd.DataFrame:
    """Синтетичні клієнти зі схемою Telco; відтік залежить від контракту, стажу, інтернету й платежу."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"customerID": [f"S-{i:05d}" for i in range(n)]})
    for col in ("gender", "Partner", "Dependents", "PhoneService", "InternetService", "Contract",
                "PaperlessBilling", "PaymentMethod"):
        df[col] = rng.choice(CATEGORICAL_ALLOWED[col], n)
    df["SeniorCitizen"] = rng.integers(0, 2, n)
    df["tenure"] = rng.integers(0, 73, n)
    df["MultipleLines"] = np.where(df["PhoneService"] == "No", NO_PHONE, rng.choice(["No", "Yes"], n))
    for col in ADDON_COLUMNS:
        df[col] = np.where(df["InternetService"] == "No", NO_INTERNET, rng.choice(["No", "Yes"], n))
    df["MonthlyCharges"] = np.round(rng.uniform(18, 118, n), 2)
    total = np.round(df["tenure"] * df["MonthlyCharges"], 2).astype(str)
    df["TotalCharges"] = np.where(df["tenure"] == 0, " ", total)  # як у реальному датасеті
    logit = (-1.0 + 1.6 * (df["Contract"] == "Month-to-month") - 0.04 * df["tenure"]
             + 0.8 * (df["InternetService"] == "Fiber optic") + 0.01 * (df["MonthlyCharges"] - 60))
    df["Churn"] = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "Yes", "No")
    columns = ["customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
               "MultipleLines", "InternetService", *ADDON_COLUMNS, "Contract", "PaperlessBilling",
               "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn"]
    return df[columns]


def tiny_config(data_path: Path, out_dir: Path) -> TrainConfig:
    return TrainConfig(data_path=data_path, model_path=out_dir / "model.joblib", reports_dir=out_dir / "reports",
                       n_iter=2, cv_folds=3, n_jobs=1, n_bootstrap=50)


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    return make_synthetic_telco()


@pytest.fixture(scope="session")
def synthetic_csv(tmp_path_factory, synthetic_df) -> Path:
    path = tmp_path_factory.mktemp("data") / "telco_synthetic.csv"
    synthetic_df.to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def trained(tmp_path_factory, synthetic_csv):
    """(metadata, cfg) після навчання на синтетичних даних зі скороченим пошуком."""
    cfg = tiny_config(synthetic_csv, tmp_path_factory.mktemp("run1"))
    return run_training(cfg), cfg


@pytest.fixture()
def valid_row(synthetic_df) -> dict:
    row = synthetic_df.drop(columns=["Churn"]).iloc[0].to_dict()
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}
