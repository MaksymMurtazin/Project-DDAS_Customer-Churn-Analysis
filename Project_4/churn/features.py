"""Підготовка ознак і побудова sklearn-пайплайна.

prepare_features — перший крок збереженого пайплайна. Тому навчання, API, CLI
й інтерфейс використовують ту саму обробку, а цільова змінна для прогнозування
не потрібна. Усі правила тут детерміновані (не навчаються на даних), тож витоку
інформації з тестової вибірки немає; навчальні статистики (медіани, масштаб,
категорії) обчислюються лише всередині Pipeline.fit.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from churn.schema import (
    CATEGORICAL_ALLOWED,
    FEATURE_COLUMNS,
    clean_string,
    to_binary,
    to_float,
)

CATEGORICAL_FEATURES = list(CATEGORICAL_ALLOWED)
NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
MODEL_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Очищує сирі колонки Telco (типи, пробіли, порожній TotalCharges). customerID/Churn ігноруються.

    Похідні ознаки (кількість додаткових послуг, різниця платежів) перевірялися на train-CV
    і не дали приросту якості, тому не використовуються.
    """
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"У даних бракує колонок: {', '.join(missing)}")

    out = pd.DataFrame(index=df.index)
    for column in CATEGORICAL_FEATURES:
        out[column] = clean_string(df[column]).astype(object)

    tenure = to_float(df["tenure"])
    monthly = to_float(df["MonthlyCharges"])
    total = to_float(df["TotalCharges"])
    # Порожній TotalCharges (нові клієнти) -> оцінка tenure * MonthlyCharges (для tenure=0 це 0).
    total = total.where(total.notna(), tenure * monthly)

    out["SeniorCitizen"] = to_binary(df["SeniorCitizen"])
    out["tenure"] = tenure
    out["MonthlyCharges"] = monthly
    out["TotalCharges"] = total
    return out[MODEL_COLUMNS]


def build_pipeline(estimator) -> Pipeline:
    """Єдиний пайплайн: підготовка -> імпутація/масштаб/one-hot -> модель."""
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    preprocess = ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline([
        ("prepare", FunctionTransformer(prepare_features, validate=False)),
        ("preprocess", preprocess),
        ("model", estimator),
    ])
