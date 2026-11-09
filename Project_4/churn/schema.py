"""Схема даних Telco Customer Churn та її валідація.

Це єдине джерело правди про колонки й допустимі значення: його використовують
навчання, API, CLI та інтерфейс (форма вводу будується з CATEGORICAL_ALLOWED).
Схему звірено з фактичним CSV (7043 рядки, 21 колонка).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"
POSITIVE_LABEL = "Yes"
NEGATIVE_LABEL = "No"

YES_NO = ("No", "Yes")
ADDON_COLUMNS = (
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
)
NO_INTERNET = "No internet service"
NO_PHONE = "No phone service"

CATEGORICAL_ALLOWED: dict[str, tuple[str, ...]] = {
    "gender": ("Female", "Male"),
    "Partner": YES_NO,
    "Dependents": YES_NO,
    "PhoneService": YES_NO,
    "MultipleLines": ("No", "Yes", NO_PHONE),
    "InternetService": ("DSL", "Fiber optic", "No"),
    **{col: ("No", "Yes", NO_INTERNET) for col in ADDON_COLUMNS},
    "Contract": ("Month-to-month", "One year", "Two year"),
    "PaperlessBilling": YES_NO,
    "PaymentMethod": (
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ),
}
BINARY_COLUMNS = ("SeniorCitizen",)
NUMERIC_COLUMNS = ("tenure", "MonthlyCharges", "TotalCharges")

# Порядок колонок як у вихідному CSV (без customerID та Churn)
FEATURE_COLUMNS = (
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "MultipleLines", "InternetService", *ADDON_COLUMNS, "Contract",
    "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges",
)


@dataclass(frozen=True)
class ValidationIssue:
    """Одна знайдена проблема. row — позиція рядка з 0 (None — проблема всієї таблиці)."""

    row: int | None
    column: str
    message: str

    def as_dict(self) -> dict:
        return asdict(self)


# ---------- допоміжні перетворення (спільні з features.py) ----------

def clean_string(series: pd.Series) -> pd.Series:
    """Рядки без крайніх пробілів; порожній рядок вважається пропуском."""
    out = series.astype("string").str.strip()
    return out.mask(out == "")


def to_float(series: pd.Series) -> pd.Series:
    """Число або NaN (нечислові значення стають NaN)."""
    return pd.to_numeric(clean_string(series), errors="coerce").astype("float64")


def to_binary(series: pd.Series) -> pd.Series:
    """0/1 з урахуванням значень '0', '1', 'true', 'false', 1.0; інше -> NaN."""
    text = clean_string(series).str.lower()
    mapped = text.replace({"true": "1", "false": "0", "1.0": "1", "0.0": "0"})
    return pd.to_numeric(mapped.where(mapped.isin(["0", "1"])), errors="coerce").astype("float64")


# ---------- валідація ----------

def validate_frame(df: pd.DataFrame, max_issues_per_column: int = 10) -> list[ValidationIssue]:
    """Перевіряє таблицю клієнтів. Колонки customerID та Churn не потрібні.

    Повертає список проблем (порожній — дані коректні). Для кожної колонки
    повертається не більше max_issues_per_column проблем, щоб відповідь не роздувалась.
    """
    issues: list[ValidationIssue] = []
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        return [ValidationIssue(None, c, "відсутня обов'язкова колонка") for c in missing]
    if len(df) == 0:
        return [ValidationIssue(None, "*", "таблиця порожня")]

    def add(column: str, mask: np.ndarray | pd.Series, message: str) -> None:
        safe_mask = pd.Series(mask, dtype="boolean").fillna(False)
        positions = np.flatnonzero(safe_mask.to_numpy(dtype=bool))

        for pos in positions[:max_issues_per_column]:
            issues.append(ValidationIssue(int(pos), column, message))

        extra = len(positions) - max_issues_per_column
        if extra > 0:
            issues.append(
                ValidationIssue(
                    None,
                    column,
                    f"ще {extra} рядків з такою самою проблемою",
                )
            )

    cleaned: dict[str, pd.Series] = {}

    for column, allowed in CATEGORICAL_ALLOWED.items():
        values = clean_string(df[column])
        cleaned[column] = values
        add(column, values.isna().to_numpy(), "значення відсутнє")
        bad = (~values.isin(allowed) & values.notna()).to_numpy()
        add(column, bad, f"недопустиме значення; дозволені: {', '.join(allowed)}")

    senior = to_binary(df["SeniorCitizen"])
    add("SeniorCitizen", senior.isna().to_numpy(), "очікується 0 або 1")

    tenure = to_float(df["tenure"])
    add("tenure", (tenure.isna() | ~np.isfinite(tenure)).to_numpy(), "очікується число")
    add("tenure", (tenure < 0).to_numpy(), "не може бути від'ємним")

    monthly = to_float(df["MonthlyCharges"])
    add("MonthlyCharges", (monthly.isna() | ~np.isfinite(monthly)).to_numpy(), "очікується число")
    add("MonthlyCharges", (monthly < 0).to_numpy(), "не може бути від'ємним")

    # TotalCharges може бути порожнім (у датасеті це клієнти з tenure = 0),
    # але якщо значення є — воно має бути скінченним невід'ємним числом.
    raw_total = clean_string(df["TotalCharges"])
    total = to_float(df["TotalCharges"])

    invalid_total = raw_total.notna() & (
        total.isna() | ~np.isfinite(total)
    )
    add(
        "TotalCharges",
        invalid_total.to_numpy(),
        "очікується скінченне число або порожнє значення",
    )
    add(
        "TotalCharges",
        (total < 0).to_numpy(),
        "не може бути від'ємним",
    )

    # Логічна узгодженість послуг (у датасеті порушень немає).
    internet = cleaned["InternetService"]
    for column in ADDON_COLUMNS:
        value = cleaned[column]
        add(column, ((internet == "No") & value.notna() & (value != NO_INTERNET)).to_numpy(),
            f"без інтернету (InternetService=No) значення має бути '{NO_INTERNET}'")
        add(column, ((internet.isin(["DSL", "Fiber optic"])) & (value == NO_INTERNET)).to_numpy(),
            f"'{NO_INTERNET}' можливе лише при InternetService=No")
    phone, lines = cleaned["PhoneService"], cleaned["MultipleLines"]
    add("MultipleLines", ((phone == "No") & lines.notna() & (lines != NO_PHONE)).to_numpy(),
        f"без телефонного зв'язку (PhoneService=No) значення має бути '{NO_PHONE}'")
    add("MultipleLines", ((phone == "Yes") & (lines == NO_PHONE)).to_numpy(),
        f"'{NO_PHONE}' можливе лише при PhoneService=No")
    return issues
