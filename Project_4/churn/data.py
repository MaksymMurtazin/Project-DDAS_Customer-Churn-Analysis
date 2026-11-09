"""Завантаження та перевірка навчального датасету."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from churn.schema import (
    FEATURE_COLUMNS,
    ID_COLUMN,
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    TARGET_COLUMN,
    validate_frame,
)


class DataError(Exception):
    """Проблема з навчальними даними (зрозуміле повідомлення для користувача)."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Читає CSV і повертає (дані з customerID та ознаками, ціль 0/1).

    Кидає DataError, якщо файлу немає або дані не відповідають схемі.
    """
    path = Path(path)
    if not path.is_file():
        raise DataError(
            f"Файл з даними не знайдено: {path}\n"
            "Покладіть CSV Telco Customer Churn у data/raw/ "
            "(WA_Fn-UseC_-Telco-Customer-Churn.csv) або вкажіть шлях через --data."
        )
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # некоректний CSV, кодування тощо
        raise DataError(f"Не вдалося прочитати CSV {path}: {exc}") from exc

    required = [ID_COLUMN, TARGET_COLUMN, *FEATURE_COLUMNS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataError(f"У файлі {path.name} бракує колонок: {', '.join(missing)}")

    if df[ID_COLUMN].isna().any() or df[ID_COLUMN].duplicated().any():
        raise DataError("customerID має бути заповненим і унікальним")

    target = df[TARGET_COLUMN].astype("string").str.strip()
    bad = ~target.isin([POSITIVE_LABEL, NEGATIVE_LABEL])
    if bad.any():
        raise DataError(
            f"Колонка {TARGET_COLUMN} має містити лише '{POSITIVE_LABEL}'/'{NEGATIVE_LABEL}'; "
            f"некоректних значень: {int(bad.sum())}"
        )
    issues = validate_frame(df)
    if issues:
        shown = "; ".join(f"рядок {i.row}, {i.column}: {i.message}" for i in issues[:10])
        raise DataError(f"Дані не відповідають схемі ({len(issues)} проблем): {shown}")

    y = (target == POSITIVE_LABEL).astype(int)
    y.name = TARGET_COLUMN
    return df[[ID_COLUMN, *FEATURE_COLUMNS]].reset_index(drop=True), y.reset_index(drop=True)
