"""Завантаження збереженої моделі та прогнозування (єдина точка для API, CLI та тестів)."""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from churn.features import prepare_features
from churn.recommendations import RuleContext, recommend
from churn.schema import ID_COLUMN, ValidationIssue, validate_frame

log = logging.getLogger(__name__)

SUPPORTED_FORMAT_VERSION = 1
HIGH_RISK_PROBABILITY = 0.5  # «ймовірніше піде, ніж залишиться»; фіксована семантична межа, не підбирається


class ModelNotAvailableError(RuntimeError):
    """Модель відсутня або не може бути завантажена."""


class InputValidationError(ValueError):
    """Вхідні дані не пройшли перевірку. issues містить деталі по рядках і колонках."""

    def __init__(self, issues: list[ValidationIssue]):
        self.issues = issues
        first = "; ".join(f"{i.column}: {i.message}" for i in issues[:3])
        super().__init__(f"Некоректні вхідні дані ({len(issues)} проблем): {first}")


class ChurnPredictor:
    def __init__(self, pipeline, metadata: dict):
        classes = list(getattr(pipeline, "classes_", [0, 1]))
        if classes != [0, 1]:
            raise ModelNotAvailableError(f"Очікувалися класи [0, 1], у моделі: {classes}")
        self.pipeline = pipeline
        self.metadata = metadata
        self.threshold = float(metadata["threshold"])
        self.high_threshold = max(HIGH_RISK_PROBABILITY, self.threshold)
        rc = metadata["rule_context"]
        self.rule_context = RuleContext(float(rc["monthly_charges_q75"]), rc["evidence"])

    @classmethod
    def load(cls, path: Path) -> "ChurnPredictor":
        path = Path(path)
        if not path.is_file():
            raise ModelNotAvailableError(
                f"Файл моделі не знайдено: {path}. Спочатку навчіть модель: python -m churn.train"
            )
        try:
            bundle = joblib.load(path)
        except Exception as exc:
            raise ModelNotAvailableError(
                f"Не вдалося завантажити модель {path} ({type(exc).__name__}: {exc}). "
                "Імовірно, версії бібліотек відрізняються від тих, з якими модель навчалась — "
                "перенавчіть її: python -m churn.train"
            ) from exc
        if not isinstance(bundle, dict) or "pipeline" not in bundle or "metadata" not in bundle:
            raise ModelNotAvailableError(f"Файл {path} не є артефактом цього проєкту")
        if bundle["metadata"].get("format_version") != SUPPORTED_FORMAT_VERSION:
            raise ModelNotAvailableError(
                f"Непідтримувана версія формату моделі: {bundle['metadata'].get('format_version')}. Перенавчіть модель."
            )
        return cls(bundle["pipeline"], bundle["metadata"])

    def risk_level(self, probability: float) -> str:
        if probability >= self.high_threshold:
            return "high"
        if probability >= self.threshold:
            return "medium"
        return "low"

    def predict(self, frame: pd.DataFrame) -> list[dict]:
        """Прогноз для таблиці клієнтів. Колонка Churn не потрібна; customerID необов'язковий.

        Кидає InputValidationError, якщо дані не відповідають схемі.
        """
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("Очікується pandas.DataFrame")
        issues = validate_frame(frame)
        if issues:
            raise InputValidationError(issues)
        probabilities = self.pipeline.predict_proba(frame)[:, 1]
        at_risk = probabilities >= self.threshold
        recommendations = recommend(prepare_features(frame), at_risk, self.rule_context)
        ids = frame[ID_COLUMN].astype("string") if ID_COLUMN in frame.columns else None
        results = []
        for i, p in enumerate(probabilities):
            customer_id = None if ids is None or pd.isna(ids.iloc[i]) else str(ids.iloc[i])
            results.append({
                "customerID": customer_id,
                "churn_probability": float(p),
                "at_risk": bool(at_risk[i]),
                "risk_level": self.risk_level(float(p)),
                "recommendations": recommendations[i],
            })
        return results

    def predict_records(self, records: list[dict]) -> list[dict]:
        return self.predict(pd.DataFrame.from_records(records))

    def info(self) -> dict:
        """Метадані навчання плюс похідна межа високого ризику (вона залежить від порога)."""
        return {**self.metadata, "high_risk_threshold": self.high_threshold}
