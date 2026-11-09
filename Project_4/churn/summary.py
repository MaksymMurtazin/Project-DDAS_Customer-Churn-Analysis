"""Агрегація результатів пакетного прогнозу для дашборду (чисті функції без Streamlit)."""
from __future__ import annotations

import pandas as pd

from churn.schema import POSITIVE_LABEL, TARGET_COLUMN


def results_to_frame(results: list[dict]) -> pd.DataFrame:
    return pd.DataFrame({
        "customerID": [r["customerID"] for r in results],
        "churn_probability": [r["churn_probability"] for r in results],
        "at_risk": [r["at_risk"] for r in results],
        "risk_level": [r["risk_level"] for r in results],
        "recommendations (rules)": ["; ".join(x["action"] for x in r["recommendations"]) for r in results],
    })


def summarize(customers: pd.DataFrame, results_frame: pd.DataFrame) -> dict:
    """Ключові показники. customers і results_frame мають однаковий порядок рядків."""
    if len(customers) != len(results_frame):
        raise ValueError("Кількість клієнтів і прогнозів не збігається")
    monthly = pd.to_numeric(customers["MonthlyCharges"], errors="coerce").to_numpy()
    prob = results_frame["churn_probability"].to_numpy()
    at_risk = results_frame["at_risk"].to_numpy(dtype=bool)
    actual = None
    if TARGET_COLUMN in customers.columns:
        labels = customers[TARGET_COLUMN].astype("string").str.strip()
        if labels.isin(["Yes", "No"]).all():
            actual = float((labels == POSITIVE_LABEL).mean())
    return {
        "n_customers": int(len(customers)),
        "avg_tenure": float(pd.to_numeric(customers["tenure"], errors="coerce").mean()),
        "mean_probability": float(prob.mean()),
        "at_risk_count": int(at_risk.sum()),
        "at_risk_share": float(at_risk.mean()),
        "high_risk_count": int((results_frame["risk_level"] == "high").sum()),
        "mrr_at_risk": float(monthly[at_risk].sum()),
        "expected_monthly_loss": float((prob * monthly).sum()),
        "actual_churn_rate": actual,
    }


def recommendation_counts(results: list[dict]) -> pd.DataFrame:
    """Скільки клієнтів у зоні ризику отримали кожну рекомендацію (правила, а не модель)."""
    counts: dict[str, int] = {}
    for r in results:
        for item in r["recommendations"]:
            counts[item["action"]] = counts.get(item["action"], 0) + 1
    frame = pd.DataFrame({"recommendation": list(counts), "customers": list(counts.values())})
    return frame.sort_values("customers", ascending=False).reset_index(drop=True)
