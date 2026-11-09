"""Рекомендації з утримання — це БІЗНЕС-ПРАВИЛА, а не прогноз моделі.

Модель оцінює лише ймовірність відтоку. Правила нижче — експертні евристики на
основі помітних закономірностей у даних: вони кореляційні (не доводять причинності),
не перевіряють ефективність втручань і спрацьовують лише для клієнтів, яких модель
позначила «у зоні ризику». Для кожного правила під час навчання рахується фактична
частка відтоку в його сегменті навчальної вибірки — це пояснення, чому правило існує.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd



@dataclass(frozen=True)
class RuleContext:
    """Параметри, обчислені на навчальній вибірці."""

    monthly_charges_q75: float
    evidence: dict  # rule_id -> {"n": int, "churn_rate": float}


@dataclass(frozen=True)
class Rule:
    id: str
    segment: str
    action: str
    condition: Callable[[pd.DataFrame, RuleContext], pd.Series]


# Умови працюють з підготовленими ознаками (churn.features.prepare_features).
RULES: tuple[Rule, ...] = (
    Rule(
        "month_to_month",
        "контракт «місяць за місяцем»",
        "Запропонувати перехід на річний або дворічний контракт зі знижкою.",
        lambda f, c: f["Contract"] == "Month-to-month",
    ),
    Rule(
        "no_tech_support",
        "інтернет-клієнти без технічної підтримки",
        "Запропонувати безкоштовний пробний період технічної підтримки.",
        lambda f, c: (f["InternetService"] != "No") & (f["TechSupport"] == "No"),
    ),
    Rule(
        "no_online_security",
        "інтернет-клієнти без онлайн-безпеки",
        "Запропонувати пакет онлайн-безпеки з пробним періодом.",
        lambda f, c: (f["InternetService"] != "No") & (f["OnlineSecurity"] == "No"),
    ),
    Rule(
        "electronic_check",
        "оплата електронним чеком",
        "Запропонувати перейти на автоматичну оплату (картка або банківське списання).",
        lambda f, c: f["PaymentMethod"] == "Electronic check",
    ),
    Rule(
        "high_monthly_charge",
        "місячний платіж вище 75-го перцентиля навчальної вибірки",
        "Переглянути тарифний план і підготувати персональну цінову пропозицію.",
        lambda f, c: f["MonthlyCharges"] > c.monthly_charges_q75,
    ),
    Rule(
        "new_customer",
        "стаж до 12 місяців включно",
        "Зв'язатися з клієнтом на ранньому етапі (онбординг, перевірка задоволеності).",
        lambda f, c: f["tenure"] <= 12,
    ),
)


def build_context(prepared: pd.DataFrame, y: pd.Series) -> RuleContext:
    """Обчислює параметри правил і фактичний відтік у сегментах за НАВЧАЛЬНОЮ вибіркою."""
    ctx = RuleContext(monthly_charges_q75=float(prepared["MonthlyCharges"].quantile(0.75)), evidence={})
    evidence = {}
    for rule in RULES:
        mask = rule.condition(prepared, ctx).to_numpy(dtype=bool)
        evidence[rule.id] = {
            "n": int(mask.sum()),
            "churn_rate": float(y.to_numpy()[mask].mean()) if mask.any() else None,
        }
    return RuleContext(ctx.monthly_charges_q75, evidence)


def recommend(prepared: pd.DataFrame, at_risk, ctx: RuleContext) -> list[list[dict]]:
    """Для кожного рядка повертає список спрацьованих правил (порожній, якщо клієнт не в зоні ризику)."""
    at_risk = pd.Series(at_risk, index=prepared.index).to_numpy(dtype=bool)
    fired = {rule.id: rule.condition(prepared, ctx).to_numpy(dtype=bool) for rule in RULES}
    result: list[list[dict]] = []
    for i in range(len(prepared)):
        items = []
        if at_risk[i]:
            for rule in RULES:
                if fired[rule.id][i]:
                    ev = ctx.evidence.get(rule.id, {})
                    items.append({
                        "rule_id": rule.id,
                        "action": rule.action,
                        "segment": rule.segment,
                        "segment_churn_rate": ev.get("churn_rate"),
                        "segment_size": ev.get("n"),
                    })
        result.append(items)
    return result
