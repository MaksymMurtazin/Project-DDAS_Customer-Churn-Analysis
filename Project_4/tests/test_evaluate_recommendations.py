import numpy as np
import pandas as pd

from churn.evaluate import bootstrap_ci, calibration_table, choose_threshold, cost_curve, threshold_report
from churn.features import prepare_features
from churn.recommendations import RULES, build_context, recommend


def test_threshold_report_counts_and_costs():
    y = np.array([1, 1, 0, 0, 0, 1])
    p = np.array([0.9, 0.2, 0.6, 0.1, 0.3, 0.7])
    r = threshold_report(y, p, 0.5, cost_fn=5, cost_fp=1)
    assert (r["tp"], r["fp"], r["fn"], r["tn"]) == (2, 1, 1, 2)
    assert r["cost"] == 5 * 1 + 1 * 1
    assert r["cost_flag_nobody"] == 15 and r["cost_flag_everybody"] == 3
    assert abs(r["precision"] - 2 / 3) < 1e-12 and abs(r["recall"] - 2 / 3) < 1e-12


def test_choose_threshold_minimises_cost():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.6, 0.25, 0.3, 0.2, 0.1, 0.05, 0.05])
    # FN дуже дорогі -> поріг опускається до 0.25, щоб упіймати всіх відтікаючих
    t, curve = choose_threshold(y, p, cost_fn=10, cost_fp=1)
    assert t <= 0.25
    assert curve["cost"].min() == curve.loc[curve["threshold"] == t, "cost"].iloc[0] == 1
    # FP дуже дорогі -> поріг піднімається вище 0.3, жертвуючи одним клієнтом
    t2, curve2 = choose_threshold(y, p, cost_fn=1, cost_fp=10)
    assert 0.3 < t2 <= 0.6
    assert curve2["cost"].min() == 1


def test_cost_curve_extremes():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.8, 0.4, 0.6, 0.2])
    c = cost_curve(y, p, [0.0, 1.01], cost_fn=5, cost_fp=1)
    assert c["cost"].tolist() == [2 * 1, 2 * 5]  # позначити всіх / нікого


def test_bootstrap_and_calibration_are_sane():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.clip(0.3 * y + rng.random(500) * 0.7, 0, 1)
    ci = bootstrap_ci(y, p, seed=1, n_boot=100)
    assert 0.5 < ci["roc_auc"][0] < ci["roc_auc"][1] <= 1.0
    cal = calibration_table(y, p)
    assert sum(b["n"] for b in cal) == 500


def _prepared(valid_row, **changes):
    base = {**valid_row, "InternetService": "Fiber optic", "PhoneService": "Yes", "MultipleLines": "No",
            "Contract": "Two year", "PaymentMethod": "Mailed check", "tenure": 40, "MonthlyCharges": 50.0,
            **{c: "Yes" for c in ("OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies")}}
    return pd.DataFrame([{**base, **changes}])


def test_rules_fire_only_for_at_risk_customers(valid_row):
    frame = pd.concat([_prepared(valid_row), _prepared(valid_row, Contract="Month-to-month", TechSupport="No",
                       PaymentMethod="Electronic check", tenure=3, MonthlyCharges=110.0)], ignore_index=True)
    prepared = prepare_features(frame)
    ctx = build_context(prepared, pd.Series([0, 1]))
    flagged = recommend(prepared, [False, True], ctx)
    assert flagged[0] == []
    assert {r["rule_id"] for r in flagged[1]} == {"month_to_month", "no_tech_support", "electronic_check",
                                                  "high_monthly_charge", "new_customer"}
    # той самий клієнт, але нижче порога моделі -> рекомендацій немає
    assert recommend(prepared, [False, False], ctx)[1] == []


def test_internet_rules_ignore_customers_without_internet(valid_row):
    frame = _prepared(valid_row, InternetService="No", **{c: "No internet service" for c in (
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies")})
    prepared = prepare_features(frame)
    ctx = build_context(prepared, pd.Series([0]))
    ids = {r["rule_id"] for r in recommend(prepared, [True], ctx)[0]}
    assert "no_tech_support" not in ids and "no_online_security" not in ids


def test_rule_evidence_uses_given_labels(valid_row):
    frame = pd.concat([_prepared(valid_row, Contract="Month-to-month")] * 4, ignore_index=True)
    prepared = prepare_features(frame)
    ctx = build_context(prepared, pd.Series([1, 1, 0, 0]))
    assert ctx.evidence["month_to_month"] == {"n": 4, "churn_rate": 0.5}
    assert {r.id for r in RULES} == set(ctx.evidence)
