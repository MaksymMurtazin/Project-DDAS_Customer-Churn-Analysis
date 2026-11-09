"""Перевірка інтерфейсу через streamlit.testing.AppTest проти справжнього HTTP-сервера API (waitress)."""
import threading
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from waitress.server import create_server

from churn.api import create_app
from churn.client import ApiClient, ApiError
from churn.summary import recommendation_counts, results_to_frame, summarize

UI_SCRIPT = str(Path(__file__).resolve().parents[1] / "churn" / "ui.py")

# значення за замовчуванням форми в churn/ui.py
UI_DEFAULT_PAYLOAD = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No", "tenure": 12,
    "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Bank transfer (automatic)", "MonthlyCharges": 70.0, "TotalCharges": None,
}


@pytest.fixture()
def api_server(trained):
    _, cfg = trained
    server = create_server(create_app(cfg.model_path), host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.effective_port}"
    server.close()


def run_app(monkeypatch, url):
    monkeypatch.setenv("CHURN_API_URL", url)
    return AppTest.from_file(UI_SCRIPT, default_timeout=60).run()


def test_ui_single_customer_matches_api(monkeypatch, api_server):
    at = run_app(monkeypatch, api_server)
    assert not at.exception
    at.button(key="btn_single").click().run()
    assert not at.exception
    expected = ApiClient(api_server).predict(UI_DEFAULT_PAYLOAD)
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Ймовірність відтоку"] == f"{expected['churn_probability']:.1%}"


def test_ui_shows_error_when_api_down(monkeypatch):
    at = run_app(monkeypatch, "http://127.0.0.1:9")
    assert not at.exception
    assert any("API недоступне" in e.value for e in at.error)


def test_ui_shows_error_when_model_missing(monkeypatch, tmp_path):
    server = create_server(create_app(tmp_path / "none.joblib"), host="127.0.0.1", port=0)
    threading.Thread(target=server.run, daemon=True).start()
    try:
        at = run_app(monkeypatch, f"http://127.0.0.1:{server.effective_port}")
        assert not at.exception
        assert any("python -m churn.train" in e.value for e in at.error)
    finally:
        server.close()


def test_client_batch_chunking_and_errors(api_server, synthetic_df):
    client = ApiClient(api_server)
    rows = synthetic_df.drop(columns=["Churn"]).head(25).to_dict("records")
    rows = [{k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()} for r in rows]
    whole = client.predict_batch(rows, chunk_size=1000)
    chunked = client.predict_batch(rows, chunk_size=10)
    assert len(whole) == len(chunked) == 25
    for a, b in zip(whole, chunked):  # останні розряди float можуть залежати від розміру пакета (BLAS)
        assert a["churn_probability"] == pytest.approx(b["churn_probability"], abs=1e-12)
        assert {k: v for k, v in a.items() if k != "churn_probability"} == {k: v for k, v in b.items() if k != "churn_probability"}
    rows[12] = {**rows[12], "Contract": "Weekly"}
    with pytest.raises(ApiError) as err:
        client.predict_batch(rows, chunk_size=10)
    assert err.value.kind == "validation_error" and err.value.details[0]["row"] == 12  # глобальний номер рядка
    with pytest.raises(ApiError) as down:
        ApiClient("http://127.0.0.1:9").health()
    assert down.value.kind == "api_unavailable"


def test_summary_functions(api_server, synthetic_df):
    customers = synthetic_df.head(60)
    rows = customers.drop(columns=["Churn"]).to_dict("records")
    results = ApiClient(api_server).predict_batch([{k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()} for r in rows])
    frame = results_to_frame(results)
    s = summarize(customers, frame)
    assert s["n_customers"] == 60
    assert s["at_risk_count"] == int(frame["at_risk"].sum())
    assert s["mrr_at_risk"] == pytest.approx(customers["MonthlyCharges"][frame["at_risk"].to_numpy()].sum())
    assert s["expected_monthly_loss"] == pytest.approx((frame["churn_probability"] * customers["MonthlyCharges"].to_numpy()).sum())
    assert s["actual_churn_rate"] == pytest.approx((customers["Churn"] == "Yes").mean())
    assert summarize(customers.drop(columns=["Churn"]), frame)["actual_churn_rate"] is None
    counts = recommendation_counts(results)
    assert counts["customers"].max() <= s["at_risk_count"]
    with pytest.raises(ValueError):
        summarize(customers.head(5), frame)


def test_ui_displays_cost_ratio_with_nonunit_fp(
    monkeypatch, trained, tmp_path
):
    import joblib

    _, cfg = trained
    bundle = joblib.load(cfg.model_path)

    # Змінюємо метадані лише для перевірки підпису в інтерфейсі.
    bundle["metadata"]["cost_fn"] = 5.0
    bundle["metadata"]["cost_fp"] = 2.0

    model_path = tmp_path / "ratio_test.joblib"
    joblib.dump(bundle, model_path)

    server = create_server(
        create_app(model_path),
        host="127.0.0.1",
        port=0,
    )
    threading.Thread(
        target=server.run,
        daemon=True,
    ).start()

    try:
        at = run_app(
            monkeypatch,
            f"http://127.0.0.1:{server.effective_port}",
        )

        assert not at.exception

        captions = " ".join(
            item.value for item in at.caption
        )
        assert "FN:FP = 5:2" in captions
        assert "співвідношення 2.5:1" in captions
        assert "є лише припущенням" in captions
    finally:
        server.close()


def test_threshold_switch_updates_test_counts_without_changing_api(monkeypatch, api_server):
    at = run_app(monkeypatch, api_server)
    assert not at.exception
    before = ApiClient(api_server).model_info()
    original = before["test_metrics"]["at_threshold"]
    standard = before["test_metrics"]["at_threshold_0_5"]
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["FN · пропущений відтік"] == f"{original['fn']:,}"
    assert len(at.get("plotly_chart")) == 5
    at.radio(key="model_threshold_view").set_value("default").run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["FN · пропущений відтік"] == f"{standard['fn']:,}"
    assert metrics["FP · хибні тривоги"] == f"{standard['fp']:,}"
    assert ApiClient(api_server).model_info()["threshold"] == before["threshold"]
    at.radio(key="model_threshold_view").set_value("trained").run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["FN · пропущений відтік"] == f"{original['fn']:,}"


def test_refresh_button_invalidates_model_info_cache(monkeypatch, api_server):
    from churn.client import ApiClient

    original = ApiClient.model_info
    calls = []

    def tracked(self):
        calls.append(1)
        return original(self)

    monkeypatch.setattr(ApiClient, "model_info", tracked)
    at = run_app(monkeypatch, api_server)
    assert not at.exception
    before = len(calls)
    at.button(key="refresh_model").click().run()
    assert not at.exception
    assert len(calls) == before + 1
