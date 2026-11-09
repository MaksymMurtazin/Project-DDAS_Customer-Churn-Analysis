import shutil

import numpy as np
import pandas as pd
import pytest

from churn.api import create_app
from churn.predict import main as predict_main
from churn.predictor import ChurnPredictor
from churn.schema import FEATURE_COLUMNS


@pytest.fixture()
def client(trained):
    _, cfg = trained
    return create_app(cfg.model_path).test_client()


def records(df, n):
    return [{k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}
            for row in df.drop(columns=["Churn"]).head(n).to_dict("records")]


def test_health_and_info(client, trained):
    meta, _ = trained
    assert client.get("/health").get_json() == {"status": "ok", "model_loaded": True, "message": None}
    info = client.get("/model/info").get_json()
    assert info["threshold"] == meta["threshold"] and info["model_name"] == meta["model_name"]
    assert info["high_risk_threshold"] == max(0.5, meta["threshold"])


def test_predict_single_matches_predictor(client, trained, synthetic_df):
    _, cfg = trained
    row = records(synthetic_df, 1)[0]
    body = client.post("/predict", json=row).get_json()
    expected = ChurnPredictor.load(cfg.model_path).predict(pd.DataFrame([row]))[0]
    assert body["result"] == expected
    assert body["result"]["customerID"] == row["customerID"]


def test_predict_batch_matches_single_and_predictor(client, trained, synthetic_df):
    _, cfg = trained
    rows = records(synthetic_df, 25)
    batch = client.post("/predict/batch", json={"customers": rows}).get_json()
    assert batch["count"] == 25
    direct = ChurnPredictor.load(cfg.model_path).predict(pd.DataFrame(rows))
    assert batch["results"] == direct
    single = client.post("/predict", json=rows[7]).get_json()["result"]
    assert single["churn_probability"] == pytest.approx(batch["results"][7]["churn_probability"], abs=1e-12)


def test_no_target_or_id_required(client, synthetic_df):
    row = {c: v for c, v in records(synthetic_df, 1)[0].items() if c in FEATURE_COLUMNS}
    assert client.post("/predict", json=row).status_code == 200


def test_invalid_requests(client, synthetic_df):
    row = records(synthetic_df, 1)[0]
    assert client.post("/predict", data="not json", content_type="application/json").status_code == 400
    assert client.post("/predict", json=[row]).status_code == 400
    assert client.post("/predict/batch", json={"customers": []}).status_code == 400
    assert client.post("/predict/batch", json={"customers": [1]}).status_code == 400
    assert client.post("/predict/batch", json=row).status_code == 400

    bad = client.post("/predict", json={**row, "Contract": "Weekly", "tenure": -3})
    assert bad.status_code == 422
    body = bad.get_json()
    assert body["error"] == "validation_error"
    assert {(d["row"], d["column"]) for d in body["details"]} == {(0, "Contract"), (0, "tenure")}

    missing = client.post("/predict", json={k: v for k, v in row.items() if k != "tenure"})
    assert missing.status_code == 422 and missing.get_json()["details"][0]["column"] == "tenure"

    batch = client.post("/predict/batch", json={"customers": [row, {**row, "gender": "X"}]})
    assert batch.status_code == 422 and batch.get_json()["details"][0]["row"] == 1

    assert client.get("/nope").status_code == 404
    assert client.get("/predict").status_code == 405
    assert client.get("/nope").get_json()["error"]


def test_missing_model_gives_503_and_recovers(tmp_path, trained, synthetic_df):
    _, cfg = trained
    path = tmp_path / "models" / "model.joblib"
    client = create_app(path).test_client()
    health = client.get("/health").get_json()
    assert health["model_loaded"] is False and "python -m churn.train" in health["message"]
    row = records(synthetic_df, 1)[0]
    for response in (client.post("/predict", json=row), client.post("/predict/batch", json={"customers": [row]}),
                     client.get("/model/info")):
        assert response.status_code == 503 and response.get_json()["error"] == "model_not_available"
    # після появи файлу моделі API підхоплює її без перезапуску
    path.parent.mkdir()
    shutil.copy(cfg.model_path, path)
    assert client.post("/predict", json=row).status_code == 200


def test_corrupted_model_gives_503(tmp_path, synthetic_df):
    path = tmp_path / "model.joblib"
    path.write_bytes(b"garbage")
    client = create_app(path).test_client()
    response = client.post("/predict", json=records(synthetic_df, 1)[0])
    assert response.status_code == 503 and "перенавчіть" in response.get_json()["message"]


def test_cli_predict(trained, synthetic_df, tmp_path, capsys):
    _, cfg = trained
    src = tmp_path / "new.csv"
    synthetic_df.drop(columns=["Churn"]).head(40).to_csv(src, index=False)
    out = tmp_path / "out.csv"
    assert predict_main(["--input", str(src), "--output", str(out), "--model-path", str(cfg.model_path)]) == 0
    result = pd.read_csv(out)
    expected = ChurnPredictor.load(cfg.model_path).predict(pd.read_csv(src))
    assert len(result) == 40
    np.testing.assert_allclose(result["churn_probability"], [r["churn_probability"] for r in expected], atol=1e-12)

    bad = tmp_path / "bad.csv"
    pd.read_csv(src).drop(columns=["tenure"]).to_csv(bad, index=False)
    assert predict_main(["--input", str(bad), "--output", str(out), "--model-path", str(cfg.model_path)]) == 2
    assert predict_main(["--input", str(src), "--output", str(out), "--model-path", str(tmp_path / "x.joblib")]) == 3
    assert predict_main(["--input", str(tmp_path / "missing.csv"), "--output", str(out)]) == 2


@pytest.mark.parametrize("column, value", [
    ("InternetService", None),
    ("PhoneService", None),
    ("TechSupport", None),
    ("TotalCharges", "inf"),
])
def test_invalid_service_fields_return_422(
    client, synthetic_df, column, value
):
    row = records(synthetic_df, 1)[0]

    response = client.post(
        "/predict",
        json={**row, column: value},
    )

    assert response.status_code == 422

    body = response.get_json()
    assert body["error"] == "validation_error"
    assert column in {
        item["column"] for item in body["details"]
    }
