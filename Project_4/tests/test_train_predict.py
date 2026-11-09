import json
import shutil

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from churn.data import DataError, load_dataset
from churn.predictor import ChurnPredictor, InputValidationError, ModelNotAvailableError
from churn.schema import FEATURE_COLUMNS
from churn.train import main as train_main, run_training

from conftest import tiny_config


def test_artifacts_and_metadata(trained):
    meta, cfg = trained
    assert cfg.model_path.is_file()
    for name in ("training_report.json", "cv_comparison.csv", "feature_importance.csv", "test_predictions.csv"):
        assert (cfg.reports_dir / name).is_file()
    assert 0 < meta["threshold"] < 1
    assert meta["data"]["n_train"] + meta["data"]["n_test"] == meta["data"]["n_rows"]
    assert meta["model_name"] in {"logreg", "random_forest", "lightgbm"}
    assert {c["model"] for c in meta["cv_comparison"]} == {"baseline_prior", "logreg", "random_forest", "lightgbm"}
    assert json.loads((cfg.reports_dir / "training_report.json").read_text(encoding="utf-8"))["threshold"] == meta["threshold"]
    # модель на синтетичних даних із реальним сигналом має бути кращою за випадкову
    assert meta["test_metrics"]["roc_auc"] > 0.6


def test_split_has_no_overlap(trained, synthetic_df):
    meta, cfg = trained
    data, y = load_dataset(cfg.data_path)
    train, test = train_test_split(data, test_size=cfg.test_size, stratify=y, random_state=cfg.seed)
    saved = pd.read_csv(cfg.reports_dir / "test_predictions.csv")
    assert set(saved["customerID"]) == set(test["customerID"])
    assert set(saved["customerID"]).isdisjoint(set(train["customerID"]))


def test_save_and_reload_gives_same_predictions(trained, synthetic_df):
    meta, cfg = trained
    saved = pd.read_csv(cfg.reports_dir / "test_predictions.csv")  # прогнози з моделі в пам'яті під час навчання
    frame = synthetic_df.set_index("customerID").loc[saved["customerID"]].reset_index()
    predictor = ChurnPredictor.load(cfg.model_path)
    reloaded = np.array([r["churn_probability"] for r in predictor.predict(frame)])
    np.testing.assert_allclose(reloaded, saved["churn_probability"].to_numpy(), rtol=0, atol=1e-9)
    again = ChurnPredictor.load(cfg.model_path).predict(frame)
    assert [r["churn_probability"] for r in again] == reloaded.tolist()


def test_predict_without_target_single_and_batch(trained, synthetic_df):
    _, cfg = trained
    predictor = ChurnPredictor.load(cfg.model_path)
    features = synthetic_df[list(FEATURE_COLUMNS)]  # без customerID та Churn
    batch = predictor.predict(features.head(30))
    assert len(batch) == 30 and all(r["customerID"] is None for r in batch)
    single = predictor.predict(features.iloc[[5]])[0]
    assert single["churn_probability"] == pytest.approx(batch[5]["churn_probability"], abs=1e-12)
    for r in batch:
        assert 0 <= r["churn_probability"] <= 1
        assert r["at_risk"] == (r["churn_probability"] >= predictor.threshold)
        assert (r["risk_level"] == "low") == (not r["at_risk"])
        assert bool(r["recommendations"]) <= r["at_risk"]  # рекомендації лише для клієнтів у зоні ризику


def test_predict_rejects_invalid_input(trained, synthetic_df):
    _, cfg = trained
    predictor = ChurnPredictor.load(cfg.model_path)
    bad = synthetic_df.drop(columns=["Churn"]).head(3).copy()
    bad.loc[1, "Contract"] = "Weekly"
    with pytest.raises(InputValidationError) as err:
        predictor.predict(bad)
    assert [(i.row, i.column) for i in err.value.issues] == [(1, "Contract")]
    with pytest.raises(InputValidationError):
        predictor.predict(bad.drop(columns=["tenure"]))
    with pytest.raises(InputValidationError):
        predictor.predict(bad.iloc[0:0])


def test_missing_and_corrupted_model(tmp_path):
    with pytest.raises(ModelNotAvailableError, match="python -m churn.train"):
        ChurnPredictor.load(tmp_path / "none.joblib")
    broken = tmp_path / "broken.joblib"
    broken.write_bytes(b"not a model")
    with pytest.raises(ModelNotAvailableError, match="перенавчіть"):
        ChurnPredictor.load(broken)
    wrong = tmp_path / "wrong.joblib"
    joblib.dump({"foo": 1}, wrong)
    with pytest.raises(ModelNotAvailableError, match="не є артефактом"):
        ChurnPredictor.load(wrong)


def test_training_is_reproducible(trained, synthetic_csv, tmp_path):
    meta1, _ = trained
    meta2 = run_training(tiny_config(synthetic_csv, tmp_path))
    assert meta2["model_name"] == meta1["model_name"]
    assert meta2["threshold"] == meta1["threshold"]
    assert meta2["best_params"] == meta1["best_params"]
    assert meta2["test_metrics"]["roc_auc"] == meta1["test_metrics"]["roc_auc"]
    assert meta2["cv_comparison"] == meta1["cv_comparison"]


def test_train_cli_reports_data_errors(tmp_path, capsys):
    code = train_main(["--data", str(tmp_path / "absent.csv"), "--model-path", str(tmp_path / "m.joblib"),
                       "--reports-dir", str(tmp_path / "r")])
    assert code == 2
    assert "ПОМИЛКА ДАНИХ" in capsys.readouterr().err
    assert not (tmp_path / "m.joblib").exists()  # фіктивних результатів немає
