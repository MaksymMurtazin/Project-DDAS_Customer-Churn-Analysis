import numpy as np
import pandas as pd
import pytest

from churn import config
from churn.data import DataError, load_dataset
from churn.features import MODEL_COLUMNS, prepare_features
from churn.schema import FEATURE_COLUMNS, validate_frame


def frame(valid_row, **changes):
    row = {**valid_row, **changes}
    return pd.DataFrame([row])


def test_real_dataset_matches_schema():
    path = config.DEFAULT_DATA_PATH
    if not path.is_file():
        pytest.skip("реальний CSV відсутній")
    data, y = load_dataset(path)
    assert data.shape == (7043, 1 + len(FEATURE_COLUMNS))
    assert set(y.unique()) == {0, 1} and 0.26 < y.mean() < 0.27


def test_valid_row_has_no_issues(valid_row):
    assert validate_frame(frame(valid_row)) == []


def test_target_and_id_not_required(valid_row):
    row = {k: v for k, v in valid_row.items() if k != "customerID"}
    assert validate_frame(pd.DataFrame([row])) == []


def test_blank_total_charges_allowed(valid_row):
    assert validate_frame(frame(valid_row, TotalCharges=" ")) == []
    assert validate_frame(frame(valid_row, TotalCharges=None)) == []


def test_whitespace_is_stripped(valid_row):
    assert validate_frame(frame(valid_row, Contract=" One year ")) == []


@pytest.mark.parametrize("changes, column", [
    ({"Contract": "Weekly"}, "Contract"),
    ({"gender": None}, "gender"),
    ({"tenure": -1}, "tenure"),
    ({"tenure": "abc"}, "tenure"),
    ({"MonthlyCharges": float("nan")}, "MonthlyCharges"),
    ({"MonthlyCharges": -5}, "MonthlyCharges"),
    ({"TotalCharges": "12,5x"}, "TotalCharges"),
    ({"SeniorCitizen": 2}, "SeniorCitizen"),
    ({"InternetService": "No", "OnlineSecurity": "Yes"}, "OnlineSecurity"),
    ({"InternetService": "DSL", "TechSupport": "No internet service"}, "TechSupport"),
    ({"PhoneService": "No", "MultipleLines": "Yes"}, "MultipleLines"),
])
def test_invalid_values_are_reported(valid_row, changes, column):
    # базовий рядок підлаштовуємо під зв'язки між колонками
    base = {**valid_row, "InternetService": "DSL", "PhoneService": "Yes", "MultipleLines": "No",
            **{c: "No" for c in ("OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies")}}
    issues = validate_frame(pd.DataFrame([{**base, **changes}]))
    assert column in {i.column for i in issues}


def test_missing_column_and_empty_frame(valid_row):
    row = {k: v for k, v in valid_row.items() if k != "Contract"}
    issues = validate_frame(pd.DataFrame([row]))
    assert [(i.row, i.column) for i in issues] == [(None, "Contract")]
    assert validate_frame(pd.DataFrame(columns=list(FEATURE_COLUMNS)))[0].message == "таблиця порожня"


def test_prepare_features_total_charges_and_columns(valid_row):
    df = pd.DataFrame([
        {**valid_row, "tenure": 0, "MonthlyCharges": 50.0, "TotalCharges": " "},
        {**valid_row, "tenure": 10, "MonthlyCharges": 20.0, "TotalCharges": None},
        {**valid_row, "tenure": 10, "MonthlyCharges": 20.0, "TotalCharges": "199.5"},
    ])
    before = df.copy()
    out = prepare_features(df)
    assert list(out.columns) == MODEL_COLUMNS
    assert out["TotalCharges"].tolist() == [0.0, 200.0, 199.5]
    assert out["tenure"].dtype == np.float64
    pd.testing.assert_frame_equal(df, before)  # вхідні дані не змінюються


def test_prepare_features_needs_no_target(valid_row):
    row = {k: v for k, v in valid_row.items() if k not in ("customerID", "Churn")}
    assert len(prepare_features(pd.DataFrame([row]))) == 1


def test_load_dataset_errors(tmp_path, synthetic_df):
    with pytest.raises(DataError, match="не знайдено"):
        load_dataset(tmp_path / "nope.csv")
    bad = synthetic_df.copy()
    bad.loc[0, "Contract"] = "Weekly"
    bad.to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(DataError, match="не відповідають схемі"):
        load_dataset(tmp_path / "bad.csv")
    synthetic_df.drop(columns=["Churn"]).to_csv(tmp_path / "no_target.csv", index=False)
    with pytest.raises(DataError, match="бракує колонок"):
        load_dataset(tmp_path / "no_target.csv")
    dup = synthetic_df.copy()
    dup.loc[1, "customerID"] = dup.loc[0, "customerID"]
    dup.to_csv(tmp_path / "dup.csv", index=False)
    with pytest.raises(DataError, match="унікальним"):
        load_dataset(tmp_path / "dup.csv")
    wrong_target = synthetic_df.copy()
    wrong_target.loc[0, "Churn"] = "Maybe"
    wrong_target.to_csv(tmp_path / "target.csv", index=False)
    with pytest.raises(DataError, match="Churn"):
        load_dataset(tmp_path / "target.csv")
