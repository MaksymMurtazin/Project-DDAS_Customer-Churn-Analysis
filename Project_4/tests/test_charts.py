"""Перевірка змісту графіків: класи, вибірка, пороги та калібрування."""
import pytest

from churn.charts import calibration_figure, comparison_figure, confusion_figure, cost_figure


def test_confusion_axes_and_counts_are_not_transposed():
    fig = confusion_figure({"tn": 653, "fp": 382, "fn": 49, "tp": 325})
    assert list(fig.data[0].z[0]) == [653, 382]
    assert list(fig.data[0].z[1]) == [49, 325]
    assert list(fig.data[0].y) == ["Залишився", "Пішов"]
    assert "FN" in fig.data[0].text[1][0]
    assert "FP" in fig.data[0].text[0][1]
    assert fig.layout.yaxis.autorange == "reversed"


def test_cost_curve_uses_oof_values_and_actual_selected_threshold():
    selection = {
        "cost_curve": [{"threshold": .5, "cost": 23}, {"threshold": .2, "cost": 17}],
        "oof_report": {"cost": 17},
    }
    fig = cost_figure(selection, .2)
    assert list(fig.data[0].x) == [.2, .5]
    assert list(fig.data[0].y) == [17, 23]
    assert list(fig.data[1].x) == [.2]
    assert list(fig.data[1].y) == [17]
    assert [s.x0 for s in fig.layout.shapes] == [.2, .5]


def test_calibration_preserves_bin_sizes_and_omits_empty_bins():
    fig = calibration_figure([
        {"n": 5, "mean_predicted": .8, "observed_rate": .6},
        {"n": 0, "mean_predicted": None, "observed_rate": None},
        {"n": 40, "mean_predicted": .1, "observed_rate": .15},
    ])
    assert list(fig.data[1].x) == [.1, .8]
    assert list(fig.data[1].y) == [.15, .6]
    assert list(fig.data[1].customdata) == [40, 5]
    assert fig.layout.yaxis.scaleanchor == "x"
    assert fig.layout.yaxis.scaleratio == 1


def test_model_comparison_shows_cv_variation_not_confidence_intervals():
    fig = comparison_figure([
        {"model": "logreg", "cv_roc_auc_mean": .8, "cv_roc_auc_std": .03},
        {"model": "lightgbm", "cv_roc_auc_mean": .85, "cv_roc_auc_std": .02},
    ], "lightgbm")
    assert list(fig.data[0].error_x.array) == pytest.approx([.03, .02])
    assert fig.data[0].marker.color[0] != fig.data[0].marker.color[1]
