"""Графіки зі збережених результатів оцінювання; модель тут не навчається."""
from __future__ import annotations

import plotly.graph_objects as go

TEAL = "#37b6c3"
AMBER = "#f0ad4e"
MODEL_LABELS = {
    "baseline_prior": "Базова модель", "logreg": "Логістична регресія",
    "random_forest": "Випадковий ліс", "lightgbm": "LightGBM",
}
FEATURE_LABELS = {
    "Contract": "Тип контракту", "tenure": "Стаж клієнта", "InternetService": "Тип інтернету",
    "OnlineSecurity": "Онлайн-безпека", "MonthlyCharges": "Щомісячний платіж",
    "TotalCharges": "Загальні платежі", "TechSupport": "Технічна підтримка",
    "PaymentMethod": "Спосіб оплати", "OnlineBackup": "Резервне копіювання",
    "DeviceProtection": "Захист пристрою", "PaperlessBilling": "Електронний рахунок",
    "SeniorCitizen": "Пенсіонер", "Partner": "Партнер", "Dependents": "Утриманці",
    "PhoneService": "Телефонний зв’язок", "MultipleLines": "Кілька телефонних ліній",
    "StreamingTV": "Онлайн-ТБ", "StreamingMovies": "Онлайн-фільми", "gender": "Стать",
}


def style_chart(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", height=height, margin=dict(l=12, r=20, t=50, b=25),
        font=dict(family="Arial, sans-serif", color="#e6edf3", size=13),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        colorway=[TEAL, AMBER, "#6574a5"],
        legend=dict(orientation="h", y=1.16, x=0),
    )
    fig.update_xaxes(gridcolor="#293544", zerolinecolor="#445365")
    fig.update_yaxes(gridcolor="#293544", zerolinecolor="#445365")
    return fig


def confusion_figure(metrics: dict) -> go.Figure:
    # Рядки = фактичний клас; колонки = рішення моделі. Без нормалізації.
    z = [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]]
    labels = [["TN · правильно", "FP · хибна тривога"],
              ["FN · пропущено", "TP · виявлено"]]
    text = [[f"{labels[r][c]}<br><b>{z[r][c]:,}</b>" for c in range(2)] for r in range(2)]
    fig = go.Figure(go.Heatmap(
        z=z, x=["Нижче порога", "У зоні ризику"], y=["Залишився", "Пішов"],
        text=text, texttemplate="%{text}", textfont=dict(size=16),
        colorscale=[[0, "#edf6f6"], [1, "#087f8c"]], showscale=False,
        xgap=6, ygap=6,
        hovertemplate="Факт: %{y}<br>Прогноз: %{x}<br>Клієнтів: %{z}<extra></extra>",
    ))
    style_chart(fig)
    fig.update_xaxes(title="Рішення моделі", side="bottom")
    fig.update_yaxes(title="Фактичний результат", autorange="reversed")
    return fig


def cost_figure(selection: dict, threshold: float) -> go.Figure:
    curve = sorted(selection["cost_curve"], key=lambda row: row["threshold"])
    fig = go.Figure(go.Scatter(
        x=[r["threshold"] for r in curve], y=[r["cost"] for r in curve],
        mode="lines", name="Вартість помилок", line=dict(color=TEAL, width=3),
        hovertemplate="Поріг: %{x:.0%}<br>Умовна вартість: %{y:,.0f}<extra></extra>",
    ))
    chosen = selection["oof_report"]
    fig.add_trace(go.Scatter(
        x=[threshold], y=[chosen["cost"]], mode="markers", name="Обраний поріг",
        marker=dict(color=TEAL, size=12, line=dict(color="white", width=2)),
        hovertemplate="Обраний поріг: %{x:.0%}<br>Умовна вартість: %{y:,.0f}<extra></extra>",
    ))
    fig.add_vline(x=threshold, line_color=TEAL, line_dash="dot")
    fig.add_vline(x=.5, line_color=AMBER, line_dash="dash", annotation_text="Поріг 50%")
    style_chart(fig, 380)
    fig.update_xaxes(title="Поріг рішення", tickformat=".0%", range=[0, 1])
    fig.update_yaxes(title="Умовна вартість FN × вага FN + FP × вага FP")
    return fig


def calibration_figure(bins: list[dict]) -> go.Figure:
    rows = sorted((r for r in bins if r["n"] > 0), key=lambda r: r["mean_predicted"])
    fig = go.Figure(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Ідеальне калібрування",
        line=dict(color="#92a0af", dash="dash"), hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=[r["mean_predicted"] for r in rows], y=[r["observed_rate"] for r in rows],
        customdata=[r["n"] for r in rows], mode="lines+markers", name="Модель на тесті",
        line=dict(color=TEAL, width=3), marker=dict(size=9),
        hovertemplate="Середній прогноз: %{x:.1%}<br>Фактичний відтік: %{y:.1%}"
                      "<br>Клієнтів у групі: %{customdata}<extra></extra>",
    ))
    style_chart(fig, 410)
    fig.update_xaxes(title="Середня прогнозована ймовірність", range=[0, 1], tickformat=".0%", constrain="domain")
    fig.update_yaxes(title="Фактична частка відтоку", range=[0, 1], tickformat=".0%", scaleanchor="x", scaleratio=1)
    return fig


def comparison_figure(rows: list[dict], selected: str) -> go.Figure:
    rows = sorted(rows, key=lambda r: r["cv_roc_auc_mean"])
    fig = go.Figure(go.Bar(
        x=[r["cv_roc_auc_mean"] for r in rows],
        y=[MODEL_LABELS.get(r["model"], r["model"]) for r in rows], orientation="h",
        error_x=dict(type="data", array=[r["cv_roc_auc_std"] for r in rows]),
        marker_color=[TEAL if r["model"] == selected else "#b7c8d3" for r in rows],
        text=[f'{r["cv_roc_auc_mean"]:.3f}' for r in rows], textposition="outside",
        hovertemplate="%{y}<br>Середній ROC-AUC: %{x:.4f}<extra></extra>",
    ))
    style_chart(fig, 290)
    fig.update_xaxes(title="ROC-AUC · середнє та ±1 стандартне відхилення між фолдами", range=[0, 1])
    return fig
