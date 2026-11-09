"""Інтерфейс Streamlit: прогноз для одного клієнта, пакетний аналіз CSV, інформація про модель.

Інтерфейс не містить власної логіки прогнозування — усі прогнози отримуються з API,
тож результати інтерфейсу завжди збігаються з результатами API.
Запуск:  python -m streamlit run churn/ui.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

# Дозволяє запуск через `streamlit run churn/ui.py` без встановлення пакета
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from churn import config  # noqa: E402
from churn.charts import FEATURE_LABELS, MODEL_LABELS, style_chart  # noqa: E402
from churn.model_view import render_model_info  # noqa: E402
from churn.client import ApiClient, ApiError  # noqa: E402
from churn.schema import ADDON_COLUMNS, CATEGORICAL_ALLOWED, NO_INTERNET, NO_PHONE  # noqa: E402
from churn.summary import recommendation_counts, results_to_frame, summarize  # noqa: E402

RISK_LABELS = {"low": "Низький", "medium": "Середній", "high": "Високий"}
YES_NO_UA = {"No": "Ні", "Yes": "Так"}

st.set_page_config(page_title="Аналіз відтоку клієнтів", page_icon="📉", layout="wide")

# Лише оформлення: семантика та доступність стандартних віджетів збережені.
st.markdown("""
<style>
.block-container {max-width: 1440px; padding-top: 2rem; padding-bottom: 3rem;}
[data-testid="stSidebar"] {border-right: 1px solid #303c49;}
[data-testid="stMetric"] {background: #18212b; border: 1px solid #303c49;
    border-radius: 14px; padding: 16px; height: 100%;}
[data-testid="stMetricValue"] {font-size: clamp(1.2rem, 2.2vw, 2rem);}
[data-baseweb="tab-list"] {gap: 1.2rem;}
h1 {letter-spacing: -0.035em;}
h2, h3 {letter-spacing: -0.02em;}
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=30, show_spinner=False)
def load_model_info(api_url: str) -> dict:
    """Метадані моделі з API. Помилки не кешуються (виняток прориває кеш)."""
    return ApiClient(api_url).model_info()


def show_api_error(exc: ApiError) -> None:
    if exc.kind == "api_unavailable":
        st.error(f"{exc.message}. Запустіть API: `python -m churn.api` (див. README).")
    elif exc.kind == "model_not_available":
        st.error(f"{exc.message}")
    elif exc.kind == "validation_error":
        st.error(exc.message)
        if exc.details:
            st.dataframe(pd.DataFrame(exc.details).rename(columns={"row": "рядок", "column": "колонка", "message": "проблема"}),
                         width="stretch", hide_index=True)
    else:
        st.error(f"Помилка API: {exc.message}")


def format_recommendations(items: list[dict]) -> None:
    st.subheader("Рекомендації з утримання")
    if not items:
        st.info("Правила не спрацювали для цього клієнта.")
    for item in items:
        evidence = ""
        if item.get("segment_churn_rate") is not None:
            evidence = (f" — у навчальній вибірці сегмент «{item['segment']}» має відтік "
                        f"{item['segment_churn_rate']:.0%} ({item['segment_size']} клієнтів)")
        st.markdown(f"- {item['action']}{evidence}")


# ---------------------------------------------------------------- бокова панель
api_url = config.get_api_url()
client = ApiClient(api_url)
with st.sidebar:
    st.header("📉 Churn Analytics")
    st.caption("Аналіз відтоку клієнтів · Telco")
    if st.button("Оновити стан", key="refresh_model", width="stretch"):
        load_model_info.clear()
        st.session_state.pop("batch", None)

try:
    info = load_model_info(api_url)
except ApiError as exc:
    st.title("📉 Аналіз відтоку клієнтів")
    show_api_error(exc)
    st.stop()

threshold = info["threshold"]
with st.sidebar:
    st.success("API та модель доступні")
    st.write(f"**Модель:** {MODEL_LABELS.get(info['model_name'], info['model_name'])}")
    st.write(f"**Навчена:** {info['trained_at_utc'][:19].replace('T', ' ')} UTC")
    st.write(f"**Поріг «у зоні ризику»:** {threshold:.0%}")
    st.write(f"**ROC-AUC (test):** {info['test_metrics']['roc_auc']:.3f}")
    st.divider()
    st.caption("Робочий поріг визначено під час навчання. Порівняння порогів доступне на вкладці «Про модель».")
    with st.expander("Підключення"):
        st.code(api_url, language=None)

st.title("Аналіз відтоку клієнтів")
st.caption("Оцініть ризик, визначте клієнтів для роботи та перевірте якість моделі.")
tab_single, tab_batch, tab_model = st.tabs(["Один клієнт", "Пакетний аналіз (CSV)", "Про модель"])

# ---------------------------------------------------------------- один клієнт
with tab_single:
    st.write("Введіть дані клієнта й отримайте ймовірність відтоку.")
    c1, c2, c3 = st.columns(3)
    payload: dict = {}
    with c1:
        st.markdown("**01 · Профіль клієнта**")
        payload["gender"] = st.selectbox("Стать (gender)", CATEGORICAL_ALLOWED["gender"], format_func=lambda v: {"Female": "Жіноча", "Male": "Чоловіча"}[v], key="f_gender")
        payload["SeniorCitizen"] = st.selectbox("Пенсіонер (SeniorCitizen)", [0, 1], format_func=lambda v: "Так" if v else "Ні", key="f_senior")
        payload["Partner"] = st.selectbox("Партнер (Partner)", ["No", "Yes"], format_func=YES_NO_UA.get, key="f_partner")
        payload["Dependents"] = st.selectbox("Утриманці (Dependents)", ["No", "Yes"], format_func=YES_NO_UA.get, key="f_dependents")
        payload["tenure"] = st.number_input("Стаж, місяців (tenure)", min_value=0, max_value=120, value=12, step=1, key="f_tenure")
    with c2:
        st.markdown("**02 · Підключені послуги**")
        payload["PhoneService"] = st.selectbox("Телефонний зв'язок (PhoneService)", ["Yes", "No"], format_func=YES_NO_UA.get, key="f_phone")
        if payload["PhoneService"] == "No":
            payload["MultipleLines"] = NO_PHONE
            st.caption(f"MultipleLines = {NO_PHONE}")
        else:
            payload["MultipleLines"] = st.selectbox("Кілька ліній (MultipleLines)", ["No", "Yes"], format_func=YES_NO_UA.get, key="f_lines")
        payload["InternetService"] = st.selectbox("Інтернет (InternetService)", CATEGORICAL_ALLOWED["InternetService"], index=1, format_func=lambda v: {"DSL": "DSL", "Fiber optic": "Оптоволокно", "No": "Немає інтернету"}.get(v, v), key="f_internet")
        for column in ADDON_COLUMNS:
            if payload["InternetService"] == "No":
                payload[column] = NO_INTERNET
            else:
                payload[column] = st.selectbox(FEATURE_LABELS.get(column, column), ["No", "Yes"], format_func=YES_NO_UA.get, key=f"f_{column}")
        if payload["InternetService"] == "No":
            st.caption(f"Додаткові послуги = {NO_INTERNET}")
    with c3:
        st.markdown("**03 · Контракт і оплата**")
        payload["Contract"] = st.selectbox("Контракт (Contract)", CATEGORICAL_ALLOWED["Contract"], format_func=lambda v: {"Month-to-month": "Помісячний", "One year": "На один рік", "Two year": "На два роки"}.get(v, v), key="f_contract")
        payload["PaperlessBilling"] = st.selectbox("Електронний рахунок (PaperlessBilling)", ["Yes", "No"], format_func=YES_NO_UA.get, key="f_paperless")
        payload["PaymentMethod"] = st.selectbox("Спосіб оплати (PaymentMethod)", CATEGORICAL_ALLOWED["PaymentMethod"], format_func=lambda v: {"Bank transfer (automatic)": "Автоматичний банківський переказ", "Credit card (automatic)": "Автоматично з картки", "Electronic check": "Електронний чек", "Mailed check": "Поштовий чек"}.get(v, v), key="f_payment")
        payload["MonthlyCharges"] = st.number_input("Щомісячний платіж (MonthlyCharges)", min_value=0.0, max_value=1000.0, value=70.0, step=0.05, key="f_monthly")
        auto_total = st.checkbox("TotalCharges = tenure × MonthlyCharges (оцінка)", value=True, key="f_auto_total")
        if auto_total:
            payload["TotalCharges"] = None  # API/модель заповнюють tenure × MonthlyCharges
        else:
            payload["TotalCharges"] = st.number_input("Загальні платежі (TotalCharges)", min_value=0.0, value=840.0, step=1.0, key="f_total")

    if st.button("Розрахувати ризик", type="primary", key="btn_single"):
        try:
            result = client.predict(payload)
        except ApiError as exc:
            show_api_error(exc)
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Ймовірність відтоку", f"{result['churn_probability']:.1%}")
            m2.metric("Рівень ризику", RISK_LABELS[result["risk_level"]])
            m3.metric("Поріг рішення моделі", f"{threshold:.0%}")
            st.progress(min(max(result["churn_probability"], 0.0), 1.0))
            if result["at_risk"]:
                st.warning("Клієнта віднесено до зони ризику (ймовірність не нижча за поріг).")
            else:
                st.success("Клієнт нижче порога ризику. Рекомендації не формуються.")
            if result["at_risk"]:
                format_recommendations(result["recommendations"])

# ---------------------------------------------------------------- пакетний аналіз
with tab_batch:
    st.write("Завантажте CSV із колонками датасету Telco. Колонка `Churn` не потрібна; `customerID` — за бажанням.")
    uploaded = st.file_uploader("CSV-файл із клієнтами", type=["csv"], key="batch_file")
    # Не показувати прогнози попереднього файлу після заміни або видалення CSV.
    upload_id = hashlib.sha256(uploaded.getvalue()).hexdigest() if uploaded is not None else None
    batch_identity = (uploaded.name, upload_id) if uploaded is not None else None
    if st.session_state.get("batch_identity") != batch_identity:
        st.session_state.pop("batch", None)
        st.session_state["batch_identity"] = batch_identity
    if uploaded is not None:
        try:
            customers = pd.read_csv(uploaded)
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
            st.error(f"Не вдалося прочитати CSV: {exc}")
            customers = None
        if customers is not None:
            st.caption(f"Файл «{uploaded.name}»: {len(customers)} рядків, {len(customers.columns)} колонок")
            st.dataframe(customers.head(5), width="stretch", hide_index=True)
            if st.button("Розрахувати прогнози", type="primary", key="btn_batch"):
                records = json.loads(customers.to_json(orient="records"))
                try:
                    with st.spinner("Розрахунок..."):
                        results = client.predict_batch(records)
                except ApiError as exc:
                    st.session_state.pop("batch", None)
                    show_api_error(exc)
                else:
                    st.session_state["batch"] = {"name": uploaded.name, "customers": customers, "results": results}

    batch = st.session_state.get("batch")
    if batch:
        customers, results = batch["customers"], batch["results"]
        frame = results_to_frame(results)
        s = summarize(customers, frame)
        st.subheader(f"Результати: {batch['name']}")
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Клієнтів", f"{s['n_customers']:,}")
        k2.metric("Середній стаж", f"{s['avg_tenure']:.1f} міс.")
        k3.metric("У зоні ризику", f"{s['at_risk_count']:,}", f"{s['at_risk_share']:.1%} клієнтів", delta_color="off")
        k4.metric("Місячний дохід у зоні ризику", f"${s['mrr_at_risk']:,.0f}")
        k5.metric("Сума платежів × ризик", f"${s['expected_monthly_loss']:,.0f}")
        st.caption(
            f"Умовна оцінка = Σ (ймовірність × MonthlyCharges); це не прогноз втрат за наступний місяць. "
            f"високий ризик (≥{info['high_risk_threshold']:.0%}): {s['high_risk_count']:,} клієнтів; "
            f"середня ймовірність відтоку: {s['mean_probability']:.1%}."
        )
        if s["actual_churn_rate"] is not None:
            st.caption(f"У файлі є колонка Churn: фактична частка відтоку {s['actual_churn_rate']:.1%}. "
                       "Якщо клієнти входили до навчальних даних, порівнювати це з прогнозом некоректно.")

        fig = px.histogram(frame, x="churn_probability", nbins=20, title="Розподіл ймовірностей відтоку",
                           labels={"churn_probability": "Ймовірність відтоку", "count": "Клієнтів"})
        fig.add_vline(x=threshold, line_dash="dash", annotation_text=f"поріг {threshold:.0%}")
        fig.update_layout(yaxis_title="Клієнтів")
        st.plotly_chart(style_chart(fig), width="stretch")

        counts = recommendation_counts(results)
        if len(counts):
            st.subheader("Рекомендації з утримання (бізнес-правила, не прогноз моделі)")
            bar = px.bar(counts, x="customers", y="recommendation", orientation="h",
                         labels={"customers": "Клієнтів у зоні ризику", "recommendation": ""})
            bar.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(style_chart(bar), width="stretch")

        st.subheader("Клієнти в зоні ризику")
        detail = pd.concat(
            [customers[[c for c in ("tenure", "Contract", "MonthlyCharges") if c in customers.columns]].reset_index(drop=True), frame],
            axis=1,
        )
        flagged = detail[detail["at_risk"]].sort_values("churn_probability", ascending=False)
        st.dataframe(flagged.head(1000), width="stretch", hide_index=True,
                     column_config={"churn_probability": st.column_config.ProgressColumn("churn_probability", min_value=0.0, max_value=1.0, format="%.3f")})
        if len(flagged) > 1000:
            st.caption(f"Показано 1000 із {len(flagged)} клієнтів; повний список — у завантаженому файлі.")
        st.download_button("Завантажити всі прогнози (CSV)", frame.to_csv(index=False).encode("utf-8"),
                           file_name="churn_predictions.csv", mime="text/csv")

# ---------------------------------------------------------------- про модель
with tab_model:
    render_model_info(info)
