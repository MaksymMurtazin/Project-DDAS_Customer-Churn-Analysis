"""Сторінка оцінювання моделі зі збережених метаданих API."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from churn.charts import (
    FEATURE_LABELS, MODEL_LABELS, calibration_figure, comparison_figure,
    confusion_figure, cost_figure, style_chart,
)


def render_model_info(info: dict) -> None:
    t, threshold = info["test_metrics"], info["threshold"]
    ci, learned = t["confidence_intervals"], t["at_threshold"]
    st.subheader("Наскільки добре модель знаходить відтік?")
    data = info["data"]
    st.caption(
        f"Навчальна вибірка: {data['n_train']:,} клієнтів · "
        f"Тестова вибірка: {data['n_test']:,} клієнтів · "
        "Тестові дані не використовуються для вибору порогу."
    )
    a, b, c, d = st.columns(4)
    a.metric("ROC-AUC", f"{t['roc_auc']:.3f}",
             f"{ci['confidence']:.0%} ДІ {ci['roc_auc'][0]:.3f}–{ci['roc_auc'][1]:.3f}", delta_color="off")
    b.metric("PR-AUC (Average Precision)", f"{t['pr_auc']:.3f}",
             f"Частка відтоку на тесті {t['base_rate']:.1%}", delta_color="off")
    c.metric("Precision / Recall", f"{learned['precision']:.2f} / {learned['recall']:.2f}",
             f"Робочий поріг {threshold:.0%}", delta_color="off")
    d.metric("Позначено клієнтів", f"{learned['flagged_share']:.1%}", "За робочим порогом", delta_color="off")

    st.divider()
    st.subheader("Правильні рішення та помилки")
    choice = st.radio(
        "Поріг для порівняння на тесті", ["trained", "default"],
        format_func=lambda v: f"Обраний під час навчання · {threshold:.0%}" if v == "trained" else "Стандартний · 50%",
        horizontal=True, key="model_threshold_view",
    )
    m = learned if choice == "trained" else t["at_threshold_0_5"]
    left, right = st.columns([1.35, 1], gap="large")
    with left:
        st.plotly_chart(confusion_figure(m), width="stretch", key="confusion_chart")
    with right:
        st.markdown(f"**Виявлено {m['tp']:,} із {m['tp'] + m['fn']:,} клієнтів, які пішли.**")
        a, b = st.columns(2)
        a.metric("Recall · знайдено відтік", f"{m['recall']:.1%}")
        b.metric("Precision · точність сигналу", f"{m['precision']:.1%}")
        a.metric("FN · пропущений відтік", f"{m['fn']:,}")
        b.metric("FP · хибні тривоги", f"{m['fp']:,}")
        st.caption("FN — клієнт пішов, але модель не позначила ризик. "
                   "FP — клієнт залишився, але модель позначила ризик.")
    st.caption(
        f"Умовні ваги помилок FN:FP = {info['cost_fn']:g}:{info['cost_fp']:g} "
        f"(співвідношення {info['cost_fn'] / info['cost_fp']:g}:1) є лише припущенням."
    )

    st.subheader("Чому обрано саме цей поріг?")
    st.plotly_chart(cost_figure(info["threshold_selection"], threshold), width="stretch", key="cost_chart")
    st.caption(
        f"Графік побудовано за out-of-fold прогнозами навчальної вибірки: кожен прогноз отримано "
        f"моделлю, яка не навчалася на відповідному фолді. Поріг {threshold:.0%} мінімізує "
        "умовну вартість помилок серед перевірених порогів за заданих ваг FN і FP. "
        "Вартість на цьому графіку та в матриці обчислена на різних вибірках."
    )

    st.divider()
    st.subheader("Порівняння моделей")
    st.caption("Крос-валідація на навчальній вибірці. Кольором виділено обрану модель; риски показують розкид між фолдами.")
    st.plotly_chart(comparison_figure(info["cv_comparison"], info["model_name"]), width="stretch", key="comparison_chart")
    with st.expander("Усі метрики та гіперпараметри"):
        cmp = pd.DataFrame(info["cv_comparison"]).drop(columns="best_params", errors="ignore")
        cmp["model"] = cmp["model"].map(lambda v: MODEL_LABELS.get(v, v))
        st.dataframe(cmp.rename(columns={"model": "Модель", "cv_roc_auc_mean": "ROC-AUC", "cv_roc_auc_std": "Std ROC-AUC", "cv_pr_auc_mean": "PR-AUC (AP)", "cv_brier_mean": "Brier"}), hide_index=True, width="stretch")
        st.json(info["best_params"])

    st.subheader("На які ознаки спирається модель?")
    imp = pd.DataFrame(info["feature_importance"]).sort_values("importance_mean", ascending=False).head(12).copy()
    imp["label"] = imp["feature"].map(lambda v: FEATURE_LABELS.get(v, v))
    fig = px.bar(imp, x="importance_mean", y="label", orientation="h", error_x="importance_std",
                 hover_data=["feature"], labels={"importance_mean": "Падіння ROC-AUC при перемішуванні", "label": ""})
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(style_chart(fig, 430), width="stretch", key="importance_chart")
    st.caption("Permutation importance на тесті: більше падіння ROC-AUC — більша важливість для прогнозу. "
               "Це не означає, що зміна ознаки спричинить зміну відтоку; пов’язані ознаки можуть ділити важливість.")

    st.divider()
    st.subheader("Чи відповідають ймовірності фактичному відтоку?")
    left, right = st.columns([1.3, 1], gap="large")
    with left:
        st.plotly_chart(calibration_figure(t["calibration"]), width="stretch", key="calibration_chart")
    with right:
        st.metric("Brier score", f"{t['brier']:.4f}", help="Середня квадратична похибка ймовірностей; менше — краще.")
        st.metric("Log loss", f"{t['log_loss']:.4f}", help="Штрафує помилкові впевнені прогнози; менше — краще.")
        st.write("Клієнтів згруповано за прогнозованою ймовірністю. Якщо середній прогноз групи — 30%, "
                 "то за доброго калібрування приблизно 30% її клієнтів мають піти.")
        st.caption("Над діагоналлю модель недооцінює відтік, під нею — переоцінює. "
                   "Наведіть курсор на точку, щоб побачити розмір групи: малі групи дають менш стійку оцінку. "
                   "Графік оцінює калібрування, але не виконує додаткового калібрування моделі.")
