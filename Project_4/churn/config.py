"""Шляхи та налаштування. Усі значення можна перевизначити змінними оточення.

Шляхи рахуються від кореня проєкту, тому локальний запуск і запуск у Docker
(де проєкт лежить у /app) використовують однакові відносні розташування.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"
DEFAULT_API_URL = "http://127.0.0.1:5000"

# Параметри експерименту (значення за замовчуванням для churn.train)
SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
N_ITER = 20

# Умовні ваги помилок для вибору порога рішення (див. README, розділ 7).
# У навчальному сценарії пропущений відтік (FN) має вп'ятеро більшу вагу,
# ніж хибна тривога (FP). Це припущення, а не виміряні витрати компанії.
# Співвідношення 5:1 не виводиться з твердження «5–10 разів» у program.pdf.
COST_FALSE_NEGATIVE = 5.0
COST_FALSE_POSITIVE = 1.0

# Розмір пакета для /predict/batch
MAX_BATCH_SIZE = 10_000


def get_data_path() -> Path:
    return Path(os.environ.get("CHURN_DATA_PATH", DEFAULT_DATA_PATH))


def get_model_path() -> Path:
    return Path(os.environ.get("CHURN_MODEL_PATH", DEFAULT_MODEL_PATH))


def get_reports_dir() -> Path:
    return Path(os.environ.get("CHURN_REPORTS_DIR", DEFAULT_REPORTS_DIR))


def get_api_url() -> str:
    return os.environ.get("CHURN_API_URL", DEFAULT_API_URL).rstrip("/")
