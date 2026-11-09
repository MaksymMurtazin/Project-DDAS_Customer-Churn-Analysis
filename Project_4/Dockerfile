# Один образ для API, інтерфейсу та навчання; команда запуску задається в docker-compose.yml
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

# LightGBM потребує OpenMP-бібліотеки libgomp, якої немає в slim-образі
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY churn ./churn
COPY .streamlit ./.streamlit

EXPOSE 5000 8501

# Дані, моделі та звіти монтуються томами (див. docker-compose.yml)
CMD ["python", "-m", "churn.api", "--host", "0.0.0.0", "--port", "5000"]
