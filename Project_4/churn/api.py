"""REST API для прогнозування відтоку (Flask + waitress).

Ендпоінти:
  GET  /health         — стан сервісу й чи завантажена модель
  GET  /model/info     — метадані моделі: метрики, поріг, важливість ознак
  POST /predict        — один клієнт (JSON-об'єкт з полями Telco; Churn не потрібен)
  POST /predict/batch  — {"customers": [ {...}, ... ]}

Модель підхоплюється зі змінної CHURN_MODEL_PATH (за замовчуванням models/churn_model.joblib)
і автоматично перезавантажується, якщо файл змінився (можна донавчити без перезапуску API).
"""
from __future__ import annotations

import argparse
import logging
import os
import threading
from pathlib import Path

from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

from churn import config
from churn.predictor import ChurnPredictor, InputValidationError, ModelNotAvailableError

log = logging.getLogger("churn.api")
MAX_REPORTED_ISSUES = 50


class _ModelStore:
    """Лінивий завантажувач моделі з перезавантаженням при зміні файлу."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._predictor: ChurnPredictor | None = None
        self._signature: tuple | None = None

    def _file_signature(self) -> tuple | None:
        try:
            st = self.path.stat()
        except FileNotFoundError:
            return None
        return (st.st_mtime_ns, st.st_size)

    def get(self) -> ChurnPredictor:
        with self._lock:
            signature = self._file_signature()
            if signature is None:
                self._predictor, self._signature = None, None
                raise ModelNotAvailableError(
                    f"Модель не знайдено ({self.path}). Навчіть її командою: python -m churn.train"
                )
            if self._predictor is None or signature != self._signature:
                log.info("Завантаження моделі %s", self.path)
                self._predictor = ChurnPredictor.load(self.path)
                self._signature = signature
            return self._predictor

    def is_loaded(self) -> bool:
        try:
            self.get()
            return True
        except ModelNotAvailableError:
            return False


def _error(status: int, code: str, message: str, details: list | None = None):
    body = {"error": code, "message": message}
    if details is not None:
        body["details"] = details
    return jsonify(body), status


def create_app(model_path: Path | None = None) -> Flask:
    app = Flask(__name__)
    app.json.ensure_ascii = False
    app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024
    store = _ModelStore(model_path or config.get_model_path())

    def validation_response(exc: InputValidationError):
        details = [i.as_dict() for i in exc.issues[:MAX_REPORTED_ISSUES]]
        return _error(422, "validation_error", str(exc), details)

    def model_meta(predictor: ChurnPredictor) -> dict:
        return {"name": predictor.metadata["model_name"], "threshold": predictor.threshold,
                "high_risk_threshold": predictor.high_threshold,
                "trained_at_utc": predictor.metadata["trained_at_utc"]}

    @app.get("/health")
    def health():
        loaded, message = True, None
        try:
            store.get()
        except ModelNotAvailableError as exc:
            loaded, message = False, str(exc)
        return jsonify({"status": "ok", "model_loaded": loaded, "message": message})

    @app.get("/model/info")
    def model_info():
        try:
            return jsonify(store.get().info())
        except ModelNotAvailableError as exc:
            return _error(503, "model_not_available", str(exc))

    @app.post("/predict")
    def predict():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return _error(400, "invalid_json", "Тіло запиту має бути JSON-об'єктом з полями клієнта")
        try:
            predictor = store.get()
            result = predictor.predict_records([payload])[0]
        except ModelNotAvailableError as exc:
            return _error(503, "model_not_available", str(exc))
        except InputValidationError as exc:
            return validation_response(exc)
        return jsonify({"result": result, "model": model_meta(predictor)})

    @app.post("/predict/batch")
    def predict_batch():
        payload = request.get_json(silent=True)
        customers = payload.get("customers") if isinstance(payload, dict) else None
        if not isinstance(customers, list) or not customers:
            return _error(400, "invalid_json", 'Очікується JSON {"customers": [ ... ]} з непорожнім списком')
        if not all(isinstance(c, dict) for c in customers):
            return _error(400, "invalid_json", "Кожен елемент customers має бути JSON-об'єктом")
        if len(customers) > config.MAX_BATCH_SIZE:
            return _error(413, "batch_too_large",
                          f"Максимум {config.MAX_BATCH_SIZE} клієнтів за запит; отримано {len(customers)}")
        try:
            predictor = store.get()
            results = predictor.predict_records(customers)
        except ModelNotAvailableError as exc:
            return _error(503, "model_not_available", str(exc))
        except InputValidationError as exc:
            return validation_response(exc)
        return jsonify({"count": len(results), "results": results, "model": model_meta(predictor)})

    @app.errorhandler(HTTPException)
    def http_error(exc: HTTPException):
        return _error(exc.code or 500, exc.name.lower().replace(" ", "_"), exc.description or exc.name)

    @app.errorhandler(Exception)
    def unexpected_error(exc: Exception):
        log.exception("Необроблена помилка під час обробки запиту")
        return _error(500, "internal_error", f"Внутрішня помилка сервера: {type(exc).__name__}. Деталі — у журналі сервера.")

    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Запуск API відтоку клієнтів")
    parser.add_argument("--host", default=os.environ.get("CHURN_API_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("CHURN_API_PORT", "5000")))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from waitress import serve

    app = create_app()
    log.info("API слухає на http://%s:%d (модель: %s)", args.host, args.port, config.get_model_path())
    serve(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
