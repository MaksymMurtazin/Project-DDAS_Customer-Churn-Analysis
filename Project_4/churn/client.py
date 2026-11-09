"""HTTP-клієнт до API для інтерфейсу Streamlit."""
from __future__ import annotations

import requests


class ApiError(Exception):
    """Помилка виклику API. kind: api_unavailable | validation_error | model_not_available | http_error."""

    def __init__(self, kind: str, message: str, details: list | None = None, status: int | None = None):
        super().__init__(message)
        self.kind, self.message, self.details, self.status = kind, message, details or [], status


class ApiClient:
    def __init__(self, base_url: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, **kwargs) -> dict:
        url = f"{self.base_url}{path}"
        try:
            response = requests.request(method, url, timeout=self.timeout, **kwargs)
        except requests.exceptions.RequestException as exc:
            raise ApiError("api_unavailable", f"API недоступне за адресою {self.base_url}: {type(exc).__name__}") from exc
        try:
            body = response.json()
        except ValueError:
            body = {}
        if response.ok:
            return body
        kind = body.get("error", "http_error")
        raise ApiError(kind, body.get("message") or f"HTTP {response.status_code}", body.get("details"), response.status_code)

    def health(self) -> dict:
        return self._request("GET", "/health")

    def model_info(self) -> dict:
        return self._request("GET", "/model/info")

    def predict(self, customer: dict) -> dict:
        return self._request("POST", "/predict", json=customer)["result"]

    def predict_batch(self, customers: list[dict], chunk_size: int = 2000) -> list[dict]:
        """Надсилає клієнтів частинами; номери рядків у помилках валідації зсуваються до глобальних."""
        results: list[dict] = []
        for start in range(0, len(customers), chunk_size):
            try:
                body = self._request("POST", "/predict/batch", json={"customers": customers[start:start + chunk_size]})
            except ApiError as exc:
                for detail in exc.details:
                    if detail.get("row") is not None:
                        detail["row"] += start
                raise
            results.extend(body["results"])
        return results
