"""Пакетний прогноз з командного рядка:  python -m churn.predict --input клієнти.csv --output прогноз.csv"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from churn import config
from churn.predictor import ChurnPredictor, InputValidationError, ModelNotAvailableError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Прогноз відтоку для CSV з клієнтами (колонка Churn не потрібна)")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=None)
    args = parser.parse_args(argv)

    if not args.input.is_file():
        print(f"ПОМИЛКА: файл {args.input} не знайдено", file=sys.stderr)
        return 2
    try:
        predictor = ChurnPredictor.load(args.model_path or config.get_model_path())
        frame = pd.read_csv(args.input)
        results = predictor.predict(frame)
    except ModelNotAvailableError as exc:
        print(f"ПОМИЛКА: {exc}", file=sys.stderr)
        return 3
    except InputValidationError as exc:
        print(f"ПОМИЛКА: {exc}", file=sys.stderr)
        for issue in exc.issues[:20]:
            print(f"  рядок {issue.row}, колонка {issue.column}: {issue.message}", file=sys.stderr)
        return 2
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        print(f"ПОМИЛКА: не вдалося прочитати CSV: {exc}", file=sys.stderr)
        return 2

    out = pd.DataFrame({
        "customerID": [r["customerID"] for r in results],
        "churn_probability": [r["churn_probability"] for r in results],
        "at_risk": [r["at_risk"] for r in results],
        "risk_level": [r["risk_level"] for r in results],
        "recommendations (rules)": ["; ".join(x["action"] for x in r["recommendations"]) for r in results],
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Збережено {len(out)} прогнозів у {args.output}; у зоні ризику (поріг {predictor.threshold:.2f}): {int(out['at_risk'].sum())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
