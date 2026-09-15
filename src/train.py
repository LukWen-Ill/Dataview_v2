"""Träningsskript. Kör: python -m src.train"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data import DEFAULT_DATA_PATH, load_dataset
from src.model import DEFAULT_MODEL_PATH, save, train


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Träna churn-modellen.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args(argv)

    X, y = load_dataset(args.data)

    result = train(X, y)
    path = save(result.pipeline, args.out)
    print(json.dumps(result.metrics.as_dict(), indent=2))
    print(f"Modell sparad: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
