"""Text-mode helpers (handy for sharing output). Run from the project root:
    python -m churn.cli inspect data/customers.csv
    python -m churn.cli train data/customers.csv --target COLUMN --positive VALUE [--id COLUMN] [--exclude COL ...]"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from . import data as D
from .models import train_and_evaluate
from .schema import Schema
from .scoring import RiskBands, make_bundle, save_bundle


def _load(path) -> pd.DataFrame:
    try:
        return D.load_table(path)
    except D.DataLoadError as exc:
        raise SystemExit(f"Error: {exc}")


def cmd_inspect(a) -> int:
    df = _load(a.file)
    print(D.overview(df))
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(D.profile(df).to_string(index=False))
    return 0


def cmd_train(a) -> int:
    df = _load(a.file)
    if a.target not in df.columns:
        print(f"Error: target column '{a.target}' not found. Columns: {list(df.columns)}")
        return 2
    skip = set(a.exclude or []) | {a.target} | ({a.id} if a.id else set())
    schema = Schema(target=a.target, positive_labels=a.positive, id_column=a.id,
                    feature_columns=a.features or [c for c in df.columns if c not in skip])
    try:
        res = train_and_evaluate(df, schema, test_size=a.test_size, seed=a.seed, balance=a.balance)
    except ValueError as exc:
        print(f"Cannot train: {exc}")
        return 2
    print("Split:", res.split)
    if res.spec.excluded:
        print("Columns not used:", res.spec.excluded)
    for n in res.spec.notes + res.warnings:
        print("Note:", n)
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(res.metrics.to_string())
    print(f"Selected by cross-validated {res.selection_metric}: {res.best_name}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    save_bundle(a.out, make_bundle(res, bands=RiskBands(a.low, a.high)))
    print(f"Saved model to {a.out}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="churn")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("inspect"); s.add_argument("file"); s.set_defaults(fn=cmd_inspect)
    s = sub.add_parser("train")
    s.add_argument("file"); s.add_argument("--target", required=True)
    s.add_argument("--positive", nargs="+", required=True, help="target value(s) meaning churned")
    s.add_argument("--id"); s.add_argument("--features", nargs="+"); s.add_argument("--exclude", nargs="+")
    s.add_argument("--out", default="models/model.joblib")
    s.add_argument("--test-size", type=float, default=0.2); s.add_argument("--seed", type=int, default=42)
    s.add_argument("--balance", action="store_true", help="weight churners more (distorts probabilities)")
    s.add_argument("--low", type=float, default=0.15); s.add_argument("--high", type=float, default=0.30)
    s.set_defaults(fn=cmd_train)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
