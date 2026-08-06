"""Command-line entry point for Home Credit EDA and training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.credit_scoring.features import ID_COLUMN, TARGET_COLUMN, build_home_credit_features
from src.credit_scoring.training import profile_training_data, train_credit_model


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Home Credit leakage-aware ML pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    eda = subparsers.add_parser("eda", help="Generate a machine-readable EDA/data quality report")
    eda.add_argument("--data-dir", type=Path, default=Path("data/raw/home-credit-default-risk"))
    eda.add_argument("--output", type=Path, default=Path("artifacts/reports/eda_report.json"))
    eda.add_argument("--sample-size", type=int)

    train = subparsers.add_parser("train", help="Train, calibrate, evaluate and serialize a POC model")
    train.add_argument("--data-dir", type=Path, default=Path("data/raw/home-credit-default-risk"))
    train.add_argument("--output-dir", type=Path, default=Path("artifacts/models"))
    train.add_argument("--feature-set", choices=("serving", "application", "full"), default="serving")
    train.add_argument("--sample-size", type=int)
    train.add_argument("--seed", type=int, default=42)
    return parser


def _run_eda(data_dir: Path, output: Path, sample_size: int | None) -> None:
    frame = build_home_credit_features(data_dir, feature_set="application", sample_size=sample_size)
    report = profile_training_data(frame, source_path=data_dir / "application_train.csv")
    report["tables"] = []
    for csv_path in sorted(data_dir.glob("*.csv")):
        try:
            sample = pd.read_csv(csv_path, nrows=3)
        except UnicodeDecodeError:
            # The official column-description file is Windows-1252 encoded,
            # while the modelling tables are UTF-8/ASCII compatible.
            sample = pd.read_csv(csv_path, nrows=3, encoding="cp1252")
        report["tables"].append(
            {
                "table": csv_path.name,
                "bytes": csv_path.stat().st_size,
                "columns": list(sample.columns),
                "examples": sample.head(1).where(sample.notna(), None).to_dict(orient="records"),
                "role": (
                    "target/application entity"
                    if TARGET_COLUMN in sample
                    else "relational history" if ID_COLUMN in sample else "metadata/submission"
                ),
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"EDA report: {output}")


def main() -> None:
    args = _parser().parse_args()
    if args.command == "eda":
        _run_eda(args.data_dir, args.output, args.sample_size)
        return
    bundle, report = train_credit_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        feature_set=args.feature_set,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "artifact": str(args.output_dir / "credit_model.joblib"),
                "model_version": bundle.model_version,
                "champion": report["champion"],
                "test_metrics": report["champion_test_metrics"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
