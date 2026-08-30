"""Prepare the datasets FraudLens trains and tests on.

Two sources:

``synthetic`` (default)
    Generates transactions locally with labelled fraud typologies. Needs no
    credentials, so a fresh clone is immediately runnable and CI never depends
    on an external service.

``kaggle``
    Downloads the real Sparkov dataset. Needs a free Kaggle API token at
    ``~/.kaggle/kaggle.json`` or ``KAGGLE_USERNAME`` / ``KAGGLE_KEY``.

Both write a full dataset to ``data/processed/`` and a small stratified sample
to ``data/sample/``. Only the sample is committed.

Usage::

    python scripts/prepare_data.py                      # synthetic, default size
    python scripts/prepare_data.py --source kaggle      # the real dataset
    python scripts/prepare_data.py --customers 1000 --merchants 800
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running as a plain script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from fraudlens.config import get_settings
from fraudlens.data.loader import (
    download_dataset,
    load_all_raw,
    summarise,
    write_sample,
)
from fraudlens.data.splits import split_frame
from fraudlens.data.synthetic import (
    TYPOLOGY_COL,
    GeneratorConfig,
    generate,
    summarise_typologies,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
logger = logging.getLogger("prepare_data")

PROCESSED_FILENAME = "transactions.parquet"


def build_synthetic(args: argparse.Namespace) -> pd.DataFrame:
    logger.info(
        "Generating synthetic transactions (%d customers, %d merchants, seed %d)",
        args.customers,
        args.merchants,
        args.seed,
    )
    return generate(
        GeneratorConfig(
            n_customers=args.customers,
            n_merchants=args.merchants,
            target_fraud_rate=args.fraud_rate,
            seed=args.seed,
        )
    )


def build_kaggle(_: argparse.Namespace) -> pd.DataFrame:
    settings = get_settings()
    cached = download_dataset()
    logger.info("Reading raw CSVs from %s", cached)
    return load_all_raw(cached if cached.exists() else settings.raw_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("synthetic", "kaggle"), default="synthetic")
    parser.add_argument("--customers", type=int, default=400)
    parser.add_argument("--merchants", type=int, default=600)
    parser.add_argument("--fraud-rate", type=float, default=0.006)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-rows", type=int, default=5_000)
    parser.add_argument(
        "--skip-full",
        action="store_true",
        help="write only the committed sample, not the full processed dataset",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    df = build_synthetic(args) if args.source == "synthetic" else build_kaggle(args)

    logger.info("Dataset summary:")
    for key, value in summarise(df).items():
        logger.info("  %-14s %s", key, value)

    if TYPOLOGY_COL in df.columns:
        logger.info("Fraud typologies:\n%s", summarise_typologies(df).to_string(index=False))

    # Report the split up front. If a fold is empty or a fraud rate is wildly
    # off, that is far cheaper to discover here than during training.
    _, report = split_frame(df)
    logger.info("Temporal split:\n%s", report.to_markdown())
    _warn_on_thin_folds(report)

    if not args.skip_full:
        settings.processed_dir.mkdir(parents=True, exist_ok=True)
        target = settings.processed_dir / PROCESSED_FILENAME
        df.to_parquet(target, index=False)
        logger.info("Wrote full dataset -> %s (%.1f MB)", target, target.stat().st_size / 1e6)

    sample_path = write_sample(
        df.drop(columns=[c for c in (TYPOLOGY_COL,) if c in df.columns]),
        n=args.sample_rows,
        seed=args.seed,
    )
    logger.info("Wrote committed sample -> %s", sample_path)
    return 0


def _warn_on_thin_folds(report: object) -> None:
    counts = getattr(report, "counts", {})
    fraud_counts = getattr(report, "fraud_counts", {})
    for fold, n in counts.items():
        if n == 0:
            logger.warning("fold %r is empty -- check the split boundaries", fold)
        elif fraud_counts.get(fold, 0) < 10:
            logger.warning(
                "fold %r has only %d fraud rows; metrics on it will be very noisy",
                fold,
                fraud_counts.get(fold, 0),
            )


if __name__ == "__main__":
    raise SystemExit(main())
