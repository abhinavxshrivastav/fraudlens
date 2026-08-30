"""Ingestion: read raw CSVs, enforce the contract, produce the committed sample.

The raw dataset is ~400 MB and is not in version control. Three entry points:

- :func:`download_dataset` fetches it from Kaggle (needs credentials, run once).
- :func:`load_raw` reads a CSV into a validated, typed frame.
- :func:`load_sample` reads the small committed slice that tests and CI use.

Only :func:`load_sample` is on the critical path for a fresh clone.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd

from fraudlens.config import constants as C
from fraudlens.config import get_settings
from fraudlens.data.schema import RAW_TRANSACTION_SCHEMA

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

KAGGLE_DATASET: Final = "kartik2112/fraud-detection"
SAMPLE_FILENAME: Final = "transactions_sample.csv"

#: Columns parsed as datetimes at read time.
_DATE_COLS: Final[tuple[str, ...]] = (C.TIMESTAMP_COL, "dob")

#: Explicit dtypes. `cc_num` in particular must not be inferred as float64 --
#: a 16-digit card number exceeds float64 integer precision and would silently
#: collide distinct cards, corrupting every card-grouped feature.
_DTYPES: Final[dict[str, str]] = {
    C.CARD_COL: "int64",
    C.AMOUNT_COL: "float64",
    "merchant": "string",
    "category": "string",
    "gender": "string",
    "city": "string",
    "state": "string",
    "job": "string",
    C.TRANSACTION_ID_COL: "string",
    "zip": "int64",
    "city_pop": "int64",
    "unix_time": "int64",
    C.HOME_LAT_COL: "float64",
    C.HOME_LON_COL: "float64",
    C.MERCH_LAT_COL: "float64",
    C.MERCH_LON_COL: "float64",
    C.TARGET_COL: "int64",
}


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the raw dataset is absent and cannot be located."""


def download_dataset(destination: Path | None = None) -> Path:
    """Download the Sparkov dataset from Kaggle into ``destination``.

    Requires Kaggle credentials (``~/.kaggle/kaggle.json`` or the
    ``KAGGLE_USERNAME`` / ``KAGGLE_KEY`` environment variables). Imported lazily
    so that ``kagglehub`` is not a hard runtime dependency of the serving path.
    """
    try:
        import kagglehub
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        msg = "kagglehub is required to download the dataset: pip install -e '.[data]'"
        raise ImportError(msg) from exc

    destination = destination or get_settings().raw_dir
    destination.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading %s from Kaggle", KAGGLE_DATASET)
    cache_path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    logger.info("Dataset cached at %s", cache_path)
    return cache_path


def find_raw_csvs(directory: Path | None = None) -> list[Path]:
    """Locate the raw ``fraudTrain.csv`` / ``fraudTest.csv`` files."""
    directory = directory or get_settings().raw_dir
    if not directory.exists():
        return []
    return sorted(directory.rglob("fraud*.csv"))


def load_raw(
    path: Path,
    *,
    validate: bool = True,
    drop_pii: bool = True,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Read one raw CSV into a typed, contract-checked frame.

    PII columns are dropped by default. ``first``, ``last`` and ``street`` are
    effectively unique per customer: keeping them would let a model memorise
    individuals instead of learning behaviour, which is both a leakage risk and
    a privacy problem.
    """
    if not path.exists():
        msg = f"raw dataset not found at {path}"
        raise DatasetNotFoundError(msg)

    df = pd.read_csv(
        path,
        dtype=_DTYPES,
        parse_dates=list(_DATE_COLS),
        nrows=nrows,
    )
    return _postprocess(df, validate=validate, drop_pii=drop_pii)


def _postprocess(df: pd.DataFrame, *, validate: bool, drop_pii: bool) -> pd.DataFrame:
    """Drop the CSV index artefact and PII, sort chronologically, validate."""
    if C.UNNAMED_INDEX_COL in df.columns:
        df = df.drop(columns=[C.UNNAMED_INDEX_COL])

    if drop_pii:
        present = [c for c in C.PII_COLS if c in df.columns]
        if present:
            df = df.drop(columns=present)

    # Every trailing-window feature downstream assumes this ordering.
    df = df.sort_values(C.TIMESTAMP_COL, kind="stable").reset_index(drop=True)

    if validate:
        RAW_TRANSACTION_SCHEMA.validate(df)
    return df


def load_all_raw(directory: Path | None = None, *, validate: bool = True) -> pd.DataFrame:
    """Read and concatenate every raw CSV, re-sorted chronologically.

    The vendor ships a pre-split ``fraudTrain`` / ``fraudTest`` pair. We ignore
    that split and re-partition by date ourselves -- see
    :mod:`fraudlens.data.splits` for why the split strategy matters.
    """
    paths = find_raw_csvs(directory)
    if not paths:
        target = directory or get_settings().raw_dir
        msg = (
            f"no raw CSVs found under {target}. "
            f"Run `python -m fraudlens.data.loader download` or `scripts/prepare_data.py`."
        )
        raise DatasetNotFoundError(msg)

    frames = [load_raw(p, validate=False, drop_pii=True) for p in paths]
    combined = pd.concat(frames, ignore_index=True)
    return _postprocess(combined, validate=validate, drop_pii=False)


def stratified_time_sample(
    df: pd.DataFrame,
    n: int = 5_000,
    *,
    seed: int = 42,
    target_col: str = C.TARGET_COL,
) -> pd.DataFrame:
    """Draw a sample that preserves both the fraud rate and the time span.

    A naive ``df.sample(n)`` on a 0.5%-positive dataset yields a handful of
    frauds and a sample whose fraud rate swings wildly with the seed. Tests that
    assert on class balance would then be flaky. This samples each class
    independently and keeps chronological order, so the result is a usable
    miniature of the real thing.
    """
    if n <= 0:
        msg = f"sample size must be positive, got {n}"
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    positives = df.index[df[target_col] == 1]
    negatives = df.index[df[target_col] == 0]

    base_rate = len(positives) / len(df) if len(df) else 0.0
    n_pos = min(len(positives), max(1, round(n * base_rate)))
    n_neg = min(len(negatives), n - n_pos)

    picked = np.concatenate(
        [
            rng.choice(positives, size=n_pos, replace=False),
            rng.choice(negatives, size=n_neg, replace=False),
        ]
    )
    return df.loc[picked].sort_values(C.TIMESTAMP_COL, kind="stable").reset_index(drop=True)


def write_sample(
    df: pd.DataFrame,
    destination: Path | None = None,
    *,
    n: int = 5_000,
    seed: int = 42,
) -> Path:
    """Write the committed test sample to ``data/sample/``."""
    directory = destination or get_settings().sample_dir
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / SAMPLE_FILENAME

    sample = stratified_time_sample(df, n=n, seed=seed)
    sample.to_csv(path, index=False)
    logger.info(
        "Wrote %d rows (%d fraud, %.3f%%) to %s",
        len(sample),
        int(sample[C.TARGET_COL].sum()),
        100 * sample[C.TARGET_COL].mean(),
        path,
    )
    return path


def load_sample(path: Path | None = None, *, validate: bool = True) -> pd.DataFrame:
    """Load the committed sample. This is what the test suite and CI use."""
    path = path or (get_settings().sample_dir / SAMPLE_FILENAME)
    if not path.exists():
        msg = (
            f"sample not found at {path}. Generate it with "
            f"`python scripts/prepare_data.py` once the raw dataset is downloaded."
        )
        raise DatasetNotFoundError(msg)
    return load_raw(path, validate=validate, drop_pii=True)


def summarise(df: pd.DataFrame) -> dict[str, object]:
    """Headline statistics, for logs, the data card and EDA."""
    return {
        "rows": len(df),
        "cards": int(df[C.CARD_COL].nunique()),
        "merchants": int(df["merchant"].nunique()),
        "categories": int(df["category"].nunique()),
        "frauds": int(df[C.TARGET_COL].sum()),
        "fraud_rate": float(df[C.TARGET_COL].mean()),
        "start": df[C.TIMESTAMP_COL].min(),
        "end": df[C.TIMESTAMP_COL].max(),
        "amount_median": float(df[C.AMOUNT_COL].median()),
        "amount_max": float(df[C.AMOUNT_COL].max()),
    }


def _column_order(df: pd.DataFrame, preferred: Sequence[str]) -> list[str]:
    """Stable column ordering: preferred names first, then the remainder."""
    head = [c for c in preferred if c in df.columns]
    return head + [c for c in df.columns if c not in head]
