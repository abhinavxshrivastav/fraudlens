"""Temporal train/validation/test splitting with an embargo gap.

Why this module exists at all
-----------------------------
Fraud data is time-ordered and the generating process drifts. A random split
lets the model see the future: it trains on transactions that occur *after* the
ones it is scored on, and card-level behavioural patterns bleed across the
boundary. Reported metrics then bear no relation to what the model would do in
production. This is the single most common flaw in public fraud-detection work,
and it typically inflates PR-AUC by a wide margin.

The embargo
-----------
A chronological split alone is not sufficient here. Behavioural features look
back up to :data:`~fraudlens.config.constants.MAX_LOOKBACK_DAYS`. A validation
row one day after the boundary computes its trailing-30-day aggregates almost
entirely from training rows, so information bleeds forward and the two folds are
not independent.

We therefore discard a gap of :data:`~fraudlens.config.constants.EMBARGO_DAYS`
between folds. This costs a small amount of data and buys genuinely independent
evaluation. The technique is standard in financial ML, where it is known as
purging and embargoing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING

import pandas as pd

from fraudlens.config import constants as C

if TYPE_CHECKING:
    from collections.abc import Iterator


class Split(StrEnum):
    """Named evaluation folds.

    ``TEST`` is scored exactly once, at the end of model selection. Every
    threshold, calibration map and early-stopping decision is made on
    ``VALIDATION``.
    """

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class SplitConfigurationError(ValueError):
    """Raised when split boundaries are internally inconsistent."""


@dataclass(frozen=True, slots=True)
class SplitBoundaries:
    """Inclusive date boundaries for each fold.

    Both ends are inclusive: a row dated ``train_end`` belongs to train.
    """

    train_start: date = C.TRAIN_START
    train_end: date = C.TRAIN_END
    valid_start: date = C.VALID_START
    valid_end: date = C.VALID_END
    test_start: date = C.TEST_START
    test_end: date = C.TEST_END
    embargo_days: int = C.EMBARGO_DAYS

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Assert the folds are ordered, non-overlapping and properly embargoed."""
        for name, start, end in (
            ("train", self.train_start, self.train_end),
            ("validation", self.valid_start, self.valid_end),
            ("test", self.test_start, self.test_end),
        ):
            if start > end:
                msg = f"{name} fold starts ({start}) after it ends ({end})"
                raise SplitConfigurationError(msg)

        required = timedelta(days=self.embargo_days)
        for earlier_name, earlier_end, later_name, later_start in (
            ("train", self.train_end, "validation", self.valid_start),
            ("validation", self.valid_end, "test", self.test_start),
        ):
            gap = later_start - earlier_end
            if gap <= timedelta(0):
                msg = (
                    f"{later_name} fold starts on {later_start}, which overlaps "
                    f"{earlier_name} ending {earlier_end}"
                )
                raise SplitConfigurationError(msg)
            if gap < required:
                msg = (
                    f"embargo violated between {earlier_name} and {later_name}: "
                    f"gap is {gap.days} day(s), need at least {self.embargo_days}. "
                    f"Trailing-window features would leak across the boundary."
                )
                raise SplitConfigurationError(msg)

    def bounds(self, split: Split) -> tuple[date, date]:
        """Return the inclusive ``(start, end)`` dates for one fold."""
        match split:
            case Split.TRAIN:
                return self.train_start, self.train_end
            case Split.VALIDATION:
                return self.valid_start, self.valid_end
            case Split.TEST:
                return self.test_start, self.test_end

    def __iter__(self) -> Iterator[tuple[Split, date, date]]:
        for split in Split:
            start, end = self.bounds(split)
            yield split, start, end


@dataclass(frozen=True, slots=True)
class SplitReport:
    """Row counts and fraud rates per fold, for logging and the model card."""

    counts: dict[str, int]
    fraud_counts: dict[str, int]
    discarded_embargo: int
    discarded_out_of_range: int

    @property
    def fraud_rates(self) -> dict[str, float]:
        return {
            name: (self.fraud_counts[name] / n if n else 0.0) for name, n in self.counts.items()
        }

    def to_markdown(self) -> str:
        """Render as a Markdown table for the model card and run logs."""
        lines = ["| Fold | Rows | Frauds | Fraud rate |", "|---|---:|---:|---:|"]
        rates = self.fraud_rates
        for name, n in self.counts.items():
            lines.append(f"| {name} | {n:,} | {self.fraud_counts[name]:,} | {rates[name]:.4%} |")
        lines.append(
            f"\nDiscarded: {self.discarded_embargo:,} rows in embargo gaps, "
            f"{self.discarded_out_of_range:,} rows outside the split range."
        )
        return "\n".join(lines)


def _as_timestamp(value: date, *, end_of_day: bool) -> pd.Timestamp:
    """Convert an inclusive date boundary to a timestamp.

    End boundaries cover the whole day, so ``2020-03-31`` includes everything up
    to ``2020-03-31 23:59:59.999999``.
    """
    moment = datetime.combine(value, datetime.max.time() if end_of_day else datetime.min.time())
    return pd.Timestamp(moment)


def assign_splits(
    df: pd.DataFrame,
    boundaries: SplitBoundaries | None = None,
    *,
    timestamp_col: str = C.TIMESTAMP_COL,
) -> pd.Series:
    """Label each row with its fold, or ``pd.NA`` if embargoed or out of range.

    Returns a string Series aligned to ``df.index``.
    """
    boundaries = boundaries or SplitBoundaries()
    if timestamp_col not in df.columns:
        msg = f"timestamp column {timestamp_col!r} not present in frame"
        raise KeyError(msg)

    ts = df[timestamp_col]
    if not pd.api.types.is_datetime64_any_dtype(ts):
        msg = f"{timestamp_col!r} must be datetime dtype, got {ts.dtype}"
        raise TypeError(msg)

    labels = pd.Series(pd.NA, index=df.index, dtype="string")
    for split, start, end in boundaries:
        mask = ts.between(
            _as_timestamp(start, end_of_day=False),
            _as_timestamp(end, end_of_day=True),
        )
        labels = labels.mask(mask, str(split))
    return labels


def split_frame(
    df: pd.DataFrame,
    boundaries: SplitBoundaries | None = None,
    *,
    timestamp_col: str = C.TIMESTAMP_COL,
    target_col: str = C.TARGET_COL,
) -> tuple[dict[Split, pd.DataFrame], SplitReport]:
    """Partition ``df`` into folds, discarding embargoed rows.

    Each returned frame is sorted by timestamp, which every downstream
    trailing-window feature depends on.
    """
    boundaries = boundaries or SplitBoundaries()
    labels = assign_splits(df, boundaries, timestamp_col=timestamp_col)

    frames: dict[Split, pd.DataFrame] = {}
    counts: dict[str, int] = {}
    fraud_counts: dict[str, int] = {}

    for split in Split:
        part = df.loc[labels == str(split)].sort_values(timestamp_col, kind="stable")
        frames[split] = part
        counts[str(split)] = len(part)
        fraud_counts[str(split)] = int(part[target_col].sum()) if target_col in part.columns else 0

    unassigned = df.loc[labels.isna()]
    ts = unassigned[timestamp_col]
    in_range = ts.between(
        _as_timestamp(boundaries.train_start, end_of_day=False),
        _as_timestamp(boundaries.test_end, end_of_day=True),
    )
    report = SplitReport(
        counts=counts,
        fraud_counts=fraud_counts,
        discarded_embargo=int(in_range.sum()),
        discarded_out_of_range=int((~in_range).sum()),
    )
    return frames, report
