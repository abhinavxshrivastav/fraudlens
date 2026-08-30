"""Tests for temporal splitting.

The properties asserted here are the ones that keep the evaluation honest, so
these tests are load-bearing rather than incidental: if the embargo silently
stopped working, every downstream metric would improve and nothing else would
fail.
"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import pytest

from fraudlens.config import constants as C
from fraudlens.data.splits import (
    Split,
    SplitBoundaries,
    SplitConfigurationError,
    assign_splits,
    split_frame,
)
from tests.conftest import make_transactions


class TestSplitBoundaries:
    def test_defaults_are_internally_consistent(self) -> None:
        # The shipped constants must themselves satisfy the embargo rule.
        SplitBoundaries()

    def test_rejects_reversed_fold(self) -> None:
        with pytest.raises(SplitConfigurationError, match=r"starts .* after it ends"):
            SplitBoundaries(train_start=date(2020, 6, 1), train_end=date(2019, 1, 1))

    def test_rejects_overlapping_folds(self) -> None:
        with pytest.raises(SplitConfigurationError, match="overlaps"):
            SplitBoundaries(
                train_end=date(2020, 3, 31),
                valid_start=date(2020, 3, 1),
                valid_end=date(2020, 6, 30),
            )

    def test_rejects_insufficient_embargo(self) -> None:
        # One day of separation is a chronological split but not an embargoed
        # one: trailing-window features would still straddle the boundary.
        with pytest.raises(SplitConfigurationError, match="embargo violated"):
            SplitBoundaries(
                train_end=date(2020, 3, 31),
                valid_start=date(2020, 4, 1),
                embargo_days=7,
            )

    def test_iteration_yields_every_fold_in_order(self) -> None:
        boundaries = SplitBoundaries()
        assert [split for split, _, _ in boundaries] == [
            Split.TRAIN,
            Split.VALIDATION,
            Split.TEST,
        ]

    def test_bounds_round_trip(self) -> None:
        boundaries = SplitBoundaries()
        assert boundaries.bounds(Split.TEST) == (C.TEST_START, C.TEST_END)


class TestAssignSplits:
    def test_labels_each_fold(self, transactions: pd.DataFrame) -> None:
        labels = assign_splits(transactions)
        assert set(labels.dropna().unique()) <= {"train", "validation", "test"}
        assert labels.notna().any()

    def test_embargoed_rows_are_unlabelled(self) -> None:
        # Place rows squarely inside each embargo gap.
        gap_rows = pd.DataFrame(
            {
                C.TIMESTAMP_COL: pd.to_datetime([datetime(2020, 4, 3), datetime(2020, 7, 3)]),
            }
        )
        assert assign_splits(gap_rows).isna().all()

    def test_boundaries_are_inclusive(self) -> None:
        edges = pd.DataFrame(
            {
                C.TIMESTAMP_COL: pd.to_datetime(
                    [
                        datetime(2019, 1, 1, 0, 0, 0),  # first instant of train
                        datetime(2020, 3, 31, 23, 59, 59),  # last instant of train
                        datetime(2020, 12, 31, 23, 59, 59),  # last instant of test
                    ]
                )
            }
        )
        assert list(assign_splits(edges)) == ["train", "train", "test"]

    def test_rejects_non_datetime_column(self) -> None:
        df = pd.DataFrame({C.TIMESTAMP_COL: ["2020-01-01", "2020-01-02"]})
        with pytest.raises(TypeError, match="must be datetime"):
            assign_splits(df)

    def test_rejects_missing_column(self) -> None:
        with pytest.raises(KeyError):
            assign_splits(pd.DataFrame({"other": [1]}))


class TestSplitFrame:
    def test_folds_are_chronologically_ordered_and_disjoint(
        self, transactions: pd.DataFrame
    ) -> None:
        frames, _ = split_frame(transactions)

        train, valid, test = (frames[s] for s in Split)
        assert not train.empty and not valid.empty and not test.empty

        # Every training row precedes every validation row, and so on. This is
        # the property a random split destroys.
        assert train[C.TIMESTAMP_COL].max() < valid[C.TIMESTAMP_COL].min()
        assert valid[C.TIMESTAMP_COL].max() < test[C.TIMESTAMP_COL].min()

        ids = [set(frames[s][C.TRANSACTION_ID_COL]) for s in Split]
        assert ids[0].isdisjoint(ids[1])
        assert ids[1].isdisjoint(ids[2])
        assert ids[0].isdisjoint(ids[2])

    def test_embargo_gap_is_respected_in_output(self, transactions: pd.DataFrame) -> None:
        frames, _ = split_frame(transactions)
        gap = frames[Split.VALIDATION][C.TIMESTAMP_COL].min() - (
            frames[Split.TRAIN][C.TIMESTAMP_COL].max()
        )
        assert gap.days >= C.EMBARGO_DAYS - 1

    def test_each_fold_is_sorted_by_timestamp(self, transactions: pd.DataFrame) -> None:
        # Trailing-window features assume this ordering.
        frames, _ = split_frame(transactions.sample(frac=1.0, random_state=3))
        for part in frames.values():
            assert part[C.TIMESTAMP_COL].is_monotonic_increasing

    def test_report_accounts_for_every_row(self, transactions: pd.DataFrame) -> None:
        frames, report = split_frame(transactions)
        assigned = sum(len(f) for f in frames.values())
        total = assigned + report.discarded_embargo + report.discarded_out_of_range
        assert total == len(transactions)

    def test_report_counts_frauds_per_fold(self, transactions: pd.DataFrame) -> None:
        frames, report = split_frame(transactions)
        for split in Split:
            assert report.fraud_counts[str(split)] == int(frames[split][C.TARGET_COL].sum())

    def test_report_renders_markdown(self, transactions: pd.DataFrame) -> None:
        _, report = split_frame(transactions)
        rendered = report.to_markdown()
        assert "| Fold |" in rendered
        assert "train" in rendered

    def test_rows_outside_range_are_discarded_not_assigned(self) -> None:
        outside = make_transactions(n=20, start=datetime(2015, 1, 1), span_days=30)
        frames, report = split_frame(outside)
        assert all(f.empty for f in frames.values())
        assert report.discarded_out_of_range == 20
        assert report.discarded_embargo == 0
