"""Adversarial tests that no feature can see its own transaction or the future.

These are the most important tests in the project. Temporal leakage does not
raise an error — it silently *improves* every metric, which means it is the one
class of bug that cannot be caught by watching for regressions. It has to be
attacked directly.

The technique throughout: construct two histories that are identical up to the
transaction under test and differ only in what happens at or after it. Any
feature that moves between the two is reading the future.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from fraudlens.config import constants as C
from fraudlens.features.pipeline import CardState, FeaturePipeline, TransactionEvent

BASE = datetime(2020, 1, 15, 12, 0, 0)


def event(
    offset_minutes: float = 0.0,
    *,
    amount: float = 50.0,
    merchant: str = "merchant_a",
    category: str = "grocery_pos",
    merch_lat: float = 40.0,
    merch_lon: float = -74.0,
    card: int = 4_000_000_000_000_000,
) -> TransactionEvent:
    return TransactionEvent(
        timestamp=BASE + timedelta(minutes=offset_minutes),
        card=card,
        amount=amount,
        merchant=merchant,
        category=category,
        home_lat=40.7,
        home_lon=-74.0,
        merch_lat=merch_lat,
        merch_lon=merch_lon,
        city_pop=50_000,
        dob=datetime(1985, 6, 1),
        transaction_id=f"txn_{offset_minutes}",
    )


def replay(events: list[TransactionEvent]) -> list[dict[str, float]]:
    """Feed events through the pipeline in order, collecting feature vectors."""
    pipeline = FeaturePipeline()
    state = CardState(card=events[0].card)
    vectors = []
    for e in events:
        state.prune(e.timestamp)
        vectors.append(pipeline.compute(e, state))
        state.update(e)
    return vectors


class TestFutureCannotLeakBackwards:
    def test_later_transactions_do_not_change_earlier_features(self) -> None:
        # The decisive test. Score the same transaction in two worlds: one where
        # nothing follows it, one where a huge burst of activity follows it.
        # Its feature vector must be byte-identical in both.
        target = event(0)
        quiet = replay([target])
        noisy = replay(
            [
                target,
                event(1, amount=9_999.0, merchant="m2"),
                event(2, amount=9_999.0, merchant="m3"),
                event(3, amount=9_999.0, merchant="m4"),
            ]
        )
        assert quiet[0] == noisy[0]

    def test_a_planted_future_spike_does_not_move_the_amount_zscore(self) -> None:
        history = [event(-60 * 24 * d, amount=50.0) for d in range(10, 0, -1)]
        target = event(0, amount=200.0)
        target_index = len(history)

        # Index the *target's* vector in both runs. Taking [-1] would compare
        # the target against the planted future transaction instead.
        without_future = replay([*history, target])[target_index]
        with_future = replay([*history, target, event(5, amount=1_000_000.0)])[target_index]

        assert without_future["amt_zscore_30d"] == pytest.approx(with_future["amt_zscore_30d"])
        assert without_future["amt_ratio_max_30d"] == pytest.approx(
            with_future["amt_ratio_max_30d"]
        )

    def test_velocity_counts_exclude_the_current_transaction(self) -> None:
        # A card's very first transaction must report zero prior activity, not
        # one. Off-by-one here would leak the label-bearing row into its own
        # features.
        first = replay([event(0)])[0]
        assert first["txn_count_1h"] == 0.0
        assert first["txn_count_24h"] == 0.0
        assert first["amt_sum_1h"] == 0.0
        assert first["card_history_count"] == 0.0

    def test_velocity_counts_only_strictly_prior_transactions(self) -> None:
        vectors = replay([event(0), event(10), event(20)])
        assert [v["txn_count_1h"] for v in vectors] == [0.0, 1.0, 2.0]

    def test_novelty_flags_do_not_see_the_current_merchant(self) -> None:
        # First visit to a merchant must be flagged new; the second must not.
        vectors = replay([event(0, merchant="new_shop"), event(10, merchant="new_shop")])
        assert vectors[0]["is_new_merchant"] == 1.0
        assert vectors[1]["is_new_merchant"] == 0.0


class TestWindowBoundaries:
    def test_transactions_outside_the_window_are_excluded(self) -> None:
        # 61 minutes earlier is outside the 1h window but inside 24h.
        vectors = replay([event(-61), event(0)])
        assert vectors[1]["txn_count_1h"] == 0.0
        assert vectors[1]["txn_count_24h"] == 1.0

    def test_history_older_than_max_lookback_is_pruned(self) -> None:
        stale = event(-60 * 24 * (C.MAX_LOOKBACK_DAYS + 5))
        vectors = replay([stale, event(0)])
        assert vectors[1]["txn_count_30d"] == 0.0
        assert vectors[1]["card_history_count"] == 0.0

    def test_pruning_keeps_state_bounded(self) -> None:
        state = CardState(card=1)
        pipeline = FeaturePipeline()
        for day in range(120):
            e = event(-60 * 24 * (120 - day))
            state.prune(e.timestamp)
            pipeline.compute(e, state)
            state.update(e)
        # Only the trailing lookback window is retained, not 120 days of history.
        assert len(state.history) <= C.MAX_LOOKBACK_DAYS + 1


class TestCardIsolation:
    def test_one_card_does_not_see_another_cards_history(self) -> None:
        pipeline = FeaturePipeline()
        frame = pd.DataFrame(
            [
                _row(event(0, card=111)),
                _row(event(1, card=111)),
                _row(event(2, card=222)),
            ]
        )
        features = pipeline.transform_frame(frame)
        # Card 222's first transaction sees no history, despite two earlier
        # transactions existing on a different card.
        assert features.iloc[2]["txn_count_24h"] == 0.0
        assert features.iloc[1]["txn_count_24h"] == 1.0


class TestBatchOrdering:
    def test_row_order_in_the_input_does_not_change_the_output(self) -> None:
        # transform_frame sorts by timestamp internally. Feeding the same
        # transactions shuffled must produce the same per-row features, or the
        # pipeline is order-dependent in a way that would differ between a
        # sorted training file and an arbitrarily ordered live batch.
        pipeline = FeaturePipeline()
        rows = [_row(event(i * 10)) for i in range(8)]
        ordered = pd.DataFrame(rows)
        shuffled = ordered.sample(frac=1.0, random_state=5)

        from_ordered = pipeline.transform_frame(ordered).sort_index()
        from_shuffled = pipeline.transform_frame(shuffled).sort_index()
        pd.testing.assert_frame_equal(from_ordered, from_shuffled)

    def test_empty_frame_returns_empty_features_with_correct_columns(self) -> None:
        pipeline = FeaturePipeline()
        result = pipeline.transform_frame(pd.DataFrame())
        assert result.empty
        assert list(result.columns) == list(FeaturePipeline.FEATURE_NAMES)

    def test_output_has_one_row_per_input_row(self) -> None:
        pipeline = FeaturePipeline()
        frame = pd.DataFrame([_row(event(i)) for i in range(25)])
        assert len(pipeline.transform_frame(frame)) == 25

    def test_no_feature_column_is_entirely_null(self) -> None:
        pipeline = FeaturePipeline()
        frame = pd.DataFrame([_row(event(i * 30)) for i in range(20)])
        features = pipeline.transform_frame(frame)
        all_null = [c for c in features.columns if features[c].isna().all()]
        assert all_null == []


def _row(e: TransactionEvent) -> dict[str, object]:
    """Render an event back into raw-frame form."""
    return {
        C.TIMESTAMP_COL: pd.Timestamp(e.timestamp),
        C.CARD_COL: e.card,
        C.AMOUNT_COL: e.amount,
        "merchant": e.merchant,
        "category": e.category,
        C.HOME_LAT_COL: e.home_lat,
        C.HOME_LON_COL: e.home_lon,
        C.MERCH_LAT_COL: e.merch_lat,
        C.MERCH_LON_COL: e.merch_lon,
        "city_pop": e.city_pop,
        "dob": pd.Timestamp(e.dob) if e.dob else pd.NaT,
        C.TRANSACTION_ID_COL: e.transaction_id,
    }
