"""Correctness tests for individual features.

Where a feature has a knowable right answer, it is pinned to that answer rather
than to a range. A geo feature that is wrong by a factor of two still produces
plausible-looking distances.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import pytest

from fraudlens.features.pipeline import (
    IMPOSSIBLE_SPEED_KMH,
    CardState,
    FeaturePipeline,
    TransactionEvent,
    haversine_km,
)
from tests.unit.test_leakage import BASE, event, replay


class TestHaversine:
    def test_zero_distance(self) -> None:
        assert haversine_km(40.0, -74.0, 40.0, -74.0) == pytest.approx(0.0)

    def test_known_distance_london_to_paris(self) -> None:
        # Great-circle distance is about 344 km.
        d = haversine_km(51.5074, -0.1278, 48.8566, 2.3522)
        assert d == pytest.approx(344.0, abs=5.0)

    def test_known_distance_new_york_to_los_angeles(self) -> None:
        d = haversine_km(40.7128, -74.0060, 34.0522, -118.2437)
        assert d == pytest.approx(3936.0, abs=15.0)

    def test_one_degree_of_latitude_is_about_111km(self) -> None:
        assert haversine_km(0.0, 0.0, 1.0, 0.0) == pytest.approx(111.19, abs=0.5)

    def test_is_symmetric(self) -> None:
        a = haversine_km(12.0, 34.0, -56.0, 78.0)
        b = haversine_km(-56.0, 78.0, 12.0, 34.0)
        assert a == pytest.approx(b)

    def test_nan_propagates(self) -> None:
        assert math.isnan(haversine_km(math.nan, 0.0, 1.0, 1.0))


class TestAmountFeatures:
    def test_log_transform(self) -> None:
        features = replay([event(0, amount=99.0)])[0]
        assert features["amt"] == pytest.approx(99.0)
        assert features["amt_log"] == pytest.approx(math.log1p(99.0))

    def test_first_transaction_gets_neutral_defaults_not_nan(self) -> None:
        # Neutral rather than NaN so that "unusual" and "unknown" are not
        # conflated in a tree split; card_history_count carries the distinction.
        f = replay([event(0, amount=500.0)])[0]
        assert f["amt_zscore_30d"] == 0.0
        assert f["amt_ratio_mean_30d"] == 1.0
        assert f["amt_ratio_max_30d"] == 1.0

    def test_zscore_detects_an_outlying_amount(self) -> None:
        history = [event(-60 * 24 * d, amount=50.0 + d) for d in range(10, 0, -1)]
        spike = replay([*history, event(0, amount=5_000.0)])[-1]
        assert spike["amt_zscore_30d"] > 3.0
        assert spike["amt_ratio_mean_30d"] > 50.0

    def test_ratio_to_mean_is_exact(self) -> None:
        history = [event(-600 + i, amount=100.0) for i in range(4)]
        f = replay([*history, event(0, amount=250.0)])[-1]
        assert f["amt_ratio_mean_30d"] == pytest.approx(2.5)
        assert f["amt_ratio_max_30d"] == pytest.approx(2.5)

    def test_constant_history_does_not_divide_by_zero_std(self) -> None:
        history = [event(-600 + i, amount=100.0) for i in range(5)]
        f = replay([*history, event(0, amount=100.0)])[-1]
        assert f["amt_zscore_30d"] == 0.0


class TestVelocityFeatures:
    def test_counts_across_windows(self) -> None:
        # Events must be supplied in chronological order: the pipeline is a
        # state machine, and feeding it the future first is not a supported
        # input (transform_frame sorts for exactly this reason).
        history = [
            event(-60 * 30),  # 30 hours ago: inside 7d only
            event(-90),  # 90 minutes ago: inside 24h, outside 1h
            event(-30),  # 30 minutes ago: inside 1h
        ]
        f = replay([*history, event(0)])[-1]
        assert f["txn_count_1h"] == 1.0
        assert f["txn_count_24h"] == 2.0
        assert f["txn_count_7d"] == 3.0

    def test_amount_sums_are_exact(self) -> None:
        history = [event(-10, amount=25.0), event(-20, amount=75.0)]
        f = replay([*history, event(0)])[-1]
        assert f["amt_sum_1h"] == pytest.approx(100.0)

    def test_burst_produces_high_short_window_count(self) -> None:
        burst = [event(-i, merchant=f"m{i}") for i in range(9, 0, -1)]
        f = replay([*burst, event(0)])[-1]
        assert f["txn_count_1h"] == 9.0


class TestGeoFeatures:
    def test_distance_home_to_merchant(self) -> None:
        f = replay([event(0, merch_lat=41.7, merch_lon=-74.0)])[0]
        # Home is at 40.7; one degree of latitude is about 111 km.
        assert f["dist_home_merch_km"] == pytest.approx(111.2, abs=1.0)

    def test_first_transaction_has_no_previous_distance(self) -> None:
        f = replay([event(0)])[0]
        assert f["dist_prev_txn_km"] == 0.0
        assert f["implied_speed_kmh"] == 0.0
        assert f["is_impossible_travel"] == 0.0

    def test_implied_speed_is_distance_over_time(self) -> None:
        # 1 degree of latitude (about 111 km) in exactly 1 hour.
        first = event(0, merch_lat=40.0, merch_lon=-74.0)
        second = event(60, merch_lat=41.0, merch_lon=-74.0)
        f = replay([first, second])[-1]
        assert f["implied_speed_kmh"] == pytest.approx(111.2, abs=1.0)
        assert f["is_impossible_travel"] == 0.0

    def test_impossible_travel_is_flagged(self) -> None:
        # New York to Los Angeles in six minutes.
        first = event(0, merch_lat=40.7128, merch_lon=-74.0060)
        second = event(6, merch_lat=34.0522, merch_lon=-118.2437)
        f = replay([first, second])[-1]
        assert f["implied_speed_kmh"] > IMPOSSIBLE_SPEED_KMH
        assert f["is_impossible_travel"] == 1.0

    def test_plausible_flight_is_not_flagged(self) -> None:
        # The same journey over six hours is an ordinary flight.
        first = event(0, merch_lat=40.7128, merch_lon=-74.0060)
        second = event(360, merch_lat=34.0522, merch_lon=-118.2437)
        f = replay([first, second])[-1]
        assert f["is_impossible_travel"] == 0.0

    def test_simultaneous_distant_transactions_are_maximally_suspicious(self) -> None:
        # Same timestamp, different cities: division by zero would be the naive
        # outcome, so this must be handled explicitly.
        pipeline = FeaturePipeline()
        state = CardState(card=1)
        first = event(0, merch_lat=40.7, merch_lon=-74.0)
        state.update(first)
        f = pipeline.compute(event(0, merch_lat=34.0, merch_lon=-118.2), state)
        assert f["is_impossible_travel"] == 1.0
        assert math.isfinite(f["implied_speed_kmh"])


class TestTemporalFeatures:
    def test_hour_and_weekday(self) -> None:
        f = replay([event(0)])[0]
        assert f["hour"] == float(BASE.hour)
        assert f["day_of_week"] == float(BASE.weekday())

    def test_night_flag(self) -> None:
        night = TransactionEvent(
            timestamp=datetime(2020, 1, 15, 2, 30),
            card=1,
            amount=10.0,
            merchant="m",
            category="c",
            home_lat=40.0,
            home_lon=-74.0,
            merch_lat=40.0,
            merch_lon=-74.0,
        )
        assert FeaturePipeline().compute(night, CardState(card=1))["is_night"] == 1.0

    def test_weekend_flag(self) -> None:
        saturday = event((datetime(2020, 1, 18) - BASE).total_seconds() / 60)
        assert replay([saturday])[0]["is_weekend"] == 1.0

    def test_seconds_since_previous(self) -> None:
        f = replay([event(0), event(5)])[-1]
        assert f["secs_since_prev_txn"] == pytest.approx(300.0)

    def test_first_transaction_uses_sentinel_gap(self) -> None:
        assert replay([event(0)])[0]["secs_since_prev_txn"] == -1.0


class TestNoveltyFeatures:
    def test_new_category_flag(self) -> None:
        vectors = replay([event(0, category="grocery_pos"), event(10, category="misc_net")])
        assert vectors[1]["is_new_category"] == 1.0

    def test_distinct_merchant_count_detects_card_testing(self) -> None:
        # Six merchants inside an hour is the card-testing signature.
        burst = [event(-i * 5, merchant=f"shop_{i}") for i in range(6, 0, -1)]
        f = replay([*burst, event(0, merchant="shop_final")])[-1]
        assert f["distinct_merchants_24h"] == 6.0

    def test_repeat_merchant_does_not_inflate_distinct_count(self) -> None:
        repeats = [event(-i * 5, merchant="same_shop") for i in range(5, 0, -1)]
        f = replay([*repeats, event(0)])[-1]
        assert f["distinct_merchants_24h"] == 1.0


class TestDemographicFeatures:
    def test_age_in_years(self) -> None:
        f = replay([event(0)])[0]
        expected = (BASE - datetime(1985, 6, 1)).days / 365.25
        assert f["age_years"] == pytest.approx(expected, abs=0.01)

    def test_missing_dob_yields_nan(self) -> None:
        no_dob = TransactionEvent(
            timestamp=BASE,
            card=1,
            amount=10.0,
            merchant="m",
            category="c",
            home_lat=40.0,
            home_lon=-74.0,
            merch_lat=40.0,
            merch_lon=-74.0,
            dob=None,
        )
        assert math.isnan(FeaturePipeline().compute(no_dob, CardState(card=1))["age_years"])

    def test_city_pop_is_log_scaled(self) -> None:
        f = replay([event(0)])[0]
        assert f["city_pop_log"] == pytest.approx(math.log1p(50_000))


class TestFeatureContract:
    def test_compute_returns_every_declared_feature(self) -> None:
        f = replay([event(0)])[0]
        assert set(f) == set(FeaturePipeline.FEATURE_NAMES)

    def test_feature_names_are_unique(self) -> None:
        names = FeaturePipeline.FEATURE_NAMES
        assert len(names) == len(set(names))

    def test_to_vector_preserves_declared_order(self) -> None:
        pipeline = FeaturePipeline()
        f = replay([event(0)])[0]
        vector = pipeline.to_vector(f)
        assert vector.shape == (len(FeaturePipeline.FEATURE_NAMES),)
        assert vector[0] == pytest.approx(f[FeaturePipeline.FEATURE_NAMES[0]])

    def test_all_features_are_finite_or_explicitly_nan(self) -> None:
        # Infinities break tree learners in ways that are hard to trace; NaN is
        # handled natively. No feature may produce inf.
        history = [event(-i, amount=float(i)) for i in range(20, 0, -1)]
        for f in replay([*history, event(0, amount=1e6)]):
            for name, value in f.items():
                assert not math.isinf(value), f"{name} produced an infinity"


class TestCardState:
    def test_pruning_removes_stale_entries(self) -> None:
        state = CardState(card=1)
        state.update(event(0))
        state.prune(BASE + timedelta(days=90))
        assert state.is_empty

    def test_last_returns_most_recent(self) -> None:
        state = CardState(card=1)
        state.update(event(0, amount=10.0))
        state.update(event(5, amount=20.0))
        assert state.last is not None
        assert state.last.amount == pytest.approx(20.0)

    def test_empty_state_has_no_last(self) -> None:
        assert CardState(card=1).last is None
