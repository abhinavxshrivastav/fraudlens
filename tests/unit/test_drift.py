"""Tests for drift detection.

PSI is easy to implement in a way that looks plausible and is wrong -- most often
by re-deriving bin edges from the live sample, which makes every window compare
against itself and reports zero drift forever. These tests pin the behaviour that
distinguishes a working monitor from a decorative one.
"""

from __future__ import annotations

import numpy as np
import pytest

from fraudlens.config import constants as C
from fraudlens.monitoring.drift import (
    CYCLICAL_FEATURES,
    DriftMonitor,
    DriftStatus,
    ks_statistic,
    population_stability_index,
    quantile_edges,
)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


class TestPopulationStabilityIndex:
    def test_identical_distributions_score_near_zero(self, rng: np.random.Generator) -> None:
        reference = rng.normal(0, 1, 10_000)
        live = rng.normal(0, 1, 10_000)
        assert population_stability_index(reference, live) < C.PSI_STABLE

    def test_a_shifted_distribution_is_detected(self, rng: np.random.Generator) -> None:
        reference = rng.normal(0, 1, 10_000)
        live = rng.normal(1.0, 1, 10_000)
        assert population_stability_index(reference, live) > C.PSI_INVESTIGATE

    def test_psi_grows_monotonically_with_the_shift(self, rng: np.random.Generator) -> None:
        reference = rng.normal(0, 1, 20_000)
        scores = [
            population_stability_index(reference, rng.normal(shift, 1, 20_000))
            for shift in (0.0, 0.25, 0.5, 1.0, 2.0)
        ]
        assert scores == sorted(scores)

    def test_a_variance_change_is_detected_even_with_the_same_mean(
        self, rng: np.random.Generator
    ) -> None:
        # A mean-only check would miss this entirely.
        reference = rng.normal(0, 1, 10_000)
        live = rng.normal(0, 3, 10_000)
        assert population_stability_index(reference, live) > C.PSI_STABLE

    def test_empty_input_is_zero_not_an_error(self) -> None:
        assert population_stability_index([], [1.0, 2.0]) == 0.0
        assert population_stability_index([1.0, 2.0], []) == 0.0

    def test_disjoint_ranges_stay_finite(self) -> None:
        # Bins empty in one distribution and full in the other would send the
        # log term to infinity without smoothing.
        psi = population_stability_index(np.arange(100.0), np.arange(1000.0, 1100.0))
        assert np.isfinite(psi)
        assert psi > C.PSI_INVESTIGATE

    def test_constant_feature_does_not_crash(self) -> None:
        assert np.isfinite(population_stability_index(np.ones(500), np.ones(500)))

    def test_nan_values_are_ignored(self, rng: np.random.Generator) -> None:
        reference = rng.normal(0, 1, 5_000)
        live = np.concatenate([rng.normal(0, 1, 5_000), np.full(100, np.nan)])
        assert np.isfinite(population_stability_index(reference, live))


class TestQuantileEdges:
    def test_edges_span_the_infinite_range(self) -> None:
        edges = quantile_edges(np.arange(1000.0), bins=10)
        assert edges[0] == -np.inf
        assert edges[-1] == np.inf

    def test_skewed_data_still_produces_usable_bins(self, rng: np.random.Generator) -> None:
        # Behavioural features are heavily skewed. Uniform bins would put nearly
        # everything in the first bucket; quantile bins must not.
        skewed = rng.gamma(1.5, 50.0, 10_000)
        edges = quantile_edges(skewed, bins=10)
        counts, _ = np.histogram(skewed, bins=edges)
        assert counts.min() > 0

    def test_constant_input_yields_a_single_usable_bin(self) -> None:
        edges = quantile_edges(np.full(100, 5.0))
        assert edges.size >= 2
        assert edges[0] < 5.0 < edges[-1]


class TestKsStatistic:
    def test_identical_distributions_score_near_zero(self, rng: np.random.Generator) -> None:
        assert ks_statistic(rng.normal(0, 1, 5_000), rng.normal(0, 1, 5_000)) < 0.1

    def test_shifted_distributions_score_high(self, rng: np.random.Generator) -> None:
        assert ks_statistic(rng.normal(0, 1, 5_000), rng.normal(3, 1, 5_000)) > 0.5

    def test_bounded_between_zero_and_one(self, rng: np.random.Generator) -> None:
        value = ks_statistic(rng.normal(0, 1, 1_000), rng.normal(10, 1, 1_000))
        assert 0.0 <= value <= 1.0


class TestDriftStatusBands:
    @pytest.mark.parametrize(
        ("psi", "expected"),
        [
            (0.0, DriftStatus.STABLE),
            (0.099, DriftStatus.STABLE),
            (0.10, DriftStatus.MODERATE),
            (0.24, DriftStatus.MODERATE),
            (0.25, DriftStatus.SIGNIFICANT),
            (5.0, DriftStatus.SIGNIFICANT),
        ],
    )
    def test_industry_standard_bands(self, psi: float, expected: DriftStatus) -> None:
        assert DriftStatus.from_psi(psi) is expected


class TestDriftMonitor:
    def _monitor(self, rng: np.random.Generator, **kwargs: object) -> DriftMonitor:
        reference = {
            "amt": rng.gamma(2.0, 40.0, 5_000),
            "txn_count_1h": rng.poisson(1.5, 5_000).astype(float),
        }
        return DriftMonitor(reference=reference, min_samples=50, **kwargs)  # type: ignore[arg-type]

    def test_reports_insufficient_data_before_the_minimum(self, rng: np.random.Generator) -> None:
        monitor = self._monitor(rng)
        for _ in range(10):
            monitor.observe({"amt": 50.0, "txn_count_1h": 1.0})
        assert monitor.measure("amt").status is DriftStatus.INSUFFICIENT_DATA

    def test_stable_when_live_matches_reference(self, rng: np.random.Generator) -> None:
        monitor = self._monitor(rng)
        for value in rng.gamma(2.0, 40.0, 1_000):
            monitor.observe({"amt": float(value), "txn_count_1h": 1.0})
        assert monitor.measure("amt").status is DriftStatus.STABLE

    def test_detects_a_genuine_shift(self, rng: np.random.Generator) -> None:
        monitor = self._monitor(rng)
        # Amounts triple: the kind of change that should raise an alarm.
        for value in rng.gamma(2.0, 120.0, 1_000):
            monitor.observe({"amt": float(value), "txn_count_1h": 1.0})
        result = monitor.measure("amt")
        assert result.status is DriftStatus.SIGNIFICANT
        assert result.mean_shift > 0.5

    def test_bin_edges_are_frozen_at_construction(self, rng: np.random.Generator) -> None:
        # The decisive test. If edges were re-derived from live data, each window
        # would be compared against itself and drift would always read zero.
        monitor = self._monitor(rng)
        for value in rng.gamma(2.0, 400.0, 1_000):
            monitor.observe({"amt": float(value), "txn_count_1h": 1.0})
        assert monitor.measure("amt").psi > C.PSI_INVESTIGATE

    def test_window_is_bounded(self, rng: np.random.Generator) -> None:
        monitor = self._monitor(rng, window=100)
        for _ in range(5_000):
            monitor.observe({"amt": 50.0, "txn_count_1h": 1.0})
        assert monitor.observations == 100

    def test_missing_features_are_skipped_not_fatal(self, rng: np.random.Generator) -> None:
        monitor = self._monitor(rng)
        for _ in range(200):
            monitor.observe({"amt": 50.0})  # txn_count_1h absent
        assert monitor.measure("amt").status is not DriftStatus.INSUFFICIENT_DATA
        assert monitor.measure("txn_count_1h").status is DriftStatus.INSUFFICIENT_DATA

    def test_drifting_lists_only_flagged_features(self, rng: np.random.Generator) -> None:
        monitor = self._monitor(rng)
        for value in rng.gamma(2.0, 400.0, 1_000):
            monitor.observe({"amt": float(value), "txn_count_1h": 1.5})
        drifting = {d.feature for d in monitor.drifting()}
        assert "amt" in drifting

    def test_measure_all_is_sorted_worst_first(self, rng: np.random.Generator) -> None:
        monitor = self._monitor(rng)
        for value in rng.gamma(2.0, 400.0, 1_000):
            monitor.observe({"amt": float(value), "txn_count_1h": 1.0})
        results = monitor.measure_all()
        assert [r.psi for r in results] == sorted([r.psi for r in results], reverse=True)

    def test_summary_shape(self, rng: np.random.Generator) -> None:
        monitor = self._monitor(rng)
        for _ in range(200):
            monitor.observe({"amt": 60.0, "txn_count_1h": 1.0})
        summary = monitor.summary()
        assert set(summary) >= {
            "observations",
            "features_tracked",
            "features_drifting",
            "worst_feature",
            "status",
        }


class TestCyclicalExclusion:
    def test_periodic_features_are_named_for_exclusion(self) -> None:
        assert {"hour", "day_of_week"} <= CYCLICAL_FEATURES

    def test_a_partial_cycle_would_produce_a_false_alarm(self) -> None:
        """Demonstrates *why* cyclical features are excluded.

        A window covering two weekdays against a reference covering all seven
        produces an enormous PSI while nothing has actually changed. This is the
        false alarm the exclusion exists to prevent.
        """
        full_week = np.repeat(np.arange(7.0), 1_000)
        two_days = np.repeat(np.array([1.0, 2.0]), 500)
        assert population_stability_index(full_week, two_days) > C.PSI_INVESTIGATE
