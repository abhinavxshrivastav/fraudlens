"""Tests for the evaluation harness.

These are written against *known-answer* cases wherever possible: a metric
implementation that is subtly wrong will still produce plausible-looking numbers
on random data, so most assertions here pin exact values computed by hand.
"""

from __future__ import annotations

import numpy as np
import pytest

from fraudlens.evaluation.metrics import (
    CostModel,
    ThresholdMetrics,
    alert_rate,
    cost_curve,
    evaluate_at_threshold,
    min_cost_threshold,
    pr_auc,
    precision_at_k,
    recall_at_fpr,
    roc_auc,
    threshold_grid,
)


class TestInputValidation:
    def test_rejects_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="different lengths"):
            pr_auc([0, 1], [0.1, 0.2, 0.3])

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            pr_auc([], [])

    def test_rejects_non_binary_labels(self) -> None:
        with pytest.raises(ValueError, match="only 0 and 1"):
            pr_auc([0, 1, 2], [0.1, 0.2, 0.3])


class TestRankingMetrics:
    def test_perfect_ranking_scores_one(self) -> None:
        y = [0, 0, 0, 1, 1]
        s = [0.1, 0.2, 0.3, 0.9, 0.95]
        assert pr_auc(y, s) == pytest.approx(1.0)
        assert roc_auc(y, s) == pytest.approx(1.0)

    def test_inverted_ranking_scores_poorly(self) -> None:
        y = [0, 0, 0, 1, 1]
        s = [0.95, 0.9, 0.8, 0.1, 0.05]
        assert roc_auc(y, s) == pytest.approx(0.0)
        assert pr_auc(y, s) < 0.5

    def test_pr_auc_of_random_scores_approaches_base_rate(self) -> None:
        # The floor for PR-AUC is the positive base rate, not 0.5 as for ROC-AUC.
        rng = np.random.default_rng(0)
        n = 20_000
        y = (rng.random(n) < 0.02).astype(int)
        s = rng.random(n)
        assert pr_auc(y, s) == pytest.approx(0.02, abs=0.01)

    def test_roc_auc_is_optimistic_relative_to_pr_auc_under_imbalance(
        self, separable_scores: tuple[np.ndarray, np.ndarray]
    ) -> None:
        # The central claim in the module docstring: on imbalanced data ROC-AUC
        # reads far more flattering than PR-AUC for the same model.
        y, s = separable_scores
        assert roc_auc(y, s) > pr_auc(y, s)


class TestRecallAtFpr:
    def test_perfect_separation_gives_full_recall_at_zero_cost(self) -> None:
        y = np.array([0] * 100 + [1] * 10)
        s = np.concatenate([np.linspace(0.0, 0.4, 100), np.linspace(0.6, 1.0, 10)])
        recall, _ = recall_at_fpr(y, s, 0.01)
        assert recall == pytest.approx(1.0)

    def test_returns_applicable_threshold(
        self, separable_scores: tuple[np.ndarray, np.ndarray]
    ) -> None:
        # The returned threshold must actually deliver the promised FPR when
        # applied -- this is the value that gets frozen and shipped.
        y, s = separable_scores
        recall, threshold = recall_at_fpr(y, s, 0.05)
        metrics = evaluate_at_threshold(y, s, threshold)
        assert metrics.false_positive_rate <= 0.05 + 1e-9
        assert metrics.recall == pytest.approx(recall, abs=1e-9)

    def test_tighter_fpr_budget_cannot_increase_recall(
        self, separable_scores: tuple[np.ndarray, np.ndarray]
    ) -> None:
        y, s = separable_scores
        loose, _ = recall_at_fpr(y, s, 0.10)
        tight, _ = recall_at_fpr(y, s, 0.01)
        assert tight <= loose

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
    def test_rejects_out_of_range_fpr(self, bad: float) -> None:
        with pytest.raises(ValueError, match="target_fpr"):
            recall_at_fpr([0, 1], [0.1, 0.9], bad)


class TestPrecisionAtK:
    def test_known_answer(self) -> None:
        # Top 4 by score are the rows scoring .9,.8,.7,.6 -> labels 1,0,1,1 -> 3/4.
        y = [1, 0, 1, 1, 0, 0]
        s = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
        assert precision_at_k(y, s, 4) == pytest.approx(0.75)

    def test_k_larger_than_population_is_clamped(self) -> None:
        y = [1, 0, 0, 0]
        s = [0.9, 0.1, 0.2, 0.3]
        assert precision_at_k(y, s, 999) == pytest.approx(0.25)

    def test_rejects_non_positive_k(self) -> None:
        with pytest.raises(ValueError, match="k must be positive"):
            precision_at_k([0, 1], [0.1, 0.9], 0)

    def test_ties_do_not_crash(self) -> None:
        assert 0.0 <= precision_at_k([1, 0, 1, 0], [0.5] * 4, 2) <= 1.0


class TestThresholdMetrics:
    def test_confusion_matrix_is_exact(self) -> None:
        y = [1, 1, 0, 0, 1, 0]
        s = [0.9, 0.4, 0.8, 0.1, 0.95, 0.2]
        m = evaluate_at_threshold(y, s, 0.5)
        # Flagged: indices 0 (fraud), 2 (legit), 4 (fraud)
        assert (m.true_positives, m.false_positives) == (2, 1)
        assert (m.true_negatives, m.false_negatives) == (2, 1)
        assert m.precision == pytest.approx(2 / 3)
        assert m.recall == pytest.approx(2 / 3)
        assert m.f1 == pytest.approx(2 / 3)
        assert m.n_alerts == 3
        assert m.alert_rate == pytest.approx(0.5)

    def test_value_detection_rate_differs_from_count_recall(self) -> None:
        # Two frauds caught, one missed -- but the missed one is the expensive
        # one. Count recall looks good; value recall tells the real story.
        y = [1, 1, 1]
        s = [0.9, 0.9, 0.1]
        amounts = [10.0, 10.0, 980.0]
        m = evaluate_at_threshold(y, s, 0.5, amounts=amounts)
        assert m.recall == pytest.approx(2 / 3)
        assert m.value_detection_rate == pytest.approx(20 / 1000)

    def test_empty_confusion_cells_do_not_divide_by_zero(self) -> None:
        m = evaluate_at_threshold([0, 0], [0.1, 0.2], 0.99)
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.f1 == 0.0

    def test_rejects_amount_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="amounts has length"):
            evaluate_at_threshold([0, 1], [0.1, 0.9], 0.5, amounts=[1.0])


class TestCostModel:
    def test_total_prices_each_error_type(self) -> None:
        costs = CostModel(
            investigation_gbp=5.0,
            false_positive_friction_gbp=20.0,
            false_negative_amount_multiplier=1.0,
        )
        # 10 alerts (x5) + 4 false positives (x20) + 300 of missed fraud value.
        assert costs.total(n_alerts=10, n_false_positives=4, missed_fraud_value=300.0) == (
            pytest.approx(50.0 + 80.0 + 300.0)
        )

    def test_expected_cost_flows_through_threshold_metrics(self) -> None:
        costs = CostModel(
            investigation_gbp=1.0,
            false_positive_friction_gbp=10.0,
            false_negative_amount_multiplier=1.0,
        )
        y = [1, 0, 1]
        s = [0.9, 0.9, 0.1]
        amounts = [50.0, 50.0, 200.0]
        m = evaluate_at_threshold(y, s, 0.5, amounts=amounts, cost_model=costs)
        # 2 alerts x1 + 1 FP x10 + 200 missed = 212
        assert m.expected_cost == pytest.approx(212.0)


class TestCostCurveAndThresholdSelection:
    def test_curve_is_monotonic_in_threshold(
        self, separable_scores: tuple[np.ndarray, np.ndarray]
    ) -> None:
        y, s = separable_scores
        curve = cost_curve(y, s, n_thresholds=50)
        thresholds = [m.threshold for m in curve]
        assert thresholds == sorted(thresholds)
        # Raising the bar can only reduce the number of alerts.
        alerts = [m.n_alerts for m in curve]
        assert alerts == sorted(alerts, reverse=True)

    def test_min_cost_threshold_is_the_curve_minimum(
        self, separable_scores: tuple[np.ndarray, np.ndarray]
    ) -> None:
        y, s = separable_scores
        curve = cost_curve(y, s, n_thresholds=50)
        best = min_cost_threshold(y, s, n_thresholds=50)
        assert best.expected_cost == pytest.approx(min(m.expected_cost for m in curve))
        assert isinstance(best, ThresholdMetrics)

    def test_expensive_false_positives_push_the_threshold_up(
        self, separable_scores: tuple[np.ndarray, np.ndarray]
    ) -> None:
        # The economic claim the whole threshold story rests on: as customer
        # friction gets costlier, the optimal system becomes more conservative
        # about alerting.
        #
        # Realistic amounts are essential here. Left at unit value, missing a
        # fraud would cost less than investigating one alert, so the optimum
        # would be "never alert" regardless of friction and the comparison
        # would be vacuous.
        y, s = separable_scores
        amounts = np.where(y == 1, 300.0, 50.0)
        cheap = min_cost_threshold(
            y,
            s,
            amounts=amounts,
            cost_model=CostModel(false_positive_friction_gbp=1.0),
            n_thresholds=100,
        )
        pricey = min_cost_threshold(
            y,
            s,
            amounts=amounts,
            cost_model=CostModel(false_positive_friction_gbp=500.0),
            n_thresholds=100,
        )
        assert pricey.threshold > cheap.threshold
        assert pricey.n_alerts < cheap.n_alerts


def _isotonic_like() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scores shaped like isotonic-calibrated output.

    A handful of distinct steps, 99.9% of the mass on the lowest two, and every
    usable operating point in a thin tail -- the shape of the real test fold,
    where 159,301 scores take only 60 distinct values.
    """
    steps = [  # (score, legitimate, fraud)
        (0.001, 89_990, 10),
        (0.003, 9_875, 5),
        (0.2, 32, 8),
        (0.5, 20, 20),
        (0.9, 2, 18),
        (1.0, 0, 20),
    ]
    scores = np.concatenate([np.full(legit + fraud, s) for s, legit, fraud in steps])
    truth = np.concatenate([np.r_[np.zeros(legit), np.ones(fraud)] for _, legit, fraud in steps])
    return truth.astype(int), scores, np.full(scores.size, 200.0)


class TestThresholdGrid:
    """Regression cover for a grid that once reported a threshold at 3.4x the true minimum.

    The grid used to be built from score quantiles. On a step function, almost
    every quantile lands on the same few low steps, so the grid jumped straight
    past the tail where the optimum lives.
    """

    def test_grid_is_every_distinct_score_when_there_are_few(self) -> None:
        _, scores, _ = _isotonic_like()
        np.testing.assert_array_equal(threshold_grid(scores), [0.001, 0.003, 0.2, 0.5, 0.9, 1.0])

    def test_finds_the_optimum_in_the_sparse_tail(self) -> None:
        y, s, amounts = _isotonic_like()
        best = min_cost_threshold(y, s, amounts=amounts)
        # By hand at t=0.2: 120 alerts x £4 + 54 false positives x £18
        # + 15 missed frauds x £200 = £4,452, the cheapest step.
        assert best.threshold == pytest.approx(0.2)
        assert best.expected_cost == pytest.approx(4_452.0)

    def test_selection_matches_brute_force_over_all_distinct_scores(self) -> None:
        y, s, amounts = _isotonic_like()
        brute = min(
            evaluate_at_threshold(y, s, float(t), amounts=amounts).expected_cost
            for t in np.unique(s)
        )
        assert min_cost_threshold(y, s, amounts=amounts).expected_cost == pytest.approx(brute)

    def test_fixture_reproduces_the_original_failure(self) -> None:
        # Guards the guard: the old 500-point quantile grid must miss the
        # optimum on this data, or the tests above would pass against it too.
        y, s, amounts = _isotonic_like()
        quantile_grid = np.unique(np.quantile(s, np.linspace(0, 1, 500)))
        quantile_best = min(
            evaluate_at_threshold(y, s, float(t), amounts=amounts).expected_cost
            for t in quantile_grid
        )
        assert 0.2 not in quantile_grid
        assert quantile_best > 2 * min_cost_threshold(y, s, amounts=amounts).expected_cost

    def test_thinned_grid_spans_the_full_range(self) -> None:
        # With many distinct scores the grid is thinned evenly across the sorted
        # distinct values, rather than packed into the dense low end.
        scores = np.random.default_rng(0).permutation(np.linspace(0.0, 1.0, 10_001))
        grid = threshold_grid(scores, max_points=50)
        assert grid.size <= 50
        assert grid[0] == 0.0
        assert grid[-1] == 1.0
        assert np.all(np.diff(grid) > 0)

    def test_non_finite_scores_are_ignored(self) -> None:
        grid = threshold_grid(np.array([0.3, np.nan, 0.1, np.inf, 0.3]))
        np.testing.assert_array_equal(grid, [0.1, 0.3])

    def test_no_finite_scores_yields_a_single_threshold(self) -> None:
        np.testing.assert_array_equal(threshold_grid(np.array([np.nan])), [0.0])


class TestAlertRate:
    def test_counts_scores_at_or_above_threshold(self) -> None:
        assert alert_rate([0.1, 0.5, 0.9, 0.95], 0.5) == pytest.approx(0.75)

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            alert_rate([], 0.5)
