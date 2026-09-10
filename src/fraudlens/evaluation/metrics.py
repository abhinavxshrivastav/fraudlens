"""Evaluation metrics for extremely imbalanced binary classification.

Accuracy is not reported anywhere in this project, deliberately. At a ~0.5% base
rate a model that predicts "never fraud" is 99.5% accurate and worthless. The
metrics here are the ones a fraud team actually operates on:

- **PR-AUC** -- threshold-free ranking quality on the minority class.
- **Recall @ fixed FPR** -- how much fraud is caught while holding false alarms
  to a level the business tolerates.
- **Precision@K** -- of the top K alerts an analyst can review in a day, how many
  are real.
- **Alert rate** -- the operational load the model imposes.
- **Expected cost** -- the only metric that answers "where should the threshold
  go?", because it prices missed fraud against customer friction.

ROC-AUC is reported for comparability with published work, but it is optimistic
under heavy imbalance: the true-negative pool is so large that the false-positive
rate barely moves, so an unusable model can still post a flattering ROC-AUC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve

from fraudlens.config import constants as C

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd

# PEP 695: lazily evaluated, so the forward references to TYPE_CHECKING-only
# imports resolve correctly without importing pandas at runtime.
type ArrayLike = np.ndarray | Sequence[float] | pd.Series


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CostModel:
    """Prices the two ways a fraud system can be wrong.

    A false negative costs the transaction amount: the issuer refunds the
    customer and absorbs the loss. A false positive costs analyst time *plus*
    the friction of wrongly declining a genuine customer -- call-centre handling
    and attrition risk. Every alert, right or wrong, costs investigation time.

    The absolute figures are order-of-magnitude estimates; what drives the
    chosen threshold is the *ratio* between them. ``docs/EVALUATION.md`` carries
    a sensitivity analysis.
    """

    investigation_gbp: float = C.INVESTIGATION_COST_GBP
    false_positive_friction_gbp: float = C.FALSE_POSITIVE_FRICTION_GBP
    false_negative_amount_multiplier: float = C.FALSE_NEGATIVE_AMOUNT_MULTIPLIER

    def total(
        self,
        *,
        n_alerts: int,
        n_false_positives: int,
        missed_fraud_value: float,
    ) -> float:
        """Total expected cost in GBP for one evaluation period."""
        return (
            n_alerts * self.investigation_gbp
            + n_false_positives * self.false_positive_friction_gbp
            + missed_fraud_value * self.false_negative_amount_multiplier
        )


# ---------------------------------------------------------------------------
# Point metrics
# ---------------------------------------------------------------------------


def _as_arrays(y_true: ArrayLike, y_score: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(y_true).astype(np.int8).ravel()
    score = np.asarray(y_score).astype(np.float64).ravel()
    if truth.shape != score.shape:
        msg = f"y_true and y_score have different lengths: {truth.shape} vs {score.shape}"
        raise ValueError(msg)
    if truth.size == 0:
        msg = "cannot evaluate on an empty array"
        raise ValueError(msg)
    if not np.isin(truth, (0, 1)).all():
        msg = "y_true must contain only 0 and 1"
        raise ValueError(msg)
    return truth, score


def pr_auc(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Area under the precision-recall curve.

    Uses ``average_precision_score``, which is the unbiased step-wise estimator.
    Computing ``auc(recall, precision)`` instead applies trapezoidal
    interpolation between operating points that are not linearly reachable, and
    is optimistically biased -- a subtle but real difference on imbalanced data.
    """
    truth, score = _as_arrays(y_true, y_score)
    return float(average_precision_score(truth, score))


def roc_auc(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Area under the ROC curve. Reported for comparability only; see module docstring."""
    truth, score = _as_arrays(y_true, y_score)
    return float(roc_auc_score(truth, score))


def brier_score(y_true: ArrayLike, y_prob: ArrayLike) -> float:
    """Mean squared error of predicted probabilities.

    Only meaningful for *calibrated* outputs. A raw gradient-boosting margin is
    not a probability, so this is computed after isotonic calibration.
    """
    truth, prob = _as_arrays(y_true, y_prob)
    return float(brier_score_loss(truth, prob))


def recall_at_fpr(y_true: ArrayLike, y_score: ArrayLike, target_fpr: float) -> tuple[float, float]:
    """Recall achievable while holding the false-positive rate at or below ``target_fpr``.

    Returns ``(recall, threshold)``. The threshold is chosen on the validation
    fold and then applied *unchanged* to test -- picking it on test would be
    another form of leakage.
    """
    if not 0.0 < target_fpr < 1.0:
        msg = f"target_fpr must be in (0, 1), got {target_fpr}"
        raise ValueError(msg)
    truth, score = _as_arrays(y_true, y_score)
    fpr, tpr, thresholds = roc_curve(truth, score)

    eligible = np.flatnonzero(fpr <= target_fpr)
    if eligible.size == 0:  # pragma: no cover - roc_curve always starts at fpr=0
        return 0.0, float("inf")
    best = eligible[-1]
    return float(tpr[best]), float(thresholds[best])


def precision_at_k(y_true: ArrayLike, y_score: ArrayLike, k: int) -> float:
    """Precision among the ``k`` highest-scoring transactions.

    Models the real constraint on a fraud team: an analyst can only work a fixed
    number of cases per shift, so what matters is the hit rate at the top of the
    ranked queue, not across the whole score range.
    """
    if k <= 0:
        msg = f"k must be positive, got {k}"
        raise ValueError(msg)
    truth, score = _as_arrays(y_true, y_score)
    k = min(k, truth.size)
    # argpartition is O(n) and sufficient -- we need the top-k set, not its order.
    top = np.argpartition(-score, k - 1)[:k]
    return float(truth[top].sum() / k)


def alert_rate(y_score: ArrayLike, threshold: float) -> float:
    """Fraction of transactions that would be flagged at ``threshold``."""
    score = np.asarray(y_score, dtype=np.float64).ravel()
    if score.size == 0:
        msg = "cannot compute alert rate on an empty array"
        raise ValueError(msg)
    return float((score >= threshold).mean())


# ---------------------------------------------------------------------------
# Threshold-dependent view
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ThresholdMetrics:
    """Confusion-matrix-derived metrics at one operating point."""

    threshold: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    alert_rate: float
    missed_fraud_value: float
    caught_fraud_value: float
    expected_cost: float

    @property
    def precision(self) -> float:
        alerts = self.true_positives + self.false_positives
        return self.true_positives / alerts if alerts else 0.0

    @property
    def recall(self) -> float:
        actual = self.true_positives + self.false_negatives
        return self.true_positives / actual if actual else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def false_positive_rate(self) -> float:
        negatives = self.false_positives + self.true_negatives
        return self.false_positives / negatives if negatives else 0.0

    @property
    def n_alerts(self) -> int:
        return self.true_positives + self.false_positives

    @property
    def value_detection_rate(self) -> float:
        """Share of fraud *value* prevented, as opposed to share of fraud count.

        Catching many small frauds while missing a few large ones can look good
        on recall and still lose money, so both are reported.
        """
        total = self.caught_fraud_value + self.missed_fraud_value
        return self.caught_fraud_value / total if total else 0.0


def evaluate_at_threshold(
    y_true: ArrayLike,
    y_score: ArrayLike,
    threshold: float,
    *,
    amounts: ArrayLike | None = None,
    cost_model: CostModel | None = None,
) -> ThresholdMetrics:
    """Compute the full confusion-matrix view at a single threshold.

    ``amounts`` supplies per-transaction values so missed fraud can be priced.
    When omitted, every transaction is treated as unit value, which keeps the
    cost figure interpretable as "fraud count" rather than currency.
    """
    truth, score = _as_arrays(y_true, y_score)
    costs = cost_model or CostModel()
    values = (
        np.ones_like(truth, dtype=np.float64)
        if amounts is None
        else np.asarray(amounts, dtype=np.float64).ravel()
    )
    if values.shape != truth.shape:
        msg = f"amounts has length {values.shape}, expected {truth.shape}"
        raise ValueError(msg)

    predicted = score >= threshold
    is_fraud = truth == 1

    tp = int(np.sum(predicted & is_fraud))
    fp = int(np.sum(predicted & ~is_fraud))
    tn = int(np.sum(~predicted & ~is_fraud))
    fn = int(np.sum(~predicted & is_fraud))

    missed_value = float(values[~predicted & is_fraud].sum())
    caught_value = float(values[predicted & is_fraud].sum())

    return ThresholdMetrics(
        threshold=float(threshold),
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        alert_rate=float(predicted.mean()),
        missed_fraud_value=missed_value,
        caught_fraud_value=caught_value,
        expected_cost=costs.total(
            n_alerts=tp + fp,
            n_false_positives=fp,
            missed_fraud_value=missed_value,
        ),
    )


def threshold_grid(scores: np.ndarray, max_points: int = 500) -> np.ndarray:
    """The set of thresholds worth evaluating.

    Built from the **distinct score values**, not from quantiles. Two thresholds
    falling between the same pair of adjacent scores classify every row
    identically, so the distinct values are the complete set of behaviourally
    different operating points.

    This matters more than it sounds. Isotonic calibration is a *step function*:
    on the test fold, 159,301 scores take only **60 distinct values**. A
    quantile grid over those is worse than useless, because 99.4% of the mass
    sits in a handful of steps near zero -- the grid jumps from 0.0028 straight
    to 1.0 and never visits the region where every usable operating point lives.
    An earlier version did exactly that and reported an "optimal" threshold
    costing 3.4x the true minimum.

    For an uncalibrated model with many distinct scores, the grid is thinned to
    an even spread **across the sorted distinct values**, which keeps coverage
    of the full range rather than concentrating on the dense low end.
    """
    unique = np.unique(scores[np.isfinite(scores)])
    if unique.size == 0:
        return np.array([0.0])
    if unique.size <= max_points:
        return unique
    indices = np.unique(np.linspace(0, unique.size - 1, max_points).astype(int))
    return np.asarray(unique[indices], dtype=np.float64)


def cost_curve(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    amounts: ArrayLike | None = None,
    cost_model: CostModel | None = None,
    n_thresholds: int = 500,
) -> list[ThresholdMetrics]:
    """Evaluate across every behaviourally distinct threshold.

    See :func:`threshold_grid` for why the grid is built from distinct score
    values rather than quantiles.
    """
    truth, score = _as_arrays(y_true, y_score)
    grid = threshold_grid(score, max_points=n_thresholds)
    return [
        evaluate_at_threshold(truth, score, float(t), amounts=amounts, cost_model=cost_model)
        for t in grid
    ]


def min_cost_threshold(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    amounts: ArrayLike | None = None,
    cost_model: CostModel | None = None,
    n_thresholds: int = 500,
) -> ThresholdMetrics:
    """Return the operating point that minimises expected cost.

    This is how the recommended production threshold is chosen: not by
    maximising F1 or picking 0.5, but by pricing the trade-off. Selected on the
    validation fold and then frozen.
    """
    curve = cost_curve(
        y_true,
        y_score,
        amounts=amounts,
        cost_model=cost_model,
        n_thresholds=n_thresholds,
    )
    return min(curve, key=lambda m: m.expected_cost)
