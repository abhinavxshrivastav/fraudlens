"""Threshold simulation over the held-out test fold.

What this is for
----------------
Every metric in the evaluation harness describes a trade-off without resolving
it. Moving the decision threshold trades recall against precision against
analyst workload against money, and no table conveys that as well as being able
to drag it and watch the numbers move.

This backs the console's threshold simulator: the page a reviewer is most likely
to understand the whole project from.

How it stays honest
-------------------
Scores are computed **once** at startup from the cached test-fold features, then
held in memory. Simulation is then pure arithmetic over three numpy arrays, so
the endpoint answers in well under a millisecond and the slider feels live.

Two things this is not:

- It is **not** threshold selection. The shipped operating point was chosen on
  the *validation* fold during training and is frozen in the model artefact.
  This visualises what other choices would have cost on test; picking a
  threshold by dragging until test looks good is exactly the leakage the whole
  project is built to avoid, and the UI says so.
- It is **not** live traffic. It is the held-out fold, scored offline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from fraudlens.config import constants as C
from fraudlens.evaluation.metrics import CostModel, evaluate_at_threshold, threshold_grid

if TYPE_CHECKING:
    from fraudlens.config import Settings
    from fraudlens.models.model import FraudModel

logger = logging.getLogger(__name__)

#: Points on the precomputed curve. Enough for a smooth line, few enough to send
#: as JSON without thinking about it.
CURVE_POINTS = 120

#: Test fold spans roughly six months; used to express alert volume per day,
#: which is the unit an analyst actually feels.
TEST_PERIOD_DAYS = 176


@dataclass(frozen=True, slots=True)
class SimulationResult:
    threshold: float
    precision: float
    recall: float
    alert_rate: float
    false_positive_rate: float
    true_positives: int
    false_positives: int
    false_negatives: int
    alerts_per_day: float
    expected_cost: float
    value_detection_rate: float


class ThresholdSimulator:
    """Holds test-fold scores and answers threshold questions against them."""

    def __init__(
        self,
        y_true: np.ndarray,
        scores: np.ndarray,
        amounts: np.ndarray,
        *,
        period_days: int = TEST_PERIOD_DAYS,
    ) -> None:
        if not (len(y_true) == len(scores) == len(amounts)):
            msg = "labels, scores and amounts must be the same length"
            raise ValueError(msg)
        self.y_true = np.asarray(y_true, dtype=np.int8)
        self.scores = np.asarray(scores, dtype=np.float64)
        self.amounts = np.asarray(amounts, dtype=np.float64)
        self.period_days = max(1, period_days)

    @property
    def size(self) -> int:
        return int(self.y_true.size)

    @property
    def fraud_count(self) -> int:
        return int(self.y_true.sum())

    def evaluate(
        self,
        threshold: float,
        *,
        investigation_cost: float = C.INVESTIGATION_COST_GBP,
        friction_cost: float = C.FALSE_POSITIVE_FRICTION_GBP,
    ) -> SimulationResult:
        """Metrics at one threshold, with a caller-supplied cost model.

        Costs are parameters rather than constants because the *ratio* is the
        interesting input: letting someone raise the friction cost and watch the
        optimal threshold climb is the clearest demonstration that the operating
        point is an economic choice, not a statistical one.
        """
        costs = CostModel(
            investigation_gbp=investigation_cost,
            false_positive_friction_gbp=friction_cost,
        )
        point = evaluate_at_threshold(
            self.y_true, self.scores, threshold, amounts=self.amounts, cost_model=costs
        )
        return SimulationResult(
            threshold=float(threshold),
            precision=point.precision,
            recall=point.recall,
            alert_rate=point.alert_rate,
            false_positive_rate=point.false_positive_rate,
            true_positives=point.true_positives,
            false_positives=point.false_positives,
            false_negatives=point.false_negatives,
            alerts_per_day=point.n_alerts / self.period_days,
            expected_cost=point.expected_cost,
            value_detection_rate=point.value_detection_rate,
        )

    def curve(
        self,
        *,
        investigation_cost: float = C.INVESTIGATION_COST_GBP,
        friction_cost: float = C.FALSE_POSITIVE_FRICTION_GBP,
        points: int = CURVE_POINTS,
    ) -> list[SimulationResult]:
        """Evaluate across a grid of thresholds.

        The grid comes from :func:`~fraudlens.evaluation.metrics.threshold_grid`,
        which uses the distinct score values. Isotonic calibration is a step
        function, so those distinct values are the complete set of meaningful
        operating points -- and a quantile grid over them misses the useful
        region entirely.
        """
        grid = threshold_grid(self.scores, max_points=points)
        return [
            self.evaluate(
                float(threshold),
                investigation_cost=investigation_cost,
                friction_cost=friction_cost,
            )
            for threshold in grid
        ]

    def optimal(
        self,
        *,
        investigation_cost: float = C.INVESTIGATION_COST_GBP,
        friction_cost: float = C.FALSE_POSITIVE_FRICTION_GBP,
    ) -> SimulationResult:
        """The cost-minimising point on the curve under the given costs."""
        return min(
            self.curve(investigation_cost=investigation_cost, friction_cost=friction_cost),
            key=lambda r: r.expected_cost,
        )

    def score_histogram(self, bins: int = 40) -> list[dict[str, float]]:
        """Score distribution split by true class, for the simulator plot."""
        edges = np.linspace(0.0, 1.0, bins + 1)
        legit, _ = np.histogram(self.scores[self.y_true == 0], bins=edges)
        fraud, _ = np.histogram(self.scores[self.y_true == 1], bins=edges)
        return [
            {
                "score": float((edges[i] + edges[i + 1]) / 2),
                "legitimate": int(legit[i]),
                "fraud": int(fraud[i]),
            }
            for i in range(bins)
        ]


def build_simulator(model: FraudModel | None, settings: Settings) -> ThresholdSimulator | None:
    """Score the cached test fold once, at startup.

    Returns ``None`` when the cached features are absent — the simulator is a
    demonstration surface, and the service must start without it.
    """
    if model is None:
        return None

    import pandas as pd

    # Prefer the committed bundle. It holds exactly the three columns this class
    # reads -- label, score, amount -- already scored, so a deployed instance
    # needs neither the 16 MB feature file nor the cost of scoring 159k rows at
    # every boot. The full features are used only when developing locally.
    bundle = settings.artifact_dir / "demo" / "test_scores.parquet"
    features_path = settings.processed_dir / "features_full" / "features_test.parquet"

    try:
        if bundle.exists():
            frame = pd.read_parquet(bundle)
            scores = frame["score"].to_numpy()
            logger.info("Threshold simulator using the committed bundle at %s", bundle)
        elif features_path.exists():
            frame = pd.read_parquet(features_path)
            scores = model.predict_proba_batch(frame[list(model.feature_names)])
            logger.info("Threshold simulator scoring cached features at %s", features_path)
        else:
            logger.info(
                "No scored test set at %s and no cached features; simulator disabled",
                bundle,
            )
            return None

        simulator = ThresholdSimulator(
            y_true=frame[C.TARGET_COL].to_numpy(),
            scores=scores,
            amounts=frame[C.AMOUNT_COL].to_numpy(),
        )
        logger.info(
            "Threshold simulator ready over %d test rows (%d fraud)",
            simulator.size,
            simulator.fraud_count,
        )
        return simulator
    except Exception:
        logger.warning("Could not build the threshold simulator", exc_info=True)
        return None
