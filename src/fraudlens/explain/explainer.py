"""Per-decision explanations via SHAP, feeding the reason-code layer.

Why TreeExplainer specifically
------------------------------
``shap.TreeExplainer`` computes exact Shapley values for tree ensembles in
polynomial time, rather than the sampling approximation the model-agnostic
explainers use. That matters twice over: explanations are deterministic, so the
same transaction always produces the same reasons (a hard requirement if a
decision has to be defended later), and it is fast enough to run inline.

Where explanations are computed
-------------------------------
On the **alert path only**. Explaining a transaction that was approved costs
latency for output nobody reads, and the approve path is ~99.5% of traffic.
:meth:`ExplainerService.explain_if_alerting` encodes that policy.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from fraudlens.explain.reason_codes import ReasonCodeInstance, derive_reason_codes

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Explanation:
    """SHAP output for one decision, plus the analyst-facing rendering."""

    contributions: dict[str, float]
    base_value: float
    reason_codes: tuple[ReasonCodeInstance, ...]

    @property
    def top_drivers(self) -> list[tuple[str, float]]:
        """Features that pushed the score toward fraud, strongest first."""
        positive = [(k, v) for k, v in self.contributions.items() if v > 0]
        return sorted(positive, key=lambda kv: kv[1], reverse=True)

    @property
    def top_mitigators(self) -> list[tuple[str, float]]:
        """Features that pushed the score away from fraud."""
        negative = [(k, v) for k, v in self.contributions.items() if v < 0]
        return sorted(negative, key=lambda kv: kv[1])

    def waterfall_data(self, limit: int = 10) -> list[dict[str, float | str]]:
        """Ordered contributions for the console's waterfall chart."""
        ranked = sorted(self.contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
        return [{"feature": name, "contribution": value} for name, value in ranked[:limit]]

    @property
    def texts(self) -> tuple[str, ...]:
        return tuple(code.text for code in self.reason_codes)


class ExplainerService:
    """Wraps a SHAP TreeExplainer and the reason-code mapping."""

    def __init__(self, model: Any, feature_names: Sequence[str], *, max_codes: int = 4) -> None:
        self._feature_names = list(feature_names)
        self._max_codes = max_codes
        self._explainer: Any | None = None
        self._model = model
        self._build(model)

    def _build(self, model: Any) -> None:
        try:
            import shap
        except ImportError:  # pragma: no cover - optional at runtime
            logger.warning("shap is not installed; explanations will be rule-based only")
            return

        booster = getattr(model, "_booster", model)
        try:
            self._explainer = shap.TreeExplainer(booster)
        except Exception:
            logger.warning(
                "Could not build a TreeExplainer for %s; falling back to rule-based reason codes",
                type(booster).__name__,
                exc_info=True,
            )
            self._explainer = None

    @property
    def available(self) -> bool:
        return self._explainer is not None

    def explain(self, features: Mapping[str, float]) -> Explanation:
        """Explain one decision.

        Degrades gracefully: if SHAP is unavailable, reason codes are still
        derived from feature materiality alone. An unexplained alert is far
        worse than an approximately explained one.
        """
        contributions = self._contributions(features)
        codes = derive_reason_codes(
            features,
            contributions or None,
            max_codes=self._max_codes,
        )
        return Explanation(
            contributions=contributions,
            base_value=self._base_value(),
            reason_codes=codes,
        )

    def explain_if_alerting(
        self, features: Mapping[str, float], *, is_alert: bool
    ) -> Explanation | None:
        """Explain only when the decision creates an alert.

        The approve path is the overwhelming majority of traffic and nobody
        reads its explanations; skipping it is what keeps p99 inside budget.
        """
        return self.explain(features) if is_alert else None

    def _contributions(self, features: Mapping[str, float]) -> dict[str, float]:
        if self._explainer is None:
            return {}
        frame = pd.DataFrame([{name: features.get(name, 0.0) for name in self._feature_names}])
        try:
            raw = _shap_values(self._explainer, frame)
        except Exception:
            logger.warning("SHAP evaluation failed for one transaction", exc_info=True)
            return {}
        values = _positive_class_row(raw)
        if values is None or len(values) != len(self._feature_names):
            return {}
        return {name: float(v) for name, v in zip(self._feature_names, values, strict=True)}

    def _base_value(self) -> float:
        if self._explainer is None:
            return 0.0
        expected = getattr(self._explainer, "expected_value", 0.0)
        if isinstance(expected, (list, tuple, np.ndarray)):
            array = np.asarray(expected).ravel()
            return float(array[-1]) if array.size else 0.0
        return float(expected)

    def explain_batch(self, frame: pd.DataFrame) -> np.ndarray:
        """SHAP values for a batch, used for global plots in the model card."""
        if self._explainer is None:
            return np.zeros((len(frame), len(self._feature_names)))
        raw = _shap_values(self._explainer, frame[self._feature_names])
        values = _positive_class_matrix(raw)
        return values if values is not None else np.zeros((len(frame), len(self._feature_names)))


def _shap_values(explainer: Any, frame: pd.DataFrame) -> Any:
    """Call SHAP, suppressing its binary-classifier output-shape warning.

    SHAP warns on every call that LightGBM binary output is now a list of
    ndarrays. :func:`_positive_class_matrix` handles that shape (and the two
    others SHAP has used) explicitly, so the warning carries no information --
    and emitted once per scored transaction it would drown the logs.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*output has changed to a list of ndarray.*",
            category=UserWarning,
        )
        return explainer.shap_values(frame, check_additivity=False)


def _positive_class_row(raw: Any) -> np.ndarray | None:
    """Extract the positive-class contributions for a single row.

    SHAP has returned several shapes across versions for binary classifiers:
    a list of two arrays (one per class), a 2-D ``(n, features)`` array, or a
    3-D ``(n, features, classes)`` array. Handling all three keeps the code
    working across upgrades instead of failing obscurely.
    """
    matrix = _positive_class_matrix(raw)
    if matrix is None or matrix.shape[0] == 0:
        return None
    return np.asarray(matrix[0]).ravel()


def _positive_class_matrix(raw: Any) -> np.ndarray | None:
    if isinstance(raw, list):
        return np.asarray(raw[-1])
    array = np.asarray(raw)
    if array.ndim == 3:  # (rows, features, classes)
        return array[:, :, -1]
    if array.ndim == 2:
        return array
    return None
