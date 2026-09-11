"""Distribution drift monitoring.

Why this is detection capability, not reporting
-----------------------------------------------
Fraud migrates toward weak spots. If a feature's distribution shifts and nobody
notices, the model degrades quietly and that segment becomes the preferred attack
route. Drift monitoring is therefore part of the detection system rather than a
dashboard nicety -- it is how the platform notices that it has stopped working.

Population Stability Index
--------------------------
PSI compares a reference distribution (the training fold) against a live window:

    PSI = sum over bins of (actual% - expected%) * ln(actual% / expected%)

It is the standard measure in credit and fraud modelling, and the action bands
below are the conventional ones used across the industry:

===========  ==========================================
PSI          Interpretation
===========  ==========================================
< 0.10       Stable; no action
0.10 - 0.25  Moderate shift; monitor
> 0.25       Significant shift; investigate and consider retraining
===========  ==========================================

Bin edges come from the *reference* distribution's quantiles and are then frozen.
Re-binning on the live data would compare each window against itself and PSI
would be near zero no matter how far the data had moved.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np

from fraudlens.config import constants as C

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

logger = logging.getLogger(__name__)

#: Smoothing floor. A bin that is empty in one distribution but populated in the
#: other would otherwise send PSI to infinity.
_EPSILON = 1e-6

DEFAULT_BINS = 10

#: Cyclical features excluded from drift monitoring.
#:
#: ``hour`` and ``day_of_week`` are periodic. Any observation window shorter than
#: a full cycle covers only part of the period, so PSI against a reference
#: spanning every hour and weekday is enormous even when nothing has changed --
#: measured at 12.05 for ``day_of_week`` over a window covering two days.
#:
#: That is a false alarm, and false alarms are expensive: they train the team to
#: ignore the drift signal, which is worse than not having one. Monitoring these
#: meaningfully requires comparing like-for-like periods (this Tuesday 14:00
#: against previous Tuesdays at 14:00), which is a different mechanism from PSI
#: against a pooled reference. Until that exists, they are excluded rather than
#: reported wrongly.
#:
#: The binary features derived from them are excluded for the same reason.
CYCLICAL_FEATURES: frozenset[str] = frozenset({"hour", "day_of_week", "is_night", "is_weekend"})


class DriftStatus(StrEnum):
    STABLE = "stable"
    MODERATE = "moderate"
    SIGNIFICANT = "significant"
    INSUFFICIENT_DATA = "insufficient_data"

    @classmethod
    def from_psi(cls, psi: float) -> DriftStatus:
        if psi < C.PSI_STABLE:
            return cls.STABLE
        if psi < C.PSI_INVESTIGATE:
            return cls.MODERATE
        return cls.SIGNIFICANT


@dataclass(frozen=True, slots=True)
class FeatureDrift:
    """Drift measurement for one feature."""

    feature: str
    psi: float
    status: DriftStatus
    reference_mean: float
    live_mean: float
    n_live: int

    @property
    def mean_shift(self) -> float:
        """Relative change in mean, for a human-readable summary."""
        if abs(self.reference_mean) < _EPSILON:
            return 0.0
        return (self.live_mean - self.reference_mean) / abs(self.reference_mean)


def population_stability_index(
    reference: Sequence[float] | np.ndarray,
    live: Sequence[float] | np.ndarray,
    *,
    bins: int = DEFAULT_BINS,
    edges: np.ndarray | None = None,
) -> float:
    """PSI between a reference and a live sample.

    ``edges`` may be supplied to reuse frozen bin boundaries across windows,
    which is what :class:`DriftMonitor` does.
    """
    ref = np.asarray(reference, dtype=np.float64).ravel()
    obs = np.asarray(live, dtype=np.float64).ravel()
    ref = ref[np.isfinite(ref)]
    obs = obs[np.isfinite(obs)]

    if ref.size == 0 or obs.size == 0:
        return 0.0

    if edges is None:
        edges = quantile_edges(ref, bins)

    ref_pct = _histogram_share(ref, edges)
    obs_pct = _histogram_share(obs, edges)

    # Smooth zero bins in both distributions so the log stays finite.
    ref_pct = np.clip(ref_pct, _EPSILON, None)
    obs_pct = np.clip(obs_pct, _EPSILON, None)

    return float(np.sum((obs_pct - ref_pct) * np.log(obs_pct / ref_pct)))


def quantile_edges(reference: np.ndarray, bins: int = DEFAULT_BINS) -> np.ndarray:
    """Bin edges at reference quantiles, deduplicated.

    Quantile bins rather than uniform ones, because most behavioural features are
    heavily skewed -- uniform bins would put almost every observation in the first
    bucket and PSI would be blind to movement within it.
    """
    ref = np.asarray(reference, dtype=np.float64).ravel()
    ref = ref[np.isfinite(ref)]
    if ref.size == 0:
        return np.array([-np.inf, np.inf])
    edges = np.unique(np.quantile(ref, np.linspace(0.0, 1.0, bins + 1)))
    if edges.size < 2:
        # A constant feature: one bin covering everything.
        return np.array([edges[0] - 1.0, edges[0] + 1.0])
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def _histogram_share(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(values, bins=edges)
    total = counts.sum()
    share = counts / total if total else counts.astype(np.float64)
    return np.asarray(share, dtype=np.float64)


def ks_statistic(
    reference: Sequence[float] | np.ndarray, live: Sequence[float] | np.ndarray
) -> float:
    """Two-sample Kolmogorov-Smirnov statistic.

    Complements PSI: KS is sensitive to a shift anywhere in the distribution,
    while PSI weights by bin mass. Used mainly on the *score* distribution, where
    a shift is the earliest sign that something upstream has changed.
    """
    ref = np.sort(np.asarray(reference, dtype=np.float64).ravel())
    obs = np.sort(np.asarray(live, dtype=np.float64).ravel())
    ref = ref[np.isfinite(ref)]
    obs = obs[np.isfinite(obs)]
    if ref.size == 0 or obs.size == 0:
        return 0.0

    grid = np.union1d(ref, obs)
    cdf_ref = np.searchsorted(ref, grid, side="right") / ref.size
    cdf_obs = np.searchsorted(obs, grid, side="right") / obs.size
    return float(np.max(np.abs(cdf_ref - cdf_obs)))


@dataclass
class DriftMonitor:
    """Tracks feature and score drift against a frozen reference.

    Live values accumulate in bounded ring buffers, so memory is constant however
    long the service runs.
    """

    reference: Mapping[str, np.ndarray]
    window: int = 5_000
    min_samples: int = 200
    bins: int = DEFAULT_BINS
    _edges: dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    _live: dict[str, deque[float]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        # Freeze bin edges at construction. Re-deriving them from live data would
        # compare each window against itself and hide all drift.
        for name, values in self.reference.items():
            array = np.asarray(values, dtype=np.float64).ravel()
            self._edges[name] = quantile_edges(array, self.bins)
            self._live[name] = deque(maxlen=self.window)

    @classmethod
    def from_frame(cls, frame: object, features: Sequence[str], **kwargs: object) -> DriftMonitor:
        """Build a monitor from a reference dataframe."""
        reference = {
            name: np.asarray(frame[name], dtype=np.float64)  # type: ignore[index]
            for name in features
            if name in frame.columns  # type: ignore[attr-defined]
        }
        return cls(reference=reference, **kwargs)  # type: ignore[arg-type]

    def observe(self, features: Mapping[str, float]) -> None:
        """Record one live feature vector."""
        for name, buffer in self._live.items():
            value = features.get(name)
            if value is not None and np.isfinite(value):
                buffer.append(float(value))

    def measure(self, feature: str) -> FeatureDrift:
        """PSI for one feature against its reference."""
        live = np.asarray(self._live.get(feature, ()), dtype=np.float64)
        ref = np.asarray(self.reference.get(feature, ()), dtype=np.float64)

        if live.size < self.min_samples:
            return FeatureDrift(
                feature=feature,
                psi=0.0,
                status=DriftStatus.INSUFFICIENT_DATA,
                reference_mean=float(ref.mean()) if ref.size else 0.0,
                live_mean=float(live.mean()) if live.size else 0.0,
                n_live=int(live.size),
            )

        psi = population_stability_index(ref, live, edges=self._edges.get(feature))
        return FeatureDrift(
            feature=feature,
            psi=psi,
            status=DriftStatus.from_psi(psi),
            reference_mean=float(ref.mean()) if ref.size else 0.0,
            live_mean=float(live.mean()),
            n_live=int(live.size),
        )

    def measure_all(self) -> list[FeatureDrift]:
        """Every tracked feature, worst drift first."""
        results = [self.measure(name) for name in self.reference]
        return sorted(results, key=lambda d: d.psi, reverse=True)

    def drifting(self) -> list[FeatureDrift]:
        """Only features at or beyond the 'monitor' band."""
        return [
            d
            for d in self.measure_all()
            if d.status in {DriftStatus.MODERATE, DriftStatus.SIGNIFICANT}
        ]

    @property
    def observations(self) -> int:
        return max((len(buffer) for buffer in self._live.values()), default=0)

    def summary(self) -> dict[str, object]:
        results = self.measure_all()
        worst = results[0] if results else None
        return {
            "observations": self.observations,
            "features_tracked": len(self.reference),
            "features_drifting": len(self.drifting()),
            "worst_feature": worst.feature if worst else None,
            "worst_psi": round(worst.psi, 4) if worst else 0.0,
            "status": str(worst.status) if worst else str(DriftStatus.INSUFFICIENT_DATA),
        }
