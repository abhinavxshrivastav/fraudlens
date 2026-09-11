"""Prometheus instrumentation.

What is measured, and why these specifically
--------------------------------------------
Four things a fraud platform must be able to answer at 3am:

``fraudlens_scoring_latency_seconds``
    Is the authorisation path still inside its budget? A histogram rather than a
    gauge, because the p99 is what matters and a mean hides it.
``fraudlens_decisions_total{band}``
    What is the system actually doing? A sudden collapse in ``block`` volume is
    an outage signal even when the process is healthy.
``fraudlens_alert_rate``
    The operational load on the fraud team. This is the number that turns into
    an angry phone call when it doubles.
``fraudlens_drift_psi{feature}``
    Has the world moved away from what the model was trained on?

Buckets for the latency histogram are chosen around the 50 ms budget so the
alerting threshold sits on a bucket boundary; default Prometheus buckets are far
too coarse at the millisecond scale this service operates at.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from fraudlens.config import constants as C

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)

#: Latency buckets in seconds, dense around the p99 budget.
LATENCY_BUCKETS = (
    0.001,
    0.002,
    0.005,
    0.010,
    0.015,
    0.020,
    0.030,
    0.050,  # the budget
    0.075,
    0.100,
    0.250,
    0.500,
    1.000,
)

SCORE_BUCKETS = (0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0)


class _NullMetric:
    """No-op stand-in used when prometheus_client is not installed.

    Instrumentation must never be the reason scoring fails, so every metric call
    degrades to nothing rather than raising.
    """

    def labels(self, *_: Any, **__: Any) -> _NullMetric:
        return self

    def inc(self, *_: Any, **__: Any) -> None: ...

    def observe(self, *_: Any, **__: Any) -> None: ...

    def set(self, *_: Any, **__: Any) -> None: ...


class Metrics:
    """Registry wrapper. Safe to construct even without prometheus_client.

    Metric attributes are typed ``Any`` because each slot holds either a real
    prometheus_client collector or a :class:`_NullMetric` standing in for it.
    Both expose the same call surface; the alternative would be a Protocol whose
    only purpose is to satisfy the type checker.
    """

    scoring_latency: Any
    decisions: Any
    alerts: Any
    rules_fired: Any
    score_distribution: Any
    alert_rate: Any
    drift_psi: Any
    stream_published: Any
    stream_dropped: Any
    model_info: Any

    def __init__(self, namespace: str = "fraudlens") -> None:
        self.enabled = False
        self.registry: Any = None
        try:
            from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
        except ImportError:  # pragma: no cover - optional dependency
            logger.warning("prometheus_client not installed; metrics disabled")
            self.scoring_latency = _NullMetric()
            self.decisions = _NullMetric()
            self.alerts = _NullMetric()
            self.rules_fired = _NullMetric()
            self.score_distribution = _NullMetric()
            self.alert_rate = _NullMetric()
            self.drift_psi = _NullMetric()
            self.stream_published = _NullMetric()
            self.stream_dropped = _NullMetric()
            self.model_info = _NullMetric()
            return

        self.registry = CollectorRegistry()
        self.enabled = True

        self.scoring_latency = Histogram(
            f"{namespace}_scoring_latency_seconds",
            "End-to-end latency of a scoring decision",
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.decisions = Counter(
            f"{namespace}_decisions_total",
            "Decisions by risk band",
            ["band"],
            registry=self.registry,
        )
        self.alerts = Counter(
            f"{namespace}_alerts_total",
            "Decisions that created an analyst alert",
            registry=self.registry,
        )
        self.rules_fired = Counter(
            f"{namespace}_rules_fired_total",
            "Rule firings by rule id",
            ["rule_id", "action"],
            registry=self.registry,
        )
        self.score_distribution = Histogram(
            f"{namespace}_score",
            "Distribution of calibrated fraud probabilities",
            buckets=SCORE_BUCKETS,
            registry=self.registry,
        )
        self.alert_rate = Gauge(
            f"{namespace}_alert_rate",
            "Rolling fraction of transactions that created an alert",
            registry=self.registry,
        )
        self.drift_psi = Gauge(
            f"{namespace}_drift_psi",
            "Population Stability Index per feature against the training reference",
            ["feature"],
            registry=self.registry,
        )
        self.stream_published = Counter(
            f"{namespace}_stream_published_total",
            "Messages published to the stream bus",
            ["topic"],
            registry=self.registry,
        )
        self.stream_dropped = Counter(
            f"{namespace}_stream_dropped_total",
            "Messages dropped due to slow subscribers",
            ["topic"],
            registry=self.registry,
        )
        self.model_info = Gauge(
            f"{namespace}_model_info",
            "Loaded model metadata (value is always 1; the labels carry the data)",
            ["version", "model_type", "calibration"],
            registry=self.registry,
        )

    # -- recording helpers -------------------------------------------------

    def record_decision(
        self,
        *,
        band: str,
        probability: float,
        latency_ms: float,
        creates_alert: bool,
        rules: Iterable[tuple[str, str]] = (),
    ) -> None:
        """Record one decision across every relevant metric."""
        self.scoring_latency.observe(latency_ms / 1000.0)
        self.decisions.labels(band=band).inc()
        self.score_distribution.observe(max(0.0, min(1.0, probability)))
        if creates_alert:
            self.alerts.inc()
        for rule_id, action in rules:
            self.rules_fired.labels(rule_id=rule_id, action=action).inc()

    def record_drift(self, feature: str, psi: float) -> None:
        self.drift_psi.labels(feature=feature).set(psi)

    def set_model_info(self, version: str, model_type: str, calibration: str) -> None:
        self.model_info.labels(version=version, model_type=model_type, calibration=calibration).set(
            1
        )

    def render(self) -> str:
        """Prometheus text exposition."""
        if not self.enabled:
            return "# prometheus_client is not installed\n"
        from prometheus_client import generate_latest

        return generate_latest(self.registry).decode("utf-8")

    @property
    def content_type(self) -> str:
        if not self.enabled:
            return "text/plain"
        from prometheus_client import CONTENT_TYPE_LATEST

        return str(CONTENT_TYPE_LATEST)


#: Latency budget, re-exported so dashboards and alert rules import one constant.
LATENCY_BUDGET_SECONDS = C.LATENCY_BUDGET_P99_MS / 1000.0
