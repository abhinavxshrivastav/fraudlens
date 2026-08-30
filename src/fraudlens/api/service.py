"""Runtime service: the object graph the API endpoints delegate to.

Holds the model, rule engine, feature store, explainer and decision engine, and
owns the scoring sequence. Keeping this separate from the FastAPI layer means
the whole scoring path is testable without an HTTP client, and the endpoints
stay thin.

The scoring sequence
--------------------
1. Read the card's state from the feature store.
2. Compute features from state + this transaction.
3. Run rules, model and policy to a decision.
4. Explain, but only if the decision creates an alert.
5. **Then** fold the transaction into the store.

Step 5 comes last for the same reason it does in training: state must reflect
only transactions strictly before the one being scored. Doing it earlier would
let a transaction contribute to its own velocity features -- the online form of
the leakage the offline tests attack.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fraudlens.api.schemas import (
    ContributionResponse,
    ReasonCodeResponse,
    RuleHitResponse,
    ScoreResponse,
    StatsResponse,
    TransactionRequest,
)
from fraudlens.config import Settings, get_settings
from fraudlens.explain.explainer import ExplainerService
from fraudlens.features.pipeline import FeaturePipeline, TransactionEvent
from fraudlens.features.store import FeatureStore, build_feature_store
from fraudlens.models.model import FraudModel
from fraudlens.rules.engine import RuleEngine
from fraudlens.scoring.decision import Decision, DecisionEngine, PolicyThresholds, RiskBand

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RULES_PATH = ROOT / "config" / "rules.yaml"


class ModelNotLoadedError(RuntimeError):
    """Raised when scoring is attempted before a model is available."""


@dataclass(slots=True)
class ServiceStats:
    """Lightweight in-process counters for the monitor page.

    Latency is kept as a bounded ring buffer rather than a full history: p99 over
    the last few thousand requests is the operationally useful number, and it
    costs constant memory.
    """

    started_at: float = field(default_factory=time.monotonic)
    scored_total: int = 0
    alerts_total: int = 0
    blocked_total: int = 0
    band_counts: dict[str, int] = field(default_factory=dict)
    latencies_ms: deque[float] = field(default_factory=lambda: deque(maxlen=5_000))

    def record(self, decision: Decision) -> None:
        self.scored_total += 1
        band = str(decision.band)
        self.band_counts[band] = self.band_counts.get(band, 0) + 1
        if decision.creates_alert:
            self.alerts_total += 1
        if decision.band is RiskBand.BLOCK:
            self.blocked_total += 1
        self.latencies_ms.append(decision.latency_ms)

    def snapshot(self) -> StatsResponse:
        latencies = sorted(self.latencies_ms)
        mean = sum(latencies) / len(latencies) if latencies else 0.0
        p99 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.99))] if latencies else 0.0
        return StatsResponse(
            scored_total=self.scored_total,
            alerts_total=self.alerts_total,
            alert_rate=(self.alerts_total / self.scored_total) if self.scored_total else 0.0,
            blocked_total=self.blocked_total,
            mean_latency_ms=mean,
            p99_latency_ms=p99,
            band_counts=dict(self.band_counts),
            uptime_seconds=time.monotonic() - self.started_at,
        )


class FraudLensService:
    """The scoring runtime."""

    def __init__(
        self,
        model: FraudModel | None,
        rule_engine: RuleEngine,
        feature_store: FeatureStore,
        *,
        thresholds: PolicyThresholds | None = None,
        explainer: ExplainerService | None = None,
    ) -> None:
        self.model = model
        self.rule_engine = rule_engine
        self.feature_store = feature_store
        self.pipeline = FeaturePipeline()
        self.stats = ServiceStats()
        self.thresholds = thresholds or PolicyThresholds()
        self.explainer = explainer
        self.decision_engine = (
            DecisionEngine(model=model, rule_engine=rule_engine, thresholds=self.thresholds)
            if model is not None
            else None
        )
        #: Recent decisions that created an alert, newest last.
        self.recent_alerts: deque[tuple[Decision, TransactionRequest]] = deque(maxlen=500)
        self.dispositions: dict[str, str] = {}

    # -- construction ------------------------------------------------------

    @classmethod
    def load(cls, settings: Settings | None = None) -> FraudLensService:
        """Build the service from configuration, tolerating a missing model.

        A missing model artefact yields a *degraded* service that still serves
        health, docs and the rule catalogue, rather than a process that refuses
        to start. Failing to boot because a file is absent turns a recoverable
        problem into an outage.
        """
        settings = settings or get_settings()
        rule_engine = _load_rules()
        store = build_feature_store(str(settings.profile), settings.redis_url)

        model: FraudModel | None = None
        thresholds = PolicyThresholds()
        explainer: ExplainerService | None = None

        model_dir = settings.model_dir / "champion"
        if (model_dir / "model.joblib").exists():
            try:
                model = FraudModel.load(model_dir)
                thresholds = _thresholds_from(model, thresholds)
                explainer = ExplainerService(model, model.feature_names)
                logger.info(
                    "Loaded model %s (calibrated=%s, explainer=%s)",
                    model.version,
                    model.is_calibrated,
                    explainer.available,
                )
            except Exception:
                logger.exception("Failed to load model from %s; starting degraded", model_dir)
        else:
            logger.warning(
                "No model at %s. Run scripts/train.py. Serving in degraded mode.", model_dir
            )

        return cls(model, rule_engine, store, thresholds=thresholds, explainer=explainer)

    # -- scoring -----------------------------------------------------------

    def score(self, request: TransactionRequest) -> ScoreResponse:
        """Score one transaction end to end."""
        if self.decision_engine is None:
            msg = "no model is loaded; run scripts/train.py and restart"
            raise ModelNotLoadedError(msg)

        event = _to_event(request)
        state = self.feature_store.get(event.card)
        state.prune(event.timestamp)

        features = self.pipeline.compute(event, state)

        decision = self.decision_engine.decide(event, features)

        explanation = None
        if self.explainer is not None and decision.creates_alert:
            explanation = self.explainer.explain_if_alerting(
                features, is_alert=decision.creates_alert
            )

        # State advances only after the decision is made. See the module docstring.
        self.feature_store.update(event.card, event)

        self.stats.record(decision)
        if decision.creates_alert:
            self.recent_alerts.append((decision, request))

        return _to_response(decision, request, explanation)

    def score_many(self, requests: Sequence[TransactionRequest]) -> list[ScoreResponse]:
        return [self.score(r) for r in requests]

    # -- introspection -----------------------------------------------------

    @property
    def is_ready(self) -> bool:
        return self.decision_engine is not None

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok" if self.is_ready else "degraded",
            "model_loaded": self.model is not None,
            "model_version": self.model.version if self.model else None,
            "rule_set_version": self.rule_engine.version,
            "explainer_available": bool(self.explainer and self.explainer.available),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_rules() -> RuleEngine:
    if DEFAULT_RULES_PATH.exists():
        engine = RuleEngine.from_yaml(DEFAULT_RULES_PATH)
        # Fail fast on a rule referencing a feature the pipeline does not emit:
        # such a rule never fires and never errors, which is the worst of both.
        engine.validate_against(FeaturePipeline.FEATURE_NAMES)
        logger.info("Loaded %d rules (version %s)", len(engine.rules), engine.version)
        return engine
    logger.warning("No rule file at %s; running with an empty rule set", DEFAULT_RULES_PATH)
    return RuleEngine()


def _thresholds_from(model: FraudModel, fallback: PolicyThresholds) -> PolicyThresholds:
    """Take the operating point recorded in the model artefact.

    Thresholds are chosen on validation during training and travel with the
    model, so serving cannot silently drift onto a different operating point.
    """
    recorded = model.metadata.thresholds.get("min_expected_cost")
    if recorded is None:
        return fallback
    review = float(min(max(recorded, 0.01), 0.98))
    return PolicyThresholds(
        challenge=max(0.0, review * 0.5),
        review=review,
        block=max(review, 0.95),
    )


def _to_event(request: TransactionRequest) -> TransactionEvent:
    return TransactionEvent(
        timestamp=request.timestamp,
        card=request.card_number,
        amount=request.amount,
        merchant=request.merchant,
        category=request.category,
        home_lat=request.home_lat,
        home_lon=request.home_lon,
        merch_lat=request.merchant_lat,
        merch_lon=request.merchant_lon,
        city_pop=request.city_population,
        dob=request.date_of_birth,
        transaction_id=request.transaction_id,
    )


def _to_response(
    decision: Decision,
    request: TransactionRequest,
    explanation: Any | None,
) -> ScoreResponse:
    contributions: list[ContributionResponse] = []
    by_code: dict[str, ReasonCodeResponse] = {}

    # Rule-derived codes come first and always. A rule block short-circuits the
    # model, and the explainer may be unavailable entirely, so sourcing reason
    # codes only from SHAP would leave exactly the most decisive outcomes --
    # deterministic blocks -- with no explanation at all.
    for hit in decision.rule_evaluation.hits:
        if not hit.reason_code:
            continue
        by_code[hit.reason_code] = ReasonCodeResponse(
            code=hit.reason_code,
            text=f"{hit.reason_code} - {hit.name}",
            category="rule",
            feature="",
            value=0.0,
            contribution=0.0,
        )

    if explanation is not None:
        # SHAP-derived codes carry the actual feature value, so they supersede
        # the rule rendering of the same code.
        for c in explanation.reason_codes:
            by_code[c.code] = ReasonCodeResponse(
                code=c.code,
                text=c.text,
                category=c.category,
                feature=c.feature,
                value=c.value,
                contribution=c.contribution,
            )
        contributions = [
            ContributionResponse(feature=str(d["feature"]), contribution=float(d["contribution"]))
            for d in explanation.waterfall_data(limit=10)
        ]

    codes = list(by_code.values())

    return ScoreResponse(
        transaction_id=decision.transaction_id or request.transaction_id,
        band=decision.band,
        probability=decision.probability,
        creates_alert=decision.creates_alert,
        decided_by_rule=decision.decided_by_rule,
        reason_codes=codes,
        rules_fired=[
            RuleHitResponse(
                rule_id=h.rule_id,
                version=h.rule_version,
                name=h.name,
                action=str(h.action),
                severity=str(h.severity),
                reason_code=h.reason_code,
            )
            for h in decision.rule_evaluation.hits
        ],
        contributions=contributions,
        model_version=decision.model_version,
        rule_set_version=decision.rule_set_version,
        policy_version=decision.policy_version,
        feature_hash=decision.feature_hash,
        latency_ms=decision.latency_ms,
        decided_at=decision.decided_at or datetime.now(UTC),
    )
