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

import asyncio
import contextlib
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from fraudlens.api.schemas import (
    ContributionResponse,
    ReasonCodeResponse,
    RuleHitResponse,
    ScoreResponse,
    StatsResponse,
    TransactionRequest,
)
from fraudlens.api.simulation import ThresholdSimulator, build_simulator
from fraudlens.config import Settings, get_settings
from fraudlens.explain.explainer import ExplainerService
from fraudlens.features.pipeline import FeaturePipeline, TransactionEvent
from fraudlens.features.store import FeatureStore, build_feature_store
from fraudlens.models.model import FraudModel
from fraudlens.monitoring.drift import CYCLICAL_FEATURES, DriftMonitor
from fraudlens.monitoring.metrics import Metrics
from fraudlens.rules.engine import RuleEngine
from fraudlens.scoring.decision import Decision, DecisionEngine, PolicyThresholds, RiskBand
from fraudlens.streaming.broker import (
    TOPIC_ALERTS,
    TOPIC_DECISIONS,
    StreamBroker,
    StreamMessage,
    build_broker,
)
from fraudlens.streaming.replay import (
    ReplayProducer,
    load_replay_source,
    load_warmup_window,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

#: A card needs at least this much history before its features are treated as
#: representative for drift measurement. See the guard in `score`.
MIN_HISTORY_FOR_DRIFT = 20


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
        broker: StreamBroker | None = None,
        metrics: Metrics | None = None,
        drift: DriftMonitor | None = None,
        simulator: ThresholdSimulator | None = None,
    ) -> None:
        self.model = model
        self.rule_engine = rule_engine
        self.feature_store = feature_store
        self.pipeline = FeaturePipeline()
        self.stats = ServiceStats()
        self.thresholds = thresholds or PolicyThresholds()
        self.explainer = explainer
        self.broker = broker
        self.metrics = metrics or Metrics()
        self.drift = drift
        self.simulator = simulator
        self.replay: ReplayProducer | None = None
        self._consumer_task: asyncio.Task[None] | None = None
        self.decision_engine = (
            DecisionEngine(model=model, rule_engine=rule_engine, thresholds=self.thresholds)
            if model is not None
            else None
        )
        #: Recent alerts, newest last. The full ScoreResponse is retained --
        #: not just the Decision -- so the case-detail view can show reason
        #: codes and SHAP contributions for an alert that has already scrolled
        #: out of the live stream. Without this the detail page can only
        #: explain alerts that happened to arrive after it was opened, which
        #: is precisely backwards for a triage queue.
        self.recent_alerts: deque[tuple[Decision, TransactionRequest, ScoreResponse]] = deque(
            maxlen=500
        )
        self.dispositions: dict[str, str] = {}

        if model is not None:
            # The ScoringModel protocol requires only `version` and
            # `predict_proba`. Richer metadata is read defensively so that any
            # object satisfying the protocol -- including a test stub -- can be
            # injected without the metrics layer dictating a wider contract.
            metadata = getattr(model, "metadata", None)
            self.metrics.set_model_info(
                version=model.version,
                model_type=getattr(metadata, "model_type", "unknown"),
                calibration=getattr(metadata, "calibration_method", "unknown"),
            )

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
        rule_engine = _load_rules(settings)
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

        return cls(
            model,
            rule_engine,
            store,
            thresholds=thresholds,
            explainer=explainer,
            broker=build_broker(str(settings.profile), settings.kafka_bootstrap_servers),
            drift=_build_drift_monitor(model, settings),
            simulator=build_simulator(model, settings),
        )

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
        self.metrics.record_decision(
            band=str(decision.band),
            probability=decision.probability,
            latency_ms=decision.latency_ms,
            creates_alert=decision.creates_alert,
            rules=[(h.rule_id, str(h.action)) for h in decision.rule_evaluation.hits],
        )
        if (
            self.drift is not None
            and features.get("card_history_count", 0.0) >= MIN_HISTORY_FOR_DRIFT
        ):
            # Only cards with real history contribute to drift measurement.
            #
            # A cold feature store makes every card look brand new: trailing
            # windows read near zero, and PSI against a training reference built
            # from mature history screams "significant drift" when nothing has
            # actually changed. That false alarm is worse than no signal, because
            # it trains the team to ignore the drift alert. Cold-start rows are
            # excluded so drift measures the world moving, not the cache filling.
            self.drift.observe(features)
        response = _to_response(decision, request, explanation)
        if decision.creates_alert:
            self.recent_alerts.append((decision, request, response))

        self._emit(decision, response)
        return response

    def score_many(self, requests: Sequence[TransactionRequest]) -> list[ScoreResponse]:
        return [self.score(r) for r in requests]

    def find_alert(self, transaction_id: str) -> ScoreResponse | None:
        """The full stored response for one alert, if still retained."""
        for _decision, _request, response in reversed(self.recent_alerts):
            if response.transaction_id == transaction_id:
                return response
        return None

    # -- streaming ---------------------------------------------------------

    def _emit(self, decision: Decision, response: ScoreResponse) -> None:
        """Publish the decision to the bus, without blocking the scoring path.

        Scoring is synchronous and may be called from a worker thread, a script
        or a test where no event loop is running. Publishing is therefore
        best-effort and fire-and-forget: an observer that is absent, slow or
        broken must never add latency to an authorisation decision, and must
        never fail one.
        """
        if self.broker is None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # no event loop (script/test context) -- nothing is listening

        payload = response.model_dump(mode="json")
        message = StreamMessage(topic=TOPIC_DECISIONS, payload=payload, key=decision.transaction_id)
        task = loop.create_task(self.broker.publish(message))
        task.add_done_callback(_log_publish_failure)

        if decision.creates_alert:
            alert = loop.create_task(
                self.broker.publish(
                    StreamMessage(topic=TOPIC_ALERTS, payload=payload, key=decision.transaction_id)
                )
            )
            alert.add_done_callback(_log_publish_failure)

    async def start_stream(self, settings: Settings | None = None) -> None:
        """Start the broker and, in demo mode, the replay producer."""
        settings = settings or get_settings()
        if self.broker is None:
            self.broker = build_broker(str(settings.profile), settings.kafka_bootstrap_servers)
        await self.broker.start()

        if not settings.demo_mode:
            return
        if not self.is_ready:
            logger.warning("Demo mode requested but no model is loaded; replay not started")
            return
        try:
            source = load_replay_source()
        except Exception:
            logger.exception("Could not load replay source; demo stream not started")
            return

        self.warm_feature_store()

        self.replay = ReplayProducer(
            transactions=source, broker=self.broker, speed=settings.replay_speed, loop=True
        )
        self.replay.start()
        self._consumer_task = asyncio.create_task(
            self._consume_replay(), name="fraudlens-replay-consumer"
        )

    def warm_feature_store(self) -> int:
        """Fold pre-replay history into the store so cards are not cold.

        These transactions build state only -- they are never scored, never seen
        by the model, and never counted in any metric. Without this the demo
        would spend its first thousands of transactions with empty velocity
        features, which is both unrepresentative and a source of false drift
        alarms.
        """
        try:
            warm = load_warmup_window()
        except Exception:
            logger.warning("Could not load the warm-up window", exc_info=True)
            return 0
        if warm.empty:
            return 0

        for row in warm.to_dict("records"):
            try:
                event = TransactionEvent.from_row(row)
            except (KeyError, ValueError):
                continue
            self.feature_store.update(event.card, event)

        logger.info(
            "Warmed the feature store with %d transactions across %d cards",
            len(warm),
            self.feature_store.size(),
        )
        return len(warm)

    async def stop_stream(self) -> None:
        """Stop the replay and close the broker."""
        if self.replay is not None:
            await self.replay.stop()
            self.replay = None
        task = self._consumer_task
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            self._consumer_task = None
        if self.broker is not None:
            await self.broker.close()

    async def _consume_replay(self) -> None:
        """Score every replayed transaction.

        This is what makes the demo real: the replayed transactions go through
        the identical scoring path a live caller would hit, so the alerts on the
        dashboard are genuine detections on data the model has never seen.
        """
        if self.broker is None:
            return
        from fraudlens.streaming.broker import TOPIC_TRANSACTIONS

        async for message in self.broker.subscribe(TOPIC_TRANSACTIONS):
            try:
                request = _request_from_payload(message.payload)
            except Exception:
                logger.debug("Skipping unparseable replay payload", exc_info=True)
                continue
            try:
                self.score(request)
            except ModelNotLoadedError:
                return
            except Exception:
                logger.exception("Scoring failed for a replayed transaction")

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


def _log_publish_failure(task: asyncio.Task[None]) -> None:
    """Surface a failed publish without letting it escape as an unretrieved exception."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.warning("Stream publish failed: %s", exc)


def _request_from_payload(payload: dict[str, Any]) -> TransactionRequest:
    """Rebuild a scoring request from a replayed raw-transaction payload."""
    from fraudlens.config import constants as C

    return TransactionRequest(
        transaction_id=str(payload[C.TRANSACTION_ID_COL]),
        timestamp=payload[C.TIMESTAMP_COL],
        card_number=int(payload[C.CARD_COL]),
        amount=float(payload[C.AMOUNT_COL]),
        merchant=str(payload["merchant"]),
        category=str(payload["category"]),
        home_lat=float(payload[C.HOME_LAT_COL]),
        home_lon=float(payload[C.HOME_LON_COL]),
        merchant_lat=float(payload[C.MERCH_LAT_COL]),
        merchant_lon=float(payload[C.MERCH_LON_COL]),
        city_population=int(payload.get("city_pop") or 0),
        date_of_birth=payload.get("dob"),
    )


def _build_drift_monitor(model: FraudModel | None, settings: Settings) -> DriftMonitor | None:
    """Build a drift monitor using the training fold as the reference.

    Returns ``None`` when the cached training features are unavailable. Drift
    monitoring is genuinely optional -- a service that cannot measure drift is
    degraded, not broken -- so its absence must not prevent startup.
    """
    if model is None:
        return None
    # The committed sample is preferred: it reproduces the reference quantiles a
    # deployed instance needs without carrying 38 MB of training features in git.
    bundle = settings.artifact_dir / "demo" / "drift_reference.parquet"
    full = settings.processed_dir / "features_full" / "features_train.parquet"
    reference_path = bundle if bundle.exists() else full

    if not reference_path.exists():
        logger.info("No drift reference at %s; drift monitoring disabled", bundle)
        return None
    try:
        import pandas as pd

        available = set(pd.read_parquet(reference_path).columns)
        wanted = [
            name
            for name in model.feature_names
            if name not in CYCLICAL_FEATURES and name in available
        ]
        frame = pd.read_parquet(reference_path, columns=wanted)
        reference = {name: frame[name].to_numpy(dtype=float) for name in wanted}
        logger.info(
            "Drift monitoring enabled over %d features (%d cyclical features excluded)",
            len(reference),
            len(CYCLICAL_FEATURES),
        )
        return DriftMonitor(reference=reference)
    except Exception:
        logger.warning("Could not build the drift reference; monitoring disabled", exc_info=True)
        return None


def _load_rules(settings: Settings | None = None) -> RuleEngine:
    settings = settings or get_settings()
    rules_path = settings.resolved_rules_path
    if rules_path.exists():
        engine = RuleEngine.from_yaml(rules_path)
        # Fail fast on a rule referencing a feature the pipeline does not emit:
        # such a rule never fires and never errors, which is the worst of both.
        engine.validate_against(FeaturePipeline.FEATURE_NAMES)
        logger.info("Loaded %d rules (version %s)", len(engine.rules), engine.version)
        return engine
    # Loud, because an empty rule set still scores transactions -- it just stops
    # enforcing every deterministic hard stop, silently.
    logger.error(
        "No rule file at %s. Running with an EMPTY rule set: deterministic blocks "
        "(impossible travel, card testing) will not fire.",
        rules_path,
    )
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
            message=hit.name,
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
                message=c.message,
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
