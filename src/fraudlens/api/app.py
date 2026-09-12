"""FastAPI application: the scoring service and analyst-console backend.

Endpoints are deliberately thin. Everything of substance lives in
:class:`~fraudlens.api.service.FraudLensService`, so the scoring path is
testable without an HTTP client and the routing layer stays reviewable.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

from fraudlens.api.schemas import (
    AlertPage,
    AlertSummary,
    BatchScoreRequest,
    BatchScoreResponse,
    DispositionRequest,
    HealthResponse,
    ModelInfoResponse,
    ScoreResponse,
    StatsResponse,
    TransactionRequest,
)
from fraudlens.api.service import FraudLensService, ModelNotLoadedError
from fraudlens.api.simulation import ThresholdSimulator
from fraudlens.config import get_settings
from fraudlens.explain.reason_codes import catalogue
from fraudlens.models.model import FeatureContractError
from fraudlens.streaming.broker import TOPIC_ALERTS, TOPIC_DECISIONS

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)


API_VERSION = "0.1.0"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model once at startup rather than per request."""
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
    )
    logger.info("Starting FraudLens API (profile=%s, env=%s)", settings.profile, settings.env)
    service = FraudLensService.load(settings)
    app.state.service = service
    await service.start_stream(settings)
    yield
    await service.stop_stream()
    logger.info("Shutting down FraudLens API")


app = FastAPI(
    title="FraudLens",
    description=(
        "Real-time fraud detection and risk intelligence. Combines a deterministic "
        "rule engine, a calibrated gradient-boosted model, and per-decision "
        "explanations rendered as analyst-readable reason codes."
    ),
    version=API_VERSION,
    lifespan=lifespan,
)

_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_service(request: Request) -> FraudLensService:
    service: FraudLensService | None = getattr(request.app.state, "service", None)
    if service is None:  # pragma: no cover - only before lifespan completes
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="service is still starting",
        )
    return service


ServiceDep = Annotated[FraudLensService, Depends(get_service)]

# Every REST endpoint lives under `/api`.
#
# Not cosmetic: the console is a single-page app with its own client-side routes,
# and `/alerts` and `/model` are natural names for *both* a UI page and an API
# resource. With the API at the root those collide, and navigating to /model in a
# browser returns raw JSON instead of the page. Namespacing the API removes the
# whole class of collision rather than renaming pages around it.
#
# `/health` and `/metrics` are additionally exposed at the root below, because
# health checks and Prometheus scrape configs conventionally expect them there.
api = APIRouter()


@app.exception_handler(ModelNotLoadedError)
async def _model_not_loaded(_: Request, exc: ModelNotLoadedError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": str(exc)},
    )


@app.exception_handler(FeatureContractError)
async def _feature_contract(_: Request, exc: FeatureContractError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"feature contract violation: {exc}"},
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@api.post("/score", response_model=ScoreResponse, tags=["scoring"])
def score(transaction: TransactionRequest, service: ServiceDep) -> ScoreResponse:
    """Score a single transaction and return a decision with reasons."""
    return service.score(transaction)


@api.post("/score/batch", response_model=BatchScoreResponse, tags=["scoring"])
def score_batch(payload: BatchScoreRequest, service: ServiceDep) -> BatchScoreResponse:
    """Score up to 1,000 transactions in one call.

    Transactions are processed in the order given, because each one updates the
    card state the next may depend on.
    """
    results = service.score_many(payload.transactions)
    return BatchScoreResponse(results=results, count=len(results))


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------


@api.get("/alerts", response_model=AlertPage, tags=["alerts"])
def list_alerts(
    service: ServiceDep,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    band: Annotated[str | None, Query()] = None,
    open_only: Annotated[bool, Query()] = False,
) -> AlertPage:
    """The analyst triage queue, newest first."""
    rows: list[AlertSummary] = []
    for decision, request, _response in reversed(service.recent_alerts):
        if band and str(decision.band) != band:
            continue
        disposition = service.dispositions.get(decision.transaction_id)
        if open_only and disposition is not None:
            continue
        rows.append(
            AlertSummary(
                transaction_id=decision.transaction_id,
                decided_at=decision.decided_at,
                band=decision.band,
                probability=decision.probability,
                amount=decision.amount,
                merchant=request.merchant,
                category=request.category,
                card_last4=str(request.card_number)[-4:],
                reason_codes=list(decision.reason_codes),
                top_reason=decision.explanation[0] if decision.explanation else "",
                disposition=disposition,
            )
        )
        if len(rows) >= limit:
            break

    open_count = sum(
        1 for d, _, _ in service.recent_alerts if d.transaction_id not in service.dispositions
    )
    return AlertPage(alerts=rows, total=len(service.recent_alerts), open_count=open_count)


@api.get("/alerts/{transaction_id}", response_model=ScoreResponse, tags=["alerts"])
def alert_detail(transaction_id: str, service: ServiceDep) -> ScoreResponse:
    """The full decision for one alert, including reason codes and SHAP values.

    Served from the retained response rather than the live stream, so a case can
    be opened long after the transaction was scored.
    """
    response = service.find_alert(transaction_id)
    if response is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"no retained alert for transaction {transaction_id!r}; "
                f"the queue holds the most recent 500"
            ),
        )
    return response


@api.post("/alerts/{transaction_id}/disposition", tags=["alerts"])
def set_disposition(
    transaction_id: str, payload: DispositionRequest, service: ServiceDep
) -> dict[str, str]:
    """Record an analyst verdict.

    These are the labels a future retrain learns from, which is why the
    vocabulary is fixed rather than free text.
    """
    known = {d.transaction_id for d, _, _ in service.recent_alerts}
    if transaction_id not in known:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"no alert found for transaction {transaction_id!r}",
        )
    service.dispositions[transaction_id] = payload.disposition
    logger.info(
        "Disposition %s recorded for %s by %s",
        payload.disposition,
        transaction_id,
        payload.analyst,
    )
    return {"transaction_id": transaction_id, "disposition": payload.disposition}


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------


@api.get("/health", response_model=HealthResponse, tags=["ops"])
def health(service: ServiceDep) -> HealthResponse:
    """Liveness and readiness.

    Reports ``degraded`` rather than failing when no model is loaded: the
    process is up and serving, it just cannot score yet, and that distinction
    matters to whatever is watching this endpoint.
    """
    info = service.health()
    return HealthResponse(
        status=info["status"],
        version=API_VERSION,
        model_loaded=info["model_loaded"],
        model_version=info["model_version"],
        rule_set_version=info["rule_set_version"],
        explainer_available=info["explainer_available"],
        profile=str(get_settings().profile),
    )


@api.get("/model", response_model=ModelInfoResponse, tags=["ops"])
def model_info(service: ServiceDep) -> ModelInfoResponse:
    """Model provenance: what was trained, on what, and how it performed."""
    if service.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="no model loaded"
        )
    meta = service.model.metadata
    return ModelInfoResponse(
        version=meta.version,
        model_type=meta.model_type,
        trained_at=meta.trained_at,
        feature_names=list(meta.feature_names),
        n_train_rows=meta.n_train_rows,
        n_train_frauds=meta.n_train_frauds,
        train_fraud_rate=meta.train_fraud_rate,
        scale_pos_weight=meta.scale_pos_weight,
        calibration_method=meta.calibration_method,
        metrics=meta.metrics,
        thresholds=meta.thresholds,
        data_window=meta.data_window,
    )


@api.get("/stats", response_model=StatsResponse, tags=["ops"])
def stats(service: ServiceDep) -> StatsResponse:
    """Live counters, including the measured p99 latency."""
    return service.stats.snapshot()


@api.get("/rules", tags=["ops"])
def rules(service: ServiceDep) -> dict[str, Any]:
    """The active rule set, so a reviewer can see what fires and why."""
    return {
        "version": service.rule_engine.version,
        "rules": [
            {
                "id": r.id,
                "name": r.name,
                "action": str(r.action),
                "severity": str(r.severity),
                "reason_code": r.reason_code,
                "description": r.description.strip(),
                "version": r.version,
                "enabled": r.enabled,
                "features": sorted(
                    __import__(
                        "fraudlens.rules.engine", fromlist=["referenced_features"]
                    ).referenced_features(r.condition)
                ),
            }
            for r in service.rule_engine.rules
        ],
    }


@api.get("/reason-codes", tags=["ops"])
def reason_codes() -> dict[str, Any]:
    """The reason-code catalogue rendered in the console and the model card."""
    return {"codes": catalogue()}


@api.get("/drift", tags=["ops"])
def drift(service: ServiceDep) -> dict[str, Any]:
    """Population Stability Index per feature against the training distribution.

    Returns ``enabled: false`` rather than erroring when the training reference
    is unavailable -- drift monitoring is a capability the service can lack
    without being broken.
    """
    if service.drift is None:
        return {
            "enabled": False,
            "reason": "no training reference available; run scripts/train.py",
            "features": [],
        }
    measurements = service.drift.measure_all()
    for measurement in measurements:
        service.metrics.record_drift(measurement.feature, measurement.psi)
    return {
        "enabled": True,
        "summary": service.drift.summary(),
        "bands": {"stable": "< 0.10", "monitor": "0.10 - 0.25", "investigate": "> 0.25"},
        "features": [
            {
                "feature": m.feature,
                "psi": round(m.psi, 5),
                "status": str(m.status),
                "reference_mean": round(m.reference_mean, 4),
                "live_mean": round(m.live_mean, 4),
                "mean_shift": round(m.mean_shift, 4),
                "n_live": m.n_live,
            }
            for m in measurements
        ],
    }


@api.get("/stream/status", tags=["ops"])
def stream_status(service: ServiceDep) -> dict[str, Any]:
    """Whether the replay producer and bus are running."""
    broker = service.broker
    zero = lambda _topic: 0  # noqa: E731
    return {
        "broker": type(broker).__name__ if broker else None,
        "replay_running": bool(service.replay and service.replay.is_running),
        "replay": service.replay.stats.as_dict() if service.replay else None,
        "subscribers": {
            TOPIC_DECISIONS: getattr(broker, "subscriber_count", zero)(TOPIC_DECISIONS),
            TOPIC_ALERTS: getattr(broker, "subscriber_count", zero)(TOPIC_ALERTS),
        },
        "dropped_total": getattr(broker, "dropped_total", 0),
    }


@app.websocket("/ws/decisions")
async def ws_decisions(websocket: WebSocket) -> None:
    """Live feed of every decision, for the console's monitor page."""
    await _stream_topic(websocket, TOPIC_DECISIONS)


@app.websocket("/ws/alerts")
async def ws_alerts(websocket: WebSocket) -> None:
    """Live feed of alerts only -- the analyst queue."""
    await _stream_topic(websocket, TOPIC_ALERTS)


async def _stream_topic(websocket: WebSocket, topic: str) -> None:
    """Fan one bus topic out to a WebSocket client.

    The subscription is torn down in ``finally`` so a client that closes its tab
    does not leak a queue for the lifetime of the process.
    """
    await websocket.accept()
    service = getattr(websocket.app.state, "service", None)
    if service is None or service.broker is None:
        await websocket.close(code=1011, reason="stream unavailable")
        return

    subscription = service.broker.subscribe(topic)
    try:
        async for message in subscription:
            await websocket.send_text(message.to_json())
    except WebSocketDisconnect:
        logger.debug("WebSocket client disconnected from %s", topic)
    except Exception:
        logger.exception("WebSocket stream failed for %s", topic)
    finally:
        await subscription.aclose()


@api.get("/simulate/threshold", tags=["simulation"])
def simulate_threshold(
    service: ServiceDep,
    threshold: Annotated[float, Query(ge=0.0, le=1.0)] = 0.5,
    investigation_cost: Annotated[float, Query(ge=0.0)] = 4.0,
    friction_cost: Annotated[float, Query(ge=0.0)] = 18.0,
) -> dict[str, Any]:
    """What one threshold would do on the held-out test fold.

    This visualises trade-offs; it does **not** select the operating point. The
    shipped threshold was chosen on the *validation* fold during training and is
    frozen in the model artefact. Tuning against these numbers would be exactly
    the test-set leakage the project exists to avoid.
    """
    simulator = _require_simulator(service)
    result = simulator.evaluate(
        threshold, investigation_cost=investigation_cost, friction_cost=friction_cost
    )
    optimal = simulator.optimal(investigation_cost=investigation_cost, friction_cost=friction_cost)
    return {
        "point": asdict(result),
        "optimal": asdict(optimal),
        "shipped_threshold": (
            service.model.metadata.thresholds.get("min_expected_cost") if service.model else None
        ),
        "test_rows": simulator.size,
        "test_frauds": simulator.fraud_count,
    }


@api.get("/simulate/curve", tags=["simulation"])
def simulate_curve(
    service: ServiceDep,
    investigation_cost: Annotated[float, Query(ge=0.0)] = 4.0,
    friction_cost: Annotated[float, Query(ge=0.0)] = 18.0,
) -> dict[str, Any]:
    """Precision, recall, alert rate and expected cost across the threshold range."""
    simulator = _require_simulator(service)
    curve = simulator.curve(investigation_cost=investigation_cost, friction_cost=friction_cost)
    optimal = min(curve, key=lambda r: r.expected_cost)
    return {
        "curve": [
            {
                "threshold": round(r.threshold, 6),
                "precision": round(r.precision, 4),
                "recall": round(r.recall, 4),
                "alert_rate": round(r.alert_rate, 6),
                "alerts_per_day": round(r.alerts_per_day, 2),
                "expected_cost": round(r.expected_cost, 2),
            }
            for r in curve
        ],
        "optimal_threshold": round(optimal.threshold, 6),
        "histogram": simulator.score_histogram(),
    }


def _require_simulator(service: FraudLensService) -> ThresholdSimulator:
    if service.simulator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "threshold simulation needs cached test features; "
                "run scripts/train.py to generate them"
            ),
        )
    return service.simulator


@api.get("/metrics", response_class=PlainTextResponse, tags=["ops"])
def metrics(service: ServiceDep) -> PlainTextResponse:
    """Prometheus exposition."""
    if service.drift is not None:
        for measurement in service.drift.measure_all():
            service.metrics.record_drift(measurement.feature, measurement.psi)
    return PlainTextResponse(
        content=service.metrics.render(), media_type=service.metrics.content_type
    )


app.include_router(api, prefix="/api")


@app.get("/health", response_model=HealthResponse, include_in_schema=False)
def root_health(service: ServiceDep) -> HealthResponse:
    """Root alias: container health checks expect /health, not /api/health."""
    return health(service)


@app.get("/metrics", response_class=PlainTextResponse, include_in_schema=False)
def root_metrics(service: ServiceDep) -> PlainTextResponse:
    """Root alias: Prometheus scrape configs conventionally use /metrics."""
    return metrics(service)


# ---------------------------------------------------------------------------
# Static console
# ---------------------------------------------------------------------------
#
# In production the API serves the built React bundle itself, so there is one
# origin, no CORS preflight, and one process to deploy. In development Vite
# serves the console and proxies `/api` here instead, which is why the client
# uses relative paths in both cases.

_CONSOLE_DIST = _settings.resolved_console_dir


def _mount_console() -> None:
    if not (_CONSOLE_DIST / "index.html").exists():
        logger.info("No built console at %s; serving the API only", _CONSOLE_DIST)
        return

    from fastapi.staticfiles import StaticFiles

    # Hashed asset filenames are immutable, so they can be cached hard.
    app.mount(
        "/assets",
        StaticFiles(directory=_CONSOLE_DIST / "assets"),
        name="console-assets",
    )

    # response_model=None: the return type is a union of Response subclasses,
    # which FastAPI would otherwise try to turn into a Pydantic response model.
    @app.get("/{full_path:path}", include_in_schema=False, response_model=None)
    async def serve_console(full_path: str) -> FileResponse | JSONResponse:
        """Serve the SPA, letting client-side routing handle deep links.

        Registered last so it never shadows an API route. An unknown path under
        `/api` must still 404 as JSON rather than silently returning HTML --
        otherwise a typo'd endpoint looks like a working page to a caller.
        """
        if full_path.startswith(("api/", "ws/")):
            return JSONResponse(status_code=404, content={"detail": "Not found"})

        candidate = _CONSOLE_DIST / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(_CONSOLE_DIST / "index.html")

    logger.info("Serving the analyst console from %s", _CONSOLE_DIST)


_mount_console()
