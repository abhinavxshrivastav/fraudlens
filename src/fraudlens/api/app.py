"""FastAPI application: the scoring service and analyst-console backend.

Endpoints are deliberately thin. Everything of substance lives in
:class:`~fraudlens.api.service.FraudLensService`, so the scoring path is
testable without an HTTP client and the routing layer stays reviewable.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

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
from fraudlens.config import get_settings
from fraudlens.explain.reason_codes import catalogue
from fraudlens.models.model import FeatureContractError

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
    app.state.service = FraudLensService.load(settings)
    yield
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


@app.post("/score", response_model=ScoreResponse, tags=["scoring"])
def score(transaction: TransactionRequest, service: ServiceDep) -> ScoreResponse:
    """Score a single transaction and return a decision with reasons."""
    return service.score(transaction)


@app.post("/score/batch", response_model=BatchScoreResponse, tags=["scoring"])
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


@app.get("/alerts", response_model=AlertPage, tags=["alerts"])
def list_alerts(
    service: ServiceDep,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    band: Annotated[str | None, Query()] = None,
    open_only: Annotated[bool, Query()] = False,
) -> AlertPage:
    """The analyst triage queue, newest first."""
    rows: list[AlertSummary] = []
    for decision, request in reversed(service.recent_alerts):
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
        1 for d, _ in service.recent_alerts if d.transaction_id not in service.dispositions
    )
    return AlertPage(alerts=rows, total=len(service.recent_alerts), open_count=open_count)


@app.post("/alerts/{transaction_id}/disposition", tags=["alerts"])
def set_disposition(
    transaction_id: str, payload: DispositionRequest, service: ServiceDep
) -> dict[str, str]:
    """Record an analyst verdict.

    These are the labels a future retrain learns from, which is why the
    vocabulary is fixed rather than free text.
    """
    known = {d.transaction_id for d, _ in service.recent_alerts}
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


@app.get("/health", response_model=HealthResponse, tags=["ops"])
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


@app.get("/model", response_model=ModelInfoResponse, tags=["ops"])
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


@app.get("/stats", response_model=StatsResponse, tags=["ops"])
def stats(service: ServiceDep) -> StatsResponse:
    """Live counters, including the measured p99 latency."""
    return service.stats.snapshot()


@app.get("/rules", tags=["ops"])
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


@app.get("/reason-codes", tags=["ops"])
def reason_codes() -> dict[str, Any]:
    """The reason-code catalogue rendered in the console and the model card."""
    return {"codes": catalogue()}


@app.get("/metrics", response_class=PlainTextResponse, tags=["ops"])
def metrics() -> str:
    """Prometheus exposition."""
    try:
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest  # noqa: F401
    except ImportError:  # pragma: no cover
        return "# prometheus_client is not installed\n"
    return generate_latest().decode("utf-8")


@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    return {
        "service": "FraudLens",
        "version": API_VERSION,
        "docs": "/docs",
        "health": "/health",
    }
