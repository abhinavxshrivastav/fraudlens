"""Request and response models for the scoring API.

Pydantic validates at the boundary so that nothing downstream has to defend
itself against malformed input. Constraints here are real domain constraints --
a latitude outside [-90, 90] or a negative amount is a client bug, and saying so
with a 422 is far more useful than scoring it and returning nonsense.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from fraudlens.scoring.decision import RiskBand

Latitude = Annotated[float, Field(ge=-90.0, le=90.0)]
Longitude = Annotated[float, Field(ge=-180.0, le=180.0)]


class TransactionRequest(BaseModel):
    """One transaction submitted for scoring."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "transaction_id": "a3f2c1d4e5b6a7c8d9e0f1a2b3c4d5e6",
                "timestamp": "2020-08-14T02:41:00",
                "card_number": 4532015112830366,
                "amount": 1897.42,
                "merchant": "fraud_Kutch-Rippin_0142",
                "category": "shopping_net",
                "home_lat": 40.7128,
                "home_lon": -74.0060,
                "merchant_lat": 34.0522,
                "merchant_lon": -118.2437,
                "city_population": 8_336_817,
                "date_of_birth": "1985-06-01T00:00:00",
            }
        },
    )

    transaction_id: str = Field(min_length=1, max_length=64)
    timestamp: datetime
    card_number: int = Field(gt=0, description="Card identifier; features are grouped by this.")
    amount: float = Field(gt=0, le=1_000_000)
    merchant: str = Field(min_length=1, max_length=128)
    category: str = Field(min_length=1, max_length=64)
    home_lat: Latitude
    home_lon: Longitude
    merchant_lat: Latitude
    merchant_lon: Longitude
    city_population: int = Field(default=0, ge=0)
    date_of_birth: datetime | None = None


class ReasonCodeResponse(BaseModel):
    code: str
    #: Code and explanation together, for plain-text consumers.
    text: str
    #: The explanation alone, for clients that render the code separately.
    message: str
    category: str
    feature: str
    value: float
    contribution: float


class RuleHitResponse(BaseModel):
    rule_id: str
    version: int
    name: str
    action: str
    severity: str
    reason_code: str


class ContributionResponse(BaseModel):
    feature: str
    contribution: float


class ScoreResponse(BaseModel):
    """The decision returned to the caller."""

    model_config = ConfigDict(protected_namespaces=())

    transaction_id: str
    band: RiskBand
    probability: float = Field(ge=0.0, le=1.0)
    creates_alert: bool
    decided_by_rule: bool = Field(
        description="True when a deterministic rule block decided the outcome, bypassing the model."
    )
    reason_codes: list[ReasonCodeResponse] = Field(default_factory=list)
    rules_fired: list[RuleHitResponse] = Field(default_factory=list)
    contributions: list[ContributionResponse] = Field(
        default_factory=list,
        description="SHAP contributions, present only on the alert path.",
    )
    model_version: str
    rule_set_version: str
    policy_version: str
    feature_hash: str
    latency_ms: float
    decided_at: datetime


class BatchScoreRequest(BaseModel):
    transactions: list[TransactionRequest] = Field(min_length=1, max_length=1_000)


class BatchScoreResponse(BaseModel):
    results: list[ScoreResponse]
    count: int


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: Literal["ok", "degraded"]
    version: str
    model_loaded: bool
    model_version: str | None = None
    rule_set_version: str | None = None
    explainer_available: bool = False
    profile: str


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    version: str
    model_type: str
    trained_at: str
    feature_names: list[str]
    n_train_rows: int
    n_train_frauds: int
    train_fraud_rate: float
    scale_pos_weight: float
    calibration_method: str
    metrics: dict[str, float] = Field(default_factory=dict)
    thresholds: dict[str, float] = Field(default_factory=dict)
    data_window: dict[str, str] = Field(default_factory=dict)


class AlertSummary(BaseModel):
    """A row in the analyst alert queue."""

    transaction_id: str
    decided_at: datetime
    band: RiskBand
    probability: float
    amount: float
    merchant: str
    category: str
    card_last4: str
    reason_codes: list[str] = Field(default_factory=list)
    top_reason: str = ""
    disposition: str | None = None


class AlertPage(BaseModel):
    alerts: list[AlertSummary]
    total: int
    open_count: int


class DispositionRequest(BaseModel):
    """An analyst's verdict on an alert.

    Captured to build the feedback loop: confirmed outcomes are the labels a
    retrained model learns from, so the disposition vocabulary is fixed rather
    than free text.
    """

    disposition: Literal["confirmed_fraud", "false_positive", "escalated"]
    note: str = Field(default="", max_length=1_000)
    analyst: str = Field(default="unknown", max_length=64)


class ThresholdSimulationRequest(BaseModel):
    threshold: float = Field(ge=0.0, le=1.0)
    investigation_cost: float = Field(default=4.0, ge=0.0)
    friction_cost: float = Field(default=18.0, ge=0.0)


class ThresholdSimulationResponse(BaseModel):
    """What moving the decision threshold would do, on held-out test data."""

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


class ThresholdCurvePoint(BaseModel):
    threshold: float
    precision: float
    recall: float
    alert_rate: float
    expected_cost: float


class StatsResponse(BaseModel):
    """Live counters for the monitor page."""

    scored_total: int
    alerts_total: int
    alert_rate: float
    blocked_total: int
    mean_latency_ms: float
    p99_latency_ms: float
    band_counts: dict[str, int] = Field(default_factory=dict)
    uptime_seconds: float


class ErrorResponse(BaseModel):
    detail: str
    context: dict[str, Any] = Field(default_factory=dict)
