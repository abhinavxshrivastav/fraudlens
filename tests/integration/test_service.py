"""Tests for the scoring service and HTTP layer.

A stub model is injected rather than loading a trained artefact, so these run in
CI without model files and exercise the wiring rather than the model's accuracy.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fraudlens.api.app import app, get_service
from fraudlens.api.schemas import TransactionRequest
from fraudlens.api.service import FraudLensService, ModelNotLoadedError, _load_rules
from fraudlens.config import Settings
from fraudlens.features.store import InMemoryFeatureStore
from fraudlens.rules.engine import RuleEngine
from fraudlens.scoring.decision import PolicyThresholds, RiskBand

BASE = datetime(2020, 8, 14, 14, 30)
CARD = 4_532_015_112_830_366


class StubModel:
    """Returns a fixed probability; records how often it was consulted."""

    def __init__(self, probability: float = 0.05) -> None:
        self.probability = probability
        self.calls = 0

    @property
    def version(self) -> str:
        return "stub-1.0"

    def predict_proba(self, features: dict[str, float]) -> float:
        self.calls += 1
        return self.probability


def make_request(
    *,
    offset_minutes: float = 0.0,
    amount: float = 60.0,
    merchant: str = "fraud_Kutch-Rippin_0001",
    merchant_lat: float = 40.71,
    merchant_lon: float = -74.01,
    txn_id: str | None = None,
) -> TransactionRequest:
    return TransactionRequest(
        transaction_id=txn_id or f"txn-{offset_minutes}-{amount}",
        timestamp=BASE + timedelta(minutes=offset_minutes),
        card_number=CARD,
        amount=amount,
        merchant=merchant,
        category="grocery_pos",
        home_lat=40.7128,
        home_lon=-74.0060,
        merchant_lat=merchant_lat,
        merchant_lon=merchant_lon,
        city_population=8_336_817,
        date_of_birth=datetime(1985, 6, 1),
    )


def build_service(probability: float = 0.05, rules_yaml: bool = True) -> FraudLensService:
    root = Path(__file__).resolve().parents[2]
    engine = RuleEngine.from_yaml(root / "config" / "rules.yaml") if rules_yaml else RuleEngine()
    return FraudLensService(
        model=StubModel(probability),  # type: ignore[arg-type]
        rule_engine=engine,
        feature_store=InMemoryFeatureStore(),
        thresholds=PolicyThresholds(challenge=0.3, review=0.6, block=0.95),
    )


class TestScoringPath:
    def test_ordinary_transaction_is_approved(self) -> None:
        service = build_service(probability=0.02)
        response = service.score(make_request())
        assert response.band is RiskBand.APPROVE
        assert not response.creates_alert
        assert response.model_version == "stub-1.0"
        assert response.feature_hash

    def test_high_probability_produces_an_alert_with_reasons(self) -> None:
        service = build_service(probability=0.99)
        response = service.score(make_request(amount=2_400.0))
        assert response.band is RiskBand.BLOCK
        assert response.creates_alert
        assert response.latency_ms >= 0.0

    def test_state_advances_only_after_scoring(self) -> None:
        # The online form of the leakage guard: a card's first transaction must
        # report no prior history.
        service = build_service()
        service.score(make_request(offset_minutes=0))
        state = service.feature_store.get(CARD)
        assert len(state.history) == 1

        service.score(make_request(offset_minutes=10, txn_id="txn-2"))
        assert len(service.feature_store.get(CARD).history) == 2

    def test_velocity_accumulates_across_requests(self) -> None:
        service = build_service()
        for i in range(6):
            service.score(make_request(offset_minutes=i * 2, txn_id=f"v{i}"))
        # The seventh transaction should see the previous six.
        state = service.feature_store.get(CARD)
        features = service.pipeline.compute(
            service.pipeline.__class__ and _event(service, make_request(offset_minutes=13)),
            state,
        )
        assert features["txn_count_1h"] == 6.0

    def test_impossible_travel_blocks_without_consulting_the_model(self) -> None:
        service = build_service(probability=0.01)
        service.score(make_request(offset_minutes=0, txn_id="local"))
        model = service.model
        before = model.calls  # type: ignore[attr-defined]

        # Same card, Los Angeles, six minutes later.
        response = service.score(
            make_request(
                offset_minutes=6,
                merchant_lat=34.0522,
                merchant_lon=-118.2437,
                merchant="fraud_Far-Away_0999",
                txn_id="cloned",
            )
        )
        assert response.band is RiskBand.BLOCK
        assert response.decided_by_rule
        assert model.calls == before  # type: ignore[attr-defined]
        # A deterministic block must still be explained, even with no explainer.
        assert "R12" in [c.code for c in response.reason_codes]
        assert response.rules_fired[0].rule_id == "R-GEO-001"

    def test_degraded_service_refuses_to_score(self) -> None:
        service = FraudLensService(
            model=None, rule_engine=RuleEngine(), feature_store=InMemoryFeatureStore()
        )
        assert not service.is_ready
        with pytest.raises(ModelNotLoadedError):
            service.score(make_request())

    def test_batch_scoring_preserves_order(self) -> None:
        service = build_service()
        requests = [make_request(offset_minutes=i, txn_id=f"b{i}") for i in range(5)]
        responses = service.score_many(requests)
        assert [r.transaction_id for r in responses] == [f"b{i}" for i in range(5)]


class TestStats:
    def test_counters_track_decisions(self) -> None:
        service = build_service(probability=0.99)
        for i in range(4):
            service.score(make_request(offset_minutes=i * 90, txn_id=f"s{i}"))
        snapshot = service.stats.snapshot()
        assert snapshot.scored_total == 4
        assert snapshot.alerts_total == 4
        assert snapshot.alert_rate == pytest.approx(1.0)
        assert snapshot.p99_latency_ms >= 0.0

    def test_alert_rate_is_zero_before_any_traffic(self) -> None:
        assert build_service().stats.snapshot().alert_rate == 0.0


class TestHttpLayer:
    @pytest.fixture
    def client(self) -> TestClient:
        service = build_service(probability=0.99)
        app.dependency_overrides[get_service] = lambda: service
        with TestClient(app) as c:
            yield c
        app.dependency_overrides.clear()

    def test_score_endpoint(self, client: TestClient) -> None:
        payload = make_request(amount=1_800.0).model_dump(mode="json")
        response = client.post("/api/score", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["band"] in {"approve", "challenge", "review", "block"}
        assert "feature_hash" in body

    def test_rejects_negative_amount(self, client: TestClient) -> None:
        payload = make_request().model_dump(mode="json")
        payload["amount"] = -5.0
        assert client.post("/api/score", json=payload).status_code == 422

    def test_rejects_out_of_range_latitude(self, client: TestClient) -> None:
        payload = make_request().model_dump(mode="json")
        payload["merchant_lat"] = 120.0
        assert client.post("/api/score", json=payload).status_code == 422

    def test_rejects_unknown_fields(self, client: TestClient) -> None:
        # extra="forbid": a typo'd field name should be a loud 422, not a
        # silently ignored value.
        payload = make_request().model_dump(mode="json")
        payload["ammount"] = 50.0
        assert client.post("/api/score", json=payload).status_code == 422

    def test_alert_queue_populates(self, client: TestClient) -> None:
        client.post("/api/score", json=make_request(amount=2_000.0).model_dump(mode="json"))
        response = client.get("/api/alerts")
        assert response.status_code == 200
        assert response.json()["total"] >= 1

    def test_disposition_round_trip(self, client: TestClient) -> None:
        request = make_request(amount=2_000.0, txn_id="disp-1")
        client.post("/api/score", json=request.model_dump(mode="json"))
        response = client.post(
            "/api/alerts/disp-1/disposition",
            json={"disposition": "confirmed_fraud", "analyst": "abhinav"},
        )
        assert response.status_code == 200
        assert response.json()["disposition"] == "confirmed_fraud"

    def test_disposition_on_unknown_alert_is_404(self, client: TestClient) -> None:
        response = client.post(
            "/api/alerts/nope/disposition", json={"disposition": "false_positive"}
        )
        assert response.status_code == 404

    def test_invalid_disposition_is_rejected(self, client: TestClient) -> None:
        request = make_request(amount=2_000.0, txn_id="disp-2")
        client.post("/api/score", json=request.model_dump(mode="json"))
        response = client.post(
            "/api/alerts/disp-2/disposition", json={"disposition": "probably_fine"}
        )
        assert response.status_code == 422

    def test_health_reports_ready(self, client: TestClient) -> None:
        body = client.get("/api/health").json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True

    def test_rules_endpoint_lists_features_per_rule(self, client: TestClient) -> None:
        body = client.get("/api/rules").json()
        assert len(body["rules"]) >= 5
        assert all("features" in r for r in body["rules"])

    def test_batch_endpoint(self, client: TestClient) -> None:
        payload = {
            "transactions": [
                make_request(offset_minutes=i, txn_id=f"bt{i}").model_dump(mode="json")
                for i in range(3)
            ]
        }
        response = client.post("/api/score/batch", json=payload)
        assert response.status_code == 200
        assert response.json()["count"] == 3

    def test_empty_batch_is_rejected(self, client: TestClient) -> None:
        assert client.post("/api/score/batch", json={"transactions": []}).status_code == 422


class TestRuleLoading:
    def test_service_loads_rules_from_the_working_directory(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Mirrors the container: rules live under WORKDIR, not beside the
        # installed package. A loader that derives the path from __file__ picks
        # up the source tree's copy instead, and in a real non-editable install
        # finds nothing and scores with an empty rule set.
        source = Path(__file__).resolve().parents[2] / "config" / "rules.yaml"
        (tmp_path / "config").mkdir()
        (tmp_path / "config" / "rules.yaml").write_text(
            source.read_text(encoding="utf-8").replace('"2026.08.1"', '"from-workdir"'),
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)

        engine = _load_rules(Settings(_env_file=None))

        assert engine.version == "from-workdir"
        assert len(engine.rules) > 0


def _event(service: FraudLensService, request: TransactionRequest):
    from fraudlens.api.service import _to_event

    return _to_event(request)
