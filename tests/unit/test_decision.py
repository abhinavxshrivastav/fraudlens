"""Tests for the decision engine: rules, model and policy combined."""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from fraudlens.features.pipeline import TransactionEvent
from fraudlens.rules.engine import RuleEngine
from fraudlens.scoring.decision import (
    Decision,
    DecisionEngine,
    PolicyThresholds,
    RiskBand,
    decision_to_json,
    hash_features,
)

FEATURES: dict[str, float] = {
    "amt": 120.0,
    "txn_count_1h": 1.0,
    "is_night": 0.0,
    "is_new_merchant": 0.0,
    "is_impossible_travel": 0.0,
    "dist_prev_txn_km": 4.0,
    "distinct_merchants_24h": 1.0,
    "amt_ratio_mean_30d": 1.2,
    "card_history_count": 30.0,
    "dist_home_merch_km": 10.0,
    "txn_count_24h": 2.0,
}


class FakeModel:
    """A model stub returning a fixed probability."""

    def __init__(self, probability: float, version: str = "fake-1.0") -> None:
        self._probability = probability
        self._version = version
        self.calls = 0

    @property
    def version(self) -> str:
        return self._version

    def predict_proba(self, features: dict[str, float]) -> float:
        self.calls += 1
        return self._probability


def make_event(amount: float = 120.0, txn_id: str = "txn-1") -> TransactionEvent:
    return TransactionEvent(
        timestamp=datetime(2020, 8, 1, 14, 30),
        card=4_000_000_000_000_000,
        amount=amount,
        merchant="merchant_a",
        category="grocery_pos",
        home_lat=40.7,
        home_lon=-74.0,
        merch_lat=40.8,
        merch_lon=-74.1,
        transaction_id=txn_id,
    )


def make_engine(probability: float, rules: dict | None = None) -> DecisionEngine:
    rule_set = RuleEngine.from_dict(rules or {"version": "test.1", "rules": []})
    return DecisionEngine(model=FakeModel(probability), rule_engine=rule_set)


class TestPolicyThresholds:
    def test_bands_by_probability(self) -> None:
        t = PolicyThresholds(challenge=0.3, review=0.6, block=0.95)
        assert t.band_for(0.05) is RiskBand.APPROVE
        assert t.band_for(0.30) is RiskBand.CHALLENGE
        assert t.band_for(0.75) is RiskBand.REVIEW
        assert t.band_for(0.99) is RiskBand.BLOCK

    def test_boundaries_are_inclusive_at_the_lower_edge(self) -> None:
        t = PolicyThresholds(challenge=0.3, review=0.6, block=0.95)
        assert t.band_for(0.2999) is RiskBand.APPROVE
        assert t.band_for(0.3) is RiskBand.CHALLENGE

    def test_unordered_thresholds_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be ordered"):
            PolicyThresholds(challenge=0.8, review=0.5, block=0.9)

    def test_out_of_range_thresholds_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="within"):
            PolicyThresholds(challenge=0.1, review=0.5, block=1.5)


class TestBandSemantics:
    @pytest.mark.parametrize(
        ("band", "alerts"),
        [
            (RiskBand.APPROVE, False),
            (RiskBand.CHALLENGE, False),
            (RiskBand.REVIEW, True),
            (RiskBand.BLOCK, True),
        ],
    )
    def test_only_review_and_block_consume_analyst_capacity(
        self, band: RiskBand, alerts: bool
    ) -> None:
        assert band.creates_alert is alerts


class TestDecisionEngine:
    def test_low_score_approves(self) -> None:
        decision = make_engine(0.02).decide(make_event(), FEATURES)
        assert decision.band is RiskBand.APPROVE
        assert not decision.creates_alert
        assert not decision.decided_by_rule

    def test_high_score_blocks(self) -> None:
        decision = make_engine(0.99).decide(make_event(), FEATURES)
        assert decision.band is RiskBand.BLOCK
        assert decision.creates_alert

    def test_rule_block_short_circuits_the_model(self) -> None:
        # A deterministic hard stop must not depend on model availability, so
        # the model is never called.
        rules = {
            "version": "test.1",
            "rules": [
                {
                    "id": "R-GEO-001",
                    "name": "Impossible travel",
                    "action": "block",
                    "severity": "critical",
                    "reason_code": "R12",
                    "when": {"feature": "is_impossible_travel", "op": "eq", "value": 1},
                }
            ],
        }
        engine = make_engine(0.01, rules)
        decision = engine.decide(make_event(), {**FEATURES, "is_impossible_travel": 1.0})

        assert decision.band is RiskBand.BLOCK
        assert decision.decided_by_rule
        assert engine.model.calls == 0  # type: ignore[attr-defined]
        assert "R12" in decision.reason_codes

    def test_advisory_rule_escalates_to_review(self) -> None:
        rules = {
            "version": "test.1",
            "rules": [
                {
                    "id": "R-VEL-001",
                    "name": "Velocity",
                    "action": "review",
                    "reason_code": "R07",
                    "when": {"feature": "txn_count_1h", "op": "gte", "value": 1},
                }
            ],
        }
        # Model alone would only reach CHALLENGE.
        decision = make_engine(0.35, rules).decide(make_event(), FEATURES)
        assert decision.band is RiskBand.REVIEW
        assert not decision.decided_by_rule

    def test_advisory_rule_never_de_escalates(self) -> None:
        rules = {
            "version": "test.1",
            "rules": [
                {
                    "id": "R-VEL-001",
                    "name": "Velocity",
                    "action": "review",
                    "reason_code": "R07",
                    "when": {"feature": "txn_count_1h", "op": "gte", "value": 1},
                }
            ],
        }
        decision = make_engine(0.98, rules).decide(make_event(), FEATURES)
        assert decision.band is RiskBand.BLOCK

    def test_allow_rule_suppresses_a_low_confidence_alert(self) -> None:
        rules = {
            "version": "test.1",
            "rules": [
                {
                    "id": "R-ALW-001",
                    "name": "Trusted",
                    "action": "allow",
                    "reason_code": "A01",
                    "when": {"feature": "amt", "op": "lt", "value": 500},
                }
            ],
        }
        decision = make_engine(0.35, rules).decide(make_event(), FEATURES)
        assert decision.band is RiskBand.APPROVE

    def test_allow_rule_cannot_rescue_a_high_risk_transaction(self) -> None:
        # Suppression protects analyst capacity; it must not override the model
        # where the model has independently raised serious concern.
        rules = {
            "version": "test.1",
            "rules": [
                {
                    "id": "R-ALW-001",
                    "name": "Trusted",
                    "action": "allow",
                    "reason_code": "A01",
                    "when": {"feature": "amt", "op": "lt", "value": 500},
                }
            ],
        }
        decision = make_engine(0.80, rules).decide(make_event(), FEATURES)
        assert decision.band is RiskBand.REVIEW


class TestAuditRecord:
    def test_carries_everything_needed_to_reproduce_the_decision(self) -> None:
        decision = make_engine(0.75).decide(make_event(), FEATURES)
        record = decision.to_audit_record()

        for key in (
            "transaction_id",
            "decided_at",
            "band",
            "probability",
            "model_version",
            "rule_set_version",
            "policy_version",
            "feature_hash",
            "rules_fired",
            "reason_codes",
            "latency_ms",
        ):
            assert key in record, f"audit record is missing {key!r}"

    def test_serialises_to_json(self) -> None:
        decision = make_engine(0.75).decide(make_event(), FEATURES)
        parsed = json.loads(decision_to_json(decision))
        assert parsed["band"] == "review"
        assert parsed["transaction_id"] == "txn-1"

    def test_records_latency(self) -> None:
        decision = make_engine(0.1).decide(make_event(), FEATURES)
        assert decision.latency_ms >= 0.0

    def test_reason_codes_are_deduplicated(self) -> None:
        decision = make_engine(0.1).decide(
            make_event(), FEATURES, extra_reason_codes=["R01", "R01", "R02"]
        )
        assert decision.reason_codes == ("R01", "R02")

    def test_is_immutable(self) -> None:
        decision = make_engine(0.1).decide(make_event(), FEATURES)
        with pytest.raises((AttributeError, TypeError)):
            decision.band = RiskBand.BLOCK  # type: ignore[misc]
        assert isinstance(decision, Decision)


class TestFeatureHash:
    def test_is_stable_across_calls(self) -> None:
        assert hash_features(FEATURES) == hash_features(FEATURES)

    def test_is_independent_of_key_order(self) -> None:
        reordered = dict(reversed(list(FEATURES.items())))
        assert hash_features(FEATURES) == hash_features(reordered)

    def test_changes_when_a_value_changes(self) -> None:
        assert hash_features(FEATURES) != hash_features({**FEATURES, "amt": 121.0})

    def test_tolerates_floating_point_noise(self) -> None:
        # Rounding keeps the hash reproducible across platforms without making
        # it insensitive to real differences.
        jittered = {**FEATURES, "amt": 120.0 + 1e-12}
        assert hash_features(FEATURES) == hash_features(jittered)

    def test_handles_nan(self) -> None:
        assert isinstance(hash_features({**FEATURES, "age_years": float("nan")}), str)
