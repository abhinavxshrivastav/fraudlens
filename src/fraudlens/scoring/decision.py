"""The decision engine: rules, model score, and policy combined into one action.

Three layers, in order:

1. **Rules** (:mod:`fraudlens.rules.engine`) -- deterministic, auditable,
   instantly changeable. A ``block`` short-circuits everything else.
2. **Model** -- a calibrated probability of fraud.
3. **Policy** -- maps the probability, plus advisory rule hits and transaction
   value, onto a :class:`RiskBand`.

Every decision produces a :class:`Decision` record carrying the model version,
the rule-set version, the exact rule instances that fired, a hash of the feature
vector, and the reason codes shown to the analyst. That record is what makes a
historical decision reproducible months later -- the concrete meaning of model
governance in a regulated setting.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol

from fraudlens.rules.engine import RuleEvaluation

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from fraudlens.features.pipeline import TransactionEvent


class RiskBand(StrEnum):
    """The action taken on a transaction.

    Four bands rather than a binary decision, because the operational responses
    genuinely differ: a step-up challenge costs the customer seconds, while a
    block costs a support call and possibly the relationship.
    """

    APPROVE = "approve"
    CHALLENGE = "challenge"
    REVIEW = "review"
    BLOCK = "block"

    @property
    def creates_alert(self) -> bool:
        """Whether this band consumes analyst capacity."""
        return self in {RiskBand.REVIEW, RiskBand.BLOCK}


class ScoringModel(Protocol):
    """What the decision engine needs from a model.

    Narrow on purpose: the engine is testable with a fake, and the concrete
    model implementation stays swappable.
    """

    @property
    def version(self) -> str: ...

    def predict_proba(self, features: Mapping[str, float]) -> float:
        """Return a calibrated probability of fraud in [0, 1]."""
        ...


@dataclass(frozen=True, slots=True)
class PolicyThresholds:
    """Probability cut-points between bands.

    Defaults are placeholders. The shipped values are chosen by minimising
    expected cost on the validation fold (see :mod:`fraudlens.evaluation`) and
    then frozen into the model artefact -- never tuned on test.
    """

    challenge: float = 0.30
    review: float = 0.60
    block: float = 0.95

    def __post_init__(self) -> None:
        if not 0.0 <= self.challenge <= self.review <= self.block <= 1.0:
            msg = (
                f"thresholds must be ordered and within [0, 1]: "
                f"challenge={self.challenge}, review={self.review}, block={self.block}"
            )
            raise ValueError(msg)

    def band_for(self, probability: float) -> RiskBand:
        if probability >= self.block:
            return RiskBand.BLOCK
        if probability >= self.review:
            return RiskBand.REVIEW
        if probability >= self.challenge:
            return RiskBand.CHALLENGE
        return RiskBand.APPROVE


@dataclass(frozen=True, slots=True)
class Decision:
    """A complete, auditable record of one scoring decision."""

    transaction_id: str
    decided_at: datetime
    band: RiskBand
    probability: float
    rule_evaluation: RuleEvaluation
    reason_codes: tuple[str, ...]
    explanation: tuple[str, ...]
    model_version: str
    rule_set_version: str
    policy_version: str
    feature_hash: str
    latency_ms: float
    amount: float = 0.0

    @property
    def creates_alert(self) -> bool:
        return self.band.creates_alert

    @property
    def decided_by_rule(self) -> bool:
        """True when a rule block determined the outcome, bypassing the model."""
        return self.rule_evaluation.blocked

    def to_audit_record(self) -> dict[str, Any]:
        """Flatten into the shape written to the decision log.

        Everything needed to reproduce the decision is present: which model,
        which rules at which versions, which policy, and a fingerprint of the
        exact feature vector.
        """
        return {
            "transaction_id": self.transaction_id,
            "decided_at": self.decided_at.isoformat(),
            "band": str(self.band),
            "probability": round(self.probability, 6),
            "amount": self.amount,
            "creates_alert": self.creates_alert,
            "decided_by_rule": self.decided_by_rule,
            "rules_fired": list(self.rule_evaluation.audit_keys),
            "reason_codes": list(self.reason_codes),
            "explanation": list(self.explanation),
            "model_version": self.model_version,
            "rule_set_version": self.rule_set_version,
            "policy_version": self.policy_version,
            "feature_hash": self.feature_hash,
            "latency_ms": round(self.latency_ms, 3),
        }


def hash_features(features: Mapping[str, float]) -> str:
    """Stable fingerprint of a feature vector.

    Recorded with every decision so that a stored decision can be tied to the
    exact inputs that produced it, without persisting the full vector for every
    transaction. Keys are sorted and floats rounded so the hash is reproducible
    across runs and platforms.
    """
    canonical = json.dumps(
        {k: round(float(v), 6) for k, v in sorted(features.items())},
        separators=(",", ":"),
        allow_nan=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


@dataclass(slots=True)
class DecisionEngine:
    """Combines the rule engine, the model and the policy into one decision."""

    model: ScoringModel
    rule_engine: Any  # RuleEngine; typed loosely to avoid a circular import
    thresholds: PolicyThresholds = field(default_factory=PolicyThresholds)
    policy_version: str = "1.0.0"

    def decide(
        self,
        event: TransactionEvent,
        features: Mapping[str, float],
        *,
        explanation: Sequence[str] = (),
        extra_reason_codes: Sequence[str] = (),
    ) -> Decision:
        """Score one transaction and return the full audit record."""
        started = time.perf_counter()

        rule_result: RuleEvaluation = self.rule_engine.evaluate(features)

        if rule_result.blocked:
            # A deterministic block does not need a probability, and skipping
            # the model keeps the hard-stop path fast and independent of model
            # availability. The probability is recorded as 1.0 to mean
            # "certain by rule", which the `decided_by_rule` flag disambiguates.
            probability = 1.0
            band = RiskBand.BLOCK
        else:
            probability = float(self.model.predict_proba(features))
            band = self._apply_policy(probability, rule_result)

        codes = (*rule_result.reason_codes, *extra_reason_codes)
        return Decision(
            transaction_id=event.transaction_id,
            decided_at=datetime.now(UTC),
            band=band,
            probability=probability,
            rule_evaluation=rule_result,
            reason_codes=tuple(dict.fromkeys(codes)),  # de-duplicate, keep order
            explanation=tuple(explanation),
            model_version=self.model.version,
            rule_set_version=rule_result.rule_set_version,
            policy_version=self.policy_version,
            feature_hash=hash_features(features),
            latency_ms=(time.perf_counter() - started) * 1000.0,
            amount=event.amount,
        )

    def _apply_policy(self, probability: float, rules: RuleEvaluation) -> RiskBand:
        """Map probability plus advisory rule hits onto a band.

        Advisory rules escalate but never de-escalate below what the model
        warrants; an explicit allow-list hit is the one exception, and it cannot
        rescue a transaction the model considers high risk.
        """
        band = self.thresholds.band_for(probability)

        if rules.review_requested and band is not RiskBand.BLOCK:
            band = max(band, RiskBand.REVIEW, key=_band_rank)

        # Suppression protects analyst capacity on the lowest-risk traffic, but
        # only where the model has not independently raised concern.
        if (
            rules.allowed
            and band in {RiskBand.CHALLENGE, RiskBand.REVIEW}
            and probability < self.thresholds.review
        ):
            band = RiskBand.APPROVE
        return band


_BAND_ORDER: dict[RiskBand, int] = {
    RiskBand.APPROVE: 0,
    RiskBand.CHALLENGE: 1,
    RiskBand.REVIEW: 2,
    RiskBand.BLOCK: 3,
}


def _band_rank(band: RiskBand) -> int:
    return _BAND_ORDER[band]


def decision_to_json(decision: Decision) -> str:
    """Serialise a decision for the append-only audit log."""
    return json.dumps(decision.to_audit_record(), separators=(",", ":"))


__all__ = [
    "Decision",
    "DecisionEngine",
    "PolicyThresholds",
    "RiskBand",
    "ScoringModel",
    "asdict",
    "decision_to_json",
    "hash_features",
]
