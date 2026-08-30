"""Deterministic rule engine.

Why a rule layer exists alongside the model
-------------------------------------------
Banks do not replace rules with ML; they run both. Rules provide things a model
cannot: a hard stop that fires with certainty on a known-bad pattern, an
immediate response to a new attack without waiting for a retraining cycle, and a
decision an investigator can explain to a regulator in one sentence.

The model provides what rules cannot: a graded assessment that generalises to
patterns nobody wrote down.

Rules as data, not code
-----------------------
Conditions are a small declarative structure evaluated by
:func:`evaluate_condition` — deliberately *not* ``eval`` on a string. Three
reasons: a YAML file that can execute arbitrary Python is a remote code
execution vector; a rule change should be reviewable by a fraud analyst who does
not write Python; and every rule carries a version, so a historical decision can
be reproduced exactly by replaying the rule set that was live at the time.

Condition grammar
-----------------
A leaf compares one feature to a constant::

    {feature: txn_count_1h, op: gte, value: 8}

Composites nest arbitrarily::

    {all: [<condition>, ...]}
    {any: [<condition>, ...]}
    {not: <condition>}
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

import yaml


class RuleAction(StrEnum):
    """What a firing rule asks the policy layer to do.

    ``BLOCK`` and ``ALLOW`` are decisive and short-circuit scoring. ``REVIEW``
    and ``FLAG`` are advisory: they raise the decision band but let the model
    score contribute.
    """

    BLOCK = "block"
    REVIEW = "review"
    FLAG = "flag"
    ALLOW = "allow"


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RuleDefinitionError(ValueError):
    """Raised when a rule definition is malformed."""


# ---------------------------------------------------------------------------
# Condition evaluation
# ---------------------------------------------------------------------------

_COMPARATORS: Final[dict[str, Callable[[Any, Any], bool]]] = {
    "gt": operator.gt,
    "gte": operator.ge,
    "lt": operator.lt,
    "lte": operator.le,
    "eq": operator.eq,
    "ne": operator.ne,
}

_COMPOSITES: Final[frozenset[str]] = frozenset({"all", "any", "not"})


def evaluate_condition(condition: Mapping[str, Any], features: Mapping[str, float]) -> bool:
    """Evaluate a declarative condition against a feature vector.

    A missing feature evaluates to ``False`` rather than raising: a rule
    referencing a feature that a particular pipeline version does not produce
    should be inert, not fatal to the whole scoring path.

    A *malformed* condition, by contrast, raises. The distinction matters: an
    inert rule is a configuration gap, a malformed one is a bug, and the YAML
    is easy to get subtly wrong (``when:`` written as a list rather than a
    mapping is the common slip).
    """
    _require_mapping(condition)

    if "all" in condition:
        return all(evaluate_condition(c, features) for c in _as_sequence(condition["all"], "all"))
    if "any" in condition:
        return any(evaluate_condition(c, features) for c in _as_sequence(condition["any"], "any"))
    if "not" in condition:
        return not evaluate_condition(condition["not"], features)

    return _evaluate_leaf(condition, features)


def _require_mapping(condition: object) -> None:
    """Reject a condition that is not a mapping.

    Typed as ``object`` rather than ``Mapping`` on purpose. Conditions are parsed
    from YAML, so at runtime this really can be a list, a string or ``None`` --
    the annotation on :func:`evaluate_condition` documents the intended contract,
    and this is where it is actually enforced. Writing ``when:`` as a YAML list
    instead of a mapping is the common slip.
    """
    if not isinstance(condition, Mapping):
        msg = (
            f"condition must be a mapping, got {type(condition).__name__}: {condition!r}. "
            f"A 'when:' clause written as a YAML list needs to be a single mapping, "
            f"or wrapped in {{all: [...]}}."
        )
        raise RuleDefinitionError(msg)


def _as_sequence(value: Any, key: str) -> Sequence[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        msg = f"{key!r} must contain a list of conditions, got {type(value).__name__}"
        raise RuleDefinitionError(msg)
    return value


def _evaluate_leaf(condition: Mapping[str, Any], features: Mapping[str, float]) -> bool:
    try:
        name = condition["feature"]
        op = condition["op"]
    except KeyError as exc:
        msg = f"condition must have 'feature' and 'op' keys, got {dict(condition)!r}"
        raise RuleDefinitionError(msg) from exc

    if name not in features:
        return False
    actual = features[name]

    # NaN comparisons are always False, which silently disables a rule. Being
    # explicit about it keeps the behaviour intentional rather than incidental.
    if isinstance(actual, float) and actual != actual:
        return False

    if op == "between":
        bounds = condition.get("value")
        if not isinstance(bounds, Sequence) or len(bounds) != 2:
            msg = f"'between' needs a two-element value, got {bounds!r}"
            raise RuleDefinitionError(msg)
        low, high = bounds
        return bool(low <= actual <= high)

    if op == "in":
        allowed = condition.get("value")
        if not isinstance(allowed, Sequence):
            msg = f"'in' needs a list value, got {allowed!r}"
            raise RuleDefinitionError(msg)
        return actual in allowed

    comparator = _COMPARATORS.get(op)
    if comparator is None:
        msg = f"unknown operator {op!r}; valid: {sorted([*_COMPARATORS, 'between', 'in'])}"
        raise RuleDefinitionError(msg)
    return bool(comparator(actual, condition.get("value")))


def referenced_features(condition: Mapping[str, Any]) -> set[str]:
    """Every feature name a condition depends on.

    Used to validate a rule set against the pipeline's feature contract at load
    time, so a typo surfaces on startup rather than as a rule that never fires.
    """
    for key in _COMPOSITES:
        if key in condition:
            children = condition[key]
            if key == "not":
                return referenced_features(children)
            return {f for c in _as_sequence(children, key) for f in referenced_features(c)}
    name = condition.get("feature")
    return {str(name)} if name is not None else set()


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Rule:
    """One versioned, auditable decision rule."""

    id: str
    name: str
    action: RuleAction
    condition: Mapping[str, Any]
    severity: Severity = Severity.MEDIUM
    version: int = 1
    description: str = ""
    reason_code: str = ""
    enabled: bool = True

    def matches(self, features: Mapping[str, float]) -> bool:
        return self.enabled and evaluate_condition(self.condition, features)

    @property
    def audit_key(self) -> str:
        """Stable identifier recorded in the decision log, e.g. ``R-VEL-001@v2``."""
        return f"{self.id}@v{self.version}"

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Rule:
        missing = {"id", "name", "action", "when"} - set(raw)
        if missing:
            msg = f"rule is missing required keys: {sorted(missing)}"
            raise RuleDefinitionError(msg)
        try:
            action = RuleAction(str(raw["action"]).lower())
        except ValueError as exc:
            msg = f"rule {raw['id']!r}: unknown action {raw['action']!r}"
            raise RuleDefinitionError(msg) from exc
        try:
            severity = Severity(str(raw.get("severity", "medium")).lower())
        except ValueError as exc:
            msg = f"rule {raw['id']!r}: unknown severity {raw['severity']!r}"
            raise RuleDefinitionError(msg) from exc

        return cls(
            id=str(raw["id"]),
            name=str(raw["name"]),
            action=action,
            condition=raw["when"],
            severity=severity,
            version=int(raw.get("version", 1)),
            description=str(raw.get("description", "")),
            reason_code=str(raw.get("reason_code", "")),
            enabled=bool(raw.get("enabled", True)),
        )


@dataclass(frozen=True, slots=True)
class RuleHit:
    """A rule that fired on a specific transaction."""

    rule_id: str
    rule_version: int
    name: str
    action: RuleAction
    severity: Severity
    reason_code: str
    description: str

    @property
    def audit_key(self) -> str:
        return f"{self.rule_id}@v{self.rule_version}"


@dataclass(frozen=True, slots=True)
class RuleEvaluation:
    """The outcome of running a whole rule set against one transaction."""

    hits: tuple[RuleHit, ...] = ()
    rule_set_version: str = "unknown"

    @property
    def blocked(self) -> bool:
        return any(h.action is RuleAction.BLOCK for h in self.hits)

    @property
    def allowed(self) -> bool:
        """An explicit allow-list hit, which suppresses alerting."""
        return any(h.action is RuleAction.ALLOW for h in self.hits)

    @property
    def review_requested(self) -> bool:
        return any(h.action is RuleAction.REVIEW for h in self.hits)

    @property
    def max_severity(self) -> Severity | None:
        order = [Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]
        present = [h.severity for h in self.hits]
        return max(present, key=order.index) if present else None

    @property
    def audit_keys(self) -> tuple[str, ...]:
        """Recorded verbatim in the decision log for reproducibility."""
        return tuple(h.audit_key for h in self.hits)

    @property
    def reason_codes(self) -> tuple[str, ...]:
        return tuple(h.reason_code for h in self.hits if h.reason_code)


@dataclass(slots=True)
class RuleEngine:
    """Evaluates an ordered rule set against feature vectors."""

    rules: tuple[Rule, ...] = ()
    version: str = "unknown"
    _enabled: tuple[Rule, ...] = field(init=False, repr=False, default=())

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for rule in self.rules:
            if rule.id in seen:
                msg = f"duplicate rule id {rule.id!r}"
                raise RuleDefinitionError(msg)
            seen.add(rule.id)
        self._enabled = tuple(r for r in self.rules if r.enabled)

    def evaluate(self, features: Mapping[str, float]) -> RuleEvaluation:
        """Run every enabled rule. All hits are collected, not just the first.

        Collecting every hit matters for explanation quality: an analyst should
        see all four reasons a transaction was stopped, not whichever rule
        happened to be listed first.
        """
        hits = tuple(
            RuleHit(
                rule_id=rule.id,
                rule_version=rule.version,
                name=rule.name,
                action=rule.action,
                severity=rule.severity,
                reason_code=rule.reason_code,
                description=rule.description,
            )
            for rule in self._enabled
            if rule.matches(features)
        )
        return RuleEvaluation(hits=hits, rule_set_version=self.version)

    def referenced_features(self) -> set[str]:
        return {f for rule in self.rules for f in referenced_features(rule.condition)}

    def validate_against(self, known_features: Sequence[str]) -> None:
        """Fail fast if any rule references a feature the pipeline does not emit."""
        unknown = self.referenced_features() - set(known_features)
        if unknown:
            msg = (
                f"rule set references features the pipeline does not produce: "
                f"{sorted(unknown)}. A rule on a non-existent feature never fires."
            )
            raise RuleDefinitionError(msg)

    @classmethod
    def from_yaml(cls, path: Path) -> RuleEngine:
        """Load a rule set from a YAML file."""
        if not path.exists():
            msg = f"rule file not found: {path}"
            raise FileNotFoundError(msg)
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> RuleEngine:
        entries = raw.get("rules", [])
        if not isinstance(entries, Sequence):
            msg = "'rules' must be a list"
            raise RuleDefinitionError(msg)
        return cls(
            rules=tuple(Rule.from_dict(entry) for entry in entries),
            version=str(raw.get("version", "unknown")),
        )
