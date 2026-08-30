"""Tests for the declarative rule engine.

Includes tests against the *shipped* rule set in ``config/rules.yaml``, not only
synthetic ones. A rule that references a misspelled feature never fires and
never errors, so the shipped configuration needs its own guard.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from fraudlens.features.pipeline import FeaturePipeline
from fraudlens.rules.engine import (
    Rule,
    RuleAction,
    RuleDefinitionError,
    RuleEngine,
    Severity,
    evaluate_condition,
    referenced_features,
)

RULES_PATH = Path(__file__).resolve().parents[2] / "config" / "rules.yaml"

FEATURES: dict[str, float] = {
    "amt": 100.0,
    "txn_count_1h": 2.0,
    "txn_count_24h": 5.0,
    "is_night": 0.0,
    "is_new_merchant": 1.0,
    "dist_home_merch_km": 12.0,
    "is_impossible_travel": 0.0,
    "amt_ratio_mean_30d": 1.5,
    "card_history_count": 20.0,
    "distinct_merchants_24h": 3.0,
    "dist_prev_txn_km": 5.0,
}


class TestLeafConditions:
    @pytest.mark.parametrize(
        ("op", "value", "expected"),
        [
            ("gt", 50, True),
            ("gt", 100, False),
            ("gte", 100, True),
            ("lt", 200, True),
            ("lte", 100, True),
            ("eq", 100, True),
            ("ne", 100, False),
        ],
    )
    def test_comparators(self, op: str, value: float, expected: bool) -> None:
        assert evaluate_condition({"feature": "amt", "op": op, "value": value}, FEATURES) is (
            expected
        )

    def test_between_is_inclusive(self) -> None:
        between = {"feature": "amt", "op": "between", "value": [100, 200]}
        assert evaluate_condition(between, FEATURES)
        assert not evaluate_condition(
            {"feature": "amt", "op": "between", "value": [101, 200]}, FEATURES
        )

    def test_in_membership(self) -> None:
        assert evaluate_condition({"feature": "amt", "op": "in", "value": [50, 100]}, FEATURES)
        assert not evaluate_condition({"feature": "amt", "op": "in", "value": [1, 2]}, FEATURES)

    def test_missing_feature_is_inert_not_fatal(self) -> None:
        # A rule referencing a feature this pipeline version does not emit must
        # not take down the whole scoring path.
        assert not evaluate_condition({"feature": "nonexistent", "op": "gt", "value": 0}, FEATURES)

    def test_nan_feature_does_not_fire(self) -> None:
        assert not evaluate_condition(
            {"feature": "age", "op": "gt", "value": 0}, {"age": float("nan")}
        )

    def test_unknown_operator_is_rejected(self) -> None:
        with pytest.raises(RuleDefinitionError, match="unknown operator"):
            evaluate_condition({"feature": "amt", "op": "approximately", "value": 1}, FEATURES)

    def test_malformed_leaf_is_rejected(self) -> None:
        with pytest.raises(RuleDefinitionError, match="must have 'feature' and 'op'"):
            evaluate_condition({"feature": "amt"}, FEATURES)

    def test_between_requires_two_bounds(self) -> None:
        with pytest.raises(RuleDefinitionError, match="two-element"):
            evaluate_condition({"feature": "amt", "op": "between", "value": [1]}, FEATURES)


class TestCompositeConditions:
    def test_all_requires_every_child(self) -> None:
        condition: dict[str, Any] = {
            "all": [
                {"feature": "amt", "op": "gte", "value": 50},
                {"feature": "is_new_merchant", "op": "eq", "value": 1},
            ]
        }
        assert evaluate_condition(condition, FEATURES)

        condition["all"].append({"feature": "is_night", "op": "eq", "value": 1})
        assert not evaluate_condition(condition, FEATURES)

    def test_any_requires_one_child(self) -> None:
        condition = {
            "any": [
                {"feature": "is_night", "op": "eq", "value": 1},
                {"feature": "is_new_merchant", "op": "eq", "value": 1},
            ]
        }
        assert evaluate_condition(condition, FEATURES)

    def test_not_inverts(self) -> None:
        negated = {"not": {"feature": "is_night", "op": "eq", "value": 1}}
        assert evaluate_condition(negated, FEATURES)

    def test_deep_nesting(self) -> None:
        condition = {
            "all": [
                {"feature": "amt", "op": "gt", "value": 10},
                {
                    "any": [
                        {"feature": "is_night", "op": "eq", "value": 1},
                        {"not": {"feature": "is_impossible_travel", "op": "eq", "value": 1}},
                    ]
                },
            ]
        }
        assert evaluate_condition(condition, FEATURES)

    def test_non_list_composite_is_rejected(self) -> None:
        with pytest.raises(RuleDefinitionError, match="must contain a list"):
            evaluate_condition({"all": {"feature": "amt", "op": "gt", "value": 1}}, FEATURES)


class TestReferencedFeatures:
    def test_collects_from_nested_conditions(self) -> None:
        condition = {
            "all": [
                {"feature": "amt", "op": "gt", "value": 1},
                {"any": [{"feature": "is_night", "op": "eq", "value": 1}]},
                {"not": {"feature": "txn_count_1h", "op": "gt", "value": 3}},
            ]
        }
        assert referenced_features(condition) == {"amt", "is_night", "txn_count_1h"}


class TestRule:
    def test_from_dict_builds_a_rule(self) -> None:
        rule = Rule.from_dict(
            {
                "id": "R-TEST-001",
                "name": "Test",
                "action": "review",
                "severity": "high",
                "version": 3,
                "reason_code": "R99",
                "when": {"feature": "amt", "op": "gt", "value": 50},
            }
        )
        assert rule.action is RuleAction.REVIEW
        assert rule.severity is Severity.HIGH
        assert rule.audit_key == "R-TEST-001@v3"
        assert rule.matches(FEATURES)

    def test_missing_keys_are_rejected(self) -> None:
        with pytest.raises(RuleDefinitionError, match="missing required keys"):
            Rule.from_dict({"id": "x", "name": "y"})

    def test_unknown_action_is_rejected(self) -> None:
        with pytest.raises(RuleDefinitionError, match="unknown action"):
            Rule.from_dict(
                {"id": "x", "name": "y", "action": "detonate", "when": {}},
            )

    def test_disabled_rule_never_matches(self) -> None:
        rule = Rule.from_dict(
            {
                "id": "R-OFF",
                "name": "Off",
                "action": "block",
                "enabled": False,
                "when": {"feature": "amt", "op": "gt", "value": 0},
            }
        )
        assert not rule.matches(FEATURES)


class TestRuleEngine:
    def _engine(self) -> RuleEngine:
        return RuleEngine.from_dict(
            {
                "version": "test.1",
                "rules": [
                    {
                        "id": "R-A",
                        "name": "Blocker",
                        "action": "block",
                        "severity": "critical",
                        "reason_code": "RA",
                        "when": {"feature": "is_impossible_travel", "op": "eq", "value": 1},
                    },
                    {
                        "id": "R-B",
                        "name": "Reviewer",
                        "action": "review",
                        "severity": "medium",
                        "reason_code": "RB",
                        "when": {"feature": "amt", "op": "gte", "value": 50},
                    },
                    {
                        "id": "R-C",
                        "name": "Flagger",
                        "action": "flag",
                        "severity": "low",
                        "reason_code": "RC",
                        "when": {"feature": "is_new_merchant", "op": "eq", "value": 1},
                    },
                ],
            }
        )

    def test_collects_every_hit_not_just_the_first(self) -> None:
        # An analyst should see all reasons a transaction was stopped.
        result = self._engine().evaluate(FEATURES)
        assert {h.rule_id for h in result.hits} == {"R-B", "R-C"}
        assert not result.blocked
        assert result.review_requested

    def test_block_is_detected(self) -> None:
        result = self._engine().evaluate({**FEATURES, "is_impossible_travel": 1.0})
        assert result.blocked
        assert result.max_severity is Severity.CRITICAL

    def test_no_hits_is_a_clean_evaluation(self) -> None:
        result = self._engine().evaluate({**FEATURES, "amt": 1.0, "is_new_merchant": 0.0})
        assert result.hits == ()
        assert result.max_severity is None
        assert not result.blocked

    def test_audit_keys_carry_versions(self) -> None:
        result = self._engine().evaluate(FEATURES)
        assert all("@v" in key for key in result.audit_keys)

    def test_reason_codes_are_exposed(self) -> None:
        result = self._engine().evaluate(FEATURES)
        assert set(result.reason_codes) == {"RB", "RC"}

    def test_duplicate_rule_ids_are_rejected(self) -> None:
        with pytest.raises(RuleDefinitionError, match="duplicate rule id"):
            RuleEngine.from_dict(
                {
                    "rules": [
                        {"id": "R-X", "name": "a", "action": "flag", "when": {}},
                        {"id": "R-X", "name": "b", "action": "flag", "when": {}},
                    ]
                }
            )

    def test_validate_against_catches_unknown_features(self) -> None:
        engine = RuleEngine.from_dict(
            {
                "rules": [
                    {
                        "id": "R-TYPO",
                        "name": "Typo",
                        "action": "flag",
                        "when": {"feature": "txn_cont_1h", "op": "gt", "value": 1},
                    }
                ]
            }
        )
        with pytest.raises(RuleDefinitionError, match="does not produce"):
            engine.validate_against(FeaturePipeline.FEATURE_NAMES)


class TestShippedRuleSet:
    """Guards on the rule set that actually ships."""

    def test_loads(self) -> None:
        engine = RuleEngine.from_yaml(RULES_PATH)
        assert len(engine.rules) >= 5
        assert engine.version != "unknown"

    def test_every_referenced_feature_exists(self) -> None:
        # The critical guard: a misspelled feature name produces a rule that
        # silently never fires.
        RuleEngine.from_yaml(RULES_PATH).validate_against(FeaturePipeline.FEATURE_NAMES)

    def test_every_rule_has_a_reason_code(self) -> None:
        # Reason codes are what the analyst console renders; a rule without one
        # produces an unexplained decision.
        for rule in RuleEngine.from_yaml(RULES_PATH).rules:
            assert rule.reason_code, f"{rule.id} has no reason_code"

    def test_every_rule_has_a_description(self) -> None:
        for rule in RuleEngine.from_yaml(RULES_PATH).rules:
            assert len(rule.description) > 20, f"{rule.id} needs a real description"

    def test_impossible_travel_blocks(self) -> None:
        engine = RuleEngine.from_yaml(RULES_PATH)
        result = engine.evaluate(
            {**FEATURES, "is_impossible_travel": 1.0, "dist_prev_txn_km": 3_000.0}
        )
        assert result.blocked

    def test_card_testing_burst_is_caught(self) -> None:
        engine = RuleEngine.from_yaml(RULES_PATH)
        result = engine.evaluate(
            {
                **FEATURES,
                "txn_count_1h": 6.0,
                "distinct_merchants_24h": 5.0,
                "amt": 3.0,
            }
        )
        assert "R-VEL-001" in {h.rule_id for h in result.hits}

    def test_ordinary_transaction_is_suppressed_by_the_allow_rule(self) -> None:
        engine = RuleEngine.from_yaml(RULES_PATH)
        result = engine.evaluate(
            {
                **FEATURES,
                "amt": 8.0,
                "is_new_merchant": 0.0,
                "txn_count_1h": 1.0,
            }
        )
        assert result.allowed
        assert not result.review_requested

    def test_rule_ids_follow_the_naming_convention(self) -> None:
        for rule in RuleEngine.from_yaml(RULES_PATH).rules:
            assert rule.id.startswith("R-"), f"{rule.id} breaks the R-XXX-NNN convention"
            assert len(rule.id.split("-")) == 3, f"{rule.id} breaks the R-XXX-NNN convention"
