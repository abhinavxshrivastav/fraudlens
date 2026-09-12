"""Turning SHAP contributions into reason codes an analyst can act on.

The gap this closes
-------------------
SHAP answers "which features moved this score, and by how much". That is the
right internal answer and the wrong external one. ``amt_ratio_mean_30d = 8.4,
contribution +0.31`` is not something an investigator can act on, quote to a
customer, or defend to a regulator.

A reason code is the same fact rendered as a claim about behaviour::

    R01 - Amount is 8.4x this card's 30-day average

This mirrors adverse-action reason codes in regulated lending, where an
institution must state *why* a decision went against someone in terms they can
understand and contest. It is the difference between a model that is
interpretable in principle and a system that is explainable in practice.

How it works
------------
SHAP supplies the ranking -- which features actually drove *this* decision,
rather than which are important in general. Each model feature maps to a
:class:`ReasonCodeSpec` carrying a template and a materiality test. A code is
emitted when the feature both pushed the score toward fraud and is materially
unusual, so codes describing ordinary values are never shown.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from fraudlens.features.pipeline import IMPOSSIBLE_SPEED_KMH


@dataclass(frozen=True, slots=True)
class ReasonCodeSpec:
    """Maps one feature to a human-readable explanation."""

    code: str
    feature: str
    template: str
    #: Materiality test. A feature can be the top SHAP contributor and still be
    #: unremarkable; without this, the top three codes on a routine transaction
    #: would read as accusations.
    is_material: Callable[[float], bool]
    category: str = "behavioural"
    #: How the raw feature value is transformed for display.
    display: Callable[[float], float] = float

    def message(self, value: float) -> str:
        """The explanation on its own, with no code prefix."""
        return self.template.format(value=self.display(value))

    def render(self, value: float) -> str:
        """Code and explanation together, for logs and plain-text output."""
        return f"{self.code} - {self.message(value)}"


def _gt(threshold: float) -> Callable[[float], bool]:
    return lambda v: v > threshold


def _is_true(v: float) -> bool:
    return v >= 0.5


#: The reason-code catalogue. Codes are stable identifiers: once published in a
#: decision record, a code's meaning must not change. Retire and supersede
#: rather than redefine.
REASON_CODES: Final[tuple[ReasonCodeSpec, ...]] = (
    ReasonCodeSpec(
        code="R01",
        feature="amt_ratio_mean_30d",
        template="Amount is {value:.1f}x this card's 30-day average",
        is_material=_gt(3.0),
        category="amount",
    ),
    ReasonCodeSpec(
        code="R02",
        feature="amt_zscore_30d",
        template="Amount is {value:.1f} standard deviations above this card's norm",
        is_material=_gt(3.0),
        category="amount",
    ),
    ReasonCodeSpec(
        code="R03",
        feature="amt_ratio_max_30d",
        template="Amount is {value:.1f}x the largest transaction on this card in 30 days",
        is_material=_gt(1.5),
        category="amount",
    ),
    ReasonCodeSpec(
        code="R04",
        feature="amt",
        template="High transaction value ({value:,.2f})",
        is_material=_gt(500.0),
        category="amount",
    ),
    ReasonCodeSpec(
        code="R05",
        feature="txn_count_1h",
        template="{value:.0f} transactions on this card in the past hour",
        is_material=_gt(3.0),
        category="velocity",
    ),
    ReasonCodeSpec(
        code="R06",
        feature="txn_count_24h",
        template="{value:.0f} transactions on this card in the past 24 hours",
        is_material=_gt(10.0),
        category="velocity",
    ),
    ReasonCodeSpec(
        code="R07",
        feature="distinct_merchants_24h",
        template="{value:.0f} distinct merchants in 24 hours, consistent with card testing",
        is_material=_gt(4.0),
        category="velocity",
    ),
    ReasonCodeSpec(
        code="R08",
        feature="amt_sum_1h",
        template="{value:,.2f} spent on this card in the past hour",
        is_material=_gt(1_000.0),
        category="velocity",
    ),
    ReasonCodeSpec(
        code="R09",
        feature="is_night",
        template="Transaction occurred overnight, when the cardholder is typically inactive",
        is_material=_is_true,
        category="temporal",
    ),
    ReasonCodeSpec(
        code="R10",
        feature="secs_since_prev_txn",
        template="Only {value:.0f} seconds since the previous transaction on this card",
        is_material=lambda v: 0 <= v < 120,
        category="temporal",
    ),
    ReasonCodeSpec(
        code="R11",
        feature="dist_home_merch_km",
        template="Merchant is {value:,.0f} km from the cardholder's registered address",
        is_material=_gt(500.0),
        category="geographic",
    ),
    ReasonCodeSpec(
        code="R12",
        feature="implied_speed_kmh",
        template=(
            "Implied travel speed of {value:,.0f} km/h since the previous transaction, "
            "which no journey could achieve"
        ),
        is_material=_gt(IMPOSSIBLE_SPEED_KMH),
        category="geographic",
    ),
    ReasonCodeSpec(
        code="R13",
        feature="dist_prev_txn_km",
        template="{value:,.0f} km from the previous transaction on this card",
        is_material=_gt(1_000.0),
        category="geographic",
    ),
    ReasonCodeSpec(
        code="R14",
        feature="is_new_merchant",
        template="First recorded transaction between this card and this merchant",
        is_material=_is_true,
        category="novelty",
    ),
    ReasonCodeSpec(
        code="R15",
        feature="is_new_category",
        template="First transaction by this card in this merchant category",
        is_material=_is_true,
        category="novelty",
    ),
    ReasonCodeSpec(
        code="R16",
        feature="distinct_categories_7d",
        template="Spending spread across {value:.0f} categories in 7 days",
        is_material=_gt(6.0),
        category="novelty",
    ),
)

_BY_FEATURE: Final[dict[str, ReasonCodeSpec]] = {spec.feature: spec for spec in REASON_CODES}
_BY_CODE: Final[dict[str, ReasonCodeSpec]] = {spec.code: spec for spec in REASON_CODES}


def spec_for_feature(feature: str) -> ReasonCodeSpec | None:
    return _BY_FEATURE.get(feature)


def spec_for_code(code: str) -> ReasonCodeSpec | None:
    return _BY_CODE.get(code)


@dataclass(frozen=True, slots=True)
class ReasonCodeInstance:
    """A reason code emitted for a specific transaction."""

    code: str
    feature: str
    value: float
    contribution: float
    #: Code and explanation, e.g. "R12 - Implied travel speed of ...".
    text: str
    #: The explanation alone. The console renders the code as a separate chip,
    #: so using `text` there would print the code twice.
    message: str
    category: str

    def __str__(self) -> str:
        return self.text


def derive_reason_codes(
    features: Mapping[str, float],
    contributions: Mapping[str, float] | None = None,
    *,
    max_codes: int = 4,
    min_contribution: float = 0.0,
) -> tuple[ReasonCodeInstance, ...]:
    """Produce ordered reason codes for one decision.

    ``contributions`` are per-feature SHAP values. When supplied, codes are
    ranked by how much each feature actually pushed *this* score toward fraud;
    features that pushed the other way are excluded, since they are not reasons
    for an adverse decision. When omitted, codes fall back to feature order,
    which is what the rule-only path uses.

    A code is emitted only when the feature is material, so an ordinary
    transaction that happens to be scored yields no spurious accusations.
    """
    candidates: list[tuple[float, ReasonCodeInstance]] = []

    for feature, value in features.items():
        spec = _BY_FEATURE.get(feature)
        if spec is None:
            continue

        numeric = float(value)
        if numeric != numeric:  # NaN
            continue
        if not spec.is_material(numeric):
            continue

        contribution = float(contributions.get(feature, 0.0)) if contributions else 0.0
        if contributions is not None and contribution <= min_contribution:
            # Pushed the score away from fraud, so not a reason for the decision.
            continue

        candidates.append(
            (
                contribution,
                ReasonCodeInstance(
                    code=spec.code,
                    feature=feature,
                    value=numeric,
                    contribution=contribution,
                    text=spec.render(numeric),
                    message=spec.message(numeric),
                    category=spec.category,
                ),
            )
        )

    # Rank by SHAP contribution where available; otherwise by catalogue order,
    # which is arranged from strongest to weakest signal.
    if contributions:
        candidates.sort(key=lambda pair: pair[0], reverse=True)
    else:
        order = {spec.code: i for i, spec in enumerate(REASON_CODES)}
        candidates.sort(key=lambda pair: order[pair[1].code])

    return tuple(instance for _, instance in candidates[:max_codes])


def summarise(codes: Sequence[ReasonCodeInstance]) -> str:
    """Render codes as a single analyst-facing block."""
    if not codes:
        return "No individually notable risk factors; score driven by the combination of features."
    return "\n".join(str(code) for code in codes)


def catalogue() -> list[dict[str, str]]:
    """The full code catalogue, for the analyst console and the model card."""
    return [
        {
            "code": spec.code,
            "feature": spec.feature,
            "category": spec.category,
            "description": spec.template,
        }
        for spec in REASON_CODES
    ]
