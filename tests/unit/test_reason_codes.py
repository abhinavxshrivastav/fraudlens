"""Tests for reason-code derivation.

Reason codes are the customer- and regulator-facing output of the system, so the
guarantees that matter are: never accuse an ordinary transaction, never cite a
factor that argued *for* the transaction, and keep code meanings stable.
"""

from __future__ import annotations

import pytest

from fraudlens.explain.reason_codes import (
    REASON_CODES,
    catalogue,
    derive_reason_codes,
    spec_for_code,
    spec_for_feature,
    summarise,
)

ORDINARY: dict[str, float] = {
    "amt": 42.0,
    "amt_ratio_mean_30d": 1.1,
    "amt_zscore_30d": 0.3,
    "amt_ratio_max_30d": 0.8,
    "txn_count_1h": 1.0,
    "txn_count_24h": 3.0,
    "distinct_merchants_24h": 2.0,
    "amt_sum_1h": 42.0,
    "is_night": 0.0,
    "secs_since_prev_txn": 7_200.0,
    "dist_home_merch_km": 12.0,
    "implied_speed_kmh": 30.0,
    "dist_prev_txn_km": 15.0,
    "is_new_merchant": 0.0,
    "is_new_category": 0.0,
    "distinct_categories_7d": 3.0,
}

SUSPICIOUS: dict[str, float] = {
    **ORDINARY,
    "amt": 2_400.0,
    "amt_ratio_mean_30d": 18.5,
    "amt_zscore_30d": 6.2,
    "txn_count_1h": 7.0,
    "distinct_merchants_24h": 6.0,
    "is_night": 1.0,
    "implied_speed_kmh": 1_240.0,
    "dist_home_merch_km": 1_847.0,
    "is_new_merchant": 1.0,
}


class TestMateriality:
    def test_ordinary_transaction_yields_no_codes(self) -> None:
        # The most important guarantee: a routine transaction must not be
        # narrated as if it were suspicious.
        assert derive_reason_codes(ORDINARY) == ()

    def test_suspicious_transaction_yields_codes(self) -> None:
        codes = derive_reason_codes(SUSPICIOUS)
        assert len(codes) > 0

    def test_borderline_values_do_not_trigger(self) -> None:
        # R01 requires a ratio above 3.0.
        assert derive_reason_codes({"amt_ratio_mean_30d": 2.9}) == ()
        assert len(derive_reason_codes({"amt_ratio_mean_30d": 3.1})) == 1

    def test_nan_values_are_skipped(self) -> None:
        assert derive_reason_codes({"amt_ratio_mean_30d": float("nan")}) == ()

    def test_unknown_features_are_ignored(self) -> None:
        assert derive_reason_codes({"some_internal_feature": 999.0}) == ()


class TestRendering:
    def test_text_contains_the_actual_value(self) -> None:
        codes = derive_reason_codes({"amt_ratio_mean_30d": 8.4})
        assert codes[0].text == "R01 - Amount is 8.4x this card's 30-day average"

    def test_impossible_travel_reads_as_a_claim_about_behaviour(self) -> None:
        codes = derive_reason_codes({"implied_speed_kmh": 1_240.0})
        assert "1,240 km/h" in codes[0].text
        assert codes[0].code == "R12"

    def test_distance_is_thousands_separated(self) -> None:
        codes = derive_reason_codes({"dist_home_merch_km": 1_847.0})
        assert "1,847 km" in codes[0].text

    def test_boolean_features_render_without_a_number(self) -> None:
        codes = derive_reason_codes({"is_new_merchant": 1.0})
        assert codes[0].text == (
            "R14 - First recorded transaction between this card and this merchant"
        )

    def test_summarise_joins_codes(self) -> None:
        rendered = summarise(derive_reason_codes(SUSPICIOUS))
        assert rendered.count("\n") >= 1

    def test_summarise_handles_no_codes(self) -> None:
        assert "combination of features" in summarise(())


class TestShapRanking:
    def test_contributions_determine_order(self) -> None:
        features = {"amt_ratio_mean_30d": 8.0, "txn_count_1h": 9.0, "dist_home_merch_km": 900.0}
        contributions = {"dist_home_merch_km": 0.5, "amt_ratio_mean_30d": 0.3, "txn_count_1h": 0.1}
        codes = derive_reason_codes(features, contributions)
        assert [c.code for c in codes] == ["R11", "R01", "R05"]

    def test_negative_contributions_are_excluded(self) -> None:
        # A feature that pushed the score *away* from fraud is not a reason for
        # an adverse decision, however unusual its value.
        features = {"amt_ratio_mean_30d": 8.0, "txn_count_1h": 9.0}
        contributions = {"amt_ratio_mean_30d": -0.4, "txn_count_1h": 0.2}
        codes = derive_reason_codes(features, contributions)
        assert [c.code for c in codes] == ["R05"]

    def test_contribution_is_recorded_on_the_instance(self) -> None:
        codes = derive_reason_codes({"amt_ratio_mean_30d": 8.0}, {"amt_ratio_mean_30d": 0.42})
        assert codes[0].contribution == pytest.approx(0.42)

    def test_falls_back_to_catalogue_order_without_contributions(self) -> None:
        features = {"dist_home_merch_km": 900.0, "amt_ratio_mean_30d": 8.0}
        codes = derive_reason_codes(features)
        assert [c.code for c in codes] == ["R01", "R11"]

    def test_max_codes_is_respected(self) -> None:
        # An analyst queue is not improved by sixteen bullet points.
        assert len(derive_reason_codes(SUSPICIOUS, max_codes=2)) == 2

    def test_default_cap_keeps_output_readable(self) -> None:
        assert len(derive_reason_codes(SUSPICIOUS)) <= 4


class TestCatalogueIntegrity:
    def test_codes_are_unique(self) -> None:
        codes = [spec.code for spec in REASON_CODES]
        assert len(codes) == len(set(codes))

    def test_features_are_unique(self) -> None:
        # One feature maps to at most one code, or ordering becomes ambiguous.
        features = [spec.feature for spec in REASON_CODES]
        assert len(features) == len(set(features))

    def test_every_code_references_a_real_model_feature(self) -> None:
        from fraudlens.features.pipeline import FeaturePipeline

        for spec in REASON_CODES:
            assert spec.feature in FeaturePipeline.FEATURE_NAMES, (
                f"{spec.code} references {spec.feature!r}, which the pipeline does not emit"
            )

    def test_lookups_work_both_ways(self) -> None:
        assert spec_for_code("R01") is not None
        assert spec_for_feature("amt_ratio_mean_30d") is not None
        assert spec_for_code("R999") is None
        assert spec_for_feature("nope") is None

    def test_catalogue_is_exportable_for_the_console(self) -> None:
        entries = catalogue()
        assert len(entries) == len(REASON_CODES)
        assert set(entries[0]) == {"code", "feature", "category", "description"}

    def test_every_template_is_human_readable(self) -> None:
        for spec in REASON_CODES:
            # No raw feature names leaking into customer-facing text.
            assert spec.feature not in spec.template, (
                f"{spec.code} template exposes the internal feature name"
            )
            assert len(spec.template) > 20
