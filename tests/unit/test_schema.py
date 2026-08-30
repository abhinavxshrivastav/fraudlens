"""Tests for the ingestion data contract."""

from __future__ import annotations

import pandas as pd
import pytest

from fraudlens.config import constants as C
from fraudlens.data.schema import (
    RAW_TRANSACTION_SCHEMA,
    ColumnSpec,
    DataFrameSchema,
    SchemaError,
)


class TestColumnSpec:
    def test_missing_column_is_reported(self) -> None:
        spec = ColumnSpec("amount", "float")
        assert spec.check(pd.DataFrame({"other": [1.0]})) == [
            "'amount': required column is missing"
        ]

    def test_nulls_rejected_when_not_nullable(self) -> None:
        spec = ColumnSpec("amount", "float")
        problems = spec.check(pd.DataFrame({"amount": [1.0, None, 3.0]}))
        assert any("null values" in p for p in problems)

    def test_nulls_allowed_when_nullable(self) -> None:
        spec = ColumnSpec("amount", "float", nullable=True)
        assert spec.check(pd.DataFrame({"amount": [1.0, None]})) == []

    def test_wrong_dtype_short_circuits_further_checks(self) -> None:
        # Range checks against a string column would produce noise, so the
        # dtype failure should be the only thing reported.
        spec = ColumnSpec("amount", "float", minimum=0.0, maximum=10.0)
        problems = spec.check(pd.DataFrame({"amount": ["a", "b"]}))
        assert len(problems) == 1
        assert "expected float dtype" in problems[0]

    def test_bounds_are_enforced(self) -> None:
        spec = ColumnSpec("lat", "float", minimum=-90.0, maximum=90.0)
        problems = spec.check(pd.DataFrame({"lat": [-100.0, 0.0, 95.0]}))
        assert any("below minimum" in p for p in problems)
        assert any("above maximum" in p for p in problems)

    def test_uniqueness_is_enforced(self) -> None:
        spec = ColumnSpec("txn", "string", unique=True)
        problems = spec.check(pd.DataFrame({"txn": ["a", "a", "b"]}))
        assert any("duplicate values" in p for p in problems)

    def test_allowed_values_are_enforced(self) -> None:
        spec = ColumnSpec("gender", "string", allowed=frozenset({"M", "F"}))
        problems = spec.check(pd.DataFrame({"gender": ["M", "F", "X"]}))
        assert any("unexpected values" in p and "X" in p for p in problems)

    def test_string_dtype_accepts_object_and_str(self) -> None:
        # pandas 3 infers `str` dtype where pandas 2 gave `object`; the contract
        # must hold on both so the codebase is not pinned to one pandas major.
        spec = ColumnSpec("merchant", "string")
        as_object = pd.Series(["a", "b"], dtype=object)
        assert spec.check(pd.DataFrame({"merchant": as_object})) == []
        assert spec.check(pd.DataFrame({"merchant": pd.Series(["a", "b"], dtype="string")})) == []


class TestDataFrameSchema:
    def test_valid_frame_passes_and_is_returned_unchanged(self, transactions: pd.DataFrame) -> None:
        result = RAW_TRANSACTION_SCHEMA.validate(transactions)
        assert result is transactions

    def test_all_violations_are_reported_together(self, transactions: pd.DataFrame) -> None:
        # Diagnosing a bad data drop one error at a time is needlessly slow, so
        # the validator must not stop at the first problem.
        broken = transactions.copy()
        broken.loc[broken.index[0], C.AMOUNT_COL] = -5.0
        broken.loc[broken.index[1], "gender"] = "X"
        broken = broken.drop(columns=["city_pop"])

        with pytest.raises(SchemaError) as exc:
            RAW_TRANSACTION_SCHEMA.validate(broken)

        assert len(exc.value.violations) >= 3
        joined = "\n".join(exc.value.violations)
        assert "below minimum" in joined
        assert "unexpected values" in joined
        assert "missing" in joined

    def test_empty_frame_is_rejected(self) -> None:
        schema = DataFrameSchema("t", (ColumnSpec("a", "integer"),))
        with pytest.raises(SchemaError, match="empty"):
            schema.validate(pd.DataFrame({"a": pd.Series([], dtype="int64")}))

    def test_error_message_names_the_frame(self, transactions: pd.DataFrame) -> None:
        broken = transactions.drop(columns=[C.AMOUNT_COL])
        with pytest.raises(SchemaError, match="raw_transactions failed schema validation"):
            RAW_TRANSACTION_SCHEMA.validate(broken)

    def test_duplicate_transaction_ids_are_caught(self, transactions: pd.DataFrame) -> None:
        broken = transactions.copy()
        broken.loc[broken.index[1], C.TRANSACTION_ID_COL] = broken.loc[
            broken.index[0], C.TRANSACTION_ID_COL
        ]
        with pytest.raises(SchemaError, match="duplicate"):
            RAW_TRANSACTION_SCHEMA.validate(broken)

    def test_schema_exposes_column_names(self) -> None:
        names = RAW_TRANSACTION_SCHEMA.names
        assert C.TIMESTAMP_COL in names
        assert C.TARGET_COL in names
        # PII columns are dropped at ingestion, so they are not in the contract.
        for pii in C.PII_COLS:
            assert pii not in names

    def test_column_lookup(self) -> None:
        spec = RAW_TRANSACTION_SCHEMA.column(C.AMOUNT_COL)
        assert spec.minimum == 0.0
