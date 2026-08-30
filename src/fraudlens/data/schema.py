"""Data contracts for the raw transaction feed.

Validation runs at the ingestion boundary. Everything downstream -- feature
engineering, training, serving -- is entitled to assume the contract holds, so
this is the single place that deals with malformed input.

Design note: this is a small in-house validator rather than ``pandera``. See
``docs/adr/0002-in-house-dataframe-validation.md`` for why. The interface is
deliberately pandera-shaped so the swap stays cheap if that changes.

All violations in a frame are collected and reported together. Failing on the
first bad column makes diagnosing a new data drop needlessly slow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import pandas as pd

from fraudlens.config import constants as C

if TYPE_CHECKING:
    from collections.abc import Sequence

DTypeFamily = Literal["datetime", "float", "integer", "string", "boolean"]


class SchemaError(ValueError):
    """Raised when a dataframe violates its declared contract.

    Carries every violation found, not just the first.
    """

    def __init__(self, frame_name: str, violations: Sequence[str]) -> None:
        self.frame_name = frame_name
        self.violations = list(violations)
        detail = "\n".join(f"  - {v}" for v in violations)
        super().__init__(f"{frame_name} failed schema validation:\n{detail}")


@dataclass(frozen=True, slots=True)
class ColumnSpec:
    """Declared contract for a single column."""

    name: str
    dtype: DTypeFamily
    nullable: bool = False
    unique: bool = False
    minimum: float | None = None
    maximum: float | None = None
    allowed: frozenset[str] | None = None
    description: str = ""

    def check(self, df: pd.DataFrame) -> list[str]:
        """Return a list of human-readable violations for this column."""
        if self.name not in df.columns:
            return [f"{self.name!r}: required column is missing"]

        series = df[self.name]
        problems: list[str] = []

        if not self.nullable and (null_count := int(series.isna().sum())) > 0:
            problems.append(f"{self.name!r}: {null_count:,} null values in a non-nullable column")

        if not self._dtype_matches(series):
            problems.append(f"{self.name!r}: expected {self.dtype} dtype, got {series.dtype}")
            # Range/membership checks on a wrong dtype produce noise, not signal.
            return problems

        if self.unique and (dupes := int(series.duplicated().sum())) > 0:
            problems.append(f"{self.name!r}: {dupes:,} duplicate values in a unique column")

        problems.extend(self._check_bounds(series))

        if self.allowed is not None:
            observed = set(series.dropna().astype(str).unique())
            if unexpected := observed - self.allowed:
                shown = sorted(unexpected)[:5]
                problems.append(
                    f"{self.name!r}: unexpected values {shown} (allowed: {sorted(self.allowed)})"
                )

        return problems

    def _check_bounds(self, series: pd.Series) -> list[str]:
        if self.dtype not in {"float", "integer", "datetime"}:
            return []
        problems: list[str] = []
        numeric = series.dropna()
        if numeric.empty:
            return []
        if self.minimum is not None and (below := int((numeric < self.minimum).sum())) > 0:
            problems.append(f"{self.name!r}: {below:,} values below minimum {self.minimum}")
        if self.maximum is not None and (above := int((numeric > self.maximum).sum())) > 0:
            problems.append(f"{self.name!r}: {above:,} values above maximum {self.maximum}")
        return problems

    def _dtype_matches(self, series: pd.Series) -> bool:
        match self.dtype:
            case "datetime":
                return bool(pd.api.types.is_datetime64_any_dtype(series))
            case "float":
                return bool(pd.api.types.is_float_dtype(series))
            case "integer":
                return bool(pd.api.types.is_integer_dtype(series))
            case "boolean":
                return bool(pd.api.types.is_bool_dtype(series))
            case "string":
                # pandas 3 defaults object columns of text to `str` dtype; accept
                # both that and legacy `object` so the contract is portable
                # across pandas 2.x and 3.x.
                return bool(pd.api.types.is_string_dtype(series)) or series.dtype == object


@dataclass(frozen=True, slots=True)
class DataFrameSchema:
    """An ordered collection of column contracts."""

    name: str
    columns: tuple[ColumnSpec, ...]
    #: Columns that must not contain nulls *jointly* -- e.g. a composite key.
    require_non_empty: bool = True
    _by_name: dict[str, ColumnSpec] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_by_name", {c.name: c for c in self.columns})

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate ``df``, raising :class:`SchemaError` on any violation.

        Returns the frame unchanged so this can be used inline in a pipeline.
        """
        violations: list[str] = []

        if self.require_non_empty and df.empty:
            violations.append("frame is empty")

        for spec in self.columns:
            violations.extend(spec.check(df))

        if violations:
            raise SchemaError(self.name, violations)
        return df

    def column(self, name: str) -> ColumnSpec:
        return self._by_name[name]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns)


# ---------------------------------------------------------------------------
# The raw transaction contract
# ---------------------------------------------------------------------------

RAW_TRANSACTION_SCHEMA = DataFrameSchema(
    name="raw_transactions",
    columns=(
        ColumnSpec(
            C.TIMESTAMP_COL,
            "datetime",
            description="Transaction timestamp; the ordering key for every temporal operation.",
        ),
        ColumnSpec(
            C.CARD_COL,
            "integer",
            description="Card number; the entity behavioural features are grouped by.",
        ),
        ColumnSpec("merchant", "string"),
        ColumnSpec("category", "string"),
        ColumnSpec(
            C.AMOUNT_COL,
            "float",
            minimum=0.0,
            description="Transaction value in USD.",
        ),
        ColumnSpec("gender", "string", allowed=frozenset({"M", "F"})),
        ColumnSpec("city", "string"),
        ColumnSpec("state", "string"),
        ColumnSpec("zip", "integer"),
        ColumnSpec(C.HOME_LAT_COL, "float", minimum=-90.0, maximum=90.0),
        ColumnSpec(C.HOME_LON_COL, "float", minimum=-180.0, maximum=180.0),
        ColumnSpec("city_pop", "integer", minimum=0),
        ColumnSpec("job", "string"),
        ColumnSpec("dob", "datetime"),
        ColumnSpec(C.TRANSACTION_ID_COL, "string", unique=True),
        ColumnSpec("unix_time", "integer", minimum=0),
        ColumnSpec(C.MERCH_LAT_COL, "float", minimum=-90.0, maximum=90.0),
        ColumnSpec(C.MERCH_LON_COL, "float", minimum=-180.0, maximum=180.0),
        ColumnSpec(
            C.TARGET_COL,
            "integer",
            minimum=0,
            maximum=1,
            description="Binary fraud label.",
        ),
    ),
)
