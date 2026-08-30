"""Domain constants: dataset schema, temporal split boundaries, cost model.

These are deliberately kept out of :mod:`fraudlens.config.settings` because they
are *modelling* decisions, not deployment configuration. Changing a split
boundary or a cost assumption invalidates trained artefacts and must go through
review -- so they live in version control, not in the environment.
"""

from __future__ import annotations

from datetime import date
from typing import Final

# --------------------------------------------------------------------------
# Raw dataset schema (Sparkov / kartik2112 "fraud-detection")
# --------------------------------------------------------------------------

#: Timestamp column, parsed to naive UTC datetimes.
TIMESTAMP_COL: Final = "trans_date_trans_time"
#: Card identifier -- the entity all behavioural features are grouped by.
CARD_COL: Final = "cc_num"
#: Binary target.
TARGET_COL: Final = "is_fraud"
#: Natural key for a transaction.
TRANSACTION_ID_COL: Final = "trans_num"
#: Transaction value.
AMOUNT_COL: Final = "amt"

#: The Kaggle CSVs carry an unnamed integer index column that must be dropped.
UNNAMED_INDEX_COL: Final = "Unnamed: 0"

#: Columns that are personally identifying and are dropped at ingestion. They
#: are never used as model inputs: `street`/`first`/`last` are unique per
#: customer and would let the model memorise individuals rather than learn
#: behaviour, which is both a leakage risk and a privacy problem.
PII_COLS: Final[tuple[str, ...]] = ("first", "last", "street")

RAW_COLUMNS: Final[tuple[str, ...]] = (
    TIMESTAMP_COL,
    CARD_COL,
    "merchant",
    "category",
    AMOUNT_COL,
    "first",
    "last",
    "gender",
    "street",
    "city",
    "state",
    "zip",
    "lat",
    "long",
    "city_pop",
    "job",
    "dob",
    TRANSACTION_ID_COL,
    "unix_time",
    "merch_lat",
    "merch_long",
    TARGET_COL,
)

#: Customer home coordinates.
HOME_LAT_COL: Final = "lat"
HOME_LON_COL: Final = "long"
#: Merchant coordinates -- the pair that makes geo-velocity features possible.
MERCH_LAT_COL: Final = "merch_lat"
MERCH_LON_COL: Final = "merch_long"

# --------------------------------------------------------------------------
# Temporal split
# --------------------------------------------------------------------------
# The dataset spans 2019-01-01 to 2020-12-31. We ignore the vendor's
# train/test files and re-split the union chronologically.
#
# An EMBARGO gap sits between each fold. Behavioural features look back up to 30
# days; without a gap, a validation row's trailing-window features would be
# computed partly from training rows, and a training row near the boundary would
# share window state with validation. The embargo makes the folds genuinely
# independent at the cost of discarding a small slice of data.

DATASET_START: Final = date(2019, 1, 1)
DATASET_END: Final = date(2020, 12, 31)

TRAIN_START: Final = date(2019, 1, 1)
TRAIN_END: Final = date(2020, 3, 31)

VALID_START: Final = date(2020, 4, 8)
VALID_END: Final = date(2020, 6, 30)

TEST_START: Final = date(2020, 7, 8)
TEST_END: Final = date(2020, 12, 31)

#: Days discarded between folds. Must be >= the longest feature lookback window.
EMBARGO_DAYS: Final = 7

#: Longest trailing window used by the feature pipeline, in days.
MAX_LOOKBACK_DAYS: Final = 30

# --------------------------------------------------------------------------
# Cost model
# --------------------------------------------------------------------------
# Used to choose the operating threshold by minimising expected cost rather
# than maximising an abstract score. Values are order-of-magnitude estimates for
# UK card fraud and are documented (with their sensitivity) in docs/EVALUATION.md.
# The *conclusion* that matters is the ratio, not the absolute figures.

#: Cost of a missed fraud. Modelled as the transaction amount itself: the
#: issuer refunds the customer and absorbs the loss.
#: (Applied per-transaction, so it is a multiplier on `amt`, not a constant.)
FALSE_NEGATIVE_AMOUNT_MULTIPLIER: Final = 1.0

#: Analyst time to triage one alert, in GBP.
INVESTIGATION_COST_GBP: Final = 4.00

#: Expected cost of wrongly declining a genuine transaction: call-centre
#: handling plus attrition risk. Dominates the investigation cost, which is why
#: precision matters more than raw alert volume.
FALSE_POSITIVE_FRICTION_GBP: Final = 18.00

# --------------------------------------------------------------------------
# Operating points
# --------------------------------------------------------------------------

#: Fixed false-positive rates at which recall is reported.
REPORTED_FPRS: Final[tuple[float, ...]] = (0.001, 0.005, 0.01, 0.05)

#: Analyst review capacity, used for Precision@K. Assumes a small fraud team
#: clearing roughly this many cases per day.
ANALYST_DAILY_CAPACITY: Final = 100

#: Population Stability Index action bands (industry convention).
PSI_STABLE: Final = 0.10
PSI_INVESTIGATE: Final = 0.25

#: Serving latency budget for POST /score, in milliseconds (p99).
LATENCY_BUDGET_P99_MS: Final = 50.0
