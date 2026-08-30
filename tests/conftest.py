"""Shared fixtures.

Everything here builds *synthetic* frames with controlled properties. Tests must
not depend on the Kaggle download: CI has no credentials, and a test suite that
needs 400 MB of data is a test suite nobody runs.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from fraudlens.config import constants as C


def make_transactions(
    n: int = 500,
    *,
    start: datetime | None = None,
    span_days: int = 730,
    fraud_rate: float = 0.02,
    n_cards: int = 20,
    seed: int = 7,
) -> pd.DataFrame:
    """Build a synthetic frame that satisfies ``RAW_TRANSACTION_SCHEMA``.

    Timestamps are spread evenly across ``span_days`` from ``start`` so that
    split boundaries and embargo gaps can be exercised deterministically.
    """
    rng = np.random.default_rng(seed)
    origin = start or datetime(C.DATASET_START.year, C.DATASET_START.month, C.DATASET_START.day)

    offsets = np.linspace(0, span_days * 86_400, n, endpoint=False)
    timestamps = [origin + timedelta(seconds=float(s)) for s in offsets]

    # Draw a fixed pool of cards and sample from it, so transactions actually
    # accumulate per-card history. Giving every row its own card number would
    # leave every trailing-window feature permanently at its cold-start value.
    card_pool = rng.choice(
        np.arange(4_000_000_000_000_000, 4_000_000_000_000_000 + n_cards * 7, 7),
        size=n_cards,
        replace=False,
    )

    categories = ["grocery_pos", "gas_transport", "misc_net", "shopping_net", "entertainment"]
    return pd.DataFrame(
        {
            C.TIMESTAMP_COL: pd.to_datetime(timestamps),
            C.CARD_COL: rng.choice(card_pool, size=n),
            "merchant": [f"fraud_Merchant_{i % 40}" for i in range(n)],
            "category": rng.choice(categories, size=n),
            C.AMOUNT_COL: np.round(rng.gamma(2.0, 40.0, size=n) + 1.0, 2),
            "gender": rng.choice(["M", "F"], size=n),
            "city": [f"City_{i % 30}" for i in range(n)],
            "state": rng.choice(["NY", "CA", "TX", "OH"], size=n),
            "zip": rng.integers(10_000, 99_999, size=n),
            C.HOME_LAT_COL: rng.uniform(25.0, 48.0, size=n),
            C.HOME_LON_COL: rng.uniform(-124.0, -70.0, size=n),
            "city_pop": rng.integers(100, 2_000_000, size=n),
            "job": [f"Job_{i % 25}" for i in range(n)],
            "dob": pd.to_datetime(
                [datetime(1950, 1, 1) + timedelta(days=int(d)) for d in rng.integers(0, 18_000, n)]
            ),
            C.TRANSACTION_ID_COL: [f"txn{i:08d}" for i in range(n)],
            "unix_time": [int(t.timestamp()) for t in timestamps],
            C.MERCH_LAT_COL: rng.uniform(25.0, 48.0, size=n),
            C.MERCH_LON_COL: rng.uniform(-124.0, -70.0, size=n),
            C.TARGET_COL: (rng.random(n) < fraud_rate).astype(np.int64),
        }
    )


@pytest.fixture
def transactions() -> pd.DataFrame:
    """A small, schema-valid transaction frame spanning the full dataset range."""
    return make_transactions()


@pytest.fixture
def separable_scores() -> tuple[np.ndarray, np.ndarray]:
    """Labels and scores with a known, imperfect separation.

    Positives are drawn from a higher-mean distribution than negatives, with
    deliberate overlap so that precision/recall trade-offs are non-trivial.
    """
    rng = np.random.default_rng(11)
    n_neg, n_pos = 1_000, 50
    y = np.concatenate([np.zeros(n_neg, dtype=np.int8), np.ones(n_pos, dtype=np.int8)])
    scores = np.concatenate(
        [
            np.clip(rng.beta(2.0, 8.0, n_neg), 0.0, 1.0),
            np.clip(rng.beta(6.0, 3.0, n_pos), 0.0, 1.0),
        ]
    )
    return y, scores
