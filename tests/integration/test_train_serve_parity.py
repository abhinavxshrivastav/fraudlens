"""Train/serve parity: the offline and online feature paths must agree exactly.

Why this test carries weight
----------------------------
Train/serve skew is the classic silent production failure. The model is trained
on features computed one way and served features computed another; nothing
raises, no test goes red, and performance quietly sits below what the offline
evaluation promised. It is usually found months later, if at all.

FraudLens has one feature implementation, so parity should hold by construction.
This test exists because "should hold by construction" is exactly the kind of
claim that stops being true after a refactor. The two paths differ in how state
is *sourced* -- a dict built during a batch replay versus a
:class:`~fraudlens.features.store.FeatureStore` queried per request -- and that
difference is where skew would enter.

The assertion is exact equality, not approximate. Any drift at all is a bug.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraudlens.config import constants as C
from fraudlens.features.pipeline import FeaturePipeline, TransactionEvent
from fraudlens.features.store import InMemoryFeatureStore
from tests.conftest import make_transactions


def score_online(df: pd.DataFrame, store: InMemoryFeatureStore) -> pd.DataFrame:
    """Replay transactions one at a time through the serving code path.

    Mirrors :meth:`fraudlens.api.service.FraudLensService.score` exactly,
    including the ordering guarantee: features are computed before the store is
    updated.
    """
    pipeline = FeaturePipeline()
    ordered = df.sort_values(C.TIMESTAMP_COL, kind="stable")
    rows = []
    for record in ordered.to_dict("records"):
        event = TransactionEvent.from_row(record)
        state = store.get(event.card)
        state.prune(event.timestamp)

        rows.append(pipeline.compute(event, state))

        store.update(event.card, event)
    return pd.DataFrame(rows, index=ordered.index, columns=list(FeaturePipeline.FEATURE_NAMES))


@pytest.fixture
def transactions_for_parity() -> pd.DataFrame:
    # Few cards, many transactions: maximises history depth per card, so the
    # trailing-window features are actually exercised rather than sitting at
    # their cold-start defaults.
    return make_transactions(n=900, n_cards=6, span_days=120, seed=99)


class TestParity:
    def test_offline_and_online_features_are_identical(
        self, transactions_for_parity: pd.DataFrame
    ) -> None:
        offline = FeaturePipeline().transform_frame(transactions_for_parity)
        online = score_online(transactions_for_parity, InMemoryFeatureStore())

        pd.testing.assert_frame_equal(
            offline.sort_index(),
            online.sort_index(),
            check_exact=True,
            check_dtype=True,
        )

    def test_parity_holds_when_input_order_is_shuffled(
        self, transactions_for_parity: pd.DataFrame
    ) -> None:
        shuffled = transactions_for_parity.sample(frac=1.0, random_state=17)
        offline = FeaturePipeline().transform_frame(shuffled)
        online = score_online(shuffled, InMemoryFeatureStore())
        pd.testing.assert_frame_equal(offline.sort_index(), online.sort_index(), check_exact=True)

    def test_parity_holds_for_the_velocity_features_specifically(
        self, transactions_for_parity: pd.DataFrame
    ) -> None:
        # Velocity features depend most heavily on accumulated state, so they
        # are where a store-vs-dict divergence would surface first.
        offline = FeaturePipeline().transform_frame(transactions_for_parity).sort_index()
        online = score_online(transactions_for_parity, InMemoryFeatureStore()).sort_index()
        for column in (
            "txn_count_1h",
            "txn_count_24h",
            "txn_count_7d",
            "txn_count_30d",
            "amt_sum_1h",
            "distinct_merchants_24h",
            "amt_zscore_30d",
        ):
            np.testing.assert_array_equal(
                offline[column].to_numpy(),
                online[column].to_numpy(),
                err_msg=f"{column} diverged between the offline and online paths",
            )

    def test_history_actually_accumulated(self, transactions_for_parity: pd.DataFrame) -> None:
        # Guards the test itself: if every row had empty history, parity would
        # hold trivially and prove nothing.
        offline = FeaturePipeline().transform_frame(transactions_for_parity)
        assert offline["card_history_count"].max() > 20
        assert (offline["txn_count_24h"] > 0).mean() > 0.5


class TestStoreSemantics:
    def test_store_eviction_degrades_gracefully(self) -> None:
        # An evicted card starts cold. That is a correctness-preserving loss of
        # signal, not corruption: card_history_count drops to zero and the model
        # can discount the other features accordingly.
        df = make_transactions(n=200, n_cards=10, span_days=30, seed=5)
        store = InMemoryFeatureStore(capacity=2)
        result = score_online(df, store)
        assert len(result) == len(df)
        assert store.size() <= 2
        assert store.evictions > 0

    def test_warm_start_reproduces_continuing_history(self) -> None:
        """Splitting a stream in two and warming the second half must match.

        This is what a service restart looks like: state is reloaded and
        processing continues. If a warm start diverged from an uninterrupted
        run, every deploy would produce a burst of subtly wrong decisions.
        """
        df = make_transactions(n=400, n_cards=4, span_days=60, seed=21).sort_values(
            C.TIMESTAMP_COL, kind="stable"
        )
        first, second = df.iloc[:250], df.iloc[250:]

        uninterrupted = score_online(df, InMemoryFeatureStore()).sort_index()

        store = InMemoryFeatureStore()
        score_online(first, store)
        resumed = score_online(second, store).sort_index()

        pd.testing.assert_frame_equal(uninterrupted.loc[resumed.index], resumed, check_exact=True)
