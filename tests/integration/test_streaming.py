"""Tests for the stream transport and replay producer.

The backpressure behaviour gets particular attention. Dropping the oldest message
when a subscriber falls behind is a deliberate design decision, not an accident,
and a silent change to it would let a slow dashboard stall the scoring path.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import pandas as pd
import pytest

from fraudlens.config import constants as C
from fraudlens.streaming.broker import (
    TOPIC_DECISIONS,
    TOPIC_TRANSACTIONS,
    InProcessBroker,
    StreamMessage,
    build_broker,
)
from fraudlens.streaming.replay import MAX_GAP_SECONDS, ReplayProducer
from tests.conftest import make_transactions


def message(n: int, topic: str = TOPIC_DECISIONS) -> StreamMessage:
    return StreamMessage(topic=topic, payload={"n": n}, key=str(n))


class TestStreamMessage:
    def test_json_round_trip(self) -> None:
        original = StreamMessage(topic="t", payload={"a": 1, "b": "x"}, key="k")
        restored = StreamMessage.from_json(original.to_json())
        assert restored.topic == original.topic
        assert restored.payload == original.payload
        assert restored.key == original.key

    def test_serialises_non_json_values(self) -> None:
        # datetimes and numpy scalars reach the bus from dataframe rows.
        payload = {"when": datetime(2020, 1, 1), "amount": 12.5}
        assert "2020-01-01" in StreamMessage(topic="t", payload=payload).to_json()


class TestInProcessBroker:
    async def test_publish_and_receive(self) -> None:
        broker = InProcessBroker()
        await broker.start()
        subscription = broker.subscribe(TOPIC_DECISIONS)

        await broker.publish(message(1))
        received = await asyncio.wait_for(anext(subscription), timeout=1.0)
        assert received.payload == {"n": 1}
        await subscription.aclose()

    async def test_fans_out_to_every_subscriber(self) -> None:
        broker = InProcessBroker()
        await broker.start()
        first = broker.subscribe(TOPIC_DECISIONS)
        second = broker.subscribe(TOPIC_DECISIONS)

        await broker.publish(message(7))
        a = await asyncio.wait_for(anext(first), timeout=1.0)
        b = await asyncio.wait_for(anext(second), timeout=1.0)
        assert a.payload == b.payload == {"n": 7}
        await first.aclose()
        await second.aclose()

    async def test_topics_are_isolated(self) -> None:
        broker = InProcessBroker()
        await broker.start()
        decisions = broker.subscribe(TOPIC_DECISIONS)

        await broker.publish(message(1, topic=TOPIC_TRANSACTIONS))
        await broker.publish(message(2, topic=TOPIC_DECISIONS))

        received = await asyncio.wait_for(anext(decisions), timeout=1.0)
        assert received.payload == {"n": 2}
        await decisions.aclose()

    async def test_publishing_with_no_subscribers_is_harmless(self) -> None:
        broker = InProcessBroker()
        await broker.start()
        await broker.publish(message(1))
        assert broker.published == 1

    async def test_slow_subscriber_drops_oldest_and_never_blocks(self) -> None:
        # The core backpressure guarantee: a subscriber that never reads must not
        # be able to stall the publisher or grow memory without bound.
        broker = InProcessBroker(queue_size=5)
        await broker.start()
        subscription = broker.subscribe(TOPIC_DECISIONS)

        for i in range(50):
            await asyncio.wait_for(broker.publish(message(i)), timeout=1.0)

        assert broker.dropped_total > 0
        # What survives is the newest, which is what matters in fraud detection.
        received = await asyncio.wait_for(anext(subscription), timeout=1.0)
        assert received.payload["n"] > 40
        await subscription.aclose()

    async def test_closing_a_subscription_removes_it(self) -> None:
        broker = InProcessBroker()
        await broker.start()
        subscription = broker.subscribe(TOPIC_DECISIONS)
        assert broker.subscriber_count(TOPIC_DECISIONS) == 1

        await subscription.aclose()
        await broker.publish(message(1))
        assert broker.subscriber_count(TOPIC_DECISIONS) == 0

    async def test_close_terminates_subscribers(self) -> None:
        broker = InProcessBroker()
        await broker.start()
        subscription = broker.subscribe(TOPIC_DECISIONS)
        await broker.close()

        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(anext(subscription), timeout=1.0)

    async def test_publish_after_close_is_ignored(self) -> None:
        broker = InProcessBroker()
        await broker.start()
        await broker.close()
        await broker.publish(message(1))  # must not raise

    def test_rejects_invalid_queue_size(self) -> None:
        with pytest.raises(ValueError, match="queue_size must be positive"):
            InProcessBroker(queue_size=0)


class TestBrokerSelection:
    def test_native_profile_binds_in_process(self) -> None:
        assert isinstance(build_broker("native", "localhost:9092"), InProcessBroker)

    def test_docker_profile_binds_kafka(self) -> None:
        from fraudlens.streaming.broker import KafkaBroker

        assert isinstance(build_broker("docker", "localhost:9092"), KafkaBroker)


class TestReplayProducer:
    @pytest.fixture
    def frame(self) -> pd.DataFrame:
        return make_transactions(n=40, n_cards=4, span_days=2, seed=3)

    def test_rejects_empty_frame(self) -> None:
        with pytest.raises(ValueError, match="empty transaction frame"):
            ReplayProducer(transactions=pd.DataFrame(), broker=InProcessBroker())

    def test_rejects_non_positive_speed(self, frame: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="speed must be positive"):
            ReplayProducer(transactions=frame, broker=InProcessBroker(), speed=0)

    def test_requires_a_timestamp_column(self) -> None:
        with pytest.raises(KeyError):
            ReplayProducer(transactions=pd.DataFrame({"x": [1]}), broker=InProcessBroker())

    def test_sorts_input_chronologically(self, frame: pd.DataFrame) -> None:
        shuffled = frame.sample(frac=1.0, random_state=7)
        producer = ReplayProducer(transactions=shuffled, broker=InProcessBroker())
        assert producer.transactions[C.TIMESTAMP_COL].is_monotonic_increasing

    async def test_publishes_transactions(self, frame: pd.DataFrame) -> None:
        broker = InProcessBroker()
        await broker.start()
        subscription = broker.subscribe(TOPIC_TRANSACTIONS)

        producer = ReplayProducer(transactions=frame, broker=broker, speed=1e7, loop=False)
        producer.start()

        received = [await asyncio.wait_for(anext(subscription), timeout=2.0) for _ in range(5)]
        await producer.stop()
        await subscription.aclose()

        assert len(received) == 5
        assert all(C.AMOUNT_COL in m.payload for m in received)
        assert all(m.topic == TOPIC_TRANSACTIONS for m in received)

    async def test_payload_is_json_serialisable(self, frame: pd.DataFrame) -> None:
        broker = InProcessBroker()
        await broker.start()
        subscription = broker.subscribe(TOPIC_TRANSACTIONS)
        producer = ReplayProducer(transactions=frame, broker=broker, speed=1e7, loop=False)
        producer.start()

        received = await asyncio.wait_for(anext(subscription), timeout=2.0)
        await producer.stop()
        await subscription.aclose()

        # Timestamps must have become strings, or the WebSocket send would fail.
        assert isinstance(received.payload[C.TIMESTAMP_COL], str)
        assert isinstance(received.to_json(), str)

    async def test_stop_is_idempotent(self, frame: pd.DataFrame) -> None:
        producer = ReplayProducer(transactions=frame, broker=InProcessBroker())
        await producer.stop()
        await producer.stop()
        assert not producer.is_running

    def test_gap_cap_prevents_a_stalled_demo(self) -> None:
        # A quiet overnight stretch must not freeze the stream for minutes.
        assert MAX_GAP_SECONDS <= 5.0

    async def test_stats_track_progress(self, frame: pd.DataFrame) -> None:
        broker = InProcessBroker()
        await broker.start()
        broker.subscribe(TOPIC_TRANSACTIONS)
        producer = ReplayProducer(transactions=frame, broker=broker, speed=1e7, loop=False)
        producer.start()
        await asyncio.sleep(0.4)
        await producer.stop()
        assert producer.stats.published > 0
        assert producer.stats.started_at is not None


class TestWarmupWindow:
    def test_warmup_precedes_the_test_period(self) -> None:
        """The warm-up window must never overlap the fold being evaluated.

        Warming builds feature-store state from transactions *before* the test
        period. If it reached into the test period itself, the replay would be
        scoring rows whose history included their own future.
        """
        from fraudlens.streaming.replay import load_warmup_window

        window = load_warmup_window()
        if window.empty:
            pytest.skip("no processed dataset available")
        assert window[C.TIMESTAMP_COL].max() < pd.Timestamp(C.TEST_START)
        earliest = pd.Timestamp(C.TEST_START) - timedelta(days=C.MAX_LOOKBACK_DAYS)
        assert window[C.TIMESTAMP_COL].min() >= earliest
