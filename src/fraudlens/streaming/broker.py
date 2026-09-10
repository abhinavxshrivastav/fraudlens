"""Stream transport: one protocol, two implementations.

``InProcessBroker`` is the default and needs no infrastructure -- asyncio queues
inside the serving process. ``KafkaBroker`` is selected by
``FRAUDLENS_PROFILE=docker`` and talks to the Redpanda/Kafka service in
``docker-compose.yml``.

Nothing above this module knows which is bound. That is the point: the streaming
architecture is real and reviewable, but running the project never requires a
broker to be installed.

Backpressure
------------
Subscriber queues are **bounded, and drop the oldest message when full**. This is
a deliberate choice rather than an oversight. A monitoring dashboard that falls
behind must not be able to stall the scoring path or grow memory without limit --
in a fraud system the newest transactions are the ones that matter, and a stale
backlog has no value. Drops are counted and exposed so the loss is visible rather
than silent.

The scoring path itself is never a subscriber, so no decision is ever dropped.
Only observers (the console feed, drift monitors) can lose messages.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

#: Default per-subscriber queue depth.
DEFAULT_QUEUE_SIZE = 1_000

TOPIC_TRANSACTIONS = "transactions"
TOPIC_DECISIONS = "decisions"
TOPIC_ALERTS = "alerts"


@dataclass(frozen=True, slots=True)
class StreamMessage:
    """One message on the bus."""

    topic: str
    payload: dict[str, Any]
    key: str = ""
    published_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_json(self) -> str:
        return json.dumps(
            {
                "topic": self.topic,
                "key": self.key,
                "published_at": self.published_at.isoformat(),
                "payload": self.payload,
            },
            separators=(",", ":"),
            default=str,
        )

    @classmethod
    def from_json(cls, raw: str | bytes) -> StreamMessage:
        data = json.loads(raw)
        return cls(
            topic=str(data.get("topic", "")),
            payload=dict(data.get("payload", {})),
            key=str(data.get("key", "")),
            published_at=datetime.fromisoformat(data["published_at"])
            if data.get("published_at")
            else datetime.now(UTC),
        )


@runtime_checkable
class StreamBroker(Protocol):
    """The transport contract the rest of the system depends on."""

    async def start(self) -> None: ...

    async def close(self) -> None: ...

    async def publish(self, message: StreamMessage) -> None: ...

    def subscribe(self, topic: str) -> AsyncIterator[StreamMessage]: ...


@dataclass(slots=True)
class SubscriberStats:
    delivered: int = 0
    dropped: int = 0


class InProcessBroker:
    """Asyncio fan-out broker. The default transport.

    Each subscriber gets its own bounded queue, so a slow consumer degrades only
    itself. Publishing is non-blocking and never awaits a consumer.
    """

    def __init__(self, queue_size: int = DEFAULT_QUEUE_SIZE) -> None:
        if queue_size <= 0:
            msg = f"queue_size must be positive, got {queue_size}"
            raise ValueError(msg)
        self._queue_size = queue_size
        self._subscribers: dict[str, list[asyncio.Queue[StreamMessage]]] = {}
        self._stats: dict[str, SubscriberStats] = {}
        self.published = 0
        self._closed = False

    async def start(self) -> None:
        self._closed = False

    async def close(self) -> None:
        """Signal every subscriber to finish."""
        self._closed = True
        for queues in self._subscribers.values():
            for queue in queues:
                with contextlib.suppress(asyncio.QueueFull):
                    queue.put_nowait(_SENTINEL)
        self._subscribers.clear()

    async def publish(self, message: StreamMessage) -> None:
        if self._closed:
            return
        self.published += 1
        stats = self._stats.setdefault(message.topic, SubscriberStats())
        for queue in self._subscribers.get(message.topic, []):
            try:
                queue.put_nowait(message)
                stats.delivered += 1
            except asyncio.QueueFull:
                # Drop the oldest, then retry. The newest transaction is the one
                # that matters; a stale backlog is worthless in fraud detection.
                with contextlib.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
                    stats.dropped += 1
                with contextlib.suppress(asyncio.QueueFull):
                    queue.put_nowait(message)

    def subscribe(self, topic: str) -> _Subscription:
        """Return an async iterator over ``topic``.

        The subscription unregisters itself on ``aclose()``, so a WebSocket
        client that disconnects does not leak a queue.
        """
        queue: asyncio.Queue[StreamMessage] = asyncio.Queue(maxsize=self._queue_size)
        self._subscribers.setdefault(topic, []).append(queue)
        self._stats.setdefault(topic, SubscriberStats())
        return _Subscription(self, topic, queue)

    def _unsubscribe(self, topic: str, queue: asyncio.Queue[StreamMessage]) -> None:
        queues = self._subscribers.get(topic)
        if queues is not None and queue in queues:
            queues.remove(queue)

    # -- introspection -----------------------------------------------------

    def subscriber_count(self, topic: str) -> int:
        return len(self._subscribers.get(topic, []))

    def stats(self, topic: str) -> SubscriberStats:
        return self._stats.get(topic, SubscriberStats())

    @property
    def dropped_total(self) -> int:
        return sum(s.dropped for s in self._stats.values())


#: Sentinel that terminates a subscriber loop on close.
_SENTINEL = StreamMessage(topic="__close__", payload={})


class _Subscription:
    """An async iterator over one topic that deterministically unregisters.

    Written as an explicit class rather than an async generator on purpose. A
    generator only runs its ``finally`` block once it has been *started*, so
    ``aclose()`` on a subscription that never yielded would silently skip
    cleanup -- and a WebSocket client that connects then disconnects before the
    first message is exactly that case. The queue would be retained for the
    lifetime of the process and keep receiving messages nobody reads.
    """

    def __init__(
        self,
        broker: InProcessBroker,
        topic: str,
        queue: asyncio.Queue[StreamMessage],
    ) -> None:
        self._broker = broker
        self._topic = topic
        self._queue = queue
        self._closed = False

    def __aiter__(self) -> _Subscription:
        return self

    async def __anext__(self) -> StreamMessage:
        if self._closed:
            raise StopAsyncIteration
        message = await self._queue.get()
        if message is _SENTINEL:
            await self.aclose()
            raise StopAsyncIteration
        return message

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._broker._unsubscribe(self._topic, self._queue)

    @property
    def closed(self) -> bool:
        return self._closed


class KafkaBroker:
    """Kafka/Redpanda transport for the ``docker`` profile.

    Not exercised in CI: this machine has no Docker daemon, so the code path is
    written against the ``aiokafka`` API and validated by the contract tests that
    every ``StreamBroker`` must satisfy, but it has not been run against a live
    cluster. That limitation is stated here rather than left for someone to
    discover.
    """

    def __init__(self, bootstrap_servers: str, *, client_id: str = "fraudlens") -> None:
        self._bootstrap = bootstrap_servers
        self._client_id = client_id
        self._producer: Any | None = None

    async def start(self) -> None:
        try:
            from aiokafka import AIOKafkaProducer
        except ImportError as exc:  # pragma: no cover - optional extra
            msg = "aiokafka is required for the docker profile: pip install -e '.[kafka]'"
            raise ImportError(msg) from exc

        self._producer = AIOKafkaProducer(
            bootstrap_servers=self._bootstrap,
            client_id=self._client_id,
            value_serializer=lambda v: v.encode("utf-8"),
            # Fraud decisions must not be silently lost in transit.
            acks="all",
            enable_idempotence=True,
        )
        await self._producer.start()
        logger.info("KafkaBroker connected to %s", self._bootstrap)

    async def close(self) -> None:
        if self._producer is not None:
            await self._producer.stop()
            self._producer = None

    async def publish(self, message: StreamMessage) -> None:
        if self._producer is None:
            msg = "KafkaBroker.publish called before start()"
            raise RuntimeError(msg)
        await self._producer.send_and_wait(
            message.topic,
            value=message.to_json(),
            key=message.key.encode("utf-8") if message.key else None,
        )

    async def _consume(self, topic: str) -> AsyncIterator[StreamMessage]:
        from aiokafka import AIOKafkaConsumer

        consumer = AIOKafkaConsumer(
            topic,
            bootstrap_servers=self._bootstrap,
            client_id=f"{self._client_id}-{topic}",
            auto_offset_reset="latest",
        )
        await consumer.start()
        try:
            async for record in consumer:
                yield StreamMessage.from_json(record.value)
        finally:
            await consumer.stop()

    def subscribe(self, topic: str) -> AsyncIterator[StreamMessage]:
        return self._consume(topic)


def build_broker(profile: str, bootstrap_servers: str) -> StreamBroker:
    """Bind the transport named by the configured profile.

    Together with :func:`fraudlens.features.store.build_feature_store`, this is
    one of only two places that branch on profile.
    """
    if profile == "docker":
        logger.info("Using KafkaBroker at %s", bootstrap_servers)
        return KafkaBroker(bootstrap_servers)
    logger.info("Using InProcessBroker")
    return InProcessBroker()
