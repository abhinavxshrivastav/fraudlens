"""Online feature store: the card history that serving needs at request time.

The problem it solves
---------------------
Behavioural features are trailing aggregates over a card's recent transactions.
At training time that history is simply the previous rows of the dataframe. At
serving time it has to come from somewhere, in single-digit milliseconds, for a
card the system may not have seen in weeks.

The store holds a bounded :class:`~fraudlens.features.pipeline.CardState` per
card -- the same object the training replay uses, which is what makes train/serve
parity structural rather than aspirational.

Implementations
---------------
``InMemoryFeatureStore``
    Default. Fast, no dependencies, lost on restart. Correct for a single
    process, which is what the native profile runs.

``RedisFeatureStore``
    Selected by ``FRAUDLENS_PROFILE=docker``. Shared across processes and
    survives restarts.

Both satisfy :class:`FeatureStore`, so nothing above this layer knows which is
bound.
"""

from __future__ import annotations

import logging
import pickle
import threading
from collections import OrderedDict
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from fraudlens.features.pipeline import CardState

if TYPE_CHECKING:
    from fraudlens.features.pipeline import TransactionEvent

logger = logging.getLogger(__name__)

#: Cards tracked before the least-recently-used is evicted. Bounds memory on an
#: unbounded card population; an evicted card simply starts cold, which the
#: `card_history_count` feature signals to the model.
DEFAULT_CAPACITY = 100_000


@runtime_checkable
class FeatureStore(Protocol):
    """The contract the scoring path depends on."""

    def get(self, card: int) -> CardState:
        """Return the card's state, creating an empty one if unknown."""
        ...

    def update(self, card: int, event: TransactionEvent) -> None:
        """Fold a scored transaction into the card's state."""
        ...

    def size(self) -> int:
        """Number of cards currently tracked."""
        ...

    def clear(self) -> None: ...


class InMemoryFeatureStore:
    """LRU-bounded in-process store.

    Thread-safe because uvicorn may run the scoring path from a worker thread
    pool. The lock is uncontended in the common case and the critical sections
    are tiny, so it costs far less than the latency budget allows.
    """

    def __init__(self, capacity: int = DEFAULT_CAPACITY) -> None:
        if capacity <= 0:
            msg = f"capacity must be positive, got {capacity}"
            raise ValueError(msg)
        self._capacity = capacity
        self._states: OrderedDict[int, CardState] = OrderedDict()
        self._lock = threading.Lock()
        self.evictions = 0

    def get(self, card: int) -> CardState:
        with self._lock:
            state = self._states.get(card)
            if state is None:
                state = CardState(card=card)
                self._states[card] = state
            self._states.move_to_end(card)
            self._evict_if_needed()
            return state

    def update(self, card: int, event: TransactionEvent) -> None:
        with self._lock:
            state = self._states.get(card)
            if state is None:
                state = CardState(card=card)
                self._states[card] = state
            state.update(event)
            self._states.move_to_end(card)
            self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        while len(self._states) > self._capacity:
            self._states.popitem(last=False)
            self.evictions += 1

    def size(self) -> int:
        with self._lock:
            return len(self._states)

    def clear(self) -> None:
        with self._lock:
            self._states.clear()

    def warm(self, states: dict[int, CardState]) -> None:
        """Preload state, e.g. from a backfill or a demo seed.

        A cold store makes the first transaction on every card look like a new
        card, which suppresses exactly the velocity signals the model relies on.
        Warming avoids a burst of degraded decisions after a deploy.
        """
        with self._lock:
            for card, state in states.items():
                self._states[card] = state
            self._evict_if_needed()
        logger.info("Warmed feature store with %d card states", len(states))


class RedisFeatureStore:
    """Redis-backed store for the ``docker`` profile.

    Security note: card state is serialised with ``pickle``, which is unsafe
    against untrusted input. That is acceptable only because this Redis instance
    is written exclusively by FraudLens itself and is not exposed outside the
    compose network. If that ever changes, this must move to an explicit
    serialisation format -- the convenience is not worth a deserialisation
    vulnerability on the scoring path.
    """

    def __init__(self, url: str, *, ttl_seconds: int = 60 * 60 * 24 * 45) -> None:
        try:
            import redis
        except ImportError as exc:  # pragma: no cover - optional extra
            msg = "redis is required for the docker profile: pip install -e '.[redis]'"
            raise ImportError(msg) from exc
        self._client = redis.Redis.from_url(url)
        self._ttl = ttl_seconds

    @staticmethod
    def _key(card: int) -> str:
        return f"fraudlens:cardstate:{card}"

    def get(self, card: int) -> CardState:
        raw = self._client.get(self._key(card))
        if raw is None:
            return CardState(card=card)
        try:
            state = pickle.loads(raw)  # noqa: S301 - see class docstring
        except Exception:
            logger.warning("Corrupt card state for %s; starting cold", card, exc_info=True)
            return CardState(card=card)
        return state if isinstance(state, CardState) else CardState(card=card)

    def update(self, card: int, event: TransactionEvent) -> None:
        state = self.get(card)
        state.update(event)
        self._client.setex(self._key(card), self._ttl, pickle.dumps(state))

    def size(self) -> int:
        return int(self._client.dbsize())

    def clear(self) -> None:
        self._client.flushdb()


def build_feature_store(profile: str, redis_url: str) -> FeatureStore:
    """Bind the implementation named by the configured profile.

    The only place in the codebase that branches on profile.
    """
    if profile == "docker":
        logger.info("Using RedisFeatureStore at %s", redis_url)
        return RedisFeatureStore(redis_url)
    logger.info("Using InMemoryFeatureStore")
    return InMemoryFeatureStore()
