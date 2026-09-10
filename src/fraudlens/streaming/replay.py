"""Replay historical transactions as a live stream.

The demo needs transactions arriving continuously. Rather than inventing traffic,
this replays the **held-out test period** at a configurable multiple of real
time -- so the live dashboard is scoring data the model has genuinely never seen,
and the alerts appearing on screen are real detections rather than a scripted
animation.

Timing
------
Inter-arrival gaps are preserved and divided by ``speed``. At ``speed=60`` an
hour of history plays in a minute, which keeps a demo lively without collapsing
the temporal structure the velocity features depend on. Gaps are clamped so a
quiet overnight stretch does not stall the stream for minutes.

The replay is deliberately *not* uniform: bursts stay bursts. That matters
because the whole point of the velocity features is detecting bursts, and a
smoothed replay would never trigger them.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

import pandas as pd

from fraudlens.config import constants as C
from fraudlens.streaming.broker import TOPIC_TRANSACTIONS, StreamMessage

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from fraudlens.streaming.broker import StreamBroker

logger = logging.getLogger(__name__)

#: Never wait longer than this between two replayed transactions, however long
#: the real gap was. Without a cap an overnight lull would freeze the demo.
MAX_GAP_SECONDS = 2.0

#: Never go faster than this, so the UI has time to render and a viewer can
#: actually follow what is happening.
MIN_GAP_SECONDS = 0.02


@dataclass(slots=True)
class ReplayStats:
    published: int = 0
    loops: int = 0
    started_at: datetime | None = None
    last_published_at: datetime | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "published": self.published,
            "loops": self.loops,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "last_published_at": (
                self.last_published_at.isoformat() if self.last_published_at else None
            ),
        }


@dataclass(slots=True)
class ReplayProducer:
    """Publishes historical transactions onto the bus at ``speed``x real time."""

    transactions: pd.DataFrame
    broker: StreamBroker
    speed: float = 60.0
    loop: bool = True
    topic: str = TOPIC_TRANSACTIONS
    stats: ReplayStats = field(default_factory=ReplayStats)
    _task: asyncio.Task[None] | None = field(default=None, init=False, repr=False)
    _stopping: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.speed <= 0:
            msg = f"speed must be positive, got {self.speed}"
            raise ValueError(msg)
        if self.transactions.empty:
            msg = "cannot replay an empty transaction frame"
            raise ValueError(msg)
        if C.TIMESTAMP_COL not in self.transactions.columns:
            msg = f"replay frame needs a {C.TIMESTAMP_COL!r} column"
            raise KeyError(msg)
        self.transactions = self.transactions.sort_values(C.TIMESTAMP_COL, kind="stable")

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Launch the replay as a background task."""
        if self._task is not None and not self._task.done():
            logger.debug("Replay already running")
            return
        self._stopping = False
        self._task = asyncio.create_task(self.run(), name="fraudlens-replay")
        logger.info(
            "Replay started: %d transactions at %.0fx real time (loop=%s)",
            len(self.transactions),
            self.speed,
            self.loop,
        )

    async def stop(self) -> None:
        """Ask the replay to finish and wait for it."""
        self._stopping = True
        if self._task is None:
            return
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None
        logger.info("Replay stopped after %d transactions", self.stats.published)

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    def _should_stop(self) -> bool:
        """Whether a stop has been requested.

        Read through a method rather than touching ``self._stopping`` directly.
        Inside ``while not self._stopping`` a type checker narrows the attribute
        to ``False`` and flags every later check as unreachable -- it cannot see
        that :meth:`stop` mutates the flag while the loop is suspended at an
        ``await``. The indirection keeps the (genuinely necessary) mid-loop check
        honest.
        """
        return self._stopping

    # -- the loop ----------------------------------------------------------

    async def run(self) -> None:
        """Replay until stopped, looping if configured."""
        self.stats.started_at = datetime.now()
        while not self._should_stop():
            async for delay, payload in self._paced():
                if self._should_stop():
                    return
                if delay > 0:
                    await asyncio.sleep(delay)
                await self.broker.publish(
                    StreamMessage(
                        topic=self.topic,
                        payload=payload,
                        key=str(payload.get(C.CARD_COL, "")),
                    )
                )
                self.stats.published += 1
                self.stats.last_published_at = datetime.now()

            self.stats.loops += 1
            if not self.loop:
                logger.info("Replay finished (%d transactions)", self.stats.published)
                return
            logger.debug("Replay looping (%d complete passes)", self.stats.loops)

    async def _paced(self) -> AsyncIterator[tuple[float, dict[str, Any]]]:
        """Yield ``(delay_seconds, payload)`` preserving relative inter-arrival gaps."""
        previous: datetime | None = None
        for row in self._rows():
            current = row[C.TIMESTAMP_COL]
            if isinstance(current, pd.Timestamp):
                current = current.to_pydatetime()

            if previous is None:
                delay = 0.0
            else:
                real_gap = (current - previous).total_seconds()
                delay = max(MIN_GAP_SECONDS, min(MAX_GAP_SECONDS, real_gap / self.speed))
            previous = current
            yield delay, _serialise(row)

    def _rows(self) -> Iterator[dict[str, Any]]:
        yield from self.transactions.to_dict("records")


def _serialise(row: dict[str, Any]) -> dict[str, Any]:
    """Convert a dataframe row into a JSON-safe payload."""
    out: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, pd.Timestamp):
            out[key] = value.to_pydatetime().isoformat()
        elif value is None or (isinstance(value, float) and value != value):
            out[key] = None
        elif hasattr(value, "item"):  # numpy scalar
            out[key] = value.item()
        else:
            out[key] = value
    return out


def load_replay_source(
    path: Any = None,
    *,
    limit: int | None = None,
) -> pd.DataFrame:
    """Load transactions to replay, preferring the held-out test fold.

    Falls back to the committed sample so the demo works on a fresh clone with no
    processed data present.
    """
    from fraudlens.config import get_settings
    from fraudlens.data.loader import load_sample
    from fraudlens.data.splits import Split, split_frame

    settings = get_settings()

    # The committed bundle is a slice of the held-out test fold, so a deployed
    # instance streams data the model has genuinely never seen without carrying
    # the full 29 MB transaction table in git.
    bundle = settings.artifact_dir / "demo" / "replay.parquet"
    if path is None and bundle.exists():
        frame = pd.read_parquet(bundle)
        logger.info("Replaying the committed demo slice (%d rows)", len(frame))
        return frame.head(limit) if limit else frame

    source = path or (settings.processed_dir / "transactions.parquet")

    if source.exists():
        frame = pd.read_parquet(source)
        frames, _ = split_frame(frame)
        test = frames[Split.TEST]
        if not test.empty:
            logger.info("Replaying the held-out test fold (%d rows)", len(test))
            return test.head(limit) if limit else test
        frame = frame
    else:
        logger.warning("No processed dataset at %s; replaying the committed sample", source)
        frame = load_sample()

    return frame.head(limit) if limit else frame


def load_warmup_window(path: Any = None) -> pd.DataFrame:
    """Transactions immediately preceding the replay period, for warming state.

    In production a feature store is never cold: cards carry weeks of history and
    the trailing-window features are meaningful from the first request. A demo
    that starts empty misrepresents the system -- every card looks brand new,
    velocity features read zero, and drift monitoring compares mature training
    data against cold-start noise.

    This returns the ``MAX_LOOKBACK_DAYS`` of transactions before the test fold
    begins. Folding them into the store reproduces the production condition.
    They are *never scored* and never reach the model -- they only build state,
    so no evaluation is contaminated.
    """
    from datetime import timedelta

    from fraudlens.config import get_settings

    settings = get_settings()

    bundle = settings.artifact_dir / "demo" / "warmup.parquet"
    if path is None and bundle.exists():
        frame = pd.read_parquet(bundle)
        logger.info("Warm-up window from the committed bundle (%d rows)", len(frame))
        return frame

    source = path or (settings.processed_dir / "transactions.parquet")
    if not source.exists():
        return pd.DataFrame()

    frame = pd.read_parquet(source)
    test_start = pd.Timestamp(C.TEST_START)
    window_start = test_start - timedelta(days=C.MAX_LOOKBACK_DAYS)

    warm = frame[(frame[C.TIMESTAMP_COL] >= window_start) & (frame[C.TIMESTAMP_COL] < test_start)]
    logger.info(
        "Warm-up window: %d transactions from %s to %s",
        len(warm),
        window_start.date(),
        test_start.date(),
    )
    return warm.sort_values(C.TIMESTAMP_COL, kind="stable")
