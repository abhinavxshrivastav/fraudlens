"""Behavioural feature engineering.

Design: one code path, not two
-------------------------------
The usual way to build this is twice: a vectorised pandas implementation for
training, and a separate online implementation for serving. The two then drift
apart, and the model is served features subtly different from the ones it was
trained on. This is *train/serve skew*, and it is the classic silent production
failure — it raises no error, it just quietly degrades performance.

FraudLens has a single implementation. Features are computed by
:meth:`FeaturePipeline.compute` from a :class:`CardState` (the card's trailing
history) plus the current transaction. Training replays transactions through the
same state machine that serving uses. Parity is therefore true by construction
rather than by discipline, and ``tests/integration/test_train_serve_parity.py``
asserts it anyway.

The cost is speed: a Python state machine over ~1.8M rows is slower than
vectorised rolling windows. That is an acceptable trade for a batch job that
runs occasionally, and the PySpark backfill exists for when it is not.

Leakage safety
--------------
:meth:`compute` reads state that reflects only transactions strictly *before*
the current one. State is advanced by :meth:`CardState.update` only *after*
features are computed. A feature therefore cannot see its own transaction or any
later one — enforced by ordering in :meth:`transform_frame`, not by convention.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd

from fraudlens.config import constants as C

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

EARTH_RADIUS_KM: Final = 6371.0088

#: Trailing windows, in seconds, over which velocity aggregates are computed.
WINDOWS: Final[dict[str, int]] = {
    "1h": 3_600,
    "24h": 86_400,
    "7d": 604_800,
    "30d": 2_592_000,
}

#: Hours considered "night". Card-present fraud in this dataset concentrates
#: heavily in the small hours, when the genuine cardholder is asleep.
NIGHT_HOURS: Final[frozenset[int]] = frozenset({22, 23, 0, 1, 2, 3})

#: Speed above which travel between two transaction locations is implausible.
#: Above commercial flight cruise speed, so this fires on genuinely impossible
#: journeys rather than on a customer who took a plane.
IMPOSSIBLE_SPEED_KMH: Final = 900.0


def _to_datetime(value: Any) -> datetime | None:
    """Coerce a timestamp-ish value to a plain ``datetime``.

    Handles ``pd.Timestamp``, ``datetime``, ``None`` and ``pd.NaT`` uniformly so
    callers do not have to know which one a given frame produced.
    """
    if value is None or pd.isna(value):
        # pd.NaT subclasses datetime, so an isinstance check alone lets it
        # through and it would silently become a real-looking date of birth.
        return None
    if isinstance(value, datetime):
        return value
    to_pydatetime = getattr(value, "to_pydatetime", None)
    if to_pydatetime is not None:
        try:
            return to_pydatetime()  # type: ignore[no-any-return]
        except (ValueError, AttributeError):
            return None
    return None


def _require_datetime(value: Any) -> datetime:
    """Like :func:`_to_datetime`, but a missing timestamp is a hard error.

    Every temporal and velocity feature is defined relative to this value, so a
    null timestamp cannot be given a sensible default.
    """
    parsed = _to_datetime(value)
    if parsed is None:
        msg = f"transaction timestamp is required but was {value!r}"
        raise ValueError(msg)
    return parsed


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in kilometres."""
    if any(math.isnan(v) for v in (lat1, lon1, lat2, lon2)):
        return math.nan
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(min(1.0, a)))


@dataclass(frozen=True, slots=True)
class TransactionEvent:
    """One transaction, in the form the feature pipeline consumes.

    Deliberately decoupled from the raw CSV schema so that serving does not
    depend on the shape of a Kaggle file.
    """

    timestamp: datetime
    card: int
    amount: float
    merchant: str
    category: str
    home_lat: float
    home_lon: float
    merch_lat: float
    merch_lon: float
    city_pop: int = 0
    dob: datetime | None = None
    transaction_id: str = ""

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> TransactionEvent:
        """Build an event from a raw row.

        Accepts any mapping -- a plain ``dict`` as well as a ``pd.Series``.
        Taking dicts matters for throughput: wrapping each of a million rows in
        a ``pd.Series`` costs more than everything else in the loop combined.
        """
        return cls(
            timestamp=_require_datetime(row[C.TIMESTAMP_COL]),
            card=int(row[C.CARD_COL]),
            amount=float(row[C.AMOUNT_COL]),
            merchant=str(row["merchant"]),
            category=str(row["category"]),
            home_lat=float(row[C.HOME_LAT_COL]),
            home_lon=float(row[C.HOME_LON_COL]),
            merch_lat=float(row[C.MERCH_LAT_COL]),
            merch_lon=float(row[C.MERCH_LON_COL]),
            city_pop=int(row.get("city_pop", 0) or 0),
            dob=_to_datetime(row.get("dob")),
            transaction_id=str(row.get(C.TRANSACTION_ID_COL, "")),
        )


@dataclass(slots=True)
class _HistoryEntry:
    timestamp: datetime
    amount: float
    merchant: str
    category: str
    lat: float
    lon: float


@dataclass(slots=True)
class CardState:
    """Trailing transaction history for a single card.

    Bounded by :data:`~fraudlens.config.constants.MAX_LOOKBACK_DAYS`: entries
    older than the longest feature window are dropped, so memory stays flat
    regardless of how long a card has been active. This is what makes the same
    state object viable as the online feature store value.
    """

    card: int
    history: deque[_HistoryEntry] = field(default_factory=deque)
    seen_merchants: set[str] = field(default_factory=set)
    seen_categories: set[str] = field(default_factory=set)

    def prune(self, now: datetime) -> None:
        """Drop history older than the longest window."""
        cutoff = now - timedelta(days=C.MAX_LOOKBACK_DAYS)
        while self.history and self.history[0].timestamp < cutoff:
            self.history.popleft()

    def update(self, event: TransactionEvent) -> None:
        """Fold a transaction into the state. Called *after* features are computed."""
        self.history.append(
            _HistoryEntry(
                timestamp=event.timestamp,
                amount=event.amount,
                merchant=event.merchant,
                category=event.category,
                lat=event.merch_lat,
                lon=event.merch_lon,
            )
        )
        self.seen_merchants.add(event.merchant)
        self.seen_categories.add(event.category)
        self.prune(event.timestamp)

    def within(self, now: datetime, seconds: int) -> Iterator[_HistoryEntry]:
        """Entries strictly within the trailing window ending at ``now``."""
        cutoff = now - timedelta(seconds=seconds)
        for entry in reversed(self.history):
            if entry.timestamp < cutoff:
                break
            yield entry

    @property
    def last(self) -> _HistoryEntry | None:
        return self.history[-1] if self.history else None

    @property
    def is_empty(self) -> bool:
        return not self.history


@dataclass(slots=True)
class _WindowAggregates:
    """Everything the window-based features need, from one pass over history.

    The trailing windows are nested (1h within 24h within 7d within 30d), so a
    single reverse walk that stops at the widest cutoff can fill all of them.
    Computing them independently walked the deque four times per transaction and
    dominated the profile -- 3.1M generator resumptions per 40k rows.
    """

    counts: dict[str, int] = field(default_factory=lambda: dict.fromkeys(WINDOWS, 0))
    sums: dict[str, float] = field(default_factory=lambda: dict.fromkeys(WINDOWS, 0.0))
    merchants_24h: set[str] = field(default_factory=set)
    categories_7d: set[str] = field(default_factory=set)
    #: Running moments over the 30-day window, for mean/std without numpy.
    month_sum: float = 0.0
    month_sum_sq: float = 0.0
    month_max: float = 0.0
    month_n: int = 0

    @property
    def month_mean(self) -> float:
        return self.month_sum / self.month_n if self.month_n else 0.0

    @property
    def month_std(self) -> float:
        """Population standard deviation, matching ``np.std(ddof=0)``."""
        if self.month_n == 0:
            return 0.0
        variance = self.month_sum_sq / self.month_n - self.month_mean**2
        # Catastrophic cancellation can push a mathematically-zero variance
        # slightly negative; clamping keeps the sqrt real.
        return math.sqrt(max(0.0, variance))


def _aggregate_windows(state: CardState, now: datetime) -> _WindowAggregates:
    """Walk the card's history once, filling every trailing window."""
    agg = _WindowAggregates()
    cutoffs = {name: now - timedelta(seconds=secs) for name, secs in WINDOWS.items()}
    widest = cutoffs["30d"]

    for entry in reversed(state.history):
        if entry.timestamp < widest:
            break
        ts = entry.timestamp
        for name, cutoff in cutoffs.items():
            if ts >= cutoff:
                agg.counts[name] += 1
                agg.sums[name] += entry.amount
        if ts >= cutoffs["24h"]:
            agg.merchants_24h.add(entry.merchant)
        if ts >= cutoffs["7d"]:
            agg.categories_7d.add(entry.category)
        agg.month_n += 1
        agg.month_sum += entry.amount
        agg.month_sum_sq += entry.amount * entry.amount
        agg.month_max = max(agg.month_max, entry.amount)
    return agg


class FeaturePipeline:
    """Computes the model feature vector for a transaction.

    Stateless itself; all state lives in :class:`CardState` objects supplied by
    the caller (a dict in training, the online feature store in serving).
    """

    #: Feature names in a fixed order. The order is part of the model contract:
    #: a trained artefact expects columns in exactly this sequence.
    FEATURE_NAMES: Final[tuple[str, ...]] = (
        # amount
        "amt",
        "amt_log",
        "amt_zscore_30d",
        "amt_ratio_mean_30d",
        "amt_ratio_max_30d",
        # velocity
        "txn_count_1h",
        "txn_count_24h",
        "txn_count_7d",
        "txn_count_30d",
        "amt_sum_1h",
        "amt_sum_24h",
        "amt_sum_7d",
        # geo
        "dist_home_merch_km",
        "dist_prev_txn_km",
        "implied_speed_kmh",
        "is_impossible_travel",
        # temporal
        "hour",
        "day_of_week",
        "is_night",
        "is_weekend",
        "secs_since_prev_txn",
        # novelty
        "is_new_merchant",
        "is_new_category",
        "distinct_merchants_24h",
        "distinct_categories_7d",
        # demographic
        "age_years",
        "city_pop_log",
        # history depth (lets the model discount unreliable early features)
        "card_history_count",
    )

    def compute(self, event: TransactionEvent, state: CardState) -> dict[str, float]:
        """Return the feature vector for ``event`` given the card's prior history.

        ``state`` must reflect only transactions strictly before ``event``.
        """
        now = event.timestamp
        windows = _aggregate_windows(state, now)

        features: dict[str, float] = {}
        features.update(self._amount_features(event, windows))
        features.update(self._velocity_features(windows))
        features.update(self._geo_features(event, state))
        features.update(self._temporal_features(event, state))
        features.update(self._novelty_features(event, state, windows))
        features.update(self._demographic_features(event))
        features["card_history_count"] = float(len(state.history))
        return features

    # -- feature groups ----------------------------------------------------

    @staticmethod
    def _amount_features(event: TransactionEvent, windows: _WindowAggregates) -> dict[str, float]:
        """How unusual is this amount *for this card*?

        An absolute amount says little: 500 is routine for one cardholder and
        unprecedented for another. The deviation features carry the signal.
        """
        amt = event.amount
        out = {"amt": amt, "amt_log": math.log1p(max(0.0, amt))}

        if windows.month_n == 0:
            # No history: report neutral values rather than NaN so that tree
            # splits on "unusual" are not conflated with "unknown". The
            # card_history_count feature lets the model learn to discount these.
            out["amt_zscore_30d"] = 0.0
            out["amt_ratio_mean_30d"] = 1.0
            out["amt_ratio_max_30d"] = 1.0
            return out

        mean = windows.month_mean
        std = windows.month_std
        maximum = windows.month_max

        out["amt_zscore_30d"] = (amt - mean) / std if std > 1e-9 else 0.0
        out["amt_ratio_mean_30d"] = amt / mean if mean > 1e-9 else 1.0
        out["amt_ratio_max_30d"] = amt / maximum if maximum > 1e-9 else 1.0
        return out

    @staticmethod
    def _velocity_features(windows: _WindowAggregates) -> dict[str, float]:
        """Burst detection. Card testing shows up as a spike in short windows."""
        out: dict[str, float] = {}
        for name in WINDOWS:
            out[f"txn_count_{name}"] = float(windows.counts[name])
        for name in ("1h", "24h", "7d"):
            out[f"amt_sum_{name}"] = float(windows.sums[name])
        return out

    @staticmethod
    def _geo_features(event: TransactionEvent, state: CardState) -> dict[str, float]:
        """Distance and implied travel speed.

        ``implied_speed_kmh`` is the strongest single geo signal: a card used in
        two places faster than any journey could connect them means one of the
        two is not the cardholder.
        """
        out = {
            "dist_home_merch_km": haversine_km(
                event.home_lat, event.home_lon, event.merch_lat, event.merch_lon
            )
        }

        previous = state.last
        if previous is None:
            out["dist_prev_txn_km"] = 0.0
            out["implied_speed_kmh"] = 0.0
            out["is_impossible_travel"] = 0.0
            return out

        distance = haversine_km(previous.lat, previous.lon, event.merch_lat, event.merch_lon)
        elapsed_hours = (event.timestamp - previous.timestamp).total_seconds() / 3600.0

        out["dist_prev_txn_km"] = distance
        if elapsed_hours <= 0 or math.isnan(distance):
            # Same-instant transactions in different places: treat as maximally
            # suspicious rather than dividing by zero.
            speed = IMPOSSIBLE_SPEED_KMH * 2 if distance > 1.0 else 0.0
        else:
            speed = distance / elapsed_hours
        out["implied_speed_kmh"] = speed
        out["is_impossible_travel"] = float(speed > IMPOSSIBLE_SPEED_KMH)
        return out

    @staticmethod
    def _temporal_features(event: TransactionEvent, state: CardState) -> dict[str, float]:
        ts = event.timestamp
        previous = state.last
        gap = (ts - previous.timestamp).total_seconds() if previous else -1.0
        return {
            "hour": float(ts.hour),
            "day_of_week": float(ts.weekday()),
            "is_night": float(ts.hour in NIGHT_HOURS),
            "is_weekend": float(ts.weekday() >= 5),
            "secs_since_prev_txn": gap,
        }

    @staticmethod
    def _novelty_features(
        event: TransactionEvent,
        state: CardState,
        windows: _WindowAggregates,
    ) -> dict[str, float]:
        """First-time merchants and category spread.

        A card suddenly transacting across many distinct merchants in a day is
        the signature of card testing — an attacker validating stolen numbers
        with small purchases before a large one.
        """
        return {
            "is_new_merchant": float(event.merchant not in state.seen_merchants),
            "is_new_category": float(event.category not in state.seen_categories),
            "distinct_merchants_24h": float(len(windows.merchants_24h)),
            "distinct_categories_7d": float(len(windows.categories_7d)),
        }

    @staticmethod
    def _demographic_features(event: TransactionEvent) -> dict[str, float]:
        if event.dob is None:
            age = math.nan
        else:
            delta = event.timestamp - event.dob
            age = delta.days / 365.25
        return {
            "age_years": age,
            "city_pop_log": math.log1p(max(0, event.city_pop)),
        }

    # -- batch driver ------------------------------------------------------

    def transform_frame(
        self,
        df: pd.DataFrame,
        *,
        states: dict[int, CardState] | None = None,
        keep_columns: tuple[str, ...] = (),
    ) -> pd.DataFrame:
        """Compute features for every row, replaying transactions in time order.

        ``states`` may carry warm state in from an earlier period. It is mutated
        in place, so the caller can chain calls across folds — which is exactly
        what must *not* happen across a split boundary, and why
        :mod:`fraudlens.data.splits` enforces an embargo.
        """
        if df.empty:
            return pd.DataFrame(columns=[*self.FEATURE_NAMES, *keep_columns])

        ordered = df.sort_values(C.TIMESTAMP_COL, kind="stable")
        card_states = states if states is not None else {}

        # Fill a preallocated array rather than accumulating one dict per row.
        # At ~1.8M rows the dicts alone ran to gigabytes and the resulting
        # paging cost more than the computation.
        names = self.FEATURE_NAMES
        matrix = np.empty((len(ordered), len(names)), dtype=np.float64)

        for i, row in enumerate(ordered.to_dict("records")):
            event = TransactionEvent.from_row(row)
            state = card_states.setdefault(event.card, CardState(card=event.card))
            state.prune(event.timestamp)

            computed = self.compute(event, state)
            for j, name in enumerate(names):
                matrix[i, j] = computed[name]

            # Ordering is the leakage guard: state advances only after the
            # feature vector for this transaction has been captured.
            state.update(event)

        features = pd.DataFrame(matrix, index=ordered.index, columns=list(names))
        for column in keep_columns:
            if column in ordered.columns:
                features[column] = ordered[column]
        return features

    def to_vector(self, features: dict[str, float]) -> np.ndarray:
        """Order a feature dict into the array layout the model expects."""
        return np.array([features[name] for name in self.FEATURE_NAMES], dtype=np.float64)
