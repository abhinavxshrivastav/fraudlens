"""Synthetic transaction generator with injected fraud typologies.

Purpose
-------
Three jobs, none of which the real dataset does well:

1. **Unblocking.** The pipeline is developable and testable before anyone has
   Kaggle credentials, and CI never needs them.
2. **Ground truth about mechanisms.** Real labels say *that* a transaction was
   fraud, not *why*. Here each fraudulent transaction is tagged with the
   typology that produced it, so detection can be measured per attack pattern —
   which is how a fraud team actually reasons about coverage.
3. **The live demo.** The replay stream needs data that keeps arriving.

Output matches the Sparkov schema exactly, so anything built against synthetic
data runs unchanged on the real download.

Typologies
----------
Modelled on genuine attack patterns rather than random label flipping:

``card_testing``
    An attacker validates a stolen card with several small purchases across
    unrelated merchants in minutes, before risking a large one.
``account_takeover``
    High-value purchases at merchants the card has never used, usually
    overnight, after credentials are compromised.
``geo_impossible``
    The card is cloned; the clone is used far from where the genuine holder
    just transacted, implying a physically impossible journey.
``spending_spree``
    Escalating purchases in a short window before the card is reported.
``subtle``
    Deliberately unremarkable: a familiar merchant, a normal amount, during the
    day. Represents a patient attacker who knows what triggers alerts.

A deliberate design constraint: fraud is *not* trivially separable. Fraudulent
amounts overlap the legitimate distribution, and legitimate customers
occasionally travel, shop at new merchants and buy expensive things. A generator
that makes fraud obvious produces a model that proves nothing.

The ``subtle`` typology exists specifically to keep the problem honest. Without
it every fraudulent transaction carries at least one loud signal, recall
approaches 1.0, and the resulting metrics say more about the generator than the
model. With it there is an irreducible floor of fraud that genuinely cannot be
caught from transaction features alone -- which is also true in production, and
is what makes per-typology recall the interesting measurement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd

from fraudlens.config import constants as C

if TYPE_CHECKING:
    from collections.abc import Sequence


class Typology(StrEnum):
    """Named fraud attack patterns. ``LEGITIMATE`` marks genuine transactions."""

    LEGITIMATE = "legitimate"
    CARD_TESTING = "card_testing"
    ACCOUNT_TAKEOVER = "account_takeover"
    GEO_IMPOSSIBLE = "geo_impossible"
    SPENDING_SPREE = "spending_spree"
    SUBTLE = "subtle"


#: Column carrying the typology label. Present in synthetic data only, and never
#: passed to the model -- it would be a perfect leak of the target.
TYPOLOGY_COL: Final = "fraud_typology"

CATEGORIES: Final[tuple[str, ...]] = (
    "grocery_pos",
    "gas_transport",
    "home",
    "shopping_pos",
    "kids_pets",
    "shopping_net",
    "entertainment",
    "food_dining",
    "personal_care",
    "health_fitness",
    "misc_pos",
    "misc_net",
    "grocery_net",
    "travel",
)

#: Typical spend by category, as (shape, scale) for a gamma distribution.
#: Gamma is right-skewed, which matches real transaction values far better than
#: a normal distribution.
CATEGORY_SPEND: Final[dict[str, tuple[float, float]]] = {
    "grocery_pos": (3.0, 20.0),
    "gas_transport": (4.0, 12.0),
    "home": (2.0, 45.0),
    "shopping_pos": (2.0, 40.0),
    "kids_pets": (2.5, 22.0),
    "shopping_net": (1.8, 55.0),
    "entertainment": (2.0, 28.0),
    "food_dining": (3.0, 18.0),
    "personal_care": (2.5, 20.0),
    "health_fitness": (2.0, 30.0),
    "misc_pos": (2.0, 25.0),
    "misc_net": (1.8, 35.0),
    "grocery_net": (3.0, 25.0),
    "travel": (1.5, 180.0),
}

JOBS: Final[tuple[str, ...]] = (
    "Systems analyst",
    "Nurse, adult",
    "Teacher, secondary school",
    "Civil engineer",
    "Graphic designer",
    "Accountant, chartered",
    "Solicitor",
    "Chef",
    "Electrician",
    "Pharmacist, community",
    "Surveyor, building",
    "Journalist, newspaper",
)

STATES: Final[tuple[str, ...]] = ("NY", "CA", "TX", "OH", "FL", "IL", "PA", "MI", "GA", "NC")

#: Number of population centres. Merchants and customers are both clustered into
#: cities rather than scattered uniformly across the map.
#:
#: This is load-bearing, not cosmetic. With uniformly scattered merchants, two
#: consecutive legitimate purchases land hundreds of kilometres apart, and the
#: geo features become noise -- an early version of this generator flagged 14%
#: of *genuine* transactions as impossible travel. Locality is what makes
#: distance and implied speed informative.
CITY_COUNT: Final = 40

#: Radius over which merchants scatter around a city centre.
CITY_RADIUS_DEG: Final = 0.22


@dataclass(frozen=True, slots=True)
class City:
    index: int
    name: str
    state: str
    lat: float
    lon: float
    population: int


@dataclass(frozen=True, slots=True)
class Customer:
    card: int
    first: str
    last: str
    gender: str
    street: str
    city: str
    state: str
    zip_code: int
    lat: float
    lon: float
    city_pop: int
    job: str
    dob: datetime
    #: Per-customer spend multiplier. Wealthier customers spend more on
    #: everything, which is why absolute amount is a weak signal and
    #: amount-relative-to-this-card is a strong one.
    spend_multiplier: float
    #: Transactions per day on average.
    activity_rate: float
    #: Categories this customer actually uses.
    preferred_categories: tuple[str, ...]
    #: Index of the customer's home city.
    city_index: int
    #: Merchants in the home city -- where routine spending happens.
    home_merchants: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Merchant:
    index: int
    name: str
    category: str
    lat: float
    lon: float
    city_index: int = 0


@dataclass(frozen=True, slots=True)
class GeneratorConfig:
    """Controls the shape of the generated dataset."""

    n_customers: int = 200
    n_merchants: int = 300
    start: datetime = datetime(2019, 1, 1)
    end: datetime = datetime(2020, 12, 31, 23, 59, 59)
    target_fraud_rate: float = 0.006
    seed: int = 42
    #: Relative frequency of each typology, ordered as
    #: (card_testing, account_takeover, geo_impossible, spending_spree, subtle).
    #: The ``subtle`` share sets the difficulty floor: it is roughly the
    #: fraction of fraud that no transaction-level model can be expected to
    #: catch, so raising it lowers achievable recall.
    typology_weights: tuple[float, ...] = (0.28, 0.18, 0.12, 0.18, 0.24)


class SyntheticGenerator:
    """Generates Sparkov-schema transactions with labelled fraud typologies."""

    def __init__(self, config: GeneratorConfig | None = None) -> None:
        self.config = config or GeneratorConfig()
        self._rng = np.random.default_rng(self.config.seed)
        self.cities = self._make_cities()
        self.merchants = self._make_merchants()
        self.customers = self._make_customers()

    # -- population --------------------------------------------------------

    def _make_cities(self) -> list[City]:
        rng = self._rng
        return [
            City(
                index=i,
                name=f"City_{i:02d}",
                state=str(rng.choice(STATES)),
                lat=float(rng.uniform(26.0, 47.0)),
                lon=float(rng.uniform(-123.0, -71.0)),
                population=int(rng.lognormal(10.5, 1.3)) + 500,
            )
            for i in range(CITY_COUNT)
        ]

    def _make_merchants(self) -> list[Merchant]:
        rng = self._rng
        merchants = []
        for i in range(self.config.n_merchants):
            # Every city gets at least one merchant before any city gets two,
            # so no customer is left without somewhere local to shop.
            city = self.cities[i % len(self.cities)]
            merchants.append(
                Merchant(
                    index=i,
                    # The index keeps names unique. Without it the random name
                    # pool collides, and two merchants in different cities share
                    # an identity -- which would silently merge them in every
                    # merchant-level feature and in the analyst console.
                    name=f"fraud_{_merchant_name(rng)}_{i:04d}",
                    category=str(rng.choice(CATEGORIES)),
                    lat=float(city.lat + rng.normal(0.0, CITY_RADIUS_DEG)),
                    lon=float(city.lon + rng.normal(0.0, CITY_RADIUS_DEG)),
                    city_index=city.index,
                )
            )
        return merchants

    def _make_customers(self) -> list[Customer]:
        rng = self._rng
        by_city: dict[int, list[int]] = {c.index: [] for c in self.cities}
        for merchant in self.merchants:
            by_city[merchant.city_index].append(merchant.index)

        customers = []
        for i in range(self.config.n_customers):
            city = self.cities[int(rng.integers(0, len(self.cities)))]
            customers.append(
                Customer(
                    card=int(rng.integers(4_000_000_000_000_000, 4_999_999_999_999_999)),
                    first=f"First{i}",
                    last=f"Last{i}",
                    gender=str(rng.choice(["M", "F"])),
                    street=f"{int(rng.integers(1, 9999))} Example Street",
                    city=city.name,
                    state=city.state,
                    zip_code=int(rng.integers(10_000, 99_999)),
                    # Home sits inside the city, so distance-from-home is small
                    # for routine spending and large for a compromised card.
                    lat=float(city.lat + rng.normal(0.0, CITY_RADIUS_DEG * 0.5)),
                    lon=float(city.lon + rng.normal(0.0, CITY_RADIUS_DEG * 0.5)),
                    city_pop=city.population,
                    job=str(rng.choice(JOBS)),
                    dob=datetime(1950, 1, 1) + timedelta(days=int(rng.integers(0, 20_000))),
                    spend_multiplier=float(rng.lognormal(0.0, 0.45)),
                    activity_rate=float(rng.uniform(0.6, 4.0)),
                    preferred_categories=tuple(
                        rng.choice(CATEGORIES, size=int(rng.integers(4, 9)), replace=False)
                    ),
                    city_index=city.index,
                    home_merchants=tuple(by_city[city.index]),
                )
            )
        return customers

    # -- generation --------------------------------------------------------

    def generate(self) -> pd.DataFrame:
        """Produce the full transaction table, sorted chronologically."""
        records: list[dict[str, object]] = []
        for customer in self.customers:
            records.extend(self._legitimate_for(customer))

        n_fraud = max(1, int(len(records) * self.config.target_fraud_rate))
        records.extend(self._fraud_episodes(n_fraud))

        df = pd.DataFrame(records)
        df = df.sort_values(C.TIMESTAMP_COL, kind="stable").reset_index(drop=True)
        df[C.TRANSACTION_ID_COL] = [f"{i:032x}" for i in range(len(df))]
        df["unix_time"] = (df[C.TIMESTAMP_COL].astype("int64") // 1_000_000_000).astype("int64")
        return df[list(_OUTPUT_COLUMNS)]

    def _legitimate_for(self, customer: Customer) -> list[dict[str, object]]:
        """Routine local spending, interrupted by occasional multi-day trips.

        Trips are modelled as contiguous *windows* rather than isolated distant
        transactions. A genuine traveller flies somewhere, spends several days
        there, and flies back -- so consecutive purchases stay close together
        even while far from home. Scattering individual distant transactions
        through the stream would manufacture impossible travel on legitimate
        cards and destroy the signal the geo features depend on.
        """
        rng = self._rng
        days = (self.config.end - self.config.start).days
        n = max(1, int(rng.poisson(customer.activity_rate * days)))

        trips = self._make_trips(customer, days, rng)

        out = []
        for _ in range(n):
            offset = float(rng.uniform(0, days * 86_400))
            when = self.config.start + timedelta(seconds=offset)
            when = _shift_to_waking_hours(when, rng)

            city_index = _city_for(when, trips) or customer.city_index
            category = str(rng.choice(customer.preferred_categories))
            merchant = self._pick_merchant(customer, category, rng, city_index=city_index)
            out.append(
                self._record(
                    customer,
                    merchant,
                    when,
                    self._amount(category, customer, rng),
                    is_fraud=0,
                    typology=Typology.LEGITIMATE,
                )
            )
        return out

    def _make_trips(
        self, customer: Customer, days: int, rng: np.random.Generator
    ) -> list[tuple[datetime, datetime, int]]:
        """Zero to a handful of trips, each a window spent in another city."""
        trips = []
        for _ in range(int(rng.integers(0, 6))):
            start = self.config.start + timedelta(seconds=float(rng.uniform(0, days * 86_400)))
            end = start + timedelta(days=float(rng.uniform(2.0, 11.0)))
            destination = int(rng.integers(0, len(self.cities)))
            if destination != customer.city_index:
                trips.append((start, end, destination))
        return trips

    def _fraud_episodes(self, n_target: int) -> list[dict[str, object]]:
        """Generate fraud as *episodes* rather than isolated transactions.

        Real fraud arrives in bursts on a compromised card. Scattering
        independent fraudulent transactions across many cards would erase the
        velocity signal that makes the problem tractable, and would make the
        generated data unrepresentative in exactly the dimension the model is
        supposed to exploit.
        """
        rng = self._rng
        out: list[dict[str, object]] = []
        typologies = [
            Typology.CARD_TESTING,
            Typology.ACCOUNT_TAKEOVER,
            Typology.GEO_IMPOSSIBLE,
            Typology.SPENDING_SPREE,
            Typology.SUBTLE,
        ]
        weights = np.array(self.config.typology_weights, dtype=float)
        weights = weights / weights.sum()

        while len(out) < n_target:
            customer = self.customers[int(rng.integers(0, len(self.customers)))]
            typology = typologies[int(rng.choice(len(typologies), p=weights))]
            start = self.config.start + timedelta(
                seconds=float(rng.uniform(0, (self.config.end - self.config.start).days * 86_400))
            )
            out.extend(self._episode(customer, typology, start, rng))
        return out[:n_target]

    def _episode(
        self,
        customer: Customer,
        typology: Typology,
        start: datetime,
        rng: np.random.Generator,
    ) -> list[dict[str, object]]:
        match typology:
            case Typology.CARD_TESTING:
                return self._card_testing(customer, start, rng)
            case Typology.ACCOUNT_TAKEOVER:
                return self._account_takeover(customer, start, rng)
            case Typology.GEO_IMPOSSIBLE:
                return self._geo_impossible(customer, start, rng)
            case Typology.SPENDING_SPREE:
                return self._spending_spree(customer, start, rng)
            case Typology.SUBTLE:
                return self._subtle(customer, start, rng)
        return []

    def _subtle(
        self, customer: Customer, start: datetime, rng: np.random.Generator
    ) -> list[dict[str, object]]:
        """Fraud that is statistically indistinguishable from normal spending.

        A familiar local merchant, an ordinary amount, in the afternoon, with no
        velocity anomaly. Nothing here is detectable from the transaction alone
        -- catching it in reality needs signals this pipeline does not have
        (device fingerprint, IP reputation, a customer report).

        Its purpose is to keep reported recall honest.
        """
        out = []
        when = start
        # Spread across days, not minutes: a burst would create the velocity
        # signature this typology is defined by not having.
        for _ in range(int(rng.integers(2, 5))):
            when += timedelta(days=float(rng.uniform(0.7, 4.0)))
            when = when.replace(hour=int(rng.integers(10, 19)), minute=int(rng.integers(0, 60)))
            category = str(rng.choice(customer.preferred_categories))
            merchant = self._pick_merchant(customer, category, rng, city_index=customer.city_index)
            out.append(
                self._record(
                    customer,
                    merchant,
                    when,
                    self._amount(category, customer, rng),
                    is_fraud=1,
                    typology=Typology.SUBTLE,
                )
            )
        return out

    def _card_testing(
        self, customer: Customer, start: datetime, rng: np.random.Generator
    ) -> list[dict[str, object]]:
        """Several small purchases at unrelated merchants within minutes."""
        out = []
        when = start
        for _ in range(int(rng.integers(4, 9))):
            when += timedelta(minutes=float(rng.uniform(0.5, 6.0)))
            merchant = self.merchants[int(rng.integers(0, len(self.merchants)))]
            out.append(
                self._record(
                    customer,
                    merchant,
                    when,
                    round(float(rng.uniform(1.0, 15.0)), 2),
                    is_fraud=1,
                    typology=Typology.CARD_TESTING,
                )
            )
        return out

    def _account_takeover(
        self, customer: Customer, start: datetime, rng: np.random.Generator
    ) -> list[dict[str, object]]:
        """High-value purchases at unfamiliar merchants, overnight."""
        night = start.replace(hour=int(rng.choice([1, 2, 3, 23])), minute=int(rng.integers(0, 60)))
        out = []
        when = night
        for _ in range(int(rng.integers(2, 5))):
            when += timedelta(minutes=float(rng.uniform(4.0, 40.0)))
            merchant = self.merchants[int(rng.integers(0, len(self.merchants)))]
            amount = round(float(rng.gamma(3.0, 120.0)) * customer.spend_multiplier + 80.0, 2)
            out.append(
                self._record(
                    customer,
                    merchant,
                    when,
                    amount,
                    is_fraud=1,
                    typology=Typology.ACCOUNT_TAKEOVER,
                )
            )
        return out

    def _geo_impossible(
        self, customer: Customer, start: datetime, rng: np.random.Generator
    ) -> list[dict[str, object]]:
        """A local purchase, then one thousands of km away minutes later."""
        local = self._pick_merchant(customer, None, rng, city_index=customer.city_index)
        out = [
            self._record(
                customer,
                local,
                start,
                self._amount(local.category, customer, rng),
                is_fraud=0,
                typology=Typology.LEGITIMATE,
            )
        ]
        # Pick the most distant merchant available, so the implied speed is
        # unambiguously impossible rather than merely improbable.
        far = max(
            self.merchants,
            key=lambda m: (m.lat - customer.lat) ** 2 + (m.lon - customer.lon) ** 2,
        )
        when = start + timedelta(minutes=float(rng.uniform(3.0, 25.0)))
        out.append(
            self._record(
                customer,
                far,
                when,
                round(float(rng.gamma(2.5, 90.0)) + 40.0, 2),
                is_fraud=1,
                typology=Typology.GEO_IMPOSSIBLE,
            )
        )
        return out

    def _spending_spree(
        self, customer: Customer, start: datetime, rng: np.random.Generator
    ) -> list[dict[str, object]]:
        """Escalating purchases before the card is reported."""
        out = []
        when = start
        amount = float(rng.uniform(30.0, 90.0))
        for _ in range(int(rng.integers(3, 7))):
            when += timedelta(minutes=float(rng.uniform(8.0, 90.0)))
            amount *= float(rng.uniform(1.4, 2.3))
            merchant = self.merchants[int(rng.integers(0, len(self.merchants)))]
            out.append(
                self._record(
                    customer,
                    merchant,
                    when,
                    round(amount, 2),
                    is_fraud=1,
                    typology=Typology.SPENDING_SPREE,
                )
            )
        return out

    # -- helpers -----------------------------------------------------------

    def _pick_merchant(
        self,
        customer: Customer,
        category: str | None,
        rng: np.random.Generator,
        *,
        city_index: int | None = None,
    ) -> Merchant:
        """Choose a merchant, preferring the given city and category.

        Falls back progressively: category within city, then any merchant in the
        city, then the customer's home merchants. A customer must always have
        somewhere to shop.
        """
        if city_index is None:
            pool = customer.home_merchants
        else:
            pool = tuple(m.index for m in self.merchants if m.city_index == city_index)
        if not pool:
            pool = customer.home_merchants or tuple(range(len(self.merchants)))

        if category is not None:
            matching = tuple(i for i in pool if self.merchants[i].category == category)
            if matching:
                pool = matching
        return self.merchants[int(rng.choice(pool))]

    def _amount(self, category: str, customer: Customer, rng: np.random.Generator) -> float:
        shape, scale = CATEGORY_SPEND.get(category, (2.0, 30.0))
        return round(max(1.0, float(rng.gamma(shape, scale)) * customer.spend_multiplier), 2)

    def _record(
        self,
        customer: Customer,
        merchant: Merchant,
        when: datetime,
        amount: float,
        *,
        is_fraud: int,
        typology: Typology,
    ) -> dict[str, object]:
        jitter = self._rng.normal(0.0, 0.25, size=2)
        return {
            C.TIMESTAMP_COL: pd.Timestamp(when),
            C.CARD_COL: customer.card,
            "merchant": merchant.name,
            "category": merchant.category,
            C.AMOUNT_COL: amount,
            "first": customer.first,
            "last": customer.last,
            "gender": customer.gender,
            "street": customer.street,
            "city": customer.city,
            "state": customer.state,
            "zip": customer.zip_code,
            C.HOME_LAT_COL: customer.lat,
            C.HOME_LON_COL: customer.lon,
            "city_pop": customer.city_pop,
            "job": customer.job,
            "dob": pd.Timestamp(customer.dob),
            C.MERCH_LAT_COL: float(merchant.lat + jitter[0]),
            C.MERCH_LON_COL: float(merchant.lon + jitter[1]),
            C.TARGET_COL: is_fraud,
            TYPOLOGY_COL: str(typology),
        }


_OUTPUT_COLUMNS: Final[tuple[str, ...]] = (
    C.TIMESTAMP_COL,
    C.CARD_COL,
    "merchant",
    "category",
    C.AMOUNT_COL,
    "first",
    "last",
    "gender",
    "street",
    "city",
    "state",
    "zip",
    C.HOME_LAT_COL,
    C.HOME_LON_COL,
    "city_pop",
    "job",
    "dob",
    C.TRANSACTION_ID_COL,
    "unix_time",
    C.MERCH_LAT_COL,
    C.MERCH_LON_COL,
    C.TARGET_COL,
    TYPOLOGY_COL,
)

_NAME_PARTS: Final[tuple[str, ...]] = (
    "Kutch",
    "Rippin",
    "Heidenreich",
    "Lind",
    "Schmidt",
    "Kuhn",
    "Boyer",
    "Predovic",
    "Stroman",
    "Bauch",
    "Rowe",
    "Kihn",
    "Herzog",
    "Bins",
    "Nitzsche",
    "Rau",
    "Dach",
    "Hilll",
    "Gutmann",
    "Reilly",
    "Cormier",
    "Sporer",
    "Effertz",
    "Ruecker",
)


def _merchant_name(rng: np.random.Generator) -> str:
    a, b = rng.choice(_NAME_PARTS, size=2, replace=False)
    return f"{a}-{b}"


def _city_for(when: datetime, trips: Sequence[tuple[datetime, datetime, int]]) -> int | None:
    """Return the destination city if ``when`` falls inside a trip window."""
    for start, end, destination in trips:
        if start <= when <= end:
            return destination
    return None


def _shift_to_waking_hours(when: datetime, rng: np.random.Generator) -> datetime:
    """Push legitimate activity toward waking hours.

    Without this, genuine transactions are uniform across the clock and the
    ``is_night`` feature carries no signal -- which would be unrepresentative,
    since overnight concentration is one of the more reliable fraud indicators.
    """
    if rng.random() < 0.93:
        return when.replace(hour=int(rng.integers(7, 23)))
    return when


def generate(
    config: GeneratorConfig | None = None,
) -> pd.DataFrame:
    """Convenience wrapper returning a generated transaction table."""
    return SyntheticGenerator(config).generate()


def summarise_typologies(df: pd.DataFrame) -> pd.DataFrame:
    """Fraud counts by typology, for the data card and per-attack recall."""
    if TYPOLOGY_COL not in df.columns:
        return pd.DataFrame(columns=["typology", "count", "share"])
    fraud = df[df[C.TARGET_COL] == 1]
    counts = fraud[TYPOLOGY_COL].value_counts()
    return pd.DataFrame(
        {
            "typology": counts.index,
            "count": counts.to_numpy(),
            "share": (counts / max(1, len(fraud))).to_numpy(),
        }
    )


def haversine_matrix(
    lat1: float, lon1: float, lats: Sequence[float], lons: Sequence[float]
) -> np.ndarray:
    """Vectorised great-circle distance, used when picking distant merchants."""
    p1 = math.radians(lat1)
    p2 = np.radians(np.asarray(lats, dtype=float))
    dl = np.radians(np.asarray(lons, dtype=float) - lon1)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return np.asarray(2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))))
