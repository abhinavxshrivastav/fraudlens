"""Measure scoring latency against the p99 budget.

The budget is 50 ms at p99 for ``POST /score``. This measures the *service*
path -- feature computation, rules, model, policy, and SHAP where it applies --
without HTTP overhead, so the number reflects the work FraudLens does rather
than the transport.

Latency is reported separately for the approve path and the alert path, because
they do genuinely different amounts of work: explanations are computed only for
alerts. Reporting a single blended figure would hide the fact that the expensive
path is the rare one.

Usage::

    python scripts/benchmark_latency.py --requests 10000
"""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from fraudlens.api.schemas import TransactionRequest
from fraudlens.api.service import FraudLensService
from fraudlens.config import constants as C
from fraudlens.config import get_settings

logging.basicConfig(level=logging.WARNING, format="%(levelname)-7s %(message)s")

BUDGET_MS = C.LATENCY_BUDGET_P99_MS


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * q))]


def to_request(row: dict[str, object]) -> TransactionRequest:
    return TransactionRequest(
        transaction_id=str(row[C.TRANSACTION_ID_COL]),
        timestamp=pd.Timestamp(row[C.TIMESTAMP_COL]).to_pydatetime(),
        card_number=int(row[C.CARD_COL]),
        amount=float(row[C.AMOUNT_COL]),
        merchant=str(row["merchant"]),
        category=str(row["category"]),
        home_lat=float(row[C.HOME_LAT_COL]),
        home_lon=float(row[C.HOME_LON_COL]),
        merchant_lat=float(row[C.MERCH_LAT_COL]),
        merchant_lon=float(row[C.MERCH_LON_COL]),
        city_population=int(row.get("city_pop", 0) or 0),
        date_of_birth=pd.Timestamp(row["dob"]).to_pydatetime() if row.get("dob") else None,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=10_000)
    parser.add_argument("--warmup", type=int, default=200)
    args = parser.parse_args(argv)

    settings = get_settings()
    service = FraudLensService.load(settings)
    if not service.is_ready:
        print("No model loaded. Run scripts/train.py first.")
        return 1

    source = settings.processed_dir / "transactions.parquet"
    if not source.exists():
        print(f"{source} not found. Run scripts/prepare_data.py first.")
        return 1

    # Sample from the tail so cards already have realistic history -- scoring a
    # cold card is cheaper and would flatter the result.
    df = pd.read_parquet(source).tail(args.requests + args.warmup)
    requests = [to_request(r) for r in df.to_dict("records")]

    print(f"Model      : {service.model.version if service.model else 'none'}")
    has_explainer = bool(service.explainer and service.explainer.available)
    print(f"Explainer  : {'available' if has_explainer else 'unavailable'}")
    print(f"Requests   : {args.requests:,} (after {args.warmup} warm-up)")
    print()

    for request in requests[: args.warmup]:
        service.score(request)

    approve_ms: list[float] = []
    alert_ms: list[float] = []
    started = time.perf_counter()
    for request in requests[args.warmup :]:
        t0 = time.perf_counter()
        response = service.score(request)
        elapsed = (time.perf_counter() - t0) * 1000.0
        (alert_ms if response.creates_alert else approve_ms).append(elapsed)
    wall = time.perf_counter() - started

    everything = approve_ms + alert_ms
    print(f"{'path':<12} {'n':>8} {'mean':>8} {'p50':>8} {'p95':>8} {'p99':>8} {'max':>9}")
    print("-" * 60)
    for name, series in (
        ("approve", approve_ms),
        ("alert", alert_ms),
        ("combined", everything),
    ):
        if not series:
            continue
        print(
            f"{name:<12} {len(series):>8,} "
            f"{statistics.fmean(series):>7.2f}ms "
            f"{percentile(series, 0.50):>7.2f}ms "
            f"{percentile(series, 0.95):>7.2f}ms "
            f"{percentile(series, 0.99):>7.2f}ms "
            f"{max(series):>8.2f}ms"
        )

    p99 = percentile(everything, 0.99)
    throughput = len(everything) / wall
    print()
    print(f"Throughput : {throughput:,.0f} transactions/second (single process)")
    print(f"Alert rate : {len(alert_ms) / max(1, len(everything)):.3%}")
    print()

    verdict = "PASS" if p99 <= BUDGET_MS else "FAIL"
    print(f"p99 budget : {BUDGET_MS:.0f}ms -> measured {p99:.2f}ms  [{verdict}]")
    return 0 if p99 <= BUDGET_MS else 1


if __name__ == "__main__":
    raise SystemExit(main())
