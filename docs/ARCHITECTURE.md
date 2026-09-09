# Architecture

How FraudLens is put together, and why each boundary sits where it does.

---

## 1. The shape of the system

```
   Replay producer                    ┌──────────────────────────────┐
   (test fold at Nx real time)  ─────▶│  StreamBroker  (protocol)    │
                                      │  ├─ InProcessBroker  native  │
   POST /score  ────────────────────▶ │  └─ KafkaBroker      docker  │
                                      └───────────────┬──────────────┘
                                                      ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │  FraudLensService.score()                                        │
   │                                                                  │
   │   1. read CardState from FeatureStore                            │
   │   2. FeaturePipeline.compute(event, state)   → 28 features       │
   │   3. DecisionEngine.decide()                                     │
   │        a. RuleEngine        deterministic, versioned             │
   │        b. FraudModel        calibrated probability               │
   │        c. PolicyThresholds  → RiskBand                           │
   │   4. ExplainerService (alerts only) → SHAP → reason codes        │
   │   5. FeatureStore.update()   ← state advances LAST               │
   └────────┬──────────────────────────────────┬──────────────────────┘
            │                                  │
   ┌────────▼─────────┐              ┌─────────▼──────────┐
   │ FeatureStore     │              │ Decision record    │
   │ ├─ InMemory (LRU)│              │ model + rule vers, │
   │ └─ Redis  docker │              │ feature hash, codes│
   └──────────────────┘              └─────────┬──────────┘
                                               ▼
                        ┌──────────────────────────────────────┐
                        │ FastAPI                              │
                        │  REST · WebSocket · /metrics /drift  │
                        └──────────────────────────────────────┘
```

**Step 5 is the load-bearing detail.** State advances only *after* the decision
is made, so a transaction can never contribute to its own velocity features.
That is the online form of the leakage the offline tests attack, and it is the
reason training and serving agree.

---

## 2. Layering

| Layer | Package | Depends on |
|---|---|---|
| Configuration | `config/` | nothing |
| Data contracts | `data/` | config |
| Features | `features/` | config |
| Models | `models/` | features |
| Rules | `rules/` | nothing (features validated at load) |
| Scoring | `scoring/` | rules, features |
| Explainability | `explain/` | features |
| Evaluation | `evaluation/` | config |
| Streaming | `streaming/` | config |
| Monitoring | `monitoring/` | config |
| Service | `api/` | everything |

Dependencies point one way. `api/` is the only package that knows about all the
others, which is what makes every layer below it testable in isolation — the
227 unit tests never construct an HTTP client.

---

## 3. The three decision layers

Rules and ML are **layered, not opposed**. Banks run both, because each provides
what the other cannot.

### 3.1 Rule engine — deterministic

Rules are **data, not code**. Conditions are a small declarative structure
evaluated by an interpreter, deliberately *not* `eval` on a string:

```yaml
- id: R-GEO-001
  action: block
  version: 1
  reason_code: R12
  when:
    all:
      - {feature: is_impossible_travel, op: eq, value: 1}
      - {feature: dist_prev_txn_km, op: gt, value: 100}
```

Three reasons: a YAML file that can execute Python is a remote-code-execution
vector; a fraud analyst should be able to change a threshold without writing
code; and every rule carries a version, so a historical decision can be
reproduced by replaying the rule set that was live at the time.

A `block` **short-circuits the model** — a hard stop must not depend on model
availability. Rule references are validated against the feature contract at
startup, because a rule naming a misspelled feature never fires *and never
errors*, which is the worst of both worlds.

### 3.2 Model — probabilistic

LightGBM producing a calibrated probability. Calibration is not cosmetic: the
cost-based threshold and the policy bands both assume the score is a real
probability.

### 3.3 Policy — economic

Maps probability plus advisory rule hits onto a band:

| Band | Meaning | Consumes analyst capacity |
|---|---|---|
| `approve` | Let through | no |
| `challenge` | Step-up authentication | no |
| `review` | Analyst queue | **yes** |
| `block` | Decline | **yes** |

Advisory rules escalate but never de-escalate. An allow-list hit can suppress a
low-confidence alert, but cannot rescue a transaction the model independently
considers high-risk.

---

## 4. Infrastructure behind protocols

Two protocols, each with a native default and a containerised implementation:

| Protocol | `native` (default) | `docker` |
|---|---|---|
| `StreamBroker` | `InProcessBroker` — asyncio queues | `KafkaBroker` — aiokafka |
| `FeatureStore` | `InMemoryFeatureStore` — LRU | `RedisFeatureStore` |

The profile is read **once**, in the dependency-injection layer. No business
logic branches on it — `build_broker()` and `build_feature_store()` are the only
two places in the codebase that do.

This is why a fresh clone runs with `pip install -e .` and nothing else, while
the containerised architecture remains real and reviewable. Full rationale in
[ADR 0001](adr/0001-infrastructure-profiles.md).

**Stated honestly:** the Kafka and Redis paths are written against their APIs and
covered by the protocol contract tests, but have never been run against live
infrastructure, because the machine this was built on has no Docker daemon.

---

## 5. One feature implementation, not two

The usual way to build this is twice — vectorised pandas for training, a
separate online path for serving. The two then drift apart and the model is
served features subtly different from those it was trained on. This is
**train/serve skew**: it raises no error and quietly degrades performance.

FraudLens has a single implementation. `FeaturePipeline.compute()` takes a
`CardState` plus a transaction. Training replays transactions through the same
state machine serving uses; only the *source* of the state differs (a dict during
batch replay, the feature store per request).

`tests/integration/test_train_serve_parity.py` asserts the two paths produce
**numerically identical** vectors — exact equality, not approximate.

The cost is speed: a Python state machine runs at ~3,900 rows/s rather than
vectorised rolling windows. Acceptable for a batch job; the trade is deliberate.

**Bounded state.** `CardState` retains only the longest lookback window (30
days), so memory is flat regardless of how long a card has been active. That is
what makes the same object viable as the online store value.

---

## 6. Explainability

SHAP `TreeExplainer` computes exact Shapley values for tree ensembles — so
explanations are **deterministic**, which matters if a decision must be defended
months later.

Explanations are computed on the **alert path only**. The approve path is ~99.5%
of traffic and nobody reads its explanations; skipping it is what keeps p99
inside budget (approve 6.8 ms mean, alert 10.7 ms).

SHAP answers "which features moved this score". A reason code turns that into a
claim a person can act on:

```
R12 - Implied travel speed of 39,354 km/h since the previous transaction,
      which no journey could achieve
R01 - Amount is 42.2x this card's 30-day average
```

A code is emitted only when the feature is both *material* and pushed the score
**toward** fraud — a factor arguing in the customer's favour is not a reason for
an adverse decision.

---

## 7. Streaming

The held-out test fold is replayed at a configurable multiple of real time.
Inter-arrival gaps are preserved and divided by `speed`, so **bursts stay
bursts** — a smoothed replay would never trigger the velocity features that
exist to detect bursts.

**Backpressure.** Subscriber queues are bounded and drop the *oldest* message
when full. A dashboard that falls behind must not stall the authorisation path,
and in fraud detection a stale backlog has no value. Drops are counted and
exposed. The scoring path is never a subscriber, so no decision is ever dropped.

**Warm start.** Before replay begins, 27,297 transactions from the 30 days
preceding the test period are folded into the feature store. They are never
scored and never reach the model — they only build state. Without this, every
card looks brand new and drift monitoring compares mature training data against
cold-start noise.

---

## 8. Observability

| Signal | Where |
|---|---|
| Latency histogram (buckets centred on the 50 ms budget) | `fraudlens_scoring_latency_seconds` |
| Decisions by band | `fraudlens_decisions_total{band}` |
| Rule firings | `fraudlens_rules_fired_total{rule_id,action}` |
| Feature drift | `fraudlens_drift_psi{feature}` |
| Live feeds | `WS /ws/decisions`, `WS /ws/alerts` |

Four Prometheus alert rules in `deploy/prometheus/alerts.yml`, each tied to a
named failure mode: latency-budget breach, alert-rate spike, scoring stopped, and
significant drift.

Instrumentation **never breaks scoring** — when `prometheus_client` is absent,
every metric call degrades to a no-op rather than raising.

---

## 9. Performance

| Path | Mean | p50 | p95 | p99 |
|---|---:|---:|---:|---:|
| approve (99.5% of traffic) | 6.78 ms | 6.53 ms | 9.00 ms | 10.33 ms |
| alert (with SHAP) | 10.70 ms | 12.22 ms | 13.97 ms | 13.97 ms |
| **combined** | 6.80 ms | 6.54 ms | 9.07 ms | **10.74 ms** |

Against a 50 ms p99 budget. Throughput 135–147 tps single-process.

Reproduce with `python scripts/benchmark_latency.py --requests 10000`.

---

## 10. Repository layout

```
src/fraudlens/
├── config/        settings (12-factor) + domain constants
├── data/          schema contracts, temporal splitter, loader, generator
├── features/      feature pipeline + online store
├── models/        training, calibration, persistence
├── rules/         declarative rule engine
├── scoring/       decision engine + audit records
├── explain/       SHAP + reason codes
├── evaluation/    metrics and cost curves
├── streaming/     broker protocol + replay producer
├── monitoring/    drift detection + Prometheus
└── api/           FastAPI service
```

Related reading: [`EVALUATION.md`](EVALUATION.md) ·
[`MODEL_CARD.md`](MODEL_CARD.md) · [`DATA_CARD.md`](DATA_CARD.md) ·
[`THREAT_MODEL.md`](THREAT_MODEL.md) · [`adr/`](adr/)
