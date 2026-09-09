# FraudLens

**Real-time fraud detection and risk intelligence platform.**

Card-transaction scoring that combines a deterministic rule engine, a calibrated
gradient-boosted model, and per-decision explanations — served behind a FastAPI
endpoint with a sub-50 ms p99 latency budget and a React analyst console.

> **Status:** under active development. See [Roadmap](#roadmap) for what is live.

---

## Why this exists

Banks screen millions of transactions a day. Classical rule engines are
auditable but blunt: they fire on thresholds, drown analysts in false positives,
and cannot adapt to new fraud typologies. Pure ML is adaptive but opaque, and an
unexplainable decline is not deployable in a regulated environment.

FraudLens treats that tension as the design problem rather than an afterthought:

- **Rules and ML are layered, not opposed.** Rules give deterministic, auditable
  hard-stops; the model supplies a calibrated probability; a policy layer
  combines them into an action band.
- **Every decision is explainable.** SHAP contributions are mapped to
  human-readable reason codes an analyst can act on.
- **Thresholds are an economic choice.** The operating point is selected by
  minimising expected cost, not by maximising an abstract score.
- **Every decision is reproducible.** Model version, feature-vector hash, fired
  rule versions, raw and calibrated score, and reason codes are all persisted.

## Results

Synthetic dataset: 662,640 transactions, 400 cards, 600 merchants, 2019-01-01 to
2020-12-31, 0.58% fraud. Temporal split with a 7-day embargo. **Test was scored
once**, with every threshold selected on validation.

| Model | PR-AUC | ROC-AUC | Recall@1%FPR | Precision | Recall | Alert rate | P@100 | Expected cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Amount only (heuristic) | 0.0298 | 0.4292 | 0.106 | 0.073 | 0.085 | 0.701% | 0.200 | £112,539 |
| Rules only | 0.4282 | 0.8889 | 0.695 | 0.551 | 0.695 | 0.751% | 0.480 | £43,594 |
| Logistic regression | 0.6534 | 0.9318 | 0.843 | 0.578 | 0.826 | 0.852% | 0.920 | £34,348 |
| **LightGBM (champion)** | 0.8277 | 0.9560 | 0.836 | **0.958** | 0.788 | **0.491%** | 1.000 | **£29,108** |
| XGBoost (challenger) | 0.8308 | 0.9578 | 0.842 | 0.926 | 0.806 | 0.519% | 1.000 | £29,119 |

**The headline is the last column, not the first.** Against the rules-only
baseline the champion cuts expected cost by **33%** — catching 79% of fraud
instead of 70%, at a *lower* alert rate (0.49% vs 0.75%) and with precision of
0.958 instead of 0.551. That is the concrete form of "minimise false alerts":
nearly double the precision on a third fewer alerts.

Three results worth dwelling on:

- **Amount alone is worse than random** (ROC-AUC 0.4292). Card-testing fraud is
  deliberately low-value, so raw transaction size is *anti*-correlated with
  fraud. Any system that leans on it is inverted.
- **The two gradient-boosted models are indistinguishable.** LightGBM and
  XGBoost land within £11 of each other (0.04%) once both are calibrated the
  same way. LightGBM ships on precision and alert rate; the honest statement is
  that there is no meaningful difference, not that one won.
- **Calibration is what makes the cost column legitimate.** Brier improves 6×
  under isotonic regression (0.008559 → 0.001409). Without it the scores are not
  probabilities and cost-based thresholding is invalid — which is also why the
  challenger is calibrated before being compared, not after.

### Recall by fraud typology

| Typology | Fraud rows | Recall |
|---|---:|---:|
| geo_impossible | 27 | 1.000 |
| account_takeover | 118 | 0.983 |
| card_testing | 460 | 0.952 |
| spending_spree | 200 | 0.840 |
| **subtle** | 145 | **0.000** |

`subtle` is fraud constructed to be indistinguishable from ordinary spending: a
familiar merchant, a normal amount, mid-afternoon, no velocity anomaly. Zero
recall on it is the **correct** result — there is no signal in the transaction
record to find. It is 15% of test fraud, generated on purpose so that headline
recall cannot be flattered by an artificially separable problem. Excluding it,
recall on detectable fraud is **93%**.

> **On these numbers.** They are measured on synthetic data and are *not* a
> benchmark result. The generator is realistic in structure but easier than
> reality. The methodology is the point; the real Sparkov dataset drops in
> unchanged via `python scripts/prepare_data.py --source kaggle`.

## Engineering positions this project takes

Most public fraud-detection projects share a handful of methodological flaws.
FraudLens takes the opposite position on each, and documents why.

| Common practice | What FraudLens does | Rationale |
|---|---|---|
| Random train/test split | **Temporal split with a 7-day embargo** between folds | Transactions are time-ordered; a random split leaks the future into training and inflates every metric |
| SMOTE / random oversampling | **Class weighting only** | Synthetic minority interpolation distorts calibration and inflates validation scores; see [`docs/EVALUATION.md`](docs/EVALUATION.md) |
| Report accuracy | **PR-AUC, recall @ fixed FPR, Precision@K, alert rate, expected cost** | At a ~0.5% base rate, a model predicting "never fraud" scores 99.5% accuracy |
| Raw model score used as a probability | **Isotonic calibration** on a held-out validation fold | Cost-based thresholding is only valid on calibrated probabilities |
| Offline features reimplemented for serving | **One pipeline, parity enforced by test** | Train/serve skew is the classic silent production failure |

## Architecture

```
                    ┌─────────────────────────────────────────┐
  Replay producer → │  StreamBroker (interface)               │
  (test period at   │  ├─ InProcessBroker  ← default, native  │
   Nx real time)    │  └─ KafkaBroker      ← docker profile   │
                    └────────────────┬────────────────────────┘
                                     ▼
              ┌──────────────────────────────────────────────┐
              │  Decision Engine                             │
              │  1. Rule engine   (YAML, versioned)          │
              │  2. ML score      (LightGBM, calibrated)     │
              │  3. Policy layer  (bands + thresholds)       │
              └────┬─────────────────────────────┬───────────┘
                   │                             │
        ┌──────────▼──────────┐      ┌───────────▼───────────┐
        │ FeatureStore (iface)│      │ Decision audit log    │
        │ ├─ InMemory/DuckDB  │      │ (model ver, feat hash,│
        │ └─ Redis (docker)   │      │  rules, score, codes) │
        └─────────────────────┘      └───────────┬───────────┘
                                                 ▼
                   FastAPI  ──WebSocket/REST──▶  React analyst console
                      │
                      └──▶ /metrics (Prometheus) · structured JSON logs
```

Every piece of infrastructure sits behind a protocol with two implementations: an
in-process default that needs no daemon, and a containerised one selected by
`FRAUDLENS_PROFILE=docker`. The platform runs on a clean laptop checkout; the
`docker-compose.yml` stack is available but never required.

## Quickstart

```bash
python -m venv .venv && .venv/Scripts/activate   # Windows
pip install -e ".[dev,data]"
```

Run the test suite (uses the committed sample; no dataset download needed):

```bash
pytest -q
```

## Dataset

[Credit Card Transactions Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
— 1,852,394 simulated transactions across 1,000 customers and 800 merchants,
generated by the Sparkov simulator, spanning 2019-01-01 to 2020-12-31.

Chosen over the more common PCA-anonymised alternatives because every column is
semantically meaningful. Customer *and* merchant coordinates make genuine
geo-velocity features possible, and SHAP output reads like an analyst's note
rather than "V14 was low".

Raw data is git-ignored. A stratified sample is committed under `data/sample/`
so that tests and CI run without credentials or a 400 MB download.

## Live streaming

The held-out **test fold** is replayed as a live stream at a configurable
multiple of real time, so the running dashboard scores data the model has
genuinely never seen — the alerts are real detections, not a scripted animation.

```bash
FRAUDLENS_DEMO_MODE=true uvicorn fraudlens.api.app:app
```

| Endpoint | Purpose |
|---|---|
| `WS /ws/decisions` | Every decision, live |
| `WS /ws/alerts` | Alerts only — the analyst queue |
| `GET /drift` | Per-feature PSI against the training distribution |
| `GET /stream/status` | Replay and broker state |
| `GET /metrics` | Prometheus exposition |

Three design decisions in this layer are worth calling out, because each one was
a bug found by testing rather than a choice made up front:

- **The feature store is warmed before replay.** 27,297 transactions from the 30
  days preceding the test period are folded into card state (never scored, never
  seen by the model). Without this every card looks brand new, velocity features
  read zero, and drift monitoring compares mature training data against
  cold-start noise.
- **Cyclical features are excluded from drift monitoring.** `hour` and
  `day_of_week` are periodic; a window shorter than a full cycle scores PSI 12.0
  against a reference spanning all seven days while nothing has changed. False
  drift alarms train the team to ignore the real ones.
- **Subscriber queues drop the oldest message when full.** A dashboard that
  falls behind must never stall the authorisation path. Drops are counted and
  exposed rather than hidden; the scoring path is not a subscriber, so no
  decision is ever dropped.

## Analyst console

React 19 + TypeScript + Vite, served by the API itself in production so there is
one origin and one process to deploy.

```bash
# API + console together (built assets)
FRAUDLENS_DEMO_MODE=true uvicorn fraudlens.api.app:app --port 8000

# or the console with hot reload against a running API
cd web && npm run dev
```

| Page | What it shows |
|---|---|
| **Live monitor** | Streaming decisions over WebSocket, band mix, score histogram, live counters |
| **Alert queue** | Triage list with filters and one-click dispositions |
| **Case detail** | Transaction facts, reason codes, rules fired, SHAP waterfall, full audit trail |
| **Threshold simulator** | Drag the threshold and watch precision, recall, workload and cost move |
| **Model** | Provenance, per-feature drift (PSI), and the live rule set |

The **threshold simulator** is the page worth opening first. It answers the
question the whole project is about — where should the line go? — by letting you
move it and see the cost. Raise the false-positive friction cost and the optimum
climbs: £1 → threshold 0.05, £18 → 0.25, £200 → 0.43. A banner on the page states
plainly that this *visualises* trade-offs and does not select the threshold; the
shipped operating point was chosen on the validation fold and is frozen in the
model artefact.

The console's types are checked against **real captured API responses**
(`web/src/types/__fixtures__/`) rather than hand-written mocks, so a backend
change that breaks the contract fails the build rather than the UI.

## Documentation

| Document | Contents |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Component design and data flow |
| [`docs/EVALUATION.md`](docs/EVALUATION.md) | Metric definitions and methodology |
| [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) | Intended use, performance, limitations, retraining triggers |
| [`docs/DATA_CARD.md`](docs/DATA_CARD.md) | Provenance, schema, known biases |
| [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md) | Fraud typologies and the layer that catches each |
| [`docs/adr/`](docs/adr/) | Architecture decision records |

## Roadmap

- [x] **Phase 0** — Foundation, tooling, dependency compatibility gate
- [ ] **Phase 1** — Data contracts, temporal splitter, evaluation harness
- [ ] **Phase 2** — Behavioural features, models, calibration
- [ ] **Phase 3** — Serving API, rule engine, explanations
- [x] **Phase 4** — Streaming, replay producer, WebSocket feeds, drift monitoring
- [x] **Phase 5** — React analyst console
- [ ] **Phase 6** — CI/CD and public deployment

## Licence

MIT
