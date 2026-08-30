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
| Amount only (heuristic) | 0.0298 | 0.4292 | 0.106 | 0.062 | 0.102 | 0.989% | 0.200 | £112,790 |
| Rules only | 0.4282 | 0.8889 | 0.695 | 0.614 | 0.653 | 0.634% | 0.480 | £75,599 |
| Logistic regression | 0.6534 | 0.9318 | 0.843 | 0.421 | 0.842 | 1.194% | 0.920 | £44,347 |
| **LightGBM (champion)** | 0.8277 | 0.9560 | 0.836 | **0.905** | 0.809 | **0.534%** | 1.000 | **£25,702** |
| XGBoost (challenger) | 0.8382 | 0.9586 | 0.843 | 0.933 | 0.806 | 0.515% | 1.000 | £28,987 |

**The headline is the last column, not the first.** Against the rules-only
baseline the champion cuts expected cost by **66%** — catching 81% of fraud
instead of 65%, at a *lower* alert rate (0.53% vs 0.63%) and with precision of
0.905 instead of 0.614. That is the concrete form of "minimise false alerts".

Three results worth dwelling on:

- **Amount alone is worse than random** (ROC-AUC 0.4292). Card-testing fraud is
  deliberately low-value, so raw transaction size is *anti*-correlated with
  fraud. Any system that leans on it is inverted.
- **XGBoost wins on PR-AUC and loses on cost.** It ranks marginally better
  (0.8382 vs 0.8277) and is more precise, yet costs £3,285 more over the test
  period because it misses slightly more high-value fraud. Selecting on PR-AUC
  alone would have picked the worse model — which is the argument for the cost
  curve in one line.
- **Calibration matters and it works.** Brier improves 6× under isotonic
  regression (0.008559 → 0.001409), which is what makes cost-based thresholding
  legitimate.

### Recall by fraud typology

| Typology | Fraud rows | Mean score | Recall |
|---|---:|---:|---:|
| geo_impossible | 27 | 0.968 | 1.000 |
| account_takeover | 118 | 0.963 | 0.983 |
| card_testing | 460 | 0.921 | 0.972 |
| spending_spree | 200 | 0.800 | 0.895 |
| **subtle** | 145 | 0.003 | **0.000** |

`subtle` is fraud constructed to be indistinguishable from ordinary spending: a
familiar merchant, a normal amount, mid-afternoon, no velocity anomaly. Zero
recall on it is the **correct** result — there is no signal in the transaction
record to find. It is 15% of test fraud, and it is generated on purpose so that
headline recall cannot be flattered by an artificially separable problem.
Excluding it, recall on detectable fraud is **95.4%**.

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
- [ ] **Phase 4** — Streaming and drift monitoring
- [ ] **Phase 5** — React analyst console
- [ ] **Phase 6** — CI/CD and public deployment

## Licence

MIT
