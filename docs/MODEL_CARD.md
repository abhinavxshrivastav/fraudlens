# Model card — FraudLens transaction risk model

Following the model-card framework (Mitchell et al., 2019), structured so it can
also serve as the model documentation a bank's Model Risk function would expect:
what the model is for, what it was trained on, how it performs, where it fails,
and when it must be retrained.

**Everything below is measured, not estimated.** Figures come from
`artifacts/reports/results-full.md`, reproducible with `python scripts/train.py`.

---

## 1. Model details

| | |
|---|---|
| **Model version** | `lightgbm-20260905.203505` |
| **Type** | LightGBM gradient-boosted trees, binary classification |
| **Trained** | 2026-09-05 |
| **Features** | 28 behavioural features (see §3) |
| **Imbalance handling** | `scale_pos_weight = 176.4` — no resampling |
| **Calibration** | Isotonic regression fitted on the validation fold |
| **Output** | Calibrated probability of fraud in [0, 1] |
| **Owner** | Abhinav Kumar |
| **Repository** | https://github.com/abhinavxshrivastav/fraudlens |

### Challengers evaluated

| Model | PR-AUC | Expected cost | Outcome |
|---|---:|---:|---|
| Amount-only heuristic | 0.0298 | £112,539 | Rejected — worse than random (ROC-AUC 0.429) |
| Rules only | 0.4282 | £43,594 | Retained as a **layer**, not as the model |
| Logistic regression | 0.6534 | £34,348 | Rejected — 18% costlier than the champion |
| **LightGBM** | 0.8277 | **£29,108** | **Champion** |
| XGBoost | 0.8308 | £29,119 | Rejected — see below |

**The two gradient-boosted models are effectively tied.** They land within £11
(0.04%) of each other once both are calibrated identically. LightGBM ships
because it is more precise (0.958 vs 0.926) at a lower alert rate (0.491% vs
0.519%), which matters when analyst capacity is the binding constraint — but the
honest statement is that the choice between them is not evidence-driven.

**A methodological note, recorded because it changed the conclusion.** An earlier
version evaluated the XGBoost challenger *without* calibration while the champion
was calibrated, and reported that XGBoost won on PR-AUC yet lost on cost by
£3,285. That gap was an artefact of the comparison, not a property of the models:
expected cost requires genuine probabilities, so comparing a calibrated model
against an uncalibrated one on that metric is invalid. Calibrating both closed
the gap to £11.

## 2. Intended use

**In scope.** Real-time risk scoring of card-present and card-not-present
transactions at authorisation time, producing a calibrated probability and an
action band (`approve` / `challenge` / `review` / `block`) with analyst-readable
reason codes.

**Out of scope, explicitly.**

- **Anti-money laundering.** A different problem with different regulatory
  obligations. Nothing here is designed for or validated against AML typologies.
- **Credit or lending decisions.** The features encode spending behaviour, not
  creditworthiness, and using them for lending would raise fair-lending issues
  this model has not been assessed for.
- **Account opening / synthetic identity.** Requires identity signals absent here.
- **Autonomous blocking without recourse.** Every `block` must be reversible by a
  human, and the reason codes exist so a customer can be told why.

**Users.** Fraud analysts triaging an alert queue; an authorisation system
consuming the band programmatically.

---

## 3. Features

28 features, all **strictly backwards-looking** and computed as-of transaction
time, grouped by card. Full definitions in `src/fraudlens/features/pipeline.py`.

| Group | Features | Signal |
|---|---|---|
| Amount | `amt`, `amt_log`, `amt_zscore_30d`, `amt_ratio_mean_30d`, `amt_ratio_max_30d` | Deviation from *this card's* norm, not absolute value |
| Velocity | `txn_count_{1h,24h,7d,30d}`, `amt_sum_{1h,24h,7d}` | Burst detection |
| Geographic | `dist_home_merch_km`, `dist_prev_txn_km`, `implied_speed_kmh`, `is_impossible_travel` | Cloned-card detection |
| Temporal | `hour`, `day_of_week`, `is_night`, `is_weekend`, `secs_since_prev_txn` | Overnight concentration |
| Novelty | `is_new_merchant`, `is_new_category`, `distinct_merchants_24h`, `distinct_categories_7d` | Card testing |
| Demographic | `age_years`, `city_pop_log` | Weak context |
| Reliability | `card_history_count` | Lets the model discount cold-start features |

**Excluded deliberately.** `first`, `last`, `street` are dropped at ingestion.
They are effectively unique per customer, so retaining them would let the model
memorise individuals rather than learn behaviour — both a leakage risk and a
privacy problem.

---

## 4. Training data

| Fold | Window | Rows | Frauds | Rate |
|---|---|---:|---:|---:|
| Train | 2019-01-01 → 2020-03-31 | 414,062 | 2,334 | 0.5637% |
| *embargo* | 7 days | discarded | | |
| Validation | 2020-04-08 → 2020-06-30 | 76,667 | 449 | 0.5856% |
| *embargo* | 7 days | discarded | | |
| Test | 2020-07-08 → 2020-12-31 | 159,301 | 950 | 0.5964% |

> ### ⚠️ The training data is synthetic
>
> This model was trained on data generated by
> `src/fraudlens/data/synthetic.py`, **not on real transactions**. It has
> learned the five fraud typologies that generator encodes. Its performance on
> real-world fraud is **unknown and should be assumed poor** until retrained.
>
> The architecture, evaluation methodology and serving path are real and
> transferable; the *model weights* are not. Retraining on the Sparkov dataset is
> a single flag (`scripts/prepare_data.py --source kaggle`) and the numbers will
> fall.

**Splitting.** Chronological with a 7-day embargo between folds, because
behavioural features look back up to 30 days and a plain split would leak across
the boundary. Rationale in [ADR 0003](adr/0003-temporal-split-with-embargo.md).

**Test discipline.** The test fold was scored **exactly once**, after model
selection was complete. Every threshold, calibration map and early-stopping
decision came from validation.

---

## 5. Performance

Test fold, thresholds frozen from validation.

| Metric | Champion | Rules baseline |
|---|---:|---:|
| PR-AUC | **0.8277** | 0.4282 |
| ROC-AUC | 0.9560 | 0.8889 |
| Recall @ 1% FPR | 0.836 | 0.695 |
| Precision | **0.958** | 0.551 |
| Recall | 0.788 | 0.695 |
| Alert rate | **0.491%** | 0.751% |
| Precision@100 | 1.000 | 0.480 |
| Value detection rate | 0.871 | — |
| Expected cost | **£29,108** | £43,594 |

Against the rules baseline: **33% lower expected cost**, catching 79% of fraud
instead of 70%, at a *lower* alert rate — nearly double the precision on a third
fewer alerts.

Operating threshold **0.25**, selected by minimising expected cost on the
validation fold.

### Calibration

Brier score improves **6×** under isotonic regression: 0.008559 → 0.001409.

Reliability on the test fold:

| Predicted band | n | Predicted | Actual |
|---|---:|---:|---:|
| 0.00 – 0.01 | 157,988 | 0.0011 | 0.0010 |
| 0.01 – 0.05 | 323 | 0.0265 | 0.0279 |
| 0.05 – 0.20 | 204 | 0.0983 | **0.1716** |
| 0.20 – 0.50 | 53 | 0.3811 | **0.6226** |
| 0.80 – 1.00 | 733 | 0.9843 | 0.9782 |

**Known weakness.** Calibration is excellent in the tails and
**under-confident in the 0.05–0.50 mid-range**, where the model predicts roughly
half the true fraud rate. Cause: very few validation examples fall in that band,
so isotonic regression has little to fit. Consequence: mid-risk transactions are
*under*-escalated. This is a genuine limitation, not a rounding artefact, and it
is the first thing to fix with more data.

### Performance by fraud typology

| Typology | Rows | Mean score | Recall |
|---|---:|---:|---:|
| geo_impossible | 27 | 0.968 | 1.000 |
| account_takeover | 118 | 0.963 | 0.983 |
| card_testing | 460 | 0.921 | 0.952 |
| spending_spree | 200 | 0.800 | 0.840 |
| **subtle** | 145 | 0.003 | **0.000** |

**Zero recall on `subtle` is expected and correct.** That typology is
constructed to be indistinguishable from ordinary spending — familiar merchant,
normal amount, mid-afternoon, no velocity anomaly. There is no signal in the
transaction record to find; catching it requires device fingerprints, IP
reputation or a customer report. It is 15% of test fraud and exists so headline
recall cannot be flattered. **Excluding it, recall on detectable fraud is 93%.**

---

## 6. Operating point

| Threshold | Value | Chosen by |
|---|---:|---|
| Recommended (min expected cost) | **0.25** | Minimising expected cost on validation |
| Recall @ 1% FPR | 0.0217 | Fixed-FPR budget on validation |

Cost model: **£4.00** per alert investigated, **£18.00** friction per false
positive, and the full transaction amount for every missed fraud. These are
order-of-magnitude estimates; what drives the threshold is the *ratio*. See
[`EVALUATION.md`](EVALUATION.md).

---

## 7. Limitations

1. **Synthetic training data** — the dominant limitation. See §4.
2. **Mid-range calibration** is under-confident (§5).
3. **No signal on patient fraud** — an attacker who stays inside normal
   behaviour is invisible to this feature set (§5, and
   [`THREAT_MODEL.md`](THREAT_MODEL.md)).
4. **Card-centric only.** No merchant-side or cross-institution view, so an
   attacker spreading activity across many cards evades the velocity features.
5. **Cold start.** A card with no history gets neutral feature defaults;
   `card_history_count` signals this but early decisions are weaker.
6. **Feature-store poisoning.** An attacker able to inject benign transactions
   could inflate a card's baseline and mask a later large purchase. The 30-day
   window bounds this but does not prevent it.
7. **Throughput** is 135–147 tps single-process. Horizontal scaling is permitted
   by the design but has not been demonstrated.
8. **Fairness has not been formally assessed.** `gender`, `age_years` and
   `city_pop` are inputs. No disparate-impact analysis has been run, and none
   should be assumed. This must be completed before any real deployment.

---

## 8. Ethical considerations

**A false positive is not costless.** A wrongly declined transaction can mean
someone unable to pay for food or medicine. The cost model prices customer
friction at 4.5× the investigation cost precisely so the optimiser does not treat
false positives as cheap.

**Demographic inputs.** `gender` and `age_years` are used. They carry genuine
behavioural signal, but a model can reach a discriminatory outcome without any
discriminatory intent. Item 8 in §7 is therefore a **blocker**, not a nice-to-have.

**Explainability is a right, not a feature.** Every alerting decision carries
reason codes stating in plain language what drove it, so a decision can be
explained to the customer it affected and contested.

---

## 9. Monitoring and retraining

**Live monitoring.** PSI per feature against the training distribution, with the
conventional bands (<0.10 stable, 0.10–0.25 monitor, >0.25 investigate), exposed
at `GET /drift` and as `fraudlens_drift_psi`. Cyclical features (`hour`,
`day_of_week`, `is_night`, `is_weekend`) are excluded — over a window shorter
than a full cycle they score PSI >12 while nothing has changed, and false alarms
teach a team to ignore the real ones.

**Retrain when any of these hold:**

| Trigger | Threshold |
|---|---|
| Feature drift | any feature PSI > 0.25 sustained 30 min |
| Precision decay | precision at the operating threshold < 0.80 over 7 days |
| Alert-rate spike | > 2% of transactions for 10 min |
| Calibration decay | Brier score > 0.003 on labelled outcomes |
| Elapsed time | 90 days, regardless of the above |
| New typology | any confirmed attack pattern absent from training |

**Auditability.** Every decision persists model version, rule-set version and
individual rule versions, policy version, a SHA-256 hash of the feature vector,
the calibrated score, band, and reason codes — enough to reproduce any historical
decision exactly.

---

## 10. Reproduction

```bash
python scripts/prepare_data.py
python scripts/train.py
```

Deterministic: `random_state=42` throughout, fixed split boundaries in
`src/fraudlens/config/constants.py`. Re-running reproduces the figures above.
