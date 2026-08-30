# Evaluation methodology

This document defines how FraudLens is measured and why each choice was made. It
is written to be read before the results, because on extremely imbalanced data
the *methodology* determines the headline number far more than the model does.

---

## 1. Why accuracy is never reported

The fraud base rate in this dataset is roughly 0.5%. A model that returns
"legitimate" for every transaction is **99.5% accurate** and catches nothing.

Accuracy, and any metric dominated by the true-negative cell of the confusion
matrix, is uninformative here. It is not reported anywhere in this project —
not in the README, not in the model card, not in the console.

---

## 2. The metric set

### 2.1 PR-AUC — primary ranking metric

Area under the precision–recall curve, computed with `average_precision_score`.

Precision and recall both ignore true negatives, so PR-AUC responds to what
actually matters: of the transactions the model ranks highly, how many are
fraud, and how much fraud does it surface at all. Its floor is the base rate
(~0.005), not 0.5, so the scale is honest about difficulty.

**Implementation note.** We use `average_precision_score`, a step-wise estimator,
rather than `auc(recall, precision)`. The latter applies trapezoidal
interpolation between operating points that are not linearly reachable and is
optimistically biased. The difference is small but systematic, and always in the
flattering direction.

### 2.2 ROC-AUC — reported, with a caveat

Reported for comparability with published work on this dataset, and treated as
secondary.

Under heavy imbalance the negative pool is so large that thousands of false
positives barely move the false-positive rate. A model can post a ROC-AUC near
0.99 while its precision at any usable threshold is unusable. Where both are
reported here, ROC-AUC reads considerably better than PR-AUC — that gap *is* the
argument, and the test suite asserts it holds.

### 2.3 Recall at fixed FPR — the operational metric

A fraud team cannot absorb unlimited false alarms. The realistic question is:
holding false positives to a rate the business tolerates, how much fraud do we
catch?

Reported at FPR ∈ {0.1%, 0.5%, 1%, 5%}.

**The threshold is selected on validation and applied unchanged to test.**
Choosing it on test would be a subtler form of the leakage this project is built
to avoid.

### 2.4 Precision@K — the analyst-capacity metric

An analyst clears a fixed number of cases per shift. What matters is the hit
rate at the top of the ranked queue, not across the whole score range.

Reported at K = 100/day (`ANALYST_DAILY_CAPACITY`).

### 2.5 Alert rate

The fraction of transactions flagged. This is the operational load the model
imposes and the direct measure of the "excessive false positives" in the problem
statement. A model with excellent recall and a 10% alert rate is not deployable.

### 2.6 Expected cost — how the threshold is actually chosen

Every metric above describes a trade-off without resolving it. Expected cost
resolves it by pricing both error types:

```
cost = n_alerts × investigation_cost
     + n_false_positives × friction_cost
     + missed_fraud_value × 1.0
```

| Term | Value | Rationale |
|---|---|---|
| `investigation_cost` | £4.00 per alert | Analyst time to triage one case. Paid on every alert, true or false. |
| `friction_cost` | £18.00 per false positive | Call-centre handling plus attrition risk from wrongly declining a genuine customer. |
| Missed fraud | transaction amount × 1.0 | The issuer refunds the customer and absorbs the loss. |

The recommended operating point is the threshold minimising this quantity on the
validation fold, then frozen.

**On the numbers.** These are order-of-magnitude estimates, not figures from a
real P&L. What drives the chosen threshold is the *ratio* between them, and
`test_expensive_false_positives_push_the_threshold_up` pins the qualitative
behaviour: as friction cost rises, the optimal system becomes more conservative.
A sensitivity analysis over the ratio is included with the results rather than a
single point estimate, because the point estimate is the least trustworthy input.

### 2.7 Value detection rate

Share of fraud *value* prevented, alongside share of fraud *count*.

Catching many small frauds while missing a few large ones scores well on recall
and still loses money. Both are reported; where they diverge, that divergence is
the finding.

---

## 3. Why there is no SMOTE

SMOTE and its variants are near-universal in public work on this dataset. They
are not used here, deliberately.

**It breaks the temporal contract.** SMOTE synthesises minority examples by
interpolating between neighbours in feature space. Those neighbours are drawn
without regard to time, so a synthetic fraud can be a blend of transactions from
either side of a split boundary — reintroducing exactly the leakage the embargo
was built to remove.

**It destroys calibration.** Oversampling changes the class prior. Predicted
scores no longer estimate P(fraud | x), so the cost model in §2.6 — which
requires genuine probabilities — becomes invalid. Cost-based thresholding is the
mechanism by which this project answers "minimise false alerts", so anything
that invalidates it is disqualifying.

**It fabricates behaviour that cannot occur.** Interpolating between two
transactions produces feature vectors that are internally inconsistent:
a trailing-24h count of 3.7, a "merchant is new to card" flag of 0.4, an implied
travel speed averaged across two unrelated journeys. Behavioural features have
semantics that linear interpolation does not respect.

**It usually is not needed.** Gradient-boosted trees handle imbalance through
`scale_pos_weight`, which reweights the loss without inventing data. Where
resampling appears to help, the gain is frequently an artefact of evaluating on
a resampled validation set — a mistake that vanishes under correct evaluation.

**What is used instead:** class weighting via `scale_pos_weight`, threshold
selection on the cost curve, and PR-AUC for model selection.

---

## 4. Calibration

Raw gradient-boosting outputs are ranking scores, not probabilities. Isotonic
regression is fitted **on the validation fold** to map them onto calibrated
probabilities.

This matters for three reasons: the cost model in §2.6 requires real
probabilities; the policy layer bands decisions by probability thresholds that
must mean the same thing across model versions; and an analyst reading "87% risk"
is entitled to expect that roughly 87 of 100 such transactions are fraud.

Reported: **Brier score** and a reliability diagram, before and after calibration.

---

## 5. Splitting

Summarised here; the full argument is in
[ADR 0003](adr/0003-temporal-split-with-embargo.md).

| Fold | Window |
|---|---|
| Train | 2019-01-01 → 2020-03-31 |
| *embargo* | 7 days, discarded |
| Validation | 2020-04-08 → 2020-06-30 |
| *embargo* | 7 days, discarded |
| Test | 2020-07-08 → 2020-12-31 |

**Test is scored exactly once**, after model selection is complete. Every
threshold, calibration map and early-stopping decision comes from validation.

**These results are not comparable to shuffled-split numbers** published on this
dataset, and will read worse. That is the intended outcome.

---

## 6. Leakage controls

Enforced by tests, not by convention. See `tests/unit/test_leakage.py`.

| Control | How it is enforced |
|---|---|
| No feature reads data at or after its own transaction time | Adversarial fixtures with a planted future spike; the feature must not move |
| Encoders never see validation or test rows | Fitted inside the training fold only; asserted by construction |
| Merchant fraud rate is not self-referential | Out-of-fold and time-shifted target encoding |
| Model is learning signal, not an artefact | Shuffling labels must collapse PR-AUC to the base rate |
| Offline and online features agree | `test_train_serve_parity.py` scores the same batch through both paths and asserts identical vectors |

The last one deserves emphasis. Train/serve skew — where the features a model
was trained on differ subtly from those it is served — is the classic silent
production failure. It does not raise an error; it quietly degrades performance.
Sharing one code path and testing the equality is the only reliable defence.

---

## 7. Baselines

Results are meaningless without a floor. Three are reported:

1. **Rules only** — the deterministic YAML engine with no ML. Represents what a
   traditional system achieves and is the honest comparator for the claim that
   ML reduces false positives.
2. **Logistic regression** — an interpretable linear model on the same features.
   Separates "the features are good" from "the gradient boosting is good".
3. **Amount-only heuristic** — flag the top N% by transaction value. A trivially
   available signal; anything not beating it is not earning its complexity.

---

## 8. Robustness reporting

Aggregate metrics conceal failure modes that matter operationally.

- **By segment** — merchant category, amount band, hour of day, customer age
  band. Uniformly strong aggregate performance with a weak segment is a real
  vulnerability, since fraud migrates toward weak spots.
- **By month across the test window** — demonstrates whether performance decays
  over a six-month horizon, which sets the retraining cadence in the model card.
- **Drift** — PSI per feature between the training distribution and a rolling
  live window, with the conventional bands: < 0.10 stable, 0.10–0.25 monitor,
  > 0.25 investigate.
