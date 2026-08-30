# Threat model

Which fraud patterns FraudLens is built to catch, which layer catches each, and
— just as importantly — what it cannot catch.

A detection system that does not state its blind spots is not trustworthy. Every
typology below is either explicitly modelled in the synthetic generator with a
labelled ground truth, or explicitly listed as out of scope.

---

## Coverage summary

| Typology | Primary signal | Caught by | Expected recall |
|---|---|---|---|
| Card testing | Many distinct merchants, small amounts, minutes apart | Rule `R-VEL-001` + model | High |
| Account takeover | High value, unfamiliar merchant, overnight | Rule `R-AMT-002` + model | High |
| Geo-impossible / cloned card | Implied travel speed | Rule `R-GEO-001` (hard block) | Very high |
| Spending spree | Escalating amounts, velocity | Model | Moderate–high |
| Subtle / patient fraud | *None* | Neither | **Near zero — by design** |

---

## 1. Card testing

**The attack.** An attacker holds a list of stolen card numbers, most of which
are dead. Before risking a large purchase they validate each card with several
small transactions at unrelated merchants — typically under £15, seconds to
minutes apart. Card-not-present merchants with weak verification are preferred.

**Why it is detectable.** The signature is not any single transaction; each looks
trivial. It is the *pattern*: a sudden spike in distinct merchants for a card
that normally uses a handful of local ones, at amounts far below its usual.

**Detection.**
- `distinct_merchants_24h` — the discriminating feature. A genuine customer
  making several purchases usually does so at one merchant.
- `txn_count_1h` — burst velocity.
- Rule `R-VEL-001` fires on the conjunction of all three conditions (velocity,
  merchant spread, low amount), so ordinary busy-shopping days do not trigger it.

**Weakness.** An attacker who spaces tests over days, or spreads them across many
cards rather than many merchants on one card, defeats the velocity window. A
merchant-side view would catch that; a card-side view cannot.

---

## 2. Account takeover

**The attack.** Credentials are compromised. The attacker makes high-value
purchases at merchants the cardholder has never used, usually overnight so the
genuine holder does not notice in time to cancel.

**Detection.**
- `amt_ratio_mean_30d` — value relative to the card's own baseline, which is far
  more informative than the absolute amount.
- `is_new_merchant`, `is_night`.
- Rule `R-AMT-002` requires all three together: high value *and* unfamiliar
  merchant *and* overnight. Any one alone is unremarkable.

**Weakness.** A cardholder who genuinely makes a large late-night purchase at a
new merchant looks identical. This is the single largest source of false
positives, and it is why the cost model prices customer friction at more than
four times the investigation cost.

---

## 3. Geo-impossible / cloned card

**The attack.** The card is physically cloned. The clone is used in one location
while the genuine card is used in another, implying a journey no traveller could
make.

**Detection.**
- `implied_speed_kmh` = distance since the previous transaction ÷ elapsed time.
- Rule `R-GEO-001` **blocks** outright above 900 km/h with at least 100 km
  travelled. This is the one typology where a deterministic hard stop is
  justified: the conclusion is a matter of physics, not statistics, so it does
  not need a probability.

**Why the threshold is 900 km/h.** Above commercial cruise speed, so a customer
who took a flight does not trigger it. The 100 km distance floor prevents GPS
jitter between two nearby merchants producing a spurious high speed over a very
short interval.

**Weakness.** Requires two transactions to compare. The first use of a cloned
card in a new region is invisible to this signal, and only
`dist_home_merch_km` (advisory) speaks to it.

---

## 4. Spending spree

**The attack.** The card is known to be compromised and the attacker extracts
maximum value before it is reported — escalating purchases over minutes to
hours.

**Detection.** Primarily the model rather than a single rule: the combination of
rising `amt_zscore_30d`, `txn_count_1h` and `amt_sum_1h` is learnable but hard
to express as a clean threshold. This is precisely the case that justifies ML
alongside rules.

**Weakness.** Early transactions in the spree look ordinary. Detection improves
as the spree progresses, which means some loss is unavoidable.

---

## 5. Subtle / patient fraud — the acknowledged blind spot

**The attack.** A familiar local merchant, an ordinary amount, mid-afternoon,
no velocity anomaly. The attacker knows what triggers alerts and stays inside
normal behaviour.

**Detection: none.** Expected recall is near zero, and the synthetic generator
emits this typology deliberately so that headline recall cannot be flattered by
an artificially separable problem. Roughly 18% of generated fraud is of this
kind.

**Why it is unsolvable here.** There is no signal in the transaction record. The
transaction *is* normal on every dimension the pipeline observes. Catching it
requires data this system does not have:

- Device fingerprint and IP reputation
- Behavioural biometrics (typing cadence, navigation patterns)
- The cardholder's own report
- Cross-institution intelligence sharing

Stating this plainly matters more than papering over it. A model claiming
near-perfect recall on fraud detection is either overfitting, leaking, or being
evaluated on a problem easier than the real one.

---

## Out of scope

Deliberately not addressed, and not claimed:

| Threat | Why it is out of scope |
|---|---|
| Merchant-side collusion | Needs a merchant-centric view; the entity here is the card |
| Money laundering / structuring | A different problem with different regulatory requirements (AML, not fraud) |
| First-party fraud (friendly chargebacks) | The genuine cardholder transacts, then disputes — indistinguishable at authorisation |
| Synthetic identity fraud | Detected at account opening, not at transaction time |
| Adversarial evasion of the model itself | See below |

---

## Adversarial considerations

**Model inversion via probing.** An attacker with repeated access to decisions
can map the decision boundary and craft transactions just under it. Mitigations
present: the rule layer is not learnable from score feedback alone, thresholds
are not exposed via the API, and the returned band is coarse (four values)
rather than a raw probability at full precision to unauthenticated callers.
Mitigations *not* present: rate limiting per card, and score-feedback
throttling. Both would be required before real deployment.

**Feature-store poisoning.** The trailing aggregates are built from a card's own
prior transactions. An attacker able to inject benign-looking transactions could
inflate the card's baseline, making a later large purchase appear ordinary — the
`amt_ratio_mean_30d` feature is the target. The 30-day window bounds how far
back poisoning can reach, but does not prevent it. A defence would compare
against a longer-horizon robust statistic (median rather than mean), which is
noted as future work rather than claimed as implemented.

**Drift as an attack surface.** Fraud migrates toward weak spots. If a segment's
performance degrades and nobody notices, that segment becomes the preferred
attack route. This is why per-segment metrics and PSI drift monitoring are
treated as detection capability rather than as reporting.
