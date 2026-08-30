# ADR 0003 — Temporal split with a 7-day embargo

**Status:** Accepted · **Date:** 2026-08-29

## Context

The dataset spans 2019-01-01 to 2020-12-31 and ships pre-split into
`fraudTrain.csv` and `fraudTest.csv`. Two decisions were needed: whether to use
the vendor split, and how to partition the data at all.

Most published work on this dataset uses `train_test_split(..., shuffle=True)`.
That is wrong here for two compounding reasons:

1. **Temporal leakage.** Transactions are time-ordered and the fraud-generating
   process drifts. A shuffled split trains on transactions that occur *after*
   the ones it is evaluated on, which is not a situation any deployed model
   faces.
2. **Entity leakage.** Behavioural features are card-grouped trailing
   aggregates. Under a shuffled split, a card appears in both folds and its
   history is shared across them, so the model is partly scored on cards whose
   behaviour it has already memorised.

A purely chronological split fixes (1) but not (2). Features look back up to 30
days, so a validation row one day past the boundary computes its trailing
aggregates almost entirely from training rows.

## Decision

Ignore the vendor split. Re-partition the union of both files chronologically,
with an embargo gap between folds:

| Fold | Window | Purpose |
|---|---|---|
| Train | 2019-01-01 → 2020-03-31 | Model and encoder fitting |
| *embargo* | 7 days | discarded |
| Validation | 2020-04-08 → 2020-06-30 | Thresholds, calibration, early stopping |
| *embargo* | 7 days | discarded |
| Test | 2020-07-08 → 2020-12-31 | Scored once, at the end |

`SplitBoundaries` enforces these invariants at construction time and raises if
folds overlap, are reversed, or are separated by less than the embargo. The
constraint is executable, not just documented.

The test period doubles as the replay stream for the live demo, so the demo is
running on data the model has genuinely never seen.

## Consequences

**Good.** Reported metrics mean what they claim. The split is reproducible and
its invariants are tested. Scoring test exactly once removes the slow leak of
selecting on the test set.

**Bad.** The embargo discards data. Metrics will read *worse* than the
shuffled-split numbers commonly published on this dataset — this is the point,
but it means the results are not directly comparable to that body of work, and
the README says so.

**Note.** The embargo is 7 days against a 30-day maximum lookback. A 30-day
embargo would be fully rigorous; 7 is a deliberate compromise that removes the
bulk of the leakage while preserving data. The residual is documented in the
model card rather than hidden.
