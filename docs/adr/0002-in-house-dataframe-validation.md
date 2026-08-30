# ADR 0002 — Validate dataframes in-house rather than with pandera

**Status:** Accepted · **Date:** 2026-08-29

## Context

Data entering the pipeline needs a contract: dtypes, nullability, ranges,
uniqueness, allowed values. `pandera` is the standard library for this and was
the first choice.

Two things argued against it here. This project runs on Python 3.14, which
forces `pandas` 3.x because no `pandas` 2.x wheels exist for 3.14. `pandas` 3.0
made copy-on-write mandatory and changed the default dtype for text columns.
Third-party libraries built against the `pandas` 2.x internals are the most
likely place for that to bite, and a validation library sits on the critical
path of every single run. Taking a hard dependency there, on a bleeding-edge
interpreter, is a poor risk trade for what is ultimately a few hundred lines of
checking logic.

The requirement is also narrow: one frame shape, validated at one boundary.
`pandera` is built for a much larger problem (schema inheritance, hypothesis
testing, synthesis strategies), none of which is needed.

## Decision

Implement `ColumnSpec` and `DataFrameSchema` in `fraudlens.data.schema`. Keep
the interface deliberately pandera-shaped — a declarative column list and a
`validate(df) -> df` method — so that swapping in `pandera` later is a contained
change.

One behaviour is a deliberate improvement on the default: **every violation in a
frame is collected and reported together**. Failing on the first bad column
makes diagnosing a new data drop a slow game of whack-a-mole.

## Consequences

**Good.** Zero dependency risk on the critical path. Full control over error
messages. The validator is small enough to be completely tested, and it is.

**Bad.** Code we own and maintain rather than code someone else maintains. It
covers only the checks this project needs; anything more exotic means either
extending it or making the swap.

**Revisit when** the project moves to a stable pandas/Python combination, or the
validation needs grow beyond declarative column checks.
