# ADR 0001 — Bind infrastructure behind protocols, with a native default

**Status:** Accepted · **Date:** 2026-08-29

## Context

FraudLens needs a message broker and a low-latency online feature store. The
obvious production choices are Kafka and Redis. Both normally arrive via Docker
Compose.

The development machine for this project has no Docker daemon, and installing
Docker Desktop is a multi-gigabyte prerequisite. More generally, a project that
cannot be run without first provisioning infrastructure is a project most
reviewers will never actually run — and for a portfolio piece, "clone and it
works" has real value.

At the same time, dropping Kafka and Redis entirely would misrepresent how a
system like this is built, and would remove the parts a reviewer from a bank
will specifically look for.

## Decision

Define `StreamBroker` and `FeatureStore` as protocols, and ship two
implementations of each:

| Protocol | `FRAUDLENS_PROFILE=native` (default) | `FRAUDLENS_PROFILE=docker` |
|---|---|---|
| `StreamBroker` | `InProcessBroker` (asyncio queue) | `KafkaBroker` (aiokafka) |
| `FeatureStore` | `DuckDBFeatureStore` | `RedisFeatureStore` |

The profile is read exactly once, in the dependency-injection layer at startup.
No business logic branches on it. `docker-compose.yml` ships in the repo and
provisions Redpanda, Postgres, Redis, Prometheus and Grafana.

## Consequences

**Good.** A fresh clone runs with `pip install -e .` and nothing else. The
containerised architecture is still visible and exercisable. The protocol
boundary is genuinely useful for testing: unit tests bind fake implementations
without patching.

**Bad.** Two implementations of each protocol must be kept in step, and only one
of them is exercised on this machine. The Kafka and Redis paths are covered by
contract tests written against the protocol, but they are not run in CI, and
this is stated plainly rather than glossed over.

**Rejected alternative — Docker only.** Closest to production, but blocks local
development on this machine and deters reviewers.

**Rejected alternative — native only.** Simplest, but discards the distributed
architecture that makes the project representative of real transaction
processing.
