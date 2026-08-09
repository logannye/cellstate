# ADR 0005: Belief-subject semantics

- **Status:** Proposed
- **Date:** 2026-08-09

## Decision to make

Schema v1 exposes a generic `CellStateBelief` identified by `subject_id`, while the biological
program requires different semantics for an individually tracked cell, a clone/lineage, a sampled
population, and a spatial niche. Before building real-data adapters, the project must decide whether
to introduce discriminated belief contracts or a typed subject descriptor in schema v2.

This is separate from the experimental dataset manifest's **sampling-subject** taxonomy. A dataset
may sample individual cells destructively while supporting only a population-level longitudinal
belief. Assay row granularity does not determine the level of the scientific estimand.

## Required outcomes

The accepted design must:

- prevent destructive cells at different times from becoming one individual history;
- state the aggregation and experimental unit of every belief and target;
- distinguish observed identity from inferred population transport or lineage coupling;
- represent clone/barcode uncertainty without treating barcode equality as certain parentage;
- define which evidence roles may update each subject kind;
- bind forecast and uncertainty semantics to the same subject level; and
- provide an explicit v1 migration or compatibility strategy.

Until this ADR is accepted, public-real adapters may catalog sampling structure but may not emit a
biological `CellStateBelief` or claim individual longitudinal inference.
