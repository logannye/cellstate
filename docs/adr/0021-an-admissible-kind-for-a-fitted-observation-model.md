# ADR 0021: An admissible artifact kind for a model fitted on real bytes that claims nothing causal

- **Status:** Accepted
- **Date:** 2026-08-13

## Context

The purpose of this project is to compute a faithful representation of hidden cellular state. After
twenty decision records and nine queue items, **no representation of cellular state has ever been
computed.** `CellStateBelief` is constructed at exactly one site in `src/` — a synthetic
linear-Gaussian reference. [ADR 0019](0019-build-the-representation-on-held-evidence.md) reordered
the queue specifically so the representation would be built on evidence already held, and named that
state as structural rather than incidental.

Attempting it reveals that the public boundary refuses, and refuses **by design**.

`estimate_cell_state` admits a model only through `_validated_public_model_descriptor`
(`src/cellstate/api.py:79`), and `ModelArtifactKind` offers exactly three labels
(`src/cellstate/ports/models.py:43`). None of them fits a model fitted on real public bytes that
makes no causal and no control claim:

| Kind | Why it does not fit |
| --- | --- |
| `BIOLOGICAL_MODEL` | Rejected unconditionally at `api.py:94` until the content-addressed admission registry resolves implementation, model, training and validation bytes. That registry is suspended, and this record does not resurrect it. |
| `CONTRACT_REFERENCE` | `ports/models.py:98` **forbids** `support_envelope_id`, `training_support_id` and `validation_evidence_*`. A model wearing this label is structurally unable to cite the fold it was fit on. |
| `SYNTHETIC_TEST_MODEL` | Permits those citations, but is a false statement about real cells. Rule 6 exists to keep synthetic and real evidence distinguishable; borrowing the synthetic label to carry real bytes destroys exactly that distinction. |

The gate at `api.py:94` is correct and this record does not weaken it. What it guards is the claim
that a runtime's outputs are *trustworthy for biological decisions* — that identification and
control have been verified against resolved bytes. That is the right bar for a biological runtime.
It is the wrong bar for the strictly smaller thing needed here: an **observation model**, fitted on
public bytes, that reports a population's state and its uncertainty and asserts nothing about what
would happen under an unseen intervention.

Without a label for that smaller thing, the project cannot compute its own object. That is not a
safety property. It is the apparatus refusing to let the purpose be attempted.

## Decision

1. **Add `ModelArtifactKind.EMPIRICAL_OBSERVATION_MODEL`** — a model whose parameters were fitted on
   real, publicly downloadable bytes, which reports state and uncertainty, and which **makes no
   identified causal claim and no control claim**.

2. **It carries the same provenance obligations as a biological model.** It joins
   `BIOLOGICAL_MODEL` and `SYNTHETIC_TEST_MODEL` in the branch at `ports/models.py:107` that
   *requires* a support envelope, a training support id, and validation evidence, each with a
   fingerprint. The whole reason `CONTRACT_REFERENCE` is unusable here is that it cannot cite what
   it was fit on; a kind added to solve that problem must not be able to decline to.

3. **It is barred from causal claims by construction, not by convention.** `estimate_cell_state`
   gains a fail-closed gate: a belief from an `EMPIRICAL_OBSERVATION_MODEL` whose causal support
   reports `IDENTIFIED_POPULATION_EFFECT` or `TRANSPORTED_UNDER_ASSUMPTIONS` raises
   `CapabilityError`. The new kind therefore cannot be used as a route around the admission
   registry: everything the registry gates remains gated, and the new label buys **only** the
   ability to say "this is a state estimate, here is its uncertainty, here is the fold it came
   from."

   A reviewer's first question about a new admissible kind is whether it is a loophole. The answer
   has to be enforced in code, because a convention that nothing checks is a guard that cannot fire
   — the defect [ADR 0017](0017-the-sufficiency-verdict-must-fail-closed.md) was written to correct.

4. **`BIOLOGICAL_MODEL` remains rejected, unchanged.** No line of the existing gate is edited, and
   its test keeps passing. When the admission registry becomes executable, biological runtimes are
   admitted through it and this kind does not become a shortcut.

5. **What this kind is not permitted to be described as.** Nothing produced under it may be called a
   sufficiency result, a faithfulness verdict, or evidence that the state is complete
   ([ADR 0019](0019-build-the-representation-on-held-evidence.md) decision 4 carries here verbatim).
   On `GSE274113` the capabilities S1, S3 and S7 remain structurally unreachable — no library spans
   a timepoint, one horizon is declared, and S7 follows from both. A belief emitted under this kind
   is a *snapshot state estimate with an earned posterior*, and the ledger entries it advances are
   named explicitly in its model card.

## What this costs

A fourth artifact kind is a widening of the public boundary, and widenings are how boundaries stop
meaning anything. Two things hold the line. The gate in decision 3 makes the causal bar
*unreachable* under this label rather than merely discouraged. And decision 2 makes the label more
expensive to wear than `CONTRACT_REFERENCE`, not less — it demands provenance the cheap label is
forbidden to supply.

The honest residual risk is different and worth stating plainly: a state estimate with an earned
posterior, emitted from real cells, **reads as more progress toward the purpose than it is.** Three
or four ledger capabilities on evidence that structurally cannot carry S1 is not a faithful
representation. Decision 5 is the part of this record most likely to be quietly eroded, and it is
the part that matters most.

## Consequences

- `CellStateBelief` acquires its first non-synthetic construction site, and the public API acquires
  its first caller carrying real cells.
- The ledger can move off 0 of 10, with its ceiling stated rather than discovered.
- A future biological runtime is unaffected: it still needs the admission registry, and this kind
  cannot stand in for it.
