# ADR 0020: RNA-first nuisance separation, and the modality test moves behind clean ATAC

- **Status:** Accepted
- **Date:** 2026-08-13

## Context

[ADR 0019](0019-build-the-representation-on-held-evidence.md) moved the observation-model item ahead
of the estimand freeze so the representation would be built on evidence already held. `Q5` was then
specified against measured facts, including a fix for the ATAC feature space: re-quantify called
peaks into a fixed genomic bin grid, since peak identifiers carry coordinates and the fragment files
are not held.

**That fix was measured, and it does not work.**

| Feature space | `r`(timepoint index, feature-space size) |
| --- | --- |
| raw per-library peak sets | −0.973 |
| after 10 kb fixed binning | **−0.963** |

Binning fixes the *coordinate system* — the grid is shared by construction, 120,468 union bins, and
99.95% of peak identifiers parse — but it does not fix the *confound*, because which bins are
occupied is still derived from per-library peak calls, and those calls track differentiation. The
grid stopped moving with the biology; the occupancy did not.

A second defect sits underneath the first, and it is one this project has already named. A bin with
no peak call in a given library is **not measured**, not zero accessibility. Filling it with zero to
obtain a rectangular matrix is imputation, which is exactly what the zero-panel doctrine carried
into this phase by [ADR 0017](0017-the-sufficiency-verdict-must-fail-closed.md) forbids: a sample
whose panel total is zero is a missing observation under an observation model, not an evaluable
target. Across the union grid, 54.3% of entries would be imputed in this way. The same defect that
made the frozen sci-Plex3 benchmark unevaluable reappears here in a different assay, and it must be
answered the same way rather than absorbed because it is inconvenient.

Clean ATAC over a fixed grid therefore requires fragment-level requantification. The fragment files
are per-sample members of `GSE274113_RAW.tar` totalling roughly 45 GB, and are not held.

The RNA half has none of these problems. Measured: **one byte-identical 36,601-gene identifier list
across all fourteen libraries.** Shared feature space, no per-library calling step, nothing imputed.

## Decision

1. **`Q5`'s nuisance separation proceeds on RNA, now.** The library remains the nuisance axis and its
   size is already measured — across-library spread at a fixed target reaches 0.53, 0.51, 0.89 and
   0.99 of across-target spread at days 7, 9, 11 and 14. S5 asks that varying a nuisance variable at
   fixed biology change the predicted observation and not the inferred state, within a predeclared
   bound on held-out units. RNA supplies that test in full.

2. **The held-out-modality test in both directions moves out of `Q5` and behind clean ATAC.** It is
   the only part of the item that requires a second modality, and the held bytes cannot supply one
   without either a declared bias or an imputation. Rather than weaken the test to fit the data, the
   test keeps its strength and waits. It becomes a separate item conditioned on one of two routes:
   the peak-call intersection grid (55,029 bins present in all fourteen libraries, where every value
   is a measurement and nothing is imputed, at the cost of a bias toward constitutively open
   chromatin because the intersection is pinned by the sparsest day-14 libraries), or fragment-level
   requantification after the ~45 GB download.

   Weakening a declared test so that available data can pass it is the failure mode this project
   exists to prevent, and it would be undetectable afterward. Moving the test is visible.

   Inserting an item renumbers everything after it, so this record carries its own mapping table and
   it supersedes [ADR 0019](0019-build-the-representation-on-held-evidence.md)'s. Read every earlier
   table through this one.

   | Before this record | After | Note |
   | --- | --- | --- |
   | `Q5` observation model | `Q5` | unchanged in ordinal; loses the held-out-modality test |
   | — | `Q6` held-out-modality test on clean ATAC | new, authorized by decision 2 |
   | `Q6` first biological belief | `Q7` | unchanged in substance |
   | `Q7` freeze the estimand | `Q8` | unchanged in substance; still blocked on source selection |
   | `Q8` state backend and verdict | `Q9` | unchanged in substance |

   Two consequences of this table are not cosmetic. The **observational floor**, assigned by
   [ADR 0017](0017-the-sufficiency-verdict-must-fail-closed.md) to the slot then holding the estimand
   freeze, is measured at `Q8` — not at `Q5`, which after two reorders holds the observation model
   and measures no floor. The **specification-only constraint** on frozen metric suites likewise
   first binds at `Q8`, the item that freezes a suite. Phase 1's graduation gate cited the old
   ordinal for both and is corrected in the same change as this record.

3. **S2 and S4 remain in the first-belief item — `Q7` after the table above — and are unaffected.**
   Neither needs a second modality. `NT` versus 19 perturbed transcription factors is an RNA-readable
   contrast, fully crossed with library at 280 of 280, so both halves of S4 survive intact.

4. **Licence status is recorded, and its scope is recorded with it.** The publication's data
   availability statement is a bare deposit statement — "Raw and processed data have been deposited
   at GEO (accession numbers GSE274110 and GSE274113)" — with no terms, and no reuse conditions
   appear anywhere in the article. The article itself is CC BY 4.0 under HHMI's open-access policy;
   **that licenses the article, not the data**, and the two are not conflated here.

   The state is therefore *no terms asserted by submitter or journal, and no grant asserted either*,
   which is the same state already recorded for `GSE141064` under `gse141064-geo-rights-unresolved`.
   The project owner has authorized use at this scope. This is recorded as an explicit, scoped
   policy rather than as a waiver: rule 6 requires the licence to be **recorded**, not to be
   permissive, so a recorded "unresolved, owner-authorized for internal method development, **not
   cleared for any published biological claim**" satisfies rule 6 as written. **No rule is amended
   or weakened by this record.** Any future publication re-opens the question rather than inheriting
   this decision.

## What this costs

`Q5` gets easier, and that deserves suspicion. The modality test was the part that would have caught
an observation model that separates nuisance from biology in one assay by quietly encoding it in the
other, and moving it means that check is not performed now. Decision 2 keeps the test at full
strength precisely so its absence stays visible in the queue rather than dissolving into a softer
version that RNA alone can pass.

The RNA-only result must therefore be described as what it is: nuisance separation demonstrated in
one modality, with the cross-modality check outstanding.

## Consequences

- `Q5` becomes buildable on held bytes with no imputation and no declared bias.
- The first `CellStateBelief` from real cells is no longer blocked on a 45 GB download.
- A new queue item carries the held-out-modality test, conditioned on route A or route C above.
- The measured falsification of the binning fix is retained in
  [the representability ledger](../data/representability/gse274113-perturb-multiome.md) rather than
  quietly replaced, because a design that was proposed, tested and rejected is evidence about the
  source.
