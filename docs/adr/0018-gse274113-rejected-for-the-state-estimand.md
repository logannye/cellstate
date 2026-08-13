# ADR 0018: GSE274113 is rejected for the state estimand and retained for the observation model

- **Status:** Accepted
- **Date:** 2026-08-13

## Context

`GSE274113` was the roadmap's primary Phase 2 candidate and its only one. The roadmap said plainly
that the intended design — day 7 as the pre-cutoff observation, days 9, 11 and 14 as horizons, the
library as the unit that spans the cutoff — was a hypothesis `Q4` had to confirm, and that if no
unit spanned the cutoff the source failed S1 and had to be redesigned or rejected.

`Q4` is now complete. The
[structural census](../data/representability/gse274113-perturb-multiome.md) computed the structure
from the bytes; the provenance review recorded in the same document supplied licence, byte identity,
and the parent-culture count. Four findings bear on the decision, in ascending order of how much
they settle.

**1. No library spans a timepoint.** Libraries appearing at more than one timepoint: **0 of 14**.
The library therefore cannot be the unit that spans the inference cutoff, and a day-7-versus-day-14
difference is a difference between two disjoint sets of libraries. S1 fails for the intended design.

**2. The longest horizon is empty.** Correlating the library-averaged effect vectors and
disattenuating by measured Spearman–Brown reliabilities, day 7 → day 14 is **ρ = −0.026** against a
target-label permutation null whose 95th percentile of `|r|` is **0.025**. The decay is monotone:
0.577 at two days, 0.179 at four, zero at seven. Disattenuation is what makes this readable — a
near-zero ρ cannot be dismissed as attenuation.

**3. The libraries are not interchangeable technical splits.** The ratio of across-library spread at
a fixed target to across-target spread is 0.53 and 0.51 at days 7 and 9, and **0.89 and 0.99** at
days 11 and 14. By day 11 the library-to-library variation is as large as the entire between-target
effect.

**4. There is one independent parent culture, and no way to measure it.** The publication describes
its robustness check as "an independent biological replicate (using HSPCs from a different donor and
sgRNAs targeting different locations in the coding sequences of each TF, for a subset of 8 TFs)",
placing the primary experiment — the one these fourteen libraries come from, confirmed by the
matching 137,604-cell count — at a **single donor**. The deposited metadata carries no donor field;
every sample records only `tissue: Hematopoietic cells`. Neither the publication nor the bytes
demonstrates more than one independent biological unit.

Findings 1 to 3 constrain the design. Finding 4 ends it, and it is the one this project's own rules
make decisive: rule 8 requires splits to follow independent experimental units, `roadmap.md` requires
a split unit that is "a genuine independent replicate, present in sufficient number to bootstrap",
and the only unit above the library available here is the donor.

## Decision

1. **`GSE274113` is rejected for the Phase 2 state-bearing estimand.** It cannot satisfy S1 with the
   library as the spanning unit, its longest declared horizon is indistinguishable from a
   permutation null, and above the library it supplies one biological unit. No redesign reaches a
   bootstrappable number of independent units, because that number is a property of the experiment
   and not of the analysis.

2. **The target-as-spanning-unit redesign is inadmissible, independently of the evidence.** Rule 8
   enumerates the units a split may follow — well, plate, library, donor, clone, or study — and a
   target is the treatment, not a container. Adopting it would also make a held-out-intervention
   fold incoherent, since the object held out and the object bootstrapped would be the same. Were
   this ever to be reconsidered, it is an amendment to rule 8 and therefore a rule 4 change in its
   own right, not a design choice available to `Q5`.

3. **The source is retained, under rule 7, for the observation-model question (`Q6`, S5).**
   Rejection in this project is claim-specific: a dataset may support one estimand and not another,
   and this one is well suited to a question that needs neither a spanning unit nor a horizon. All
   **280 of 280** `(library, target)` combinations are populated, so intervention is fully crossed
   with library and cleanly separable from it; the series carries paired same-cell RNA and ATAC; and
   finding 3 is not merely a defect here but the object of study — S5 asks that varying a nuisance
   variable at fixed biology change the predicted observation and not the inferred state, and a
   measured library nuisance axis of that size is exactly the material such a test needs. The
   weakness that disqualifies the source for one claim is the asset that qualifies it for the other.

4. **Licence status is recorded as unresolved, and unresolved is not permissive.** The series page
   carries no licence and no data-use statement. This takes the same posture already recorded for
   `GSE141064` under `gse141064-geo-rights-unresolved`. Public downloadability is not a grant. Any
   `Q6` use must resolve terms before the bytes are used as evidence for a published claim, and rule
   6 makes that a precondition rather than a follow-up.

5. **Byte identity is recorded as corroborated, not established, and re-verification is deferred to
   first use.** GEO publishes no checksum for these artifacts, and the series `filelist.txt` covers
   only the members of `GSE274113_RAW.tar`, not the fourteen matrices. All fifteen local artifacts
   agree with the directory listing's megabyte-rounded sizes — fifteen of fifteen — which excludes a
   substituted file and does not exclude a modified one. Since the source no longer sits on the
   state path, spending three gigabytes of download to close the gap now would be spending ahead of
   a claim. It becomes a precondition of `Q6`, not of this record.

6. **`Q5` does not proceed on this source, and does not proceed on any source until one is
   selected against criteria fixed in advance.** `Q4`'s completion unblocks `Q5` procedurally and
   leaves it with nothing to freeze. The selection criteria and the candidate landscape are the
   subject of the next record; they are deliberately not set here, because setting them in the same
   document that rejects a candidate invites fitting them to that rejection.

## What this costs, stated plainly

The roadmap had one Phase 2 candidate and now has none. Phase 2 is the phase that acquires an
estimand in which a hidden state exists to be inferred, and every later phase — observation models,
posterior inference, the sufficiency verdict itself — is downstream of it. The deferral of Vertical B
was conditioned on Vertical A producing a sufficiency verdict, and this record moves that verdict
further away rather than closer.

The alternative was to proceed on a source whose spanning unit does not exist, whose longest horizon
is empty, and whose independent-unit count is one. A sufficiency verdict computed there would have
been a number with an interval and no meaning — which is precisely the failure
[ADR 0017](0017-the-sufficiency-verdict-must-fail-closed.md) was written to make impossible, arriving
by a different route. Rejecting a source for cause is a result. Recording it is how the project
avoids paying for it twice.

## Consequences

- `Q4` is **decided but not complete.** The census, the provenance review, and this decision are its
  substance; its done-when names a reviewed manifest, and that manifest is not yet written. The
  claim-specific split decision 3 makes — ineligible for the state estimand, retained for `Q6` — is
  prose until a `ClaimAssessment` carries it, and prose is not machine-checkable. This record does
  not relax the done-when to match what has been produced.
- `Q5` remains blocked, and is now blocked on source selection rather than on this source.
- `Q6` gains a candidate source with three preconditions: resolved licence terms, established byte
  identity, and a determination of which of the two per-replicate artifact families it reads — the
  series carries both a 27–94 MB `filtered_feature_bc_matrix_N.h5` family and the 160–306 MB
  `repN_` family this census measured, and per-claim eligibility is a property of exact artifacts.
- The ATAC peak-set sharing question, deferred by the census, follows the source to `Q6`.
