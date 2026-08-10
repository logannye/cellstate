# ADR 0006: Independently scoped biological-data eligibility

- **Status:** Accepted
- **Date:** 2026-08-09

ADR 0007 subsequently advances the manifest to `0.3-experimental`; the scoped eligibility and
permission decisions below remain in force.

## Context

A public dataset is not simply eligible or ineligible for `cellstate`. One experimental slice may
support an assay likelihood, another a population transition, and neither an individual-cell
trajectory. The same scientific claim may also be defensible at one horizon, endpoint, or biological
scope and indefensible at another.

The experimental manifest `0.1-experimental` allowed only one assessment per scientific-claim kind.
It had no stable assessment identity, could not distinguish repeated functional endpoints sharing an
ontology term, and could not state whether an exact slice was suitable for a particular training loss
or evaluation metric. Downstream code could therefore be tempted to promote a narrow claim into a
dataset-wide capability.

## Decision

1. The dataset-manifest contract advances independently to `0.2-experimental`. No `0.1` manifest is
   silently coerced.
2. Claim, loss, and metric eligibility are separate immutable assessment records. Every record has a
   globally unique assessment ID, an exact canonical scope, evidence, status, assumptions or
   blockers, and a content fingerprint.
3. The same claim, loss, or metric may be assessed at multiple distinct scopes. A duplicate semantic
   assessment at the same exact scope is rejected, even under a different ID or tuple order.
4. Loss and metric assessments cite exact claim-assessment IDs and fingerprints. In this version,
   their scopes must equal the supporting claim scopes; implicit subset, superset, union, and ontology
   similarity are deliberately unsupported. A supported objective's data sources exactly cover the
   cited claim-evidence sources, preventing scientific or legal requirements from disappearing by
   omission.
5. Functional readouts have stable readout IDs. A scope names exact readouts rather than borrowing
   any endpoint with the same ontology label.
6. A supported metric or loss declares a leakage-safe experimental split unit. This is a structural
   requirement only. A future split manifest must still prove immutable row membership.
7. Scientific eligibility and legal permission remain separate. Resolution combines the exact data
   sources with the most restrictive applicable use-policy layers and reports both outcomes.
8. The only downstream eligibility key is the tuple `(dataset manifest fingerprint, assessment ID,
   assessment fingerprint)`. Dataset ID, accession, claim kind, or a valid manifest alone never
   authorizes use.

## Consequences

- Loaders, trainers, evaluators, and benchmark builders must request exact assessment references and
  fail closed when a scope, fingerprint, source, split unit, or permission differs.
- Conditional scientific support remains conditional; a loss or metric cannot become stronger than
  any supporting claim.
- Training-loss eligibility does not imply model-selection, calibration, or locked-test eligibility.
- Loss assessments are training-only. A held-out-modality loss masks each exact scoped modality in
  turn; composing horizons or scopes requires resolving every exact assessment reference.
- Repeated outputs, horizons, and overlapping scopes can coexist without being merged.
- Metric IDs remain local role names until roadmap item 8 freezes content-addressed metric
  definitions. Metric families requiring event/censoring, OOD axes, paired-history tasks, exact
  candidate sets, or decision utility remain ineligible until those semantics exist.
- Dataset manifests still describe source evidence. They do not claim that a trained model passed a
  validation criterion, and they never gain downstream model or run back-references.
- Field-clock lag semantics, cross-study transport, exact repeated intervention-design instances,
  normalized-row lineage, and immutable split membership remain fail-closed future contracts.
