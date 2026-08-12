# Data contracts

All biological time is experiment-relative seconds. Wall-clock timestamps are provenance only.
Histories are canonicalized by `(time_seconds, event_id)` and reject duplicate IDs, mixed subjects,
and evidence after the request time.

Observations preserve raw/minimally processed values, assay metadata, quality, uncertainty, and an
explicit missingness state. Observed zero and missing are different. Censored and below-detection
records are different from technical failure. Below-detection records carry their detection limit;
one-sided censored records carry a direction and limit; interval-censored records carry ordered,
unit-consistent bounds. Their `value` remains empty so a bound is never mistaken for an observed or
imputed value.

Interventions separate intended exposure from actual perturbation realization. A guide assignment
or drug exposure is not interpreted as complete target engagement. Environment snapshots, lineage,
and neighborhood graphs are first-class inputs. Completeness declarations distinguish “no event”
from “no record supplied.”

Environment variable keys are case-folded to one canonical form. A backend must reject conflicting
assignments to the same variable and time unless its model defines an explicit precedence or fusion
rule; provenance IDs are never biological ordering rules. Realized-perturbation measurements cannot
predate the intervention whose efficacy they establish.

Lineage and population observations retain the target history subject plus an explicit evidence role
and source subject. Backends that cannot model sibling, ancestor, or population evidence must reject
it rather than treating it as a direct measurement.

Query outputs and forecast predictions declare units. Precision requirements bind a target and a
named horizon, and planning targets use typed quantities in those same units. A backend cannot make
numeric outputs comparable merely by assigning them the same ontology label.

Next-measurement selection has its own v2 input and output contracts. A
`MeasurementDecisionRequest` binds the parent belief and query to an intervention objective,
at least two semantically distinct ordered candidate scenarios, ordered candidate assay IDs,
collection and decision times, utility
units, the assay-cost conversion, and explicit delay and destructiveness penalties. A
`MeasurementRecommendation` is a separate provenance-bearing decision artifact; it is not embedded
in, or implied by, the belief. Their checked-in wire schemas are
`schemas/v2/measurement-decision-request.schema.json` and
`schemas/v2/measurement-recommendation.schema.json`.

A supported assay evaluation may report intervention-decision EVSI only when the backend has a
calibrated assay-outcome model, performs a hypothetical update for possible outcomes, replans the
same counterfactual intervention problem, and evaluates a declared decision utility. Unevaluated
components produce `NOT_EVALUATED`; failed or out-of-support components produce `UNSUPPORTED`;
supported numeric evaluations below the decision threshold produce `ABSTAINED`. Every unavailable
result carries explicit reasons and empty numeric fields, never zero-valued sentinels. The causal
claim is bound to the exact ordered decision-set fingerprint, including dose
and schedule, and transported value is allowed only when the query permits transport and the causal
and transport domains agree. Posterior covariance reduction alone is not a measurement
recommendation.

Every assay evaluation persists four typed `MeasurementEvidenceTrace` records—assay-outcome model,
hypothetical update, exact-candidate counterfactual replanning, and decision utility—bound to the
exact measurement capability scope. A passing trace requires content-addressed evidence in result
provenance. A numeric EVSI requires all four to pass; the contract reference emits four explicit
`NOT_EVALUATED` traces instead.

Run `make schemas` after any contract change. ADR 0005 accepts the breaking move to schema v2 for
typed belief subjects and associated biological semantics. Checked-in v1 schemas remain immutable;
v1 produced beliefs, forecasts, and plans must be regenerated rather than relabeled. The active
runtime is v2; `cellstate.schema.inspect_v1_payload` inventories the explicit annotations required
for legacy inputs without coercing or mutating them.

Every v2 query carries a typed subject, positive temporal resolution, bounded intervention and
environment domains, evidence policy, acceptance thresholds, and explicit constraints. Targets
declare whether they are future assay observations, latent quantities, or versioned transforms;
they also bind protocol/model identity, an exact value-schema reference, units, aggregation,
missingness, censoring, and supported horizons. Assay purposes distinguish a fixed target endpoint
from a measurement-selection candidate; only the latter carries decision cost and turnaround
economics. Actions bind assignment and randomization units, matched-control semantics, schedule,
washout, typed reversible/irreversible/unknown status, and per-action realization-evidence
requirements. Missing realization evidence remains
represented as uncertainty and may force abstention; it is never silently promoted to successful
target engagement.

Estimation compiles that complete query into a fingerprinted `CompiledStateSpecification`; the
specification embeds the bounded query contract, partitions known factor families into active and
excluded blocks, and names any active context, perturbation-realization `R`, and
observation-nuisance `Xi` dimensions. It is embedded unchanged in beliefs and forecasts.

Calculation support is scoped to the exact estimation request, evolution scenario, intervention
planning problem, or measurement-decision request. Scientific readiness is a separate result:
support, sufficiency, identifiability, calibration, causal evidence, and decision uncertainty may
be unevaluated, failed, or unsupported
even when a backend can numerically return a distribution. A malformed or computationally
unsupported request fails closed; a well-formed scientific calculation returns its typed
unsupported/abstaining artifact so callers can inspect the reasons. There is no capability or
scientific-validity bypass flag.

Provenance binds results to model/configuration and posterior-schema fingerprints, content-addressed
support envelopes, training support, and validation claim artifacts, the query, context, non-event
history structure, and the exact content hash of every eligible source event. Recursive filtering
rejects changed payloads hidden behind a previously assimilated event ID. A passing identified or
transported population-effect claim additionally binds a typed target, horizon, aggregation,
experimental unit, action/environment contrast, comparator, and randomized or quasi-experimental
design. Ordinary history events may condition a belief but cannot, by themselves, establish that
population effect. Until a typed local assignment/control/outcome graph exists, local identification
fails closed; passing claims must cite matching content-addressed external validation artifacts.
Forecast and candidate estimands also bind the exact scenario ID and SHA-256 fingerprint. Their
declared action contrast must equal the effective concrete scenario actions, including inherited
controls, so a no-action or different-dose branch cannot borrow another candidate's causal label.

Sample-backed posteriors use an explicit two-axis convention: sample artifacts have shape
`(sample_count, state_dimension_count)` with axes `("sample", "state_dimension")`; optional weights
have one `"sample"` axis and exactly one entry per sample.

Public models prevent top-level field assignment and freeze nested mappings and JSON lists in
serialization-compatible containers that reject in-place mutation. Adapters must still serialize
and revalidate contracts at process and storage boundaries before trusting their fingerprints;
immutability prevents local drift but does not authenticate an external payload.

## Public-real dataset manifests

`cellstate.data.DatasetManifest` is the **experimental** evidence boundary for biological data. Its
independent version is `0.3-experimental`; it does not change or claim stability under core runtime
schema v2. A manifest requires a publicly downloadable real-data origin, remote acquired artifacts
with retrieval timestamps and SHA-256 hashes, source-scoped use terms, an experimental-unit
hierarchy, sampling/linkage semantics, modality alignment, an exact whole-artifact or
content-addressed slice, and independently scoped claim, loss, and metric assessments.

`RepresentabilityProof` is a separate experimental artifact. It binds an exact manifest, slice,
assessment ledger, declared source-byte digests/locators, and typed representability criteria. Its
current verifier checks whether the reviewed subject/evidence declarations are structurally
expressible; it does not fetch source bytes, recompute slice membership, replay selectors, evaluate
legal permission, or admit a training loss, metric, benchmark, or biological model.

Experimental units distinguish donor, organism, tissue, clone, culture, organoid, plate, well,
sample, cell, and spatial region. The manifest binds alignment keys and sampling subjects to those
units. It rejects a default split or biological-replicate unit finer than a shared sampling or
randomization unit, preventing cells from one well from silently becoming independent replicates.

Sampling explicitly distinguishes destructive endpoints, repeated population snapshots,
lineage-linked endpoints, partial nondestructive sampling, and longitudinal nondestructive
measurement. Destructive sampling cannot claim same-cell linkage. Supported and conditionally
supported assessments carry a structured scope over subject, system boundary, biological system,
modalities, interventions, environments, exact functional readout IDs, cutoff, and horizons. Claim
gates run independently for every scope and fail closed when evidence, temporal coverage, alignment,
controls, or replication are absent.

The same claim may be assessed repeatedly only at distinct canonical scopes. Loss and metric
assessments are separate records and bind exact supporting claim IDs and fingerprints at the same
scope. Their data-source set exactly covers the supporting claim evidence so that restrictive
source policies cannot be bypassed by omission; no subset, union, or ontology-similarity inference
is performed. Each supported empirical
objective declares a leakage-safe split unit, but that declaration is not proof that a frozen split
exists. Downstream use is keyed by `(manifest fingerprint, assessment ID, assessment fingerprint)`,
never by dataset accession or claim kind alone.

Loss assessments authorize training objectives only. A held-out-modality assessment treats each
scoped modality as the masked target in turn; it does not authorize a broader modality set. Metric
IDs are local role identifiers under a typed metric family, not frozen formula definitions. Event
or survival scoring, OOD/selective-risk, predictive-sufficiency, intervention-ranking, and
decision-utility families remain fail-closed until the frozen `sciplex3-k562-24h-v1` benchmark contract (historical Item 8) binds
their missing target, held-out-domain, paired-history, candidate-set, and utility semantics.

Use policies cover every source and explicitly assess research training, commercial training,
benchmark evaluation, source redistribution, derived-model distribution, and publication. A policy
record makes restrictions machine-readable. Policy layers may overlap, and the most restrictive
applicable status wins. Scientific eligibility and legal permission remain separate; later workflows
must choose an intended use and reject `unknown` or `prohibited` permission.

Dataset manifests describe source evidence; they do not download, transform, impute, pair, or
interpret measurements. Normalization, split, and training-run manifests will be separate,
content-addressed contracts. Do not create a dataset manifest with placeholder checksums, guessed
licensing, or aspirational eligibility.

## Benchmark artifacts

`cellstate.data.BenchmarkArtifact` is an independent experimental boundary. It embeds one exact v2
query, maps each query target/action/horizon to exact claim, loss, and metric assessments, and
separates physical dataset identity from assessment views so one set of real records is split only
once. Split plans materialize record and protected experimental-unit membership; authoritative
evaluation cases bind every subject, static context, action or control, target, horizon, and
matching stratum.

Metric definitions bind an exact output schema, evaluation partitions, independent evaluation
unit, weighting, missingness, dependence blocks, and either a specification-only or executable
implementation. Specification-only metrics, uncertainty methods, and baselines may freeze a
component design but can never satisfy scientific admission. Baseline-relative gates consume
paired, block-aware candidate-minus-baseline results rather than comparing unrelated marginal
confidence intervals. The acceptance policy is an explicit `ALL`/`ANY` tree, and every declared
metric must be reported on every declared evaluation partition before admission.

Verification exposes assessment-and-permission, performance, and admission-ready states
separately. A structurally verified `COMPONENT_BENCHMARK` may pass exact source and permission
resolution while remaining non-admitted because implementations, leakage checks, baselines, or
performance thresholds have not passed. `verified` never means biologically validated.

The generated schemas capture structural JSON constraints. Several scientific invariants—including
cross-field status/value rules, fingerprint binding, interval coverage, selection order, and EVSI
arithmetic—require Pydantic validators and therefore **must** be revalidated through the Python
model. This applies to v2 runtime artifacts as well as experimental dataset manifests;
JSON-Schema-only acceptance is never a scientific, eligibility, or decision-validity judgment.
