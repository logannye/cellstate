# Add a model backend

A backend exists to compute a faithful representation of hidden cellular state for a declared query.
Everything below about contracts, admission, provenance, and verification is instrumental to that. A
backend that satisfies every contract in this guide and cannot be shown faithful has delivered
nothing.

Read [the roadmap](../roadmap.md) first. Its state-capability ledger `S1`-`S10` is the sole measure
of progress, and the query your backend serves must be able to satisfy `S1` — an admissible
pre-cutoff observation in the target modality, on a unit also observed after the cutoff — and `S3` —
at least two horizons after the cutoff — before any of the rest is worth building. Under a query
that satisfies neither, the predictive-sufficiency test is not merely unrun; it is inapplicable,
because the belief-only and belief-plus-history predictors receive the same inputs. Confirm this
before you write code, not after.

Before implementing any runtime protocol, decide whether the task actually estimates hidden state.
An endpoint-only mapping from static experimental context and assignment to a future destructive
assay distribution is a `PopulationAssayResponseModel` component, not a `CellStateEstimator` or
`StateEvolutionModel`. Bind such a component through the experimental biological support envelope,
complete stage-port map, frozen query and benchmark, and evidence-derived lifecycle. Do not create
an `EstimatorDescriptor`, current-state prior, or public operation merely to make the component fit
the four-operation API. See [ADR 0009](../adr/0009-population-response-component-boundary.md).

Bundle contract v0.1 keeps hashes, artifact URIs, and Python entry-point strings as declarations.
Execution requires the separate trusted admission context described in
[ADR 0010](../adr/0010-trusted-admission-verification.md); a serialized receipt is not authority.
Do not work around that boundary with a manually constructed biological descriptor.
`SYNTHETIC_TEST_MODEL` exists only for software contract tests and is never biological evidence or
an admission path.

## Prepare trusted admission

This section is machinery, not science. It establishes that the bytes you executed are the bytes you
declared, and nothing more; clearing all of it proves nothing about cellular state. Keep verifier
authority outside the submitted bundle and every serialized artifact. Provision
capability-scoped `TrustedAdmissionVerifier` objects with external HMAC keys, and provision
`TrustedRuntimeInterface` entries from an application-owned interface registry. Never put those
secrets beside code being loaded, copy them into receipts, or accept the report submitter's live
interface object as the contract.

Derive real-data execution sources through the typed science-and-permission workflow. Authenticate
that exact selection and include its resolution artifacts in byte coverage; do not infer execution
inputs from every review-only source in a manifest. Resolve every consumed declaration from its
exact URI and stream its actual bytes through the artifact verifier. Metadata, a matching filename,
or an HTTP checksum header is not a byte observation.

Load a declared implementation in an application-owned isolation boundary. Authenticate the
bounded loader observation, then have the receipt issuer compare it with the exact implementation
requirement and trusted interface registry. The loader and verifier roles may use separate trust
roots. Specification-only bindings, inherited protocol stubs, abstract classes, substituted code,
or merely similar signatures must fail.

Produce one canonical typed result manifest for every required validation-evidence binding. It must
cover the exact evidence role, partitions, authoritative cases, model and implementation scope,
ports or operations, required semantic criteria, and supporting result artifacts. Semantic
verification and scientific pass/fail are separate: a correctly parsed failed result is verified,
but it still blocks execution. Reuse the admission batch's authoritative byte receipts rather than
attaching a second claim about the same result bytes.

Finally, derive query prerequisites from the exact query, envelope, bundle, and target surface.
Pass the runtime-only `AdmissionVerificationContext` to the appropriate execution guard together
with a code-only provider. The provider reacquires the admitted bytes; it must not supply a loaded
object. The guard seals and verifies that stream once, then gives that exact immutable snapshot to
the registry-owned `TrustedJITLoader` bound to the authenticated loader identity and key. Invoke
only the object carried by the returned `VerifiedRuntimeHandle`; never treat the persisted receipt
or a previously imported object as authority.

## Implement the estimator

Implement `QueryCompiler` and `CellStateEstimator` together. Compilation must turn the exact
`StateQuery` into an auditable `CompiledStateSpecification`: activate only factors required by the
query, explicitly exclude the others, bind target and horizon names, and declare any context,
realized-perturbation `R`, and observation-nuisance `Xi` dimensions. Compiler identity and the
compiled specification travel with every belief and forecast.

Estimator capability preflight is scoped to the complete request, not merely a list of modality
names. It must assess the typed subject and aggregation, biological and static context, evidence
roles and collection effects, history completeness, intervention realization, action and
environment domains, output decoders, horizons, precision, and constraints. Return all blockers;
never advertise a scope as supported while silently ignoring a supplied field.

The posterior joint distribution is authoritative. Return marginals for every query-active factor
and every compiled context, `R`, and `Xi` block, and make those marginals agree with the joint where
the distribution family permits the comparison. Do not emit all eight factor families by habit:
the compiler's active/excluded partition defines the state for this query. Preserve observed,
inferred-with-support, weakly identified, interventionally unidentified, and unavailable semantics
without numeric sentinels.

Return diagnostic availability separately from scientific pass/fail. Support, predictive
sufficiency, identifiability, calibration, causal identification, and decision uncertainty must use
the query's declared thresholds. Derive independent prediction, control, and measurement-selection
readiness flags. Producing a distribution is not evidence that any readiness gate passed.

For a passing identified or transported population effect, emit typed causal estimands that exactly
cover the relevant query/objective target-horizon scope. Bind their aggregation, experimental unit,
action/environment contrast, comparator, and randomized or quasi-experimental design. Each cited
validation claim must carry a SHA-256 fingerprint matching the descriptor and result provenance.
Do not use ordinary conditioning-history events as a shortcut for causal identification; the local
path remains unsupported until a typed assignment/control/outcome graph is implemented.
For forecasts and candidate evaluations, bind every causal estimand to the exact scenario ID and
fingerprint and to its effective intervention contrast. A plan's selected causal report must be the
selected candidate's report, not a stronger claim copied from another branch.

If the backend supports controlled propagation, implement `StateEvolutionModel` with an exact
scenario capability scope. Preserve the joint posterior, propagate uncertainty and active
intervention/environment context, and attach causal and transport status to every target
prediction. Planned actions cannot claim retrospectively measured realization; integrate over a
declared realization model instead. A forecast must cover each query target exactly once at the
named horizon.

Planning backends implement `InterventionPlanner` over an exact objective and ordered candidate
set. Candidates remain inside the query's bounded dose, duration, schedule, delivery,
reversibility, combination, and environment domains. Score typed query-target predictions—not
private latent coordinates. Return a typed abstention when prediction, calibration, causal,
transport, constraint, or control-readiness evidence is insufficient.

Measurement backends implement `MeasurementPolicy` for the separate
`recommend_next_measurement` operation. Preflight support against the exact parent belief and
`MeasurementDecisionRequest`, including its objective, ordered intervention candidates, ordered
assays, collection and decision times, utility units, cost conversion, delay penalty, and every
collection-effect penalty. The result must evaluate every requested assay once and in order.
There must be at least two semantically distinct candidate regimes, and every numeric causal claim
must bind their exact ordered decision-set fingerprint; an intervention-type union is not an
adequate contrast. A transported result must also be permitted by the query and use the same source
and target domains as its causal-support report.

Do not equate posterior variance, entropy, or covariance reduction with value of information. A
supported numeric assay evaluation is intervention-decision EVSI and requires four independently
auditable capabilities:

1. a calibrated model for the distribution of possible assay outcomes;
2. a hypothetical posterior update conditional on each possible outcome;
3. counterfactual replanning over the same bounded intervention decision; and
4. a declared utility over the objective's query targets and horizon.

Gross EVSI is the expected post-measurement best utility minus the baseline best utility. Net value
then subtracts assay, delay, and destructiveness penalties in the declared utility units. Bind
one typed, exact-scope, content-addressed evidence trace for each of the four criteria into result
provenance. The assay-outcome trace is assay-specific through its containing assay evaluation.
Use `NOT_EVALUATED` only when the calculation was not performed, `UNSUPPORTED` when a required
component failed or lies outside support, and `ABSTAINED` when supported numeric values do not clear
the declared net-value threshold. Unavailable assay evaluations contain explicit reasons and empty
numeric fields. The contract reference returns `NOT_EVALUATED` and must not be presented as an EVSI
implementation.

Every returned artifact is revalidated at the public boundary and must preserve exact input,
subject, compiler, model, support, seed, event-content, and provenance identity. Sample-backed
posteriors use the public sample-axis convention. Matching dimension labels alone never establish
posterior compatibility.

## What the backend must be shown to do

Contract conformance is not evidence about biology. Before a biological backend may be called
usable, each of the following must be measured, not declared.

- **Splits at the independent experimental unit.** Well, plate, library, donor, animal, clone, cell
  line, or study — never randomly sampled cells — and frozen before model selection. A random
  cell-level split reports memorization as generalization and is prohibited for scientific claims.
- **Every mandatory baseline beaten individually.** The mandatory set is ledger entry `S9` in
  [the roadmap](../roadmap.md), plus the temporal state-space baseline where the query permits it —
  each on its own, on the frozen proper scores, with a bootstrap interval grouped at the split unit
  that excludes zero. Beating a pooled or averaged floor is not the test. Marginal error and
  all-gene correlation are maximized by predicting no change and never stand alone; a
  differential-expression-weighted and a rank-based metric accompany them.
- **Calibration.** Absolute marginal coverage error within the query's predeclared threshold at
  every declared level, reported as an upper confidence bound grouped at the split unit.
- **A predictive-sufficiency verdict.** The history-information gain between a predictor given the
  belief alone and one given the belief plus the raw history, with a bootstrap interval grouped at
  the split unit, on genuinely held-out future evidence. A gain reported without an interval is not
  a verdict. A verdict of insufficiency, reported with its interval, is a result; suppressing it is
  not.
- **Abstention that works.** On a deliberately out-of-support partition the abstention rate exceeds
  the in-support rate, and discarding the lowest-confidence decile does not increase held-out risk.

Two design constraints follow from a candidate family already retired here, recorded in
[ADR 0013](../adr/0013-state-first-roadmap-reordering.md). Actions must enter through features, not
through a free parameter per observed action: a per-action free parameter cannot generalize to an
unseen action in principle and forecloses external replication. And equal weighting of experimental
units is a property of the objective, not of the reporting — every term of the objective and of
every derivative carries the same unit normalization. A context parameter must be accompanied by a
measured effective-context diagnostic; a context dimension whose effective count collapses toward
one is unidentified, and that diagnostic belongs in the model's own gates.

Publish, alongside those results, the model card, content-addressed training and calibration
support, query-specific support envelope, external validation evidence, OOD behavior, and
state-versus-history sufficiency results. A measurement-capable backend additionally publishes
assay-outcome calibration, hypothetical-update validation, replanning validation, and utility/regret
evaluation for the supported decision scope. `ModelArtifactKind.CONTRACT_REFERENCE` artifacts may
exercise software contracts but cannot carry biological support evidence.
