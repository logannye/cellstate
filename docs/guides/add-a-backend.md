# Add a model backend

Before implementing any runtime protocol, decide whether the task actually estimates hidden state.
An endpoint-only mapping from static experimental context and assignment to a future destructive
assay distribution is a `PopulationAssayResponseModel` component, not a `CellStateEstimator` or
`StateEvolutionModel`. Bind such a component through the experimental biological support envelope,
complete stage-port map, frozen query and benchmark, and evidence-derived lifecycle. Do not create
an `EstimatorDescriptor`, current-state prior, or public operation merely to make the component fit
the four-operation API. See [ADR 0009](../adr/0009-population-response-component-boundary.md).

Bundle contract v0.1 is declaration-only and intentionally non-executable. It cannot trust a hash,
artifact URI, or Python entry-point string as proof that bytes were resolved or an interface was
loaded and validated. Every v0.1 bundle remains `SCAFFOLD`; do not work around that boundary with a
manually constructed biological descriptor. `SYNTHETIC_TEST_MODEL` exists only for software
contract tests and is never biological evidence or an admission path.

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

Before calling a biological backend usable, publish its model card, content-addressed training and
calibration support, query-specific support envelope, external validation evidence, OOD behavior,
and state-versus-history sufficiency results. A measurement-capable backend additionally publishes
assay-outcome calibration, hypothetical-update validation, replanning validation, and utility/regret
evaluation for the supported decision scope. `ModelArtifactKind.CONTRACT_REFERENCE` artifacts may
exercise software contracts but cannot carry biological support evidence.
