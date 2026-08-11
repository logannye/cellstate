# Architecture overview

The framework uses a functional core and replaceable model backends.

- `domain/` owns strict, deeply immutable, schema-versioned boundary objects. Nested mappings and
  JSON lists reject in-place mutation; serialized values are still revalidated at process/storage
  boundaries.
- `engine/` owns deterministic history/registry mechanics and no biology.
- `ports/` defines estimation, transition, planning, observation, fusion, constraint, diagnostic,
  inheritance, and measurement-policy interfaces.
- `backends/` contains experimental content-addressed biological-bundle admission contracts and
  fail-closed biological component scaffolds; its presence never implies runtime admission.
- `reference/` contains an opt-in linear-Gaussian vertical slice for contract testing.
- `evaluation/` contains backend-independent calibration and sufficiency diagnostics.
- `training/` names the composable intervention-focused training objective.

The four top-level operations are deliberately distinct:

```python
belief = estimate_cell_state(request, estimator=model)
forecast = evolve_cell_state(belief, scenario=scenario, evolution_model=model)
plan = choose_intervention(belief, objective=objective, candidates=candidates, planner=planner)
measurement = recommend_next_measurement(
    belief,
    request=measurement_request,
    policy=measurement_policy,
)
```

Estimation first invokes the backend's query compiler and embeds the resulting
`CompiledStateSpecification` in the belief. The specification records the typed subject, active and
excluded state factors, target/horizon contract, evidence roles, acceptance thresholds, and any
context, perturbation-realization `R`, or nuisance `Xi` dimensions. Capability preflights are
fingerprinted to the exact estimation request, scenario, intervention-planning problem, or
measurement-decision request; a query-wide Boolean is not a support envelope.

Every stochastic operation receives an explicit seed. Large modality arrays and posterior samples
are referenced through content-addressed artifacts rather than embedded in manifests.

`StateForecast` contains both the propagated joint latent posterior and typed predictive
distributions for every query target at one named query horizon. A backend must return an explicit
unavailable prediction when it lacks an output decoder. Planning objectives name that horizon and
query targets, and planners evaluate those output distributions—not backend-private latent
coordinates.

Measurement selection is not an optional annotation embedded in `CellStateBelief`. A
`MeasurementDecisionRequest` binds the parent belief to one intervention objective, at least two
semantically distinct ordered candidate regimes, candidate assays, collection time, decision
deadline, utility scale, and
all assay, delay, and collection penalties. The separate `MeasurementRecommendation` evaluates
each assay exactly once and either selects one, abstains below threshold, reports `NOT_EVALUATED`
when calculation was not performed, or reports `UNSUPPORTED` for a failed/out-of-support decision.
Unavailable outcomes never use numeric sentinels.

A supported numeric assay value is intervention-decision EVSI. Computing it requires all four of:

1. a calibrated predictive distribution for each possible assay outcome;
2. a valid hypothetical belief update conditional on each outcome;
3. counterfactual replanning over the same bounded intervention candidates; and
4. a declared decision utility on query targets at the objective horizon.

Numeric EVSI is tied to the exact ordered decision-set fingerprint rather than a loose union of
intervention labels. Thus different doses, schedules, and environment trajectories remain distinct,
and a one-option or duplicated decision set cannot manufacture value. Transported EVSI additionally
requires query permission and matching causal and transport domains.

Each assay evaluation stores one content-addressed evidence trace for each of those four criteria,
with the exact request-scope fingerprint and a typed outcome. These traces are model-validation
claims; a dataset eligibility record may identify suitable evidence but cannot itself claim that a
trained model passed.

Generic entropy, variance, or covariance reduction may be reported as a separately named diagnostic
only when supported. It is not a substitute for EVSI and cannot justify an assay recommendation.

Recursive estimation, evolution, intervention planning, and measurement decisions require
compatible model ID/version, complete model configuration fingerprint, posterior-schema identifier,
and training-support identity. Matching dimension labels alone do not establish compatible latent
semantics.

Public API boundaries revalidate backend results and bind them to their request, parent belief,
query, scenario, objective, candidates, candidate assays, seed, and backend descriptor. Third-party
adapters cannot silently return a valid-looking artifact for a different computation.

Calculation availability and scientific validity are independent. Beliefs and forecasts expose
support, sufficiency, identifiability, calibration, causal evidence, decision uncertainty, and
derived prediction/control/measurement readiness. A valid artifact with unsupported or unevaluated
scientific gates returns with a typed abstention and auditable reasons; malformed contracts and
unsupported computational scopes still fail closed. The synthetic reference artifact never
constitutes a biological support claim.

The reference measurement policy has no calibrated assay-outcome model, hypothetical update,
counterfactual replanning model, or decision utility. It therefore returns a complete
`NOT_EVALUATED` recommendation with explicit reasons and no numeric value fields. It never labels
generic covariance reduction as assay value of information.

Direct endpoint response is not automatically a fifth cell-state operation. The first sci-Plex3
task maps static K562 well context and intended compound-dose assignment directly to a future
recovered-nucleus RNA distribution. Because it has no pre-cutoff molecular state observation, it
implements the separate `PopulationAssayResponseModel` component protocol and may not return a
`CellStateBelief`. Its biological-bundle contract classifies every proposed model stage, binds exact
artifacts, and derives a lifecycle from training through locked evaluation. The checked-in scaffold
has no weights and rejects execution; even a future admitted component would authorize only that
exact assay-response surface, not estimation, evolution, planning, or measurement selection.

The reference backend requires complete past intervention, environment, lineage, and
neighborhood/contact records. It rejects input features its synthetic likelihood/dynamics do not
consume, including population or lineage context, assay confounder metadata, regional environment,
discrete division/contact events, and target-specific controls. Ongoing interventions and the current
environment are carried in the belief context; every scenario explicitly chooses whether they
persist. A production backend may support additional features only by advertising and implementing
their semantics. The reference prior has a configured time origin; record presence never determines
the prior epoch.
