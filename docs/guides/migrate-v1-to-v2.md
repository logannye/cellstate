# Migrate schema v1 inputs to v2

Schema v2 is a breaking scientific-contract migration. It adds typed belief subjects, explicit
source-to-target evidence, collection effects, bounded action and environment domains,
query-compiled state, perturbation-realization and nuisance beliefs, and scientific readiness and
causal-support results.

The migration is intentionally not a field-renaming exercise.

## What remains stable

Checked-in files under `schemas/v1/` are an immutable record of the former wire format. They may be
used to validate or inventory legacy artifacts. A v1 payload is never accepted by the v2 public API
through coercion or a changed default.

## Inputs require explicit reconstruction

Use `cellstate.schema.inspect_v1_payload` to obtain an auditable list of missing decisions. Then
construct a new v2 query or history from the source experiment and reviewed annotations. At minimum:

- choose the belief subject independently of assay row granularity;
- declare source-to-target evidence relationships and collection effects;
- bind output aggregation and experimental units;
- provide bounded interventions, environments, schedules, and assay semantics; and
- set query-scoped support, calibration, sufficiency, identifiability, and uncertainty thresholds.

No migration tool will assume that destructive cells form a longitudinal individual, that missing
history means no event, or that any numeric dose with matching units is supported.

Intervention reversibility must use `ReversibilityStatus`: `REVERSIBLE`, `IRREVERSIBLE`, or
`UNKNOWN`. A washout interval records the administration schedule; it does not prove reversal.
Boolean `reversible` fields and boolean action-domain values are rejected.

Each available assay must declare one or both explicit purposes: `TARGET_ENDPOINT` and
`MEASUREMENT_SELECTION`. A fixed target-only endpoint omits cost, cost units, and turnaround, and a
query with no measurement-selection assay omits the corresponding budget fields. A selectable
assay requires all of those economics, and only assays declaring `MEASUREMENT_SELECTION` may appear
in a `MeasurementDecisionRequest`.

Every `OutputSpec` also binds a distinct `value_schema_reference`. Use it for the exact scalar,
vector, tensor, category, feature-order, and support convention of the returned value; an assay
protocol, latent model, or transformation reference does not substitute for this value schema.

## Produced artifacts require re-estimation

V1 beliefs, forecasts, and intervention plans cannot be upgraded. Their serialized values do not
contain the active-state compilation, intervention-realization posterior, nuisance state, causal
status, or readiness results required by v2. Rebuild the v2 query and history, re-estimate the
belief, and regenerate downstream forecasts and decisions.

V1 has no standalone measurement-decision request or recommendation contract. Any legacy embedded
“next measurement” hint is not a v2 recommendation and must not be copied forward. To evaluate an
assay under v2, construct a `MeasurementDecisionRequest` that explicitly binds the re-estimated
belief to an intervention objective, ordered candidate scenarios, candidate assays, collection and
decision times, utility units, cost conversion, and delay and collection-effect penalties. Then run
`recommend_next_measurement` with a measurement policy.

A migrated backend may return a supported numeric measurement value only if it implements a
calibrated assay-outcome model, hypothetical belief updates, counterfactual replanning, and declared
decision utility. Otherwise return `NOT_EVALUATED` when the calculation was not performed or
`UNSUPPORTED` when a required component failed or lies outside support. Reserve `ABSTAINED` for
supported numeric values below threshold. None may use zero EVSI or a covariance-reduction proxy as
a sentinel.

Posterior dimension labels or matching identifiers are not proof of compatible semantics.

## Fingerprints

Every migrated input receives a new v2 fingerprint. Recursive updates reject v1 previous beliefs.
Model, compiler, posterior-schema, support, and calibration identities must match before a v2 belief
can be propagated or used for decisions.
