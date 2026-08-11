# ADR 0012: Define a sci-Plex3 p1-only trained-candidate boundary

- **Status:** Accepted
- **Date:** 2026-08-10
- **Extends:** ADR 0009 population-response component boundary, ADR 0010 trusted admission
  verification, and ADR 0011 p1-only loader and baselines

## Context

Item 11 established an exact, close-reauthenticated `p1-train` count surface and fitted all six
mandatory baseline algorithms without opening `p2`, `p3`, or `p4`. The next lifecycle transition
needs one nonvacuous population-response candidate trained from those same `p1` bytes. A model file
cannot mint that transition by declaring itself trained: the training plan, selected source, count
stream, code, runtime, fitted behavior, sealed model bytes, and loaded candidate interface all need
independent verification.

Three earlier real-`p1` attempts or diagnostics failed closed without issuing a candidate artifact:

1. Candidate v1 was a rank-16 Gamma--Poisson factor model with a separate `q`/capture term and 16
   independently fitted factor-shape parameters. Variance moved between `q`, capture, and the free
   shapes until a shape reached the rejected numerical boundary.
2. Candidate v2 removed `q`/capture but estimated one pooled factor shape. The exact fit drove that
   shape through `0.1` and down to approximately `0.073524`, toward the `0.05` guard, while the
   outer fit was still drifting at its predeclared 50-pass limit. Solving the pooled-shape inner
   root exactly did not make the outer trajectory converge.
3. Candidate v3 was a nonissuing counterfactual that fixed the shape at exactly `0.1`. It still
   failed the predeclared relative-ELBO convergence and terminal factor-order stability gates at
   pass 50. Factor order changed 20 times, including a factor 7/11 crossing at pass 49, and the
   final order was stable for only two passes.

The v3 characterization separated a portability ambiguity from those scientific failures. The
equal-well activation matrix had raw smallest-to-largest singular-value ratio
`8.472584996122713e-7`, about 56.85 times the strict `1.4902161193847657e-8` rank gate. NumPy SVD
and both SciPy LAPACK drivers produced the same spectrum. The provisional 12-decimal ratio lay
within tolerance of a quantization half-boundary, but both adjacent quantized values passed the
gate. The candidate was therefore not rejected for scientifically low rank; it remained rejected
for nonconvergence and real factor-order movement.

The same characterization also rejected v3's parametric unseen-plate nuisance. The `p1` plate rows
implied a mean-one independent lognormal scale with `sigma_plate=5.66126548675`. Its median was
approximately `1.10e-7` and its upper tail carried the mean, making new-plate behavior tail-dominated
and scientifically unsuitable. This is not a numerical threshold to relax.

These are scientific and numerical design results, not operational incidents to bypass. Relaxing a
boundary, increasing a pass cap after seeing the trajectory, clipping a collapsed parameter, or
treating a diagnostic output as a trained model would create an unreviewed recovery path.

## Decision

Retain the p1-only trusted-training boundary and its incompatible v4 candidate implementation. The
audited exact-reference v4 execution failed closed during initial inner equilibration. The v4
candidate and artifact are therefore `NO-GO`; this decision records no successful fit, model
artifact, training evidence, or lifecycle transition.

### Candidate family v4

Candidate v4 is a continuous rank-16 Gamma--Poisson factor model with:

- no separate `q` or capture random variable;
- factor shape fixed at exactly `r_theta=0.1`, never estimated from `p1`;
- exact equal-well fitting on the ordered 2,000-feature raw-count panel;
- deterministic NNDSVD initialization and immutable tensor and behavior manifests;
- deterministic sampling from an externally authenticated model artifact; and
- no calibration result, held-out embedding, outcome lookup, or public-runtime authority.

Fixing `r_theta` removes the variance-allocation drift observed in v1 and v2. It does not by itself
establish convergence, as the v3 counterfactual demonstrated.

### Local inner equilibration

V4 adds deterministic row-local `phi`/`theta` fixed-point equilibration inside each canonical
sparse batch. Each row uses fixed rate `0.1 / m_well_factor + 1`, updates shape as
`0.1 + allocated_counts`, and must pass both shape and expected-log residual checks at tolerance
`1e-8` for two consecutive sweeps. At least two and at most 50 sweeps are allowed; there is no
damping or acceleration, and inner nonconvergence fails without a candidate.

This change addresses the mismatch between stale local posterior coordinates and the outer
loading/action updates. It does not weaken the outer 50-pass limit, `1e-7` relative-ELBO tolerance,
three-pass convergence streak, or terminal factor-order stability requirement.

### Empirical whole-row unseen-plate proposal

V4 removes the independent parametric lognormal plate draw. For one target-plate/seed pair it
selects one complete, observed 16-factor `rho` row uniformly from the eight `p1` plate contexts and
shares that row across requested nuclei and actions. Selecting a whole row preserves observed
cross-factor dependence. Factorwise arithmetic normalization gives mean one over the eight rows,
so every strictly positive context value is bounded above by eight.

That arithmetic bound did not validate the context scientifically. The failed run's provisional
maximum was `rho=7.999995552807402`; because each factor column sums to eight, this is near single-
plate concentration: at least one factor places `99.9999444101%` of its eight-context mass on one
plate and has an effective context count of approximately `1.0000011118`. This is a candidate-
family risk, not a fitted-model pathology, because no fit was accepted. The empirical whole-row
proposal must therefore be reexamined and must not be described as a valid unseen-plate model. It
is not a learned held-out plate embedding, a transport claim, or evidence about `p2`, `p3`, or `p4`.

### Calibration declaration only

The v4 specification predeclares a future one-dimensional p2 calibration grid

```text
tau_j = exp(j / 20),  j = -20, -19, ..., 6.
```

For a future authorized calibration only, the declaration maps the factor shape to
`0.1 / tau_j^2` and maps each plate context to
`rho[p,k]^tau_j / mean_q(rho[q,k]^tau_j)`, preserving whole-row selection and all other means,
weights, and support. The `tau=1` branch preserves the original `rho` bytes. This declaration does
not open or read `p2`, select a value, calibrate uncertainty, or grant authority.

### Exact training runtime

The audited v4 execution was restricted to the complete reference runtime identity:

- Linux `x86_64`;
- CPython `3.11.15`;
- NumPy `2.4.6`;
- SciPy `1.17.1`; and
- `scipy-openblas` `0.3.31.188.0`, with the frozen single-thread environment.

Another BLAS backend, an unknown BLAS version, a different thread policy, or code loaded from bytes
that differ from the repository binding fails before a model can be accepted.

### Audited v4 execution outcome

The first launch produced report
`4677fc8ef1a458bf3616abc507250572c2da7a8d53c1c8a7a03d4b097f3d4877`. Its audit hook rejected
h5py's nonpersistent `/dev/null` probe before fitting. That attempt is infrastructure-invalid,
contains no fit, and must not be interpreted as a scientific result.

The replacement launch was a valid exact-reference, p1-only, nonissuing execution. Report
`66e9debc1a402e7aa68cbc934f7c5f641529eea3187ec15606364c912af8faa8` passed its integrity and
resource gates and kept every provisional tensor finite. It failed initial, untraced inner
equilibration at the predeclared sweep limit of 50, with passing streak 0,
`Rshape=0.24714465227035654`, and `Relog=3.750385840630546`. The failure occurred before any outer
update or ELBO trace. The residuals are respectively 24,714,465 and 375,038,584 times the `1e-8`
tolerance. The provisional loading-rank ratio `0.101239623839`, activation-rank ratio
`0.001249342162`, and minimum contribution share `0.001237124212` describe initialization only;
they are not evidence that rescues the failed fit.

The execution issued no model, plan, observation, training evidence, or materialization and derived
no lifecycle state. It did not open `p2`, `p3`, or `p4`, inspect held-out cases or outcomes, run a
metric, or create biological or performance evidence.

Before any separately authorized real-source rerun, the next milestone is:

### Item 12.1 — local-map and plate-context characterization, nonissuing

Its one bounded, software-only first-failing-batch diagnostic must record per-sweep `Rshape` and
`Relog`, local ELBO, one- and two-step state distances and update cosine, the worst
row/factor/count aggregate, shape/rate/mass invariants, the 16-by-16 Jacobian spectral radius, exact
replay equality, state/allocation digests, and all 16 `rho` effective-context counts and maximum
shares.

An objective decrease routes to an implementation fix. An intact objective plus a two-cycle or
noncontractive map routes to a versioned v5 safeguarded local solver. Near-zero prior/context
behavior or effective context near one routes to v5 plate regularization or neutralization. This
decision does not prefreeze damping or shrinkage and does not authorize cap or tolerance
relaxation, an artifact, p2 access, or any biological or performance claim.

### Pre-fit and post-fit evidence

The pre-fit `CandidateTrainingPlan` is immutable and cannot name a future model hash. It binds the
opaque query, benchmark, and support-envelope identities; the exact typed `p1` role; loader and
count-stream closure; feature, action, target, design, specification, schema, trainer, factory, and
runtime identities; and deterministic fit settings. Computing this plan does not parse protected
partition descendants.

After fitting, a model would have to be closed, reread, rehashed, loaded through the exact
registered candidate class, and checked against its fitted-state and behavior manifests before any
training evidence could be issued. Deterministic files are evidence, never access authority.

Source selection and fit semantics are authenticated separately with capability-scoped HMAC
receipts held in a runtime-only `TrainingVerificationContext`. Secrets, live interface objects, and
signed receipt bytes are not deterministic bundle dependencies. Trusted verification must also
resolve the exact deterministic artifact set and verify the loaded candidate-factory interface
before deriving a lifecycle state.

### Lifecycle boundary

Only a future separately authorized, successful, and audited candidate fit may derive
`TRAINED_CANDIDATE`. At that state:

- the component bundle may declare exactly the population assay-response distribution port as
  provided by the candidate factory;
- the other seven query-required component ports remain required;
- calibration, model selection, validation, benchmark performance, scientific admission, and
  every public runtime operation remain false; and
- `sample_response` and all four public cell-state operations remain unavailable.

Artifact presence alone cannot derive even `TRAINED_CANDIDATE`. Without fresh trusted source, fit,
byte-resolution, query-prerequisite, and loaded-interface verification, readiness remains at
`SCAFFOLD`. The valid v4 run failed before artifact emission; no v4 model artifact exists and no
lifecycle transition has occurred.

### Held-out partitions remain sealed

Item 12 opens only `p1-train`. It does not read `p2`, `p3`, or `p4` membership, cases, or outcomes
during planning or fitting.

- `p2-calibration` requires a future one-use calibration grant freshly bound to the exact verified
  `TRAINED_CANDIDATE`.
- `p3-model-selection-validation` remains unavailable until an exact calibrated candidate exists.
- `p4-untouched-test` remains unavailable until an exact candidate is selected and frozen.

Neither this ADR nor the v4 calibration declaration authorizes those grants.

## Consequences

- V1, v2, and v3 leave no model artifact and no lifecycle residue. Their diagnostics remain
  reproducible evidence for rejecting those parameterizations or fitting schemes.
- V4 removes estimated factor-shape drift and attempts to equilibrate local posterior coordinates
  before outer updates, but its exact run did not equilibrate within the frozen limit. Its empirical
  context is arithmetically bounded yet near single-plate concentration and remains scientifically
  unresolved.
- V4 is incompatible with all v1, v2, and v3 candidate bytes, schemas, specifications, plans,
  observations, fixtures, and manifests; they must be rejected rather than migrated or relabeled.
- Exact BLAS binding narrows reproducibility claims but prevents materially different numerical
  kernels from borrowing the same runtime label.
- A future checked-in model could be a trained candidate without being calibrated, selected,
  evaluated, admitted, or publicly executable.
- Later calibration must prove that it changes only the declared `tau` state and cannot silently
  rewrite base weights, support, or the eventual accepted plate-context rule.

## Acceptance criteria

Item 12 is complete only when a final audit demonstrates that it:

- fits a separately authorized candidate version from the exact close-reauthenticated real `p1`
  stream under its frozen Linux/Python/NumPy/SciPy/BLAS runtime;
- passes the predeclared inner-equilibration, outer convergence, factor-order, identifiability, and
  finite-state gates without post hoc cap or tolerance changes;
- emits no artifact for an unconverged, nonfinite, stale, substituted, or behavior-inconsistent
  model;
- closes, rereads, rehashes, and exactly reloads the model and reproduces its frozen golden sample;
- binds the exact source, scan, assembly, design, feature, action, target, code, runtime, plan,
  training observation, and model identities;
- proves no `p2`, `p3`, or `p4` descendant was opened and all later authority flags remain false;
- rejects every incompatible earlier candidate artifact at the active candidate boundary;
- derives `TRAINED_CANDIDATE` only with fresh external trust and exact candidate-factory interface
  verification; and
- leaves calibration, selection, metrics, admission, component execution, and public runtime
  closed.

## Current implementation status

The p1-only training and verification contracts exist, and the v4 software implementation passed
its source-free audit. V1 failed at its shape boundary; v2 retained pooled-shape drift; v3 fixed the
shape but failed relative-ELBO and factor-order stability, while its characterization also rejected
the parametric lognormal plate nuisance. The valid exact-reference v4 run then failed initial inner
equilibration before any outer update or ELBO trace. V4 is `NO-GO`: there is no v4 training plan,
observation, fitted model, training evidence, materialization, or trusted lifecycle result. Item 12
remains in progress, and the component remains a non-runnable `SCAFFOLD`. The next authorized work
is only the bounded first-failing-batch inner-map diagnostic described above; another real-source
run requires a separate audit authorization.
