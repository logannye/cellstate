# sci-Plex3 K562 population assay-response component boundary

## Status

The checked-in component is a non-runnable `SCAFFOLD`, not a biological model and not one of the
four public cell-state operations. `SciPlex3K562PopulationAssayResponseScaffold` implements the
separate typed `PopulationAssayResponseModel` component port only to resolve exact artifacts,
preflight one frozen task, and reject execution. It has no accepted model weights, training run,
calibration evidence, model-selection freeze, locked evaluation, or admitted benchmark
performance.

Item 11 adds a separate, non-public software path for this frozen scope. Its immutable H5AD loader
can open only `p1-train`, authenticates the exact source and `p1` closure, and yields sparse raw-count
batches on the ordered 2,000-feature panel. The loader refuses `p2`, `p3`, and `p4` before source
access; future access requires grants bound to the preceding lifecycle state. Six probabilistic
baseline algorithms fit from `p1` only. The exact source scan and all six fitted-state identities
are recorded, but no predictions, metrics, baseline comparisons, or performance results exist; the
software path does not make this scaffold executable.

Item 12 defines the next, still non-public lifecycle boundary. Its original rank-16
Gamma--Poisson candidate used a separate `q`/capture term and 16 independently fitted factor
shapes. The exact real-`p1` fit failed closed at the accepted Gamma-shape boundary before a model
artifact was emitted. V2 removed `q`/capture, but its free pooled factor shape fell from `0.1` to
approximately `0.073524` and its outer fit remained unconverged at pass 50. V3 fixed the shape at
`0.1`, but its nonissuing counterfactual still failed the relative-ELBO and terminal factor-order
gates. A follow-up characterization cleared the activation-rank gate and rejected the independent
lognormal unseen-plate nuisance as tail-dominated at `sigma_plate=5.66126548675`.

The audited v4 software design keeps the fixed shape, attempts to equilibrate row-local posterior
coordinates within every canonical batch, and proposes replacing that parametric tail with uniform
selection of one complete observed `p1` `rho` row. Its valid exact-reference, p1-only, nonissuing run
failed initial inner equilibration before an outer update or ELBO trace. No candidate artifact
exists. V4 is `NO-GO`, so this page makes no trained-candidate or lifecycle claim.

Its only eventual computation is:

```text
P(raw nonnegative integer UMI vector on the exact ordered 2,000-feature panel
  in recovered nuclei at 24 hours
  | static K562 source-well/plate context, intended compound-dose assignment or no-action control)
```

Static context is not a prior over hidden state. The destructive 24-hour RNA assay and matched
vehicle wells are future target/comparator observations, never pre-cutoff evidence. The forecast
label is `predictive_association`, intervention realization remains `unknown`, and the component
cannot claim viability, survival, target engagement, individual-cell dynamics, transport, or a
current-state belief.

## Item 12 trained-candidate boundary

Candidate v4 is a continuous Gamma--Poisson model with exactly 16 factors, exact equal-well
`p1-train` fitting, immutable tensor and behavior manifests, and raw-count sampling on the frozen
ordered 2,000-feature panel. It has no `q`/capture term, estimated factor shape, calibration result,
held-out plate embedding, outcome lookup, or public-runtime authority. Its factor shape is fixed at
exactly `r_theta=0.1` and never estimated from `p1`.

Within each canonical sparse batch, deterministic row-local `phi`/`theta` fixed-point updates must
equilibrate before their sufficient statistics can reach an outer update. Shape and expected-log
residuals must pass `1e-8` for two consecutive sweeps, with a minimum of two and maximum of 50
sweeps. Inner nonconvergence fails without a candidate; the outer 50-pass limit, `1e-7`
relative-ELBO tolerance, three-pass convergence streak, and factor-order gate are unchanged.

For an unseen target plate and seed, v4 proposes uniformly selecting one complete 16-factor `rho`
row from the eight observed `p1` plate contexts and sharing it across nuclei and actions. Whole-row
selection preserves the observed cross-factor dependence. Factorwise mean-one normalization makes
every positive context value at most eight, but the failed run's provisional maximum was
`7.999995552807402`. Because each factor column sums to eight, this is near single-plate
concentration: at least one factor places `99.9999444101%` of its eight-context mass on one plate
and has effective context count approximately `1.0000011118`. This is a candidate-family risk, not
a fitted-model pathology, because no fit was accepted. The whole-row proposal must be reexamined and
is not a validated unseen-plate model; v4 makes no parametric-tail or held-out transport claim.

The specification also predeclares a future p2-only calibration grid
`tau_j=exp(j/20)`, `j=-20,...,6`. It would map shape to `0.1/tau_j^2` and map each context to
factorwise-renormalized `rho^tau_j` while retaining whole-row selection. This declaration has not
opened `p2`, selected `tau`, or produced calibration evidence.

The reference fit is bound to single-thread Linux `x86_64`, CPython 3.11.15, NumPy 2.4.6, SciPy
1.17.1, and `scipy-openblas` 0.3.31.188.0. A conforming immutable pre-fit plan names only the exact
typed `p1` role and binds the count stream, model specification, output schema, trainer and
candidate-factory code, runtime, query, benchmark, and support identities. After fitting, the
workflow must close, reread, rehash, exactly reload, and semantically verify the model before
issuing training evidence.

The first v4 launch produced report
`4677fc8ef1a458bf3616abc507250572c2da7a8d53c1c8a7a03d4b097f3d4877`, but its audit hook
rejected h5py's nonpersistent `/dev/null` probe before fitting. It is infrastructure-invalid,
contains no fit, and must not be interpreted as science. The replacement exact-reference, p1-only,
nonissuing run produced report
`66e9debc1a402e7aa68cbc934f7c5f641529eea3187ec15606364c912af8faa8`. It passed integrity and
resource gates with every provisional tensor finite, then failed initial, untraced inner
equilibration at sweep 50, passing streak 0, `Rshape=0.24714465227035654`, and
`Relog=3.750385840630546`. No outer update or ELBO trace occurred. The residuals are respectively
24,714,465 and 375,038,584 times the `1e-8` tolerance. The provisional loading-rank ratio
`0.101239623839`, activation-rank ratio `0.001249342162`, and minimum contribution share
`0.001237124212` characterize initialization only and are not rescue evidence.

Deterministic plan, training, and model files are evidence, not execution authority. Source
selection and fit semantics are independently HMAC-authenticated in a runtime-only verification
context, and the candidate must be loaded through the application-owned exact interface registry.
Only that complete fresh verification may derive `TRAINED_CANDIDATE`. If exact model and training
artifacts exist, a static component bundle may declare `COMPONENT_MODEL` and mark only the
population assay-response distribution port `PROVIDED`; all seven other query-required ports stay
`REQUIRED`. That declaration alone leaves trusted readiness at `SCAFFOLD`.

Neither a model file nor `TRAINED_CANDIDATE` supplies calibration, model selection, benchmark
performance, scientific admission, component execution, or any public cell-state operation.
`sample_response` remains unavailable.

The v4 execution issued no plan, model, observation, training evidence, or materialization. None of
those conditional declarations apply, trusted readiness remains `SCAFFOLD`, and Item 12 remains in
progress.

## Frozen component and historical diagnostic identities

| Artifact | SHA-256 |
| --- | --- |
| Reviewed dataset manifest | `6248e63237a4c0c7ae53538666a1294cf1108569792eb54702ec15f439d9cb31` |
| StateQuery | `d0fa67f31a8ea1d7b2e8839dfe7629fd6f359ea7eed4f6d336e2cd1d8813971e` |
| Benchmark artifact | `97bfb8f00f9efd93ad19635ce1a843a126c3c1b23ae6002102353c5e3bded76e` |
| Component support envelope | `17aa440c14b40981f97358119085f44b2ffeb9bed75ba322114ffe2c1c53dd9f` |
| Component bundle contract | `69eddd15eb87b167ea0ef484d54234f6af5ebd6b353ad4aa72a96c2dca3f6343` |
| Item 11 p1 loader contract | `3de5be54b60ba1403995ba79d122ee8232218be5c027da1bf530cb610ae80f90` |
| Item 11 baseline golden fixture | `59fd7410df297ce8a63e37068fc7d5727ebd12268526a5f43be6bded553dde49` |
| Item 11 real p1 materialization | `7dd28d3ddca5d09d81779bfc3e02ec15d09428be354f6972e1ceda20ee1dd0e6` |
| Historical v2 candidate implementation source | `87e08b4d65596b9a1e2234d2db234293fdf1392ad443d2611283a6911cbcb3c0` |
| Historical v2 candidate runner source | `0c963dd035577567f28f11cd62727141a1c7bc627dc1326a3278591633953344` |
| Historical v2 candidate specification | `7bd027ee95a238c039d35f1aa5547d48158b2515b19da3372ebede41a24ee670` |
| Historical v2 output-model schema | `8ce0511161df45ea434aeda3292a534cae4668d80d2ba82a84dd527db76911ff` |
| Historical v2 synthetic golden model | `2ab05dc29bcad67aaa60640b8c6b3090127023fdcea745f2bba31f84c44ad64f` |
| Historical v2 synthetic golden sample | `26b601ce6779cb5bdca9337ed1f6eaeb41bd3e10c728ae1d111c15ba1bca8e01` |
| Infrastructure-invalid v4 no-fit report | `4677fc8ef1a458bf3616abc507250572c2da7a8d53c1c8a7a03d4b097f3d4877` |
| Valid exact-reference v4 nonissuing report | `66e9debc1a402e7aa68cbc934f7c5f641529eea3187ec15606364c912af8faa8` |

The v2 rows identify the software used for a rejected diagnostic trajectory. They are incompatible
with v4 and do not identify a real-p1 fitted model, training plan, current candidate family, or
trusted lifecycle result. The v4 report rows identify unsuccessful execution records, not a model,
training evidence, or trusted lifecycle result.

There is deliberately no constructible or exported response/provenance model yet. A future admitted
response contract must resolve and inspect its payload bytes—not trust `ArtifactRef` metadata—and
verify shape `(sample_count, 2000)`, integer dtype, nonnegative raw-UMI values, nonzero totals,
ordered-feature identity
`8b9e0e71d9bfc79a5e1db29f73bc30006bf117a29db8abf63b1607403629401f`, and target-value-schema
identity `b2463271246eca932824ad4d0089aaf3c924afcedec865dec8e04c4bbf7b23e2`. Their provenance must
bind the exact bundle, support envelope, query, benchmark, training run, model artifact, validation
evidence, evaluation case, task fingerprint, and seed. Until that byte-verifying boundary exists,
`sample_response` is typed `Never` and always raises.

## Partition access

Partition identity and access purpose are one-to-one:

| Purpose | Exact partition | Permitted lifecycle use |
| --- | --- | --- |
| `TRAIN_PARAMETERS` | `p1-train` | Fit candidate model parameters |
| `FIT_CALIBRATION` | `p2-calibration` | Fit/freeze uncertainty calibration only |
| `MODEL_SELECTION` | `p3-model-selection-validation` | Select and freeze a candidate |
| `UNTOUCHED_EVALUATION` | `p4-untouched-test` | Evaluate the already frozen component only |

Opening `p2`, `p3`, or `p4` for parameter training fails. In particular, `p4` cannot tune weights,
calibration, thresholds, feature order, baselines, or hyperparameters.

The Item 11 loader is narrower still: its current session type opens `p1-train` only. `p2` requires
a future trusted grant bound to an exact `TRAINED_CANDIDATE`; `p3` requires one bound to an exact
`CALIBRATED_CANDIDATE`; and `p4` requires a locked-evaluation grant bound to an exact
`MODEL_SELECTED_FROZEN` candidate. The `p1` session does not parse held-out outcome or membership
ledgers. Item 12 retains the same seal: candidate planning, fitting, model reload, and training
verification do not read `p2`, `p3`, or `p4` membership, cases, or outcomes.

## Mandatory baseline software

The six frozen-scope implementations are:

1. matched-vehicle resampling;
2. exact-condition replicate-1 empirical resampling;
3. exact-condition negative binomial;
4. hierarchical well negative binomial;
5. low-rank compound-dose response; and
6. nearest-supported-dose resampling.

All six implement the no-action target. Matched-vehicle pools use only same-plate vehicle wells from
`p1`; held-out controls are scoring comparators, not fit data. Nearest-supported-dose resampling
excludes the exact requested dose, minimizes absolute log10-dose distance, and resolves a tie toward
the lower dose. The fixed prediction campaign requests 512 raw-count samples for each case and each
seed, uses seeds `0`, `1`, `2`, `3`, and `4`, and instantiates NumPy `PCG64DXSM` explicitly.

Fitted-state manifests bind implementation, feature order, random-number contract, statistical
semantics, and the exact arrays or empirical pools. Runner scaffolding content-addresses fitted
state before held-out access and writes prediction shards incrementally so a complete campaign need
not be materialized as one dense array. Item 11 recorded the exact close-reauthenticated `p1` scan
and six software-only fitted-state identities. These are reproducibility mechanisms, not evidence
that a prediction campaign, metric, acceptance comparison, or benchmark performance gate passed.

## Fail-closed preflight

The gate re-reads and hashes all five frozen artifacts on every preflight. It rejects:

- query, bundle, support-envelope, manifest, or benchmark byte drift;
- unknown or modified well cases, plate contexts, action/dose assignments, targets, or horizons;
- partition/access-purpose mismatches and untouched-test mutation;
- future endpoint RNA supplied before the inference cutoff;
- environment or transport requests;
- point-only output in place of a predictive distribution;
- identified/transported causal labels, known realization, viability/survival, or hidden-state
  interpretations; and
- missing weights, calibration/model-selection/locked-test evidence, executable metrics and
  uncertainty, completed and passed baseline runs, source-duplicate audit, or scientific
  admission.

Even a request with exact scope receives readiness blockers and `sample_response` raises before
reading or returning biological values. The component exposes no `estimate`, `evolve`, planning,
measurement, or condition-response lookup entry point.

Regenerate and verify the two canonical component artifacts with:

```bash
PYTHONPATH=src uv run --no-sync python scripts/build_sciplex3_k562_component_scaffold.py --check
```

That command checks the historical scaffold artifacts. The trained-candidate builder remains
fail-closed until exact candidate plan, model, observation, scan, assembly, and materialization
artifacts exist at their canonical paths. After those inputs pass their independent fit and
artifact audit, the currentness boundary is:

```bash
PYTHONPATH=src uv run --no-sync python scripts/build_sciplex3_k562_trained_candidate.py --check
```

Bundle contract v0.1 now has its trusted artifact, loaded-interface, result-semantic, and
query-prerequisite verifier boundary. Item 11 supplies the `p1`-only loader and mandatory baseline
algorithms, plus content-addressed streaming-run scaffolding. Its exact scan of source SHA-256
`603ed16c5e25401c8a7f5bb0b2b045179701017d65dcfc6aeea71722a66cd10a` covered 94,785 `p1` records
across 768 wells, retained seven zero-panel records, and emitted six fitted-state identities without
opening a held-out partition. The later sequence remains a candidate distribution model, p2
calibration, p3 freeze, and one locked p4 evaluation. V1 and v2 failed their real fits; the v3
counterfactual and characterization were deliberately nonissuing. V4's audited exact p1 execution
also failed closed, and `TRAINED_CANDIDATE` cannot be derived.

### Item 12.1 — local-map and plate-context characterization, nonissuing

This next milestone's one bounded, software-only first-failing-batch diagnostic must capture per-
sweep `Rshape`/`Relog`, local ELBO, one- and two-step state distances and update cosine, the worst
row/factor/count aggregate, shape/rate/mass invariants, the 16-by-16 Jacobian spectral radius, exact
replay equality, state/allocation digests, and all 16 `rho` effective-context counts and maximum
shares. Objective decrease routes to an implementation fix; an intact objective plus a two-cycle or
noncontractive map routes to a versioned v5 safeguarded local solver; near-zero prior/context
behavior or effective context near one routes to v5 plate regularization or neutralization. It does
not prefreeze damping or shrinkage and does not authorize a real-source rerun, cap or tolerance
relaxation, an artifact, p2 access, or any biological or performance claim.

Passing a later complete component lifecycle could authorize only this direct component surface;
it would not authorize a hidden-state estimator.
