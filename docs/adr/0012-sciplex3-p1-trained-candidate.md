# ADR 0012: Define a sci-Plex3 p1-only trained-candidate boundary

- **Status:** Accepted; amended 2026-08-11 to retire v4, complete source-free Item 12.2,
  and assemble the source-free Item 12.3 execution-control design
- **Date:** 2026-08-10
- **Extends:** ADR 0009 population-response component boundary, ADR 0010 trusted admission
  verification, and ADR 0011 p1-only loader and baselines

## Context

Item 11 established an exact, close-reauthenticated `p1-train` count surface and fitted all six
mandatory baseline algorithms without reading or decoding protected `p2`, `p3`, or `p4`
expression/raw-count values or endpoints and without scoring authority. Public frozen split
membership and outcome-free case design were already checked in. The next lifecycle transition
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

Retain the p1-only trusted-training boundary. Preserve the incompatible v4 implementation only as
historical source and audit lineage. The audited exact-reference v4 execution failed closed during
initial inner equilibration, and a later source-code audit confirmed an objective-scale inconsistency
before the planned Item 12.1b replay. The v4 candidate and artifact are therefore retired and
`NO-GO`; this decision records no successful fit, model artifact, training evidence, or lifecycle
transition.

### 2026-08-11 amendment: v4 dose-objective inconsistency

The tracked full ELBO gives every cell in well `w` weight `94785/(768*n_w)`. Summing the
mean-dependent Gamma-prior term over that well therefore multiplies it by `94785/768`, while the
global magnitude and second-difference dose penalty is subtracted once. V4's dose Newton block
instead minimizes the unweighted per-well Gamma term plus that same global penalty. Its gradient and
Hessian omit the `94785/768` multiplier, placing the dose penalty on the wrong relative scale and
breaking the claimed correspondence between the action update and tracked objective.

This is a source-level implementation and objective-design defect, not a result inferred from a
held-out partition. It does not reinterpret the historical v4 run, which failed before an outer
update; it independently establishes that v4 must not be revived or characterized with another
real-source run. A v5 design must reconcile the full objective and M-step before any new
authorization.

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
not read or decode `p2` calibration values, select a value, calibrate uncertainty, or grant
authority.

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
post-hoc resource acceptance checks and kept every provisional tensor finite. It failed initial,
untraced inner
equilibration at the predeclared sweep limit of 50, with passing streak 0,
`Rshape=0.24714465227035654`, and `Relog=3.750385840630546`. The failure occurred before any outer
update or ELBO trace. The residuals are respectively 24,714,465 and 375,038,584 times the `1e-8`
tolerance. The provisional loading-rank ratio `0.101239623839`, activation-rank ratio
`0.001249342162`, and minimum contribution share `0.001237124212` describe initialization only;
they are not evidence that rescues the failed fit.

The execution issued no model, plan, observation, training evidence, or materialization and derived
no lifecycle state. It did not read or decode protected `p2`, `p3`, or `p4` expression/raw-count
values or endpoints, inspect held-out outcomes, run a metric, or create biological or performance
evidence.

Before any separately authorized real-source replay, the software milestone is now complete:

### Item 12.1a — local-map and plate-context characterization software freeze

The exact historical harness (`f4e6b768...`), tests (`8989618e...`), parent driver (`795c5929...`),
and both reports (`4677fc8e...` and `66e9debc...`) are canonical under
[`audits/item12_1a`](https://github.com/logannye/cellstate/tree/main/audits/item12_1a). The 26 focused
tests, Ruff check/format, compilation,
and independent SHA-bound audit reported no P0/P1 finding within the bounded local-map and
containment scope; they did not validate the outer dose objective. The frozen contract records 50
production maps and exactly one diagnostic `A51` lookahead;
per-sweep `Rshape` and `Relog`; synchronized local objectives; one- and two-step state distances and
update cosine; deterministic worst row/factor/count aggregates; shape/rate/mass invariants;
per-sweep analytic 16-by-16 Jacobian spectral radii; exact replay equality; state/allocation
digests; and all 16 bounded `rho` effective-context counts and maximum shares.

An objective decrease routes to an implementation fix. An intact objective plus a two-cycle or
noncontractive map routes to a versioned v5 safeguarded local solver. Near-zero prior/context
behavior or effective context near one routes to v5 plate regularization or neutralization. This
decision does not prefreeze damping or shrinkage and does not authorize cap or tolerance
relaxation, an artifact, p2 access, or any biological or performance claim.

### Item 12.1b — retired before execution

The planned one-use real-`p1` v4 characterization is no longer pending and must not run. Item 12.1a
opened no source and did not run Docker, and no Item 12.1b report exists. Once the dose-objective
inconsistency was known, another v4 source execution could not establish a candidate worth advancing
and would spend authorization on a retired objective.

### Item 12.2 — source-free v5 objective, M-step, sampling, publication, and containment

This source-free milestone is complete. V5 defines one exact fixed-`q` action/context contribution
to the equal-well full ELBO,

```text
Q = -(94785/768) * 0.1 * sum_wk(eta_wk + t_wk * exp(-eta_wk)) - P(delta),
```

and applies it over all 768 wells in the compatible `alpha`, arithmetic-mean-one constrained
`log-rho`, and `delta` dose-effect M-step. A separately implemented scalar objective, feasible-
coordinate finite differences, dose and joint arrowhead Hessian checks, a treated-well adversarial
fixture, strict one-ULP decrease rejection, and accepted-substep and complete-block checks all
verify nondecrease of that same canonical objective.

V5 exactly conditions the whole raw-count panel to be positive through the zero-truncated
superposition of the Gamma--Poisson model's compound-Poisson/log-series factor counts. An exact
`CandidateSampleRequest` caps one request at 512 draws; target-only support fails. One global
certificate covers all 753 actions (752 compound-dose actions plus no action), the one neutral unit
unseen-plate context, and all 27 declared calibration `tau` values. For each complete request, its
conservative conditional signed-`int64` Chernoff tail bound is at most `2^-64`; compound-Poisson
intensity, log-series RNG support, allocation overflow, and positive-panel validation are part of
the same fail-closed decision.

Publication builds, verifies, and seals one immutable generation on the pointer filesystem before
atomically replacing only `current.json`. Tests cover concurrent old-or-new reader visibility,
process death at each visibility boundary, stable-lock release, orphan and temporary recovery,
resealing, and symlink/path tampering. The pre-fit software contract binds a canonical Python code
closure and the exact mounted code plus declared public JSON/runtime inputs. The worker and parent
share a typed no-follow staged-output inventory, and the combined containment observation binds it
to source pre/post authentication and parent process-tree cleanup.

Three independent no-cache, provenance-disabled builds at `SOURCE_DATE_EPOCH=1786406400`—Docker
Desktop `amd64` emulation, the separate local `tinyzkp` `docker-container` builder pinned to the
locked BuildKit image, and native Linux `amd64` GitHub Actions—produced the same byte-for-byte OCI
archive under frozen inputs and toolchain. Its SHA-256 is
`37c2fa5846acfbd8357476859bd7f8f0ac6591261d79c2f6f46f0aa22fb76454`; the index is
`sha256:e0f0afd6c66197a37d0ab7a05e7cccfe5990da1fd8497e175fdf3ab909a67812`, runnable child manifest is
`sha256:12c2faa6019fb60cdcabaa8f38f70e99be7998997b97ddb0ca59fbe2e82f1e25`, and config is
`sha256:80ed48f278d7a46c0ae7811285efc69181ae59872a358cc9b176079aa09f3cc8`. The Dockerfile SHA-256 is
`a3a71c3d61c71235d9c1a99c16aa00568b398971adfc2da65388b0c7ea3987a0`. The multi-stage clean final
copies only the curated runtime tree and excludes build-host caches, including Rosetta's
`/root/.cache/rosetta`; the workflow and runtime lock freeze the exact Buildx/BuildKit builder
identity.
Native-Linux execution uses the frozen `host-effective-uid-gid` policy. The worker's numeric UID/GID
matches the bounded mode-`0700` tmpfs UID/GID, while the sole anonymous snapshot volume initializes
from the empty mode-`1777` image directory. The declared `0400`/`0700` host binds therefore remain
usable without running the worker as root or weakening their permissions.
The parent accounts its 3,600-second budget before public staging and actively bounds every Docker
command and wait; a returned staging overrun fails before container creation. A hard in-container
3,540-second watchdog begins before protected-source open and covers snapshot, fit, and close-
reauthentication even if the supervisor dies. Docker's memory and total-memory-plus-swap limits are
both 4,294,967,296 bytes, disabling additional swap. Source-free live probes cover success, timeout
with descendants, cgroup OOM, supervisor death/watchdog recovery, anonymous-volume cleanup, no
canonical publication, and parent re-inventory and sealing of the exact worker stage.

No protected source was opened for Item 12.2, no real-`p1` fit ran, and no candidate artifact,
training plan, observation, evidence, materialization, or lifecycle result was issued. Completing
this software boundary grants no authority to open `p1` or any held-out endpoint partition.

### Pre-fit and post-fit evidence

The pre-fit `CandidateTrainingPlan` is immutable and cannot name a future model hash. It binds the
opaque query, benchmark, and support-envelope identities; the exact typed `p1` role; loader and
count-stream closure; feature, action, target, design, specification, schema, trainer, factory, and
runtime identities; contained-execution policy and runtime-image lock; canonical Python code and
exact mounted-input closures; a pre-render immutable-generation seed; and deterministic fit
settings. Computing this plan does not parse protected held-out endpoint values. Item 12.2 validates
the plan type and source-free closure construction; it did not issue a real plan artifact.

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
lifecycle transition has occurred. The source-free v5 implementation likewise has no real fitted
artifact or trusted verification context.

### Item 12.3 — source-free execution-control design assembled; protected execution pending

The source-free authorization, runtime-preparation, containment, and terminal-evidence design for one
version-bound, nonissuing real-`p1` v5 execution is now implemented. It binds the exact Item 12.2
candidate and closures, one native runtime, fixed limits and paths, an exact stage allowlist, and
fail-closed stop and nonissuance policies. Those checked-in bytes grant no execution authority. No
canonical pending proposal is checked in, no exact proposal digest has been approved, no one-use
attempt has been consumed, and no Item 12.3 protected-source run or terminal report exists.

Authorization separates execution code from the reviewed proposal. A clean execution commit `C`
must not contain the proposal. Its one-parent child `D` may add only the canonical proposal at
`audits/item12_3/sciplex3-k562-v5-pending-proposal.json`. That proposal binds `C`, has a validity
interval of at most 24 hours, declares itself pending and nonauthoritative, and fixes the public
source locator and the acquired source path rather than accepting a caller path. The authorized
actor dispatches the trusted workflow from `refs/heads/main` at exact `C`, supplying both the exact
proposal SHA-256 and exact `D` commit. The workflow checks out only `C`, authenticates `D` as inert
Git object data before any repository execution, and never checks out or executes `D`. The immutable,
asset-free attempt release targets `D` and globally consumes that proposal digest; a private local
ledger and a reauthenticated execution-start receipt close local replay and concurrency. Execution
remains on exact `C` before and during the source-touching wrapper. Failure at any post-consumption
step is terminal and permits no retry or resume.

The existing `sciplex3-v5-runtime-20260811-locked` release is a separate, asset-bearing runtime
dependency. Its target commit records immutable runtime provenance and is intentionally not required
to equal `C`. Only after attempt consumption, and still before source acquisition, may the workflow
download and reauthenticate the immutable release, its sole attested OCI asset, complete archive and
image closure, held archive descriptor, and exact native Linux `x86_64` Docker 29.7.2/cgroup-v2
boundary. Runtime distribution is not source or execution authority.

The source disclosure distinguishes physical access from semantic decode. The H5AD is one opaque
2,526,631,614-byte asset containing all partitions, so it must be transferred and snapshotted whole.
Resolving `p1-train` also requires decoding the complete source-axis selector metadata, including
held-out selector metadata. Only `p1-train` expression/raw-count values may be read or decoded;
held-out expression/count values, endpoints, and outcomes remain unread or unresolved, and held-out
rows cannot be selected for training, scored, or emitted.

The execution remains nonissuing even on contained success. The source, stage, and model are
destroyed with the ephemeral runner, canonical publication is forbidden, and only a sanitized
terminal report of at most 4,096 bytes may leave. The immutable attempt release carries only a
generic unknown-state fallback for a post-consumption terminalization failure; it is not the actual
run outcome. The actual terminal report is retained as a GitHub Actions artifact for 90 days and
then requires exact, separately reviewed persistence at
`audits/item12_3/sciplex3-k562-v5-terminal.json`. Terminal persistence grants no lifecycle,
scientific, calibration, evaluation, or publication authority.

### Held-out source data and lifecycle authority remain sealed

The repository intentionally publishes frozen benchmark-design metadata: exact split-membership
arrays, record/well/plate identities, well-level cases, action assignments, matched-control
identities, and the outcome-free prediction schedule. Reading that metadata is not permission to
resolve protected expression/raw-count values or endpoints, inspect an outcome, score a prediction,
or issue lifecycle evidence.

Item 12 parameter fitting semantically decodes expression/raw-count values only for `p1-train`.
Because the physical H5AD is monolithic, Item 12.3 must nevertheless transfer and snapshot the whole
asset and decode full-axis selector metadata, including held-out selector metadata, to resolve the
training rows. It does not read or decode held-out expression/raw-count values, resolve protected
`p2`, `p3`, or `p4` endpoint values or outcomes, score results, or acquire lifecycle authority. The
training session also does not read the public held-out membership files; their checked-in existence
is not itself a protected boundary.

- `p2-calibration` requires a future one-use calibration grant freshly bound to the exact verified
  `TRAINED_CANDIDATE`.
- `p3-model-selection-validation` remains unavailable until an exact calibrated candidate exists.
- `p4-untouched-test` source outcomes and locked scoring remain unavailable until an exact
  candidate is selected and frozen; its public frozen design metadata grants no such access.

Neither this ADR nor the v4 calibration declaration authorizes those grants.

## Consequences

- V1, v2, and v3 leave no model artifact and no lifecycle residue. Their diagnostics remain
  reproducible evidence for rejecting those parameterizations or fitting schemes.
- V4 removes estimated factor-shape drift and attempts to equilibrate local posterior coordinates
  before outer updates, but its exact run did not equilibrate within the frozen limit. Its empirical
  context is arithmetically bounded yet near single-plate concentration and remains scientifically
  unresolved.
- V4's dose objective is inconsistently scaled relative to its tracked full equal-well ELBO. V4 and
  the planned Item 12.1b replay are retired; historical bytes remain evidence, not executable work.
- V4 is incompatible with all v1, v2, and v3 candidate bytes, schemas, specifications, plans,
  observations, fixtures, and manifests; they must be rejected rather than migrated or relabeled.
- Exact BLAS binding narrows reproducibility claims but prevents materially different numerical
  kernels from borrowing the same runtime label.
- A future checked-in model could be a trained candidate without being calibrated, selected,
  evaluated, admitted, or publicly executable.
- Later calibration must prove that it changes only the declared `tau` state and cannot silently
  rewrite base weights, support, or the eventual accepted plate-context rule.
- V5 now has a source-free exact objective/M-step, neutral-context request-level sampler,
  crash-safe publisher, exact execution/evidence closures, reproducible OCI identity, and hard
  parent containment. None is a real-data fit or a trained-candidate lifecycle result.

## Acceptance criteria

The source-free criteria assigned to Item 12.2 are complete. Item 12 as a whole remains incomplete
and is complete only when a final audit demonstrates the full set below, revalidating the source-
free guarantees alongside the separately authorized fit:

- fits a separately authorized candidate version from the exact close-reauthenticated real `p1`
  stream under its frozen Linux/Python/NumPy/SciPy/BLAS runtime;
- proves source-free that its action/context M-step matches the exact full ELBO through finite-
  difference gradients for `alpha`, constrained `log-rho`, and `delta`; dose and joint Hessian
  finite differences; treated-well adversarial coverage; strict decrease rejection; and accepted-
  substep and complete-block fixed-`q` nondecrease;
- exactly conditions the whole sampled panel to be positive, requires an exact request, caps it at
  512 draws, and revalidates the global conditional signed-`int64` tail budget of at most `2^-64`
  over all `753 * 1 * 27` action/context/calibration states, including compound-Poisson intensity,
  RNG, allocation, and positivity overflow;
- publishes one immutable, verified artifact generation through one atomic pointer and recovers
  after forced termination without mixed files or stale locks;
- binds the canonical executable-code closure, exact mounted code plus declared public JSON/runtime
  input closure, typed no-follow staged-output inventory, and worker/parent containment observations
  without absorbing the separately authenticated protected source into public inputs;
- enforces the parent-owned 3,600-second deadline and 4,294,967,296-byte Docker memory and total-
  memory-plus-swap limits over the complete source-touching container; runs with host-effective
  UID/GID plus matching tmpfs ownership and the mode-`1777` empty-image snapshot-volume
  initialization; durably resolves the exact locked OCI archive; and revalidates the frozen OCI
  identity, timeout, OOM, descendant-cleanup, no-publication, volume-cleanup, parent inventory/seal,
  and next-run recovery tests;
- passes the predeclared inner-equilibration, outer convergence, factor-order, identifiability, and
  finite-state gates without post hoc cap or tolerance changes;
- emits no artifact for an unconverged, nonfinite, stale, substituted, or behavior-inconsistent
  model;
- closes, rereads, rehashes, and exactly reloads the model and reproduces its frozen golden sample;
- binds the exact source, scan, assembly, design, feature, action, target, code, runtime, plan,
  training observation, and model identities;
- proves the unavoidable whole-asset transfer, snapshot, and full-axis selector-metadata decode did
  not read or decode protected `p2`, `p3`, or `p4` expression/raw-count values, resolve held-out
  endpoints or outcomes, score a row, or acquire lifecycle authority, and that all later authority
  flags remain false;
- rejects every incompatible earlier candidate artifact at the active candidate boundary;
- derives `TRAINED_CANDIDATE` only with fresh external trust and exact candidate-factory interface
  verification; and
- leaves calibration, selection, metrics, admission, component execution, and public runtime
  closed.

## Current implementation status

The p1-only training and verification contracts exist. V1 failed at its shape boundary; v2 retained
pooled-shape drift; v3 fixed the shape but failed relative-ELBO and factor-order stability, while its
characterization also rejected the parametric lognormal plate nuisance. The valid exact-reference
v4 run then failed initial inner equilibration before any outer update or ELBO trace. Item 12.1a's
exact historical bytes are canonical, but the later source-code audit found v4's dose-objective scale
inconsistency outside that local-map audit's scope. V4 and Item 12.1b are retired: there is no v4
training plan, observation, fitted model, training evidence, materialization, or trusted lifecycle
result. Item 12.2's source-free v5 software acceptance is complete, but it opened no protected
source, ran no real-`p1` fit, and issued no plan, model, observation, evidence, materialization, or
lifecycle result. Item 12 remains in progress, and the component remains a non-runnable `SCAFFOLD`.
Item 12.3's source-free execution-control design is assembled, including exact C/D proposal topology,
one-use consumption, immutable runtime reauthentication, fixed source acquisition, contained
terminal evidence, and nonissuance. The canonical pending proposal file is absent, no proposal digest
has been approved, and no Item 12.3 protected-source run is currently authorized or has occurred.
