# Changelog

## 0.2.0 - Unreleased

- Move the active public runtime to schema v2 while preserving immutable v1 JSON Schemas and
  fail-closed legacy migration inspection.
- Add typed individual, clone/lineage, population, and spatial-niche subjects; explicit evidence
  transfer and collection effects; bounded, interval-aware action/environment spaces; assignment,
  randomization, matched-control, and realization-evidence semantics; and target aggregation.
- Add strict target estimands for future assay observations, latent quantities, and versioned
  transforms, including protocol/model identity, missingness, censoring, and target-specific
  horizons.
- Add query compilation, active/excluded state factors, and authoritative joint posterior blocks
  for context, realized perturbation, and assay nuisance.
- Add request/scenario/decision-scoped capability preflights, numerical acceptance thresholds,
  support/sufficiency/identifiability/calibration/causal diagnostics, independent readiness flags,
  causal and transport labels, and typed planning abstention.
- Correct contract-boundary defects found in the pre-roadmap audit: assimilate interval observations
  at collection end; treat zero-duration environment records as points; bind returned context,
  forecast realization/nuisance provenance, and planning transport to exact inputs; use absolute
  marginal tolerances; validate distribution-support means and nonnegative dynamics; require
  same-horizon matched controls and the benchmark's exact training partitions; and expose that
  representability resolutions neither fetch source bytes nor replay selectors. Deep-freeze nested
  schema mappings and JSON lists so provenance and context cannot drift after boundary validation.
- Return structured scientific abstentions while keeping computational capability preflights
  fail-closed; remove capability and scientific-validity bypass flags.
- Bind passing population-effect claims to typed target/horizon/aggregation/action estimands,
  eligible experimental designs, and content-addressed external validation evidence; bind result
  provenance to content-addressed support, complete eligible events, and validation artifacts.
- Prevent pre-action observations from being counted as evidence of realized perturbation.
- Bind forecast and planning causal estimands to exact scenario fingerprints and effective actions,
  preventing a no-action or different candidate from inheriting another branch's causal claim.
- Add `recommend_next_measurement` as a standalone, decision-conditioned fourth operation with
  exact ordered candidate-set binding, explicit timing and collection costs, independent readiness,
  strict transport and provenance checks, and fail-closed destructive-sampling semantics. The
  contract reference returns a complete `NOT_EVALUATED` result and never relabels covariance
  reduction as EVSI.
- Persist four exact-scope, content-addressed EVSI evidence traces and distinguish recommended,
  threshold-abstained, not-evaluated, and unsupported measurement decisions.
- Advance the independent public-real dataset manifest to `0.3-experimental`: repeated canonical
  claim assessments, stable functional-readout IDs, separately scoped loss/metric eligibility,
  exact supporting evidence and split-unit closure, layered most-restrictive use policy, and an
  exact science-plus-permission resolver. Add content-addressed dataset slices, interval-valued
  evidence clocks, derived-readout provenance, and machine-checked reviewed representability
  ledgers. Unsupported benchmark metric semantics remain fail-closed.
- Add reviewed real-data proofs for a Replogle K562 destructive population snapshot and an exact
  GSE141064 Live-seq individual functional-recorder slice. Both proofs preserve explicit negative
  claims and keep data-use authorization and benchmark admission closed.
- Add typed intervention reversibility, distinguish fixed target assays from measurement-selection
  candidates, remove unused assay-cost sentinels, and bind every output to an exact value schema.
- Add an experimental content-addressed benchmark contract with physical-unit split closure,
  authoritative evaluation cases, exact metric and baseline specifications, paired block-aware
  comparisons, complete metric reporting, leakage audits, and separate assessment/permission,
  performance, and admission-ready states.
- Freeze the corrected scPerturb v1.4 sci-Plex3 K562 24-hour component benchmark: 173,652 real
  nuclei nested in 1,536 wells and 16 plates, 752 exact compound-dose actions, four immutable
  partitions, and a training-derived 2,000-feature output schema. The artifact remains
  `COMPONENT_BENCHMARK`; planned scoring/baseline implementations and performance admission are
  explicitly blocked rather than represented by numeric sentinels.
- Add experimental content-addressed biological-bundle and support-envelope contracts with an
  exhaustive original-stage port map, operation-specific prerequisites, exact training,
  calibration, selection, validation, and implementation bindings, and an evidence-derived
  component lifecycle. Add a non-runnable sci-Plex3 K562 direct population assay-response scaffold
  that enforces partition roles and exact scope while exposing no hidden-state estimator, evolution,
  planner, or measurement policy.
- Keep bundle contract v0.1 declarations hard-closed: they cannot authorize execution without an
  exact external admission context and just-in-time object reverification. Reject caller-
  constructed biological descriptors at every public operation; retain a separately labeled
  synthetic-test artifact kind only for software-boundary tests.
- Add scope-bound biological admission verification: streaming exact-byte receipts, authenticated
  workflow-derived execution sources, isolated loaded-interface observations, capability-scoped
  external HMAC trust roots, typed validation-result verification with separate verified/passed
  state, deterministic query-derived prerequisites, and just-in-time reverified runtime handles.
  These checks advance infrastructure only; the sci-Plex3 component remains a non-runnable
  `SCAFFOLD` pending its immutable loader and mandatory baseline suite.
- Add the Item 11 sci-Plex3 software path without advancing biological admission: a permanently
  `p1-train`-scoped immutable H5AD loader; six `p1`-fit probabilistic raw-count baselines with
  explicit no-action behavior and an exact-dose-excluding nearest-dose comparator; fixed
  512-sample, five-seed `PCG64DXSM` execution semantics; and streamed, content-addressed fitted-
  state and prediction-run scaffolding. Protected `p2`, `p3`, and `p4` raw UMI endpoint values,
  outcome scoring, and lifecycle authority remain hard sealed behind future grants; frozen split
  membership and outcome-free case design are public. Record the exact close-reauthenticated source
  scan of 94,785 `p1` rows across
  768 wells and all six content-addressed fitted-state identities, retaining seven genuine zero-
  panel rows. These are software provenance, not prediction or performance results: the benchmark
  stays `COMPONENT_BENCHMARK`, the bundle stays `SCAFFOLD`, and no public cell-state runtime is
  added.
- Add the Item 12 p1-only trusted-training boundary: an immutable candidate plan, typed training
  evidence, runtime-only HMAC source and fit attestations, close/reread/rehash model verification,
  and an exact registry-owned candidate-factory interface. Record three fail-closed real-p1
  investigations with no candidate artifact: v1's separate `q`/capture term and 16 free shapes
  reached a rejected shape boundary; v2's free pooled shape drifted to approximately `0.073524`
  while its outer fit remained unconverged; and the fixed-`r_theta=0.1` v3 counterfactual still
  failed the relative-ELBO and terminal factor-order gates. Record that v3's activation-rank
  spectrum was about 56.85 times above its strict gate, making the provisional rank issue a
  quantization-portability ambiguity, while its independent lognormal plate nuisance was
  tail-dominated at `sigma_plate=5.66126548675`.
- Define the audited, incompatible v4 software-only design as a 16-factor continuous
  Gamma--Poisson model without `q`/capture, with fixed `r_theta=0.1`, deterministic row-local inner
  equilibration, and proposed unseen-plate context formed by uniformly selecting one complete
  observed `p1` `rho` row. Predeclare the future p2 calibration grid
  `tau_j=exp(j/20)`, `j=-20,...,6`, without reading p2 calibration values or selecting a value. The
  exact reference
  runtime remains single-thread Linux `x86_64`, CPython 3.11.15, NumPy 2.4.6, SciPy 1.17.1, and
  `scipy-openblas` 0.3.31.188.0.
- Record the audited v4 executions without issuing an artifact. The first report,
  `4677fc8ef1a458bf3616abc507250572c2da7a8d53c1c8a7a03d4b097f3d4877`, is infrastructure-
  invalid and contains no fit; h5py's nonpersistent `/dev/null` probe triggered the filesystem
  audit, so that report must not be interpreted scientifically. The replacement exact-reference,
  p1-only report, `66e9debc1a402e7aa68cbc934f7c5f641529eea3187ec15606364c912af8faa8`,
  passed integrity checks and post-hoc resource acceptance checks with finite provisional tensors,
  then failed initial,
  untraced inner equilibration at sweep 50, streak 0, `Rshape=0.24714465227035654`, and
  `Relog=3.750385840630546`, before any outer update or ELBO trace. These are 24,714,465 and
  375,038,584 times the `1e-8` tolerance. Initialization-only loading-rank, activation-rank, and
  minimum-share values were respectively `0.101239623839`, `0.001249342162`, and
  `0.001237124212`; they are not rescue evidence. Provisional `rho` reached
  `7.999995552807402`. With factor-column sum eight, at least one factor places
  `99.9999444101%` of its context mass on one plate and has effective context count approximately
  `1.0000011118`. This is a candidate-family risk, not a fitted-model pathology, and requires
  reexamination of the empirical whole-row proposal. V4 is `NO-GO`: no model, plan, observation,
  training evidence, materialization, or lifecycle result was issued. Item 12 remains in progress
  and the component remains `SCAFFOLD`; protected `p2`, `p3`, and `p4` raw UMI endpoint values,
  outcome scoring, and lifecycle authority remain sealed.
- Completed **Item 12.1a — local-map and plate-context characterization software freeze**. The
  historical harness SHA-256 is `f4e6b76847bd926952995d66233389768f091135699fb60a38d7d9762bb03ff1`
  and its source-free test SHA-256 is
  `8989618e259fb4aed0e0798bc010e40092c45e6bd30234bb3a7b534cdc562903`.
  The exact harness, test, parent driver, and both historical reports are now canonical under
  [`audits/item12_1a`](audits/item12_1a/). Twenty-six focused tests, Ruff check/format, compilation,
  and an independent math/containment audit reported no P0/P1 finding within the bounded local-map
  and containment scope. The harness predeclares one bounded first-failing-batch replay
  with per-sweep `Rshape`/`Relog`, local ELBO, one- and two-step state distances and update cosine,
  the worst
  row/factor/count aggregate, shape/rate/mass invariants, the 16-by-16 Jacobian spectral radius,
  exact replay equality, state/allocation digests, and all 16 `rho` effective-context counts and
  maximum shares. Objective decrease routes to an implementation fix; an intact objective plus a
  two-cycle or noncontractive map routes to a versioned v5 safeguarded local solver; near-zero
  prior/context behavior or effective context near one routes to v5 plate regularization or
  neutralization. Item 12.1a opened no source and does not prefreeze damping or shrinkage or
  authorize a real-source run, cap or tolerance relaxation, an artifact, p2 access, or any biology
  or performance claim.
- Retire v4 and its planned Item 12.1b real-`p1` replay before execution after a source-code audit
  confirmed that the dose Newton objective omits the tracked full ELBO's `94785/768` equal-well
  multiplier. No v4 model or lifecycle evidence was issued. At that checkpoint, record a source-
  free v5 objective/M-step redesign as the next milestone, requiring finite-difference full-ELBO
  gradient tests, fixed-`q` M-step nondecrease tests, one-objective action/context updates, request-
  level sampler/RNG support gates, immutable-generation atomic publication, and whole-process-tree
  hard wall-clock and memory containment before any new real-`p1` authorization. That milestone is
  completed below as Item 12.2.
- Complete **Item 12.2 — source-free v5 objective, M-step, sampling, publication, and containment**.
  Replace the inconsistent v4 action block with one exact fixed-`q`, equal-well objective and an
  all-768-well `alpha`/constrained-`log-rho`/`delta` M-step. Independently implemented objective,
  finite-difference gradient and Hessian, treated-well adversarial, one-ULP decrease, accepted-
  substep, and complete-block checks all bind updates to nondecrease of that same objective.
- Add exact positive-panel sampling through the zero-truncated superposed compound-Poisson/log-
  series representation. Exact `CandidateSampleRequest` support caps requests at 512 draws and
  certifies a conservative conditional signed-`int64` tail bound no larger than `2^-64` over all
  `753 * 1 * 27 = 20,331` action, neutral-context, and calibration-state combinations. Sampling is
  prefix-stable and provenance-bound; target-only support, unsupported global envelopes, and RNG or
  count overflow fail closed.
- Add crash-safe immutable-generation publication through one atomic `current.json` pointer,
  including concurrent old-or-new reader visibility, forced-termination recovery, stable-lock
  release, orphan cleanup, and exact resealing. Bind the complete Python executable closure, exact
  mounted public-control input closure, and typed no-follow staged-output inventory into the plan,
  worker observation, parent containment observation, and future stage verification.
- Freeze the source-free Linux `amd64` OCI runtime from byte-identical no-cache,
  provenance-disabled builds in three independent environments under frozen inputs and toolchain:
  Docker Desktop `amd64` emulation, the separate local `tinyzkp` `docker-container` builder pinned
  to the locked BuildKit image, and native Linux `amd64` GitHub Actions. The exact archive SHA-256 is
  `37c2fa5846acfbd8357476859bd7f8f0ac6591261d79c2f6f46f0aa22fb76454`; its index is
  `sha256:e0f0afd6c66197a37d0ab7a05e7cccfe5990da1fd8497e175fdf3ab909a67812`, runnable child is
  `sha256:12c2faa6019fb60cdcabaa8f38f70e99be7998997b97ddb0ca59fbe2e82f1e25`, and config is
  `sha256:80ed48f278d7a46c0ae7811285efc69181ae59872a358cc9b176079aa09f3cc8`; the Dockerfile SHA-256 is
  `a3a71c3d61c71235d9c1a99c16aa00568b398971adfc2da65388b0c7ea3987a0`. The multi-stage clean final
  copies only the curated runtime tree and excludes build-host caches; the lock and workflow freeze
  the exact Buildx/BuildKit builder identity.
  Freeze native-Linux execution as `host-effective-uid-gid`, mount the bounded mode-`0700` tmpfs
  with that same numeric UID/GID, and initialize the anonymous snapshot volume from an empty mode-
  `1777` image directory. The contained non-root worker can therefore use the declared
  `0400`/`0700` host binds without broadening their permissions.
  The parent accounts its 3,600-second budget before public staging and actively bounds every
  Docker command and wait; any returned staging overrun fails before container creation. An
  independent 3,540-second in-container watchdog begins before protected-source open and covers
  snapshot, fit, and close-reauthentication. Docker's memory and total-memory-plus-swap limits are
  both 4 GiB, disabling additional swap. Source-free live probes pass for success, timeout with
  descendants, cgroup OOM, supervisor death/watchdog recovery, anonymous-volume cleanup, no
  canonical publication, and parent re-inventory and sealing of the exact worker stage.
- Keep this completion strictly software-only. Item 12.2 opened no protected source, ran no real-
  `p1` fit, and issued no candidate artifact, plan, observation, evidence, materialization, or
  lifecycle result; sci-Plex3 remains `SCAFFOLD`. The version-bound, nonissuing Item 12.3 protected
  execution remains unauthorized and unrun. Its exact OCI layout is durably published as the
  immutable, attested
  `sciplex3-v5-runtime-20260811-locked` GitHub Release; reauthenticating that release under an exact
  one-use authorization remains a pre-source execution gate.
- Assemble the source-free **Item 12.3 authorization and execution-control plane** without creating
  an authorization. Canonical proposal bytes bind a clean execution commit `C`; proposal-only child
  `D` may add only those bytes. Manual dispatch from trusted `refs/heads/main` at exact `C` supplies
  both the proposal SHA-256 and `D` commit. The workflow authenticates `D` only as inert object data
  and the immutable asset-free attempt-release target, never checks it out or executes it, and remains
  on `C`. A private one-use ledger and execution-start receipt, exact runtime preparation, a fixed
  protected-source path, and no-retry/no-resume stop policy fail closed before source execution.
- Record the complete source scope honestly. The 2,526,631,614-byte monolithic H5AD must transfer and
  snapshot whole, and full-axis selector metadata including held-out selectors must be decoded to
  resolve `p1`. Expression/raw-count decoding remains `p1-train`-only; held-out values, endpoints,
  outcomes, scoring, emission, and lifecycle authority remain unavailable.
- Separate three evidence roles: the asset-bearing immutable runtime release is provenance and
  distribution only; the future immutable asset-free attempt release would globally consume one
  proposal and permanently retain only a generic fallback; and the actual at-most-4,096-byte
  sanitized terminal uses 90-day Actions artifact transport followed by exact reviewed repository
  persistence. Source, stage, model, and canonical publication remain runner-confined and
  nonissuing. No canonical pending proposal, approved digest, protected Item 12.3 run, or terminal
  report currently exists.

## 0.1.0 - Unreleased

- Establish the query-conditioned belief-state domain model.
- Add pluggable estimation, evolution, and intervention-planning ports.
- Add a biologically non-authoritative linear-Gaussian reference backend.
- Bind recursive updates and derived artifacts to exact event, query, model, scenario, and objective
  fingerprints.
- Add unit-aware target predictions and objectives, explicit sample-axis contracts, and strict
  abstention for unsupported biological or measurement semantics.
- Establish the public-real-data roadmap, evidence policy, biological verticals, and scientific
  graduation gates as durable project documentation and repository memory.
- Add an experimental typed dataset-manifest scaffold for source hashes, use restrictions,
  experimental units, sampling linkage, modality alignment, and scoped capability assessment.
