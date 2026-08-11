# ADR 0011: Keep the first sci-Plex3 executable path p1-only

- **Status:** Accepted
- **Date:** 2026-08-10
- **Extends:** ADR 0008 first Vertical A component benchmark, ADR 0009 population-response
  component boundary, and ADR 0010 trusted admission verification

## Context

The frozen sci-Plex3 K562 benchmark defines four physically separated lifecycle partitions. The
first executable data path needs the source counts in `p1-train` to fit mandatory baselines and,
later, a candidate distribution model. It does not yet need calibration outcomes from `p2`, model-
selection outcomes from `p3`, or untouched-test outcomes from `p4`.

A general source-partition reader at this stage would make protected held-out endpoint values or
scoring available before the artifact that justifies their lifecycle use exists. Exact frozen split
membership and outcome-free case design are already public checked-in metadata; the current session
still does not need to parse them. The software milestone could otherwise look like a completed
scientific benchmark even if the exact source had not been scanned, the baselines had not been run,
and no metric or acceptance gate had passed. The first path must instead demonstrate deterministic
raw-count ingestion and executable probabilistic algorithms while preserving the benchmark's sealed
outcome-evaluation sequence.

The target remains the direct population endpoint from ADR 0009: raw nonnegative integer UMI counts
on the exact ordered 2,000-feature panel in recovered K562 nuclei at 24 hours, conditional on static
well context and intended compound-dose assignment or no-action vehicle. This is not hidden-state
inference and cannot authorize any public cell-state operation.

## Decision

Item 11 introduces a permanently single-purpose `p1-train` loader, six `p1`-fit probabilistic
baseline algorithms, and content-addressed streaming-run scaffolding. It does not advance the
component lifecycle or benchmark admission.

### Immutable p1 source boundary

The loader authenticates the exact corrected H5AD bytes and the checked-in `p1` loader contract
before yielding any counts. It resolves only the frozen `p1` record, well, condition, and feature
mappings; emits immutable sparse raw-count batches in canonical membership order; preserves source-
row provenance; and binds the exact manifest, query, benchmark, target schema, scoring transform,
feature panel, and ordered-feature identities.

The same seekable source object is hashed before HDF5 parsing and reauthenticated after HDF5 closes.
The `p1` session neither parses the public held-out membership ledgers nor protected outcome values,
and it exposes no caller-selectable partition. `p2`, `p3`, and `p4` fail before protected
raw-endpoint access. Their future gates are respectively:

- `p2-calibration`: a trusted calibration grant bound to an exact `TRAINED_CANDIDATE`;
- `p3-model-selection-validation`: a trusted selection grant bound to an exact
  `CALIBRATED_CANDIDATE`; and
- `p4-untouched-test`: a locked-evaluation grant bound to an exact `MODEL_SELECTED_FROZEN`
  candidate.

A source-scan receipt explicitly records that it is not lifecycle evidence, is not scientifically
admissible, and carries no trusted workflow authority by itself.

### Frozen baseline algorithms

Every baseline accepts `p1` training data and a design-only prediction request. There is no outcome
or comparator-count input on the prediction surface. The suite is:

1. **Matched-vehicle resampling:** choose from same-plate vehicle wells associated with the exact
   `p1` condition counterpart, using `p1` controls only.
2. **Exact-condition replicate-1 empirical resampling:** resample an equal-weight `p1` well and then
   a nucleus from the requested exact condition.
3. **Exact-condition negative binomial:** fit featurewise Gamma--Poisson means and dispersions with
   frozen global pseudo-well smoothing.
4. **Hierarchical well negative binomial:** pool at exact-condition, compound, and global well
   levels with frozen weights.
5. **Low-rank compound-dose response:** fit a deterministic truncated-SVD approximation to `p1`
   log1p effects relative to matched `p1` vehicles, then sample with the frozen count likelihood.
6. **Nearest-supported-dose resampling:** use the nearest alternate same-compound `p1` dose in
   absolute log10-dose distance, explicitly excluding the requested dose and breaking ties toward
   the lower dose.

All six have an explicit no-action path. Empirical draws weight wells equally before nuclei. The
exact p1 membership is retained in full during fitting, including any nucleus with a positive
full-transcriptome library but zero counts on the selected panel; such rows are never silently
excluded or imputed. Predictive outputs remain strictly positive on the panel: empirical samplers
use bounded same-well nucleus redraws, and stochastic count models use the same bounded support
rule. The frozen campaign requests 512 samples per case and seed for seeds `0` through `4` and uses
NumPy `PCG64DXSM`; implicit default random generators are not part of the protocol.

### Content-addressed execution without performance admission

Each fit produces a canonical manifest that binds the implementation version, baseline identity,
feature order, statistical and random-number semantics, and the fitted arrays or empirical pools.
The runner hashes fitted state before any later-stage access. Prediction output is written in
bounded shards that bind baseline, fit, case, seed, draw interval, feature order, and content bytes;
the full campaign is never required to exist as one dense in-memory matrix.

These artifacts are software provenance only. Item 11 close-reauthenticated and scanned the exact
2,526,631,614-byte source with SHA-256
`603ed16c5e25401c8a7f5bb0b2b045179701017d65dcfc6aeea71722a66cd10a`, covered all 94,785 `p1`
records across 768 wells, retained seven genuine zero-panel rows, and recorded content-addressed
fitted-state identities for all six baselines. That proves the loader-to-fit software closure; it
does not record a prediction campaign, metric, baseline comparison, benchmark performance result,
or scientific admission as passed.

The exact synthetic fixture was reproduced byte-for-byte on macOS arm64 with Accelerate and Linux
amd64 with OpenBLAS under Python 3.11 and NumPy 2.4.6. That dual-runtime check supports the current
fixture only; it does not broaden the declared runtime or substitute for real-data tests.

## Consequences

- Loader and baseline unit tests can mature without exposing held-out outcomes or fabricating
  biological evidence.
- Matched controls used for fit are restricted to `p1`; later `p4` controls remain evaluation
  comparators only.
- Exact-dose empirical reproduction and alternate-dose generalization remain distinct baselines.
- Deterministic random-number, feature-order, fitted-state, and prediction-shard identities make
  drift observable without holding the multi-gigabyte source or full prediction campaign in Git.
- The scientific benchmark remains `COMPONENT_BENCHMARK`; mandatory baseline result status remains
  unrun until exact executions and comparisons exist.
- The component bundle remains `SCAFFOLD`, with no trained candidate, calibration, selection freeze,
  locked evaluation, passed performance gate, or public cell-state runtime.

## Acceptance criteria

Item 11 is complete only when tests and checked-in receipts demonstrate that it:

- authenticates the exact source object before and after use and validates the full frozen H5AD
  structure needed for `p1`;
- resolves complete, one-to-one `p1` record, well, condition, row, and ordered-feature closure while
  never parsing public held-out membership files or protected outcomes in the `p1` session;
- refuses protected `p2`, `p3`, and `p4` raw-endpoint access without future lifecycle-bound
  trusted grants;
- fits all six algorithms only from `p1`, including no-action cases, p1-only matched controls, and
  exact-dose-excluding nearest-dose behavior;
- reproduces the 512-draw, five-seed `PCG64DXSM` contract and content-addresses fitted state and
  streamed prediction shards;
- runs the exact real-source scan and records its non-admissible software receipt; and
- leaves baseline performance, benchmark admission, bundle lifecycle, and all four public
  cell-state operations unchanged.
