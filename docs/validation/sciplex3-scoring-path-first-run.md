# The scoring path's first run on real cells

- **Run date:** 2026-08-12
- **Partition:** `p1-train` only. **In-sample.** No protected partition was opened.
- **Status:** software diagnostic. **Not** an observational floor, **not** a benchmark result, and
  **not** admissible evidence for any scientific claim.
- **Source:** `SrivatsanTrapnell2020_sciplex3.h5ad`, 2,526,631,614 bytes, sha256
  `603ed16c5e25401c8a7f5bb0b2b045179701017d65dcfc6aeea71722a66cd10a`, verified before the run
  against the value `benchmarks/artifacts/sciplex3-k562-24h-v1/source-verification.json` declares.

## Why this run happened

`Q1` implemented the metrics and the clustered bootstrap. `Q2` implemented the two faithfulness
harnesses. Neither could be pointed at biology, because the span between them and the data did not
exist: the frozen panel-only scoring transform appeared in `src/` only as a content hash, nothing
joined a baseline's raw-count predictions to a metric, and `MetricResult` had no construction site
anywhere in the package. `src/cellstate/evaluation/scoring.py` is that span, and this is the first
time any part of this repository has computed a metric value from biological bytes.

Scoring a baseline against the partition it was fitted on is not a result. It is the cheapest
available proof that the path executes. What made the run worth doing is that it produced two
findings that no synthetic fixture could have produced, and both are about the frozen benchmark
rather than about the code.

## Finding 1 — the frozen scoring policy makes this benchmark unevaluable as written

The transform refused to run. That was correct behaviour.

`support/scoring-transform.json` declares:

```
zero_panel_total_policy: "error_fail_evaluation_no_exclusion_or_imputation"
failure_action:          "fail the metric evaluation and therefore block benchmark admission;
                          do not drop, exclude, impute, renormalize, clip, or substitute the sample"
```

and `p1-train` contains records whose 2,000-feature panel total is exactly zero — every UMI for
that nucleus fell outside the declared panel. Measured from the bytes:

| | Measured |
| --- | --- |
| Zero-panel records in `p1-train` | **7** |
| Distinct wells carrying one | **7** of 768 (each carries exactly one) |
| Treated wells affected | **7** of 752 |
| Declared by the assembly receipt | `zero_panel_record_count: 7` |

The project has always *counted* these records — the receipt records them and
`scripts/materialize_sciplex3_k562_p1_baselines.py` pins `EXPECTED_ZERO_PANEL_RECORD_COUNT = 7` —
but nothing had ever tried to score them, so the conflict stayed latent. On first contact the
policy fires and blocks the evaluation, and the policy forbids the only three responses that would
let it proceed: dropping the record, imputing it, or renormalizing it.

**Consequence for `Q3`.** The roadmap records `Q3` as blocked on an ADR 0011 access decision. This
is a second, independent blocker: even with protected-partition access granted, the floor cannot be
measured if `p4-untouched-test` carries the same defect, because the frozen policy makes the
evaluation fail rather than proceed. Whether it does carry it is unknown here — `p4` was not
opened. At `p1`'s rate (7 in 94,785 records, 0.0074%) a partition of comparable size would be
expected to carry a handful.

The numbers below were produced by excluding the 7 affected **wells** outright. That is a
deviation from the frozen policy, it is recorded here rather than buried, and it is why nothing
below is admissible.

## Finding 2 — three of the ten frozen metrics cannot discriminate on this data

Marginal coverage error is `|empirical coverage - nominal|`, which is sign-blind. The errors alone
looked alarming, so the empirical coverage itself was computed directly over 150 wells:

| Nominal | Empirical coverage | Reported error | Frozen acceptance |
| --- | --- | --- | --- |
| 0.50 | **0.9477** | 0.4477 | UCB <= 0.03 |
| 0.80 | **0.9573** | 0.1573 | UCB <= 0.03 |
| 0.95 | **0.9750** | 0.0250 | UCB <= 0.03 |

Coverage barely moves — 0.948 to 0.975 — across a nominal range of 0.50 to 0.95. The cause is
measured and simple: **94.63% of transformed coordinates in a well are exactly zero.** For a
feature that is zero in every predictive sample the central interval is degenerate at zero, and the
observation is zero too, so the feature is "covered" whatever the nominal level. Coverage is
therefore an estimate of panel sparsity, not of calibration.

This is the pathology the roadmap already names for a different metric family — *"Marginal error
and all-gene correlation are maximized by predicting no change and must not stand alone"* — arriving
in the coverage family, which the frozen suite carries three of.

Two consequences follow, and neither is about any model:

- `sciplex3.marginal-coverage-error-p50` and `-p80` **cannot** meet the frozen acceptance
  criterion of a 0.03 upper confidence bound. Their errors are 15x and 5x the threshold and are
  driven by sparsity, so no candidate can improve them.
- `sciplex3.marginal-coverage-error-p95` **passes** at 0.0250, for the same reason: coverage
  happens to sit near 0.95. A passing verdict there is close to uninformative.

A frozen suite is not retrofitted, and `Q1`'s completion condition deliberately does not depend on
this. But `Q5` freezes the next suite, and this is the evidence it should read: a coverage metric on
sparse counts needs to be conditioned on features with support, or replaced.

## What the run produced

745 of 752 treated wells, 188 compound clusters, 8 plates, 64 predictive samples per well,
rank-50 train-fitted projection, 400 bootstrap resamples. **In-sample; the exact-condition model is
scored on the wells it was fitted on and its apparent advantage is memorisation.**

| `metric_id` | exact-condition NB | matched-vehicle resampling |
| --- | --- | --- |
| `marginal-crps-logcp10k` | 0.2073 [0.2041, 0.2102] | 0.2102 [0.2064, 0.2133] |
| `joint-energy-train-pca` | 10.8467 [10.7095, 11.0085] | 10.9293 [10.7099, 11.1975] |
| `vehicle-relative-pseudobulk-rmse` | 0.1321 [0.1295, 0.1351] | 0.1730 [0.1631, 0.1844] |
| `four-dose-profile-diagnostic` | 0.1321 [0.1296, 0.1345] | 0.1728 [0.1633, 0.1818] |
| `marginal-coverage-error-p50` | 0.4469 [0.4458, 0.4482] | 0.4464 [0.4452, 0.4478] |
| `marginal-coverage-error-p80` | 0.1567 [0.1561, 0.1575] | 0.1551 [0.1543, 0.1561] |
| `marginal-coverage-error-p95` | 0.0246 [0.0244, 0.0249] | 0.0242 [0.0238, 0.0247] |
| `marginal-interval-width-p50` | 0.0591 [0.0545, 0.0634] | 0.0475 [0.0428, 0.0523] |
| `marginal-interval-width-p80` | 0.7407 [0.7256, 0.7540] | 0.6554 [0.6368, 0.6728] |
| `marginal-interval-width-p95` | 2.2630 [2.2461, 2.2770] | 2.1309 [2.1035, 2.1543] |

Every one of the ten identifiers the frozen suite declares resolved to a value with a grouped
interval. On the effect metrics the two baselines separate cleanly; on CRPS their intervals
overlap. Neither observation means anything scientifically, because both baselines are being
scored in-sample.

## What this does not establish

- No observational floor. That needs held-out predictions and both blockers cleared.
- No comparison between baselines that survives contact with held-out data.
- No claim about any candidate model; none was run.
- The frozen artifact's metric bindings remain `specification_only`, and re-versioning them stays
  `Q3`'s decision under ADR 0014.
