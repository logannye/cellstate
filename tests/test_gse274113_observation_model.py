"""The GSE274113 observation model, from frozen bytes to a belief about real cells.

The headline test is :func:`test_a_belief_is_emitted_from_real_cells`.  Until it passed, every
scheduled item in this project had built apparatus that *judges* a representation while the thing
being judged had never been built.

Several tests here exist specifically to have a reachable failing branch.  A guard that cannot fire
is the defect this repository keeps finding in its own work, so the leakage refusal, the
one-guide requirement and the ADR 0021 causal gate are each exercised from the side that fails.
"""

from __future__ import annotations

import hashlib
import json
import math
import tomllib
from pathlib import Path

import numpy as np
import pytest

from cellstate.api import estimate_cell_state
from cellstate.backends.gse274113 import usage as usage_module
from cellstate.backends.gse274113.arm_request import arm_query, arm_request
from cellstate.backends.gse274113.estimator import GSE274113ObservationEstimator
from cellstate.backends.gse274113.fit import (
    DISPERSION_FLOOR,
    NULL_TARGET,
    PLACEBO_TARGETS,
    ArmSlice,
    FittedFold,
    fit_fold,
)
from cellstate.backends.gse274113.likelihood import (
    log_composition,
    posterior,
    stabilize,
    technical_variance,
)
from cellstate.backends.gse274113.usage import artifact_directory
from cellstate.domain.belief import (
    CalibrationReport,
    CausalStatus,
    CellStateBelief,
    CriterionOutcome,
    EvaluationStatus,
)
from cellstate.domain.distributions import UnavailableDistribution
from cellstate.errors import CapabilityError
from cellstate.evaluation.gse274113_reports import (
    measure_calibration_coverage,
)
from cellstate.ports.models import ModelArtifactKind

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "backends" / "vertical-a" / "gse274113-rna-obs-v1"
PANEL_PATH = ARTIFACTS / "panel.json"
SLICE_PATH = ARTIFACTS / "arms.json"

# Pinned so a regenerated artifact is a deliberate, visible act rather than a silent drift.
PANEL_SHA256 = "710ed885667d8bff2f3803ab12762899116722c20cce1668b7808d223ddc127d"
SLICE_SHA256 = "3de8ee39ba5dbf30aa383d1ae49193d2b818a031dff40098fdda4084d3cc6cfb"

HELD_OUT = "rep1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def panel() -> dict[str, object]:
    return json.loads(PANEL_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def arm_slice() -> ArmSlice:
    return ArmSlice.from_payload(json.loads(SLICE_PATH.read_text(encoding="utf-8")))


@pytest.fixture(scope="module")
def fold(arm_slice: ArmSlice) -> FittedFold:
    return fit_fold(arm_slice, HELD_OUT)


@pytest.fixture(scope="module")
def calibration(arm_slice: ArmSlice) -> CalibrationReport:
    thresholds = arm_query(arm_slice.targets, model_fingerprint="0" * 64).acceptance_thresholds
    return measure_calibration_coverage(
        arm_slice,
        minimum_coverage=thresholds.minimum_calibration_coverage,
        maximum_calibration_error=thresholds.maximum_calibration_error,
    )


@pytest.fixture(scope="module")
def estimator(
    fold: FittedFold, arm_slice: ArmSlice, calibration: CalibrationReport
) -> GSE274113ObservationEstimator:
    panel_fingerprint = _sha256(PANEL_PATH)
    slice_fingerprint = _sha256(SLICE_PATH)
    seed = GSE274113ObservationEstimator(
        fold,
        query=arm_query(arm_slice.targets, model_fingerprint="0" * 64),
        slice_fingerprint=slice_fingerprint,
        panel_fingerprint=panel_fingerprint,
        calibration=calibration,
    )
    query = arm_query(arm_slice.targets, model_fingerprint=seed.model_fingerprint)
    return GSE274113ObservationEstimator(
        fold,
        query=query,
        slice_fingerprint=slice_fingerprint,
        panel_fingerprint=panel_fingerprint,
        calibration=calibration,
    )


def _request(
    arm_slice: ArmSlice, estimator: GSE274113ObservationEstimator, library: str, target: str
):
    composition, depth = arm_slice.log_composition(library, target)
    return arm_request(
        library,
        target,
        query=estimator._query,
        log_composition=tuple(float(value) for value in composition),
        cells=arm_slice.cells[(library, target)],
        panel_total=int(depth),
    )


# --------------------------------------------------------------------- artifacts


def test_frozen_artifacts_match_their_pinned_digests() -> None:
    """A regenerated panel or slice changes the model; it may not change silently."""

    assert _sha256(PANEL_PATH) == PANEL_SHA256
    assert _sha256(SLICE_PATH) == SLICE_SHA256


def test_the_wheel_ships_the_slice_outside_the_package() -> None:
    """The build's force-include destination must be a SIBLING of `cellstate`, never inside it.

    Two defects meet at this one line of `pyproject.toml` and they pull in opposite directions:

    * Ship nothing, and an installed `cellstate` imports cleanly and then answers nothing --
      `artifact_directory` has no `backends/` tree to resolve. That was true of every wheel until
      the force-include was added.
    * Ship it *inside* the package, and **editable** installs break. Hatchling applies
      `force-include` to editable installs too, where the modules live in `src/`, so the slice's two
      JSON files became the only occupants of `site-packages/cellstate` -- turning
      `cellstate.backends` into a namespace package that shadowed the source tree.
      `from cellstate.backends import X` failed with `(unknown location)` while `import cellstate`
      still worked, and CI never saw it because CI syncs `--no-editable`.

    A sibling destination satisfies both. This test pins the *contract between the declaration and
    the code*: `pyproject.toml` says where the build puts it, `artifact_directory` says where the
    runtime looks, and nothing else made those two agree.
    """

    manifest = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    force_include = manifest["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    sources = list(force_include)
    assert sources == ["backends/vertical-a/gse274113-rna-obs-v1"], (
        "the slice is the only thing force-included; a new entry needs its own reasoning"
    )
    destination = force_include[sources[0]]

    assert not destination.startswith("cellstate/"), (
        f"force-include destination {destination!r} sits inside the package and will shadow the "
        "source tree in editable installs"
    )
    assert "/" not in destination, "a nested destination re-creates the namespace-package problem"

    # And the runtime looks exactly where the build writes. `artifact_directory`'s packaged branch
    # is `<dir containing the package>/<destination>`; recompute it here rather than trusting it.
    package_root = Path(usage_module.__file__).resolve().parents[3]
    assert (package_root / destination).name == destination
    assert artifact_directory().is_dir()
    assert (artifact_directory() / "arms.json").is_file()


def test_the_panel_carries_every_perturbed_target(panel: dict[str, object]) -> None:
    """On-target knockdown has to be readable, or the panel cannot see the perturbation at all."""

    genes = panel["genes"]
    assert isinstance(genes, list)
    assert panel["gene_count"] == len(genes) == 100
    symbols = {str(gene["symbol"]) for gene in genes}
    targets = {str(gene["symbol"]) for gene in genes if gene["category"] == "target_tf"}
    assert len(targets) == 19
    assert targets <= symbols
    assert len({str(gene["row_index"]) for gene in genes}) == len(genes)


def test_the_slice_covers_every_arm_computed_from_its_own_arrays(arm_slice: ArmSlice) -> None:
    """Recomputed from the membership arrays, never read from a declared count."""

    assert len(arm_slice.libraries) == 14
    assert len(arm_slice.targets) == 20
    observed = {key for key in arm_slice.counts if not key[1].startswith("NT_")}
    expected = {
        (library, target) for library in arm_slice.libraries for target in arm_slice.targets
    }
    assert observed == expected, "280 of 280 (library, target) arms must be populated"
    assert all(count.shape == (100,) for count in arm_slice.counts.values())
    assert all(int(count.sum()) > 0 for count in arm_slice.counts.values())


def test_the_placebo_split_exists_for_every_library(arm_slice: ArmSlice) -> None:
    for library in arm_slice.libraries:
        for half in PLACEBO_TARGETS:
            assert (library, half) in arm_slice.counts


# ------------------------------------------------------------------------- fit


def test_a_fold_never_sees_the_library_it_answers(arm_slice: ArmSlice) -> None:
    for library in arm_slice.libraries:
        fitted = fit_fold(arm_slice, library)
        assert library not in fitted.fit_library_ids
        assert len(fitted.fit_library_ids) == 13


def test_the_subspaces_are_orthogonal(fold: FittedFold) -> None:
    """S5 is claimed structurally, so the structure has to actually hold."""

    biology, nuisance = fold.biology_basis, fold.nuisance_basis
    assert np.abs(biology.T @ nuisance).max() < 1e-10
    assert np.abs(biology.T @ biology - np.eye(biology.shape[1])).max() < 1e-10
    assert np.abs(nuisance.T @ nuisance - np.eye(nuisance.shape[1])).max() < 1e-10
    shared = np.hstack([biology, nuisance])
    for target, direction in fold.target_directions.items():
        assert np.abs(shared.T @ direction).max() < 1e-10, f"{target} leaks into the shared basis"


def test_the_declared_null_direction_is_not_structurally_zero(fold: FittedFold) -> None:
    """The inert-``do`` defect ADR 0019 names by name.

    If NT's direction were fixed at zero, the S4 null half would be unfailable: a check whose
    failing branch cannot be reached.  It is estimated from a placebo split instead, so it is a
    genuine direction whose *expected* magnitude is zero.
    """

    direction = fold.target_directions[NULL_TARGET]
    assert np.linalg.norm(direction) == pytest.approx(1.0, abs=1e-10)
    assert fold.residual_norm_by_target[NULL_TARGET] > 0.0


def test_the_fitted_dispersion_is_fitted_rather_than_clamped(fold: FittedFold) -> None:
    """``psi^2`` must be a fitted quantity, not the floor wearing its name (ADR 0022 decision 4).

    This assertion could not have failed before: the guard read
    ``assert fold.biological_observation_variance > 0.0`` against a value produced by
    ``max(..., 1e-6)``, so it passed on the clamp in all fourteen folds while ``likelihood.py``
    claimed the term was fitted.  The pre-clamp value is now carried on the fold precisely so this
    can fail.
    """

    assert not fold.dispersion_is_clamped
    assert fold.biological_observation_variance_before_clamp > DISPERSION_FLOOR
    assert fold.biological_observation_variance == pytest.approx(0.060434, rel=1e-3)
    assert fold.biology_prior_variance > 0.0
    assert fold.nuisance_prior_variance > 0.0
    assert fold.realization_prior_variance > 0.0


def test_an_unknown_library_is_refused(arm_slice: ArmSlice) -> None:
    with pytest.raises(ValueError, match="unknown library"):
        fit_fold(arm_slice, "rep999")


# ------------------------------------------------------------------ likelihood


def test_an_unmeasured_arm_has_no_log_composition() -> None:
    """The zero-panel doctrine: a zero total is missing, never a vector of zeros."""

    with pytest.raises(ValueError, match="not measured"):
        log_composition(np.zeros(4, dtype=np.int64))


def test_technical_variance_is_evaluated_at_the_pooled_rate() -> None:
    """``lambda/(lambda + 1/2)^2 - 1/(n + G/2)`` at a pooled ``lambda``, not at the arm's own count.

    The property that matters is the one ADR 0022 was written for: **the variance depends on the
    pooled rate and not at all on what this particular arm observed.**  The old form could not
    satisfy that -- it read ``y`` directly -- and it is why a zero count claimed a variance of 2.0
    however small its true rate was.
    """

    rate = np.array([1e-8, 1e-5, 1e-3, 1e-1], dtype=np.float64)
    depth = 1e5
    variance = technical_variance(rate, depth)
    expected_counts = depth * rate
    expected = expected_counts / (expected_counts + 0.5) ** 2 - 1.0 / (depth + rate.shape[0] / 2.0)
    assert np.allclose(variance, np.clip(expected, 0.0, None))

    # The correction that matters: at a vanishing rate the variance goes to ZERO, because a gene
    # that is never detected produces the same log-composition every time.  The old plug-in put its
    # MAXIMUM there -- a flat 2.0 -- and that single entry carried 79.8% of the panel's claimed
    # technical mass.
    # 0.0040 here against the flat 2.0 the old plug-in returned for the same gene: 500x.
    assert variance[0] < 0.01, "a vanishing rate must carry vanishing sampling variance"
    assert variance[0] < variance[1] < 0.5, "variance rises toward the lambda = 1/2 peak"
    assert variance[1] > variance[2] > variance[3], "and falls again as the expected count grows"

    # Deeper sequencing lowers sampling noise for any gene past that peak.
    assert np.all(technical_variance(rate, 10 * depth)[2:] < variance[2:])


def test_stabilize_removes_negative_eigenvalues() -> None:
    matrix = np.array([[1.0, 2.0], [2.0, 1.0]])
    clipped = stabilize(matrix)
    assert np.all(np.linalg.eigvalsh(clipped) >= -1e-12)
    assert np.allclose(clipped, clipped.T)


def test_the_posterior_tightens_as_depth_grows(fold: FittedFold) -> None:
    """Spread has to respond to evidence, or it is a declared number rather than a posterior."""

    design = fold.design(NULL_TARGET)
    precision = fold.prior_precision()
    widths = []
    for scale in (1, 100):
        counts = np.full(100, 10 * scale, dtype=np.int64)
        composition, depth = log_composition(counts)
        _, covariance = posterior(
            composition,
            intercept=fold.intercept,
            design=design,
            prior_precision=precision,
            observation_variance_diagonal=technical_variance(fold.pooled_rate, depth),
        )
        widths.append(float(np.trace(covariance)))
    assert widths[1] < widths[0]


def test_a_nonpositive_observation_variance_is_refused(fold: FittedFold) -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        posterior(
            np.zeros(100),
            intercept=fold.intercept,
            design=fold.design(NULL_TARGET),
            prior_precision=fold.prior_precision(),
            observation_variance_diagonal=np.zeros(100),
        )


# ------------------------------------------------------------------ the belief


def test_a_belief_is_emitted_from_real_cells(
    arm_slice: ArmSlice, estimator: GSE274113ObservationEstimator
) -> None:
    """The headline: real human cells in, a typed belief with a posterior out."""

    request = _request(arm_slice, estimator, HELD_OUT, "GATA1")
    belief = estimate_cell_state(request, estimator=estimator)

    assert belief.subject.subject_id == f"gse274113:{HELD_OUT}:GATA1"
    assert belief.subject.experimental_unit_kind == "library"
    assert estimator.descriptor.artifact_kind is ModelArtifactKind.EMPIRICAL_OBSERVATION_MODEL
    assert CellStateBelief.model_validate_json(belief.model_dump_json()) == belief


def test_every_block_is_an_exact_marginal_of_the_joint(
    arm_slice: ArmSlice, estimator: GSE274113ObservationEstimator
) -> None:
    """The pieces are slices of one covariance, so they cannot drift from the whole."""

    belief = estimate_cell_state(
        _request(arm_slice, estimator, HELD_OUT, "TAL1"), estimator=estimator
    )
    joint_mean = np.array(belief.joint_posterior.mean)
    joint_dimensions = list(belief.joint_posterior.dimensions)

    for block in (belief.factors[0].posterior, belief.nuisance.posterior):
        offset = joint_dimensions.index(block.dimensions[0])
        width = len(block.dimensions)
        assert np.allclose(block.mean, joint_mean[offset : offset + width], atol=1e-12)

    realization = belief.intervention_realizations[0].posterior
    assert np.allclose(
        realization.mean[0],
        joint_mean[joint_dimensions.index(realization.dimensions[0])],
        atol=1e-12,
    )


def test_the_belief_declares_what_this_evidence_cannot_support(
    arm_slice: ArmSlice, estimator: GSE274113ObservationEstimator
) -> None:
    """S1, S3 and S7 are unreachable here, and the belief says so in fields, not prose."""

    belief = estimate_cell_state(
        _request(arm_slice, estimator, HELD_OUT, NULL_TARGET), estimator=estimator
    )

    assert isinstance(belief.dynamics.velocity, UnavailableDistribution)
    assert belief.diagnostics.sufficiency.outcome is CriterionOutcome.NOT_EVALUATED
    assert belief.diagnostics.causal_support.causal_status is CausalStatus.UNSUPPORTED
    assert belief.readiness.abstention_required is True
    assert belief.readiness.valid_for_prediction is False
    assert belief.readiness.valid_for_control is False
    assert belief.readiness.reasons

    # Every readiness criterion this backend reports PASSED must be backed by a diagnostics report
    # that was actually EVALUATED.  This fired twice before ADR 0021's gate was repaired: `support`
    # said PASSED on a hardcoded SupportReport, and `measurement_model` said PASSED with no
    # diagnostics counterpart to contradict it -- `coherent_contract` cross-checks the other six
    # criteria and cannot reach that one.  A belief reading "abstention required, not valid for
    # prediction, not valid for control -- but valid for measurement selection", from a query
    # declaring no assays, is what that bought.
    reports = {
        "support": belief.diagnostics.support,
        "sufficiency": belief.diagnostics.sufficiency,
        "identifiability": belief.diagnostics.identifiability,
        "decision_uncertainty": belief.diagnostics.decision_uncertainty,
        "calibration": belief.diagnostics.calibration,
        "causal": belief.diagnostics.causal_support,
    }
    for name, report in reports.items():
        if getattr(belief.readiness, name) is CriterionOutcome.PASSED:
            assert report.evaluation_status is EvaluationStatus.EVALUATED, (
                f"readiness.{name} claims PASSED while its diagnostics report was never evaluated"
            )

    # `measurement_model` is the criterion with NO diagnostics counterpart, so the loop above cannot
    # reach it and it has to be asserted directly.  Until a measurement-model report exists to be
    # cross-checked, this backend cannot honestly claim it.
    assert belief.readiness.measurement_model is CriterionOutcome.NOT_EVALUATED
    assert belief.readiness.valid_for_measurement_selection is False

    # The support report declines to invent scores rather than asserting perfect ones.
    assert belief.diagnostics.support.evaluation_status is EvaluationStatus.NOT_EVALUATED
    assert belief.diagnostics.support.in_distribution_score is None
    assert belief.diagnostics.support.ood_score is None
    assert belief.diagnostics.support.notes


def test_the_support_report_is_not_the_same_certificate_for_impossible_data(
    arm_slice: ArmSlice, estimator: GSE274113ObservationEstimator
) -> None:
    """A support verdict that ignores its input is not a measurement of support.

    Before the repair this report read EVALUATED / PASSED / ``in_distribution_score=1.0`` /
    ``ood_score=0.0``, and a composition no CD34+ progenitor could produce -- every one of the 100
    panel genes at an identical rate -- received the identical certificate as a real held-out arm.

    The repair is to stop claiming it, not to invent a score: a real one needs a decided estimand
    and a threshold predeclared before the measurement, and this query's ``maximum_ood_score`` of
    0.99 would leave even a correctly computed score unable to fail.  So what this test pins is
    that the report is honest, and that it stays honest for impossible input.
    """

    composition, depth = arm_slice.log_composition(HELD_OUT, "GATA1")
    cells = arm_slice.cells[(HELD_OUT, "GATA1")]
    real = tuple(float(value) for value in composition)
    # Every panel gene at an identical rate: housekeeping anchors and lineage markers alike.
    uniform = tuple(math.log(1.0 / len(real)) for _ in real)
    # And the real arm reflected about its own mean, so the most expressed genes become the least.
    centre = sum(real) / len(real)
    inverted = tuple(2.0 * centre - value for value in real)

    verdicts = set()
    for panel in (real, uniform, inverted):
        request = arm_request(
            HELD_OUT,
            "GATA1",
            query=estimator._query,
            log_composition=panel,
            cells=cells,
            panel_total=int(depth),
        )
        support = estimate_cell_state(request, estimator=estimator).diagnostics.support
        assert support.evaluation_status is EvaluationStatus.NOT_EVALUATED
        assert support.outcome is CriterionOutcome.NOT_EVALUATED
        assert support.abstention_required is True
        verdicts.add((support.in_distribution_score, support.ood_score))

    # The three verdicts are still identical -- that is not the defect and never was.  The defect
    # was claiming EVALUATED and a perfect score while being identical.  Declining to score is a
    # defensible constant; a fabricated measurement is not.
    assert verdicts == {(None, None)}


def test_uncertainty_separates_measurement_from_biology(
    arm_slice: ArmSlice, estimator: GSE274113ObservationEstimator
) -> None:
    belief = estimate_cell_state(
        _request(arm_slice, estimator, HELD_OUT, "MYB"), estimator=estimator
    )
    components = {component.kind.value: component for component in belief.uncertainty.components}
    assert components["measurement"].magnitude is not None
    assert components["biological_stochasticity"].magnitude is not None
    for unsupported in ("parameter", "model", "counterfactual"):
        assert components[unsupported].magnitude is None


def test_every_arm_of_the_held_out_library_emits_a_belief(
    arm_slice: ArmSlice, estimator: GSE274113ObservationEstimator
) -> None:
    for target in arm_slice.targets:
        belief = estimate_cell_state(
            _request(arm_slice, estimator, HELD_OUT, target), estimator=estimator
        )
        assert belief.subject.subject_id.endswith(target)


# ----------------------------------------------------- guards that can fire


def test_an_in_fold_library_is_refused(
    arm_slice: ArmSlice, estimator: GSE274113ObservationEstimator
) -> None:
    """The leakage guard.  ``rep2`` is inside this fold's fit set, so a belief for it would be
    an in-sample estimate wearing the authority of a held-out one."""

    request = _request(arm_slice, estimator, "rep2", "GATA1")
    with pytest.raises(CapabilityError, match="in-sample"):
        estimate_cell_state(request, estimator=estimator)


def test_the_causal_gate_refuses_an_identified_claim(
    arm_slice: ArmSlice, estimator: GSE274113ObservationEstimator
) -> None:
    """ADR 0021 decision 3, enforced in code rather than by convention.

    Without this the new artifact kind would be a route around the admission registry.
    """

    from cellstate.domain.belief import (
        CausalEstimandBinding,
        CausalSupportReport,
        EvaluationStatus,
    )
    from cellstate.domain.common import OntologyTerm
    from cellstate.domain.events import AssignmentMechanism, InterventionEvent
    from cellstate.domain.subjects import AggregationStatistic, SubjectKind, TargetAggregation

    class OverclaimingEstimator(GSE274113ObservationEstimator):
        """Produces a STRUCTURALLY VALID belief that claims an identified population effect.

        The domain model already refuses a bare identified claim -- it demands a typed estimand, a
        randomized design, scopes, evidence, provenance agreement and matching readiness.  All of
        that is satisfied here on purpose, so what the test exercises is ADR 0021's gate and not one
        of the layers beneath it.  Without that gate, this belief would be returned.
        """

        def estimate(self, request, *, options=None):  # type: ignore[no-untyped-def]
            belief = super().estimate(request, options=options)
            guide = next(
                event for event in request.history.events if isinstance(event, InterventionEvent)
            )
            identified = CausalSupportReport(
                evaluation_status=EvaluationStatus.EVALUATED,
                outcome=CriterionOutcome.PASSED,
                causal_status=CausalStatus.IDENTIFIED_POPULATION_EFFECT,
                identification_basis="pooled CRISPRi transduction",
                identification_design=AssignmentMechanism.RANDOMIZED,
                source_scope=f"GSE274113 {HELD_OUT}",
                target_scope=f"GSE274113 {HELD_OUT}",
                # Identified support must cite EXTERNAL validation artifacts; the contract
                # refuses ordinary history events here, so the descriptor's validation evidence
                # is what an overclaiming model would have to point at.
                evidence_ids=belief.provenance.validation_evidence_ids,
                evidence_fingerprints=dict(belief.provenance.validation_evidence_fingerprints),
                estimands=(
                    CausalEstimandBinding(
                        target=OntologyTerm(label="panel log composition"),
                        horizon_name="now",
                        aggregation=TargetAggregation(
                            subject_kind=SubjectKind.POPULATION,
                            statistic=AggregationStatistic.DISTRIBUTION,
                            experimental_unit="library",
                        ),
                        intervention_spec_ids=(guide.intervention_spec_id,),
                        comparator="NT",
                    ),
                ),
            )
            return belief.model_copy(
                update={
                    "diagnostics": belief.diagnostics.model_copy(
                        update={"causal_support": identified}
                    ),
                    "readiness": belief.readiness.model_copy(
                        update={"causal": CriterionOutcome.PASSED}
                    ),
                }
            )

    overclaiming = OverclaimingEstimator(
        estimator._fold,
        query=estimator._query,
        slice_fingerprint=estimator._slice_fingerprint,
        panel_fingerprint=estimator._panel_fingerprint,
        calibration=estimator._calibration,
    )
    request = _request(arm_slice, overclaiming, HELD_OUT, "GATA1")
    with pytest.raises(CapabilityError, match="identified or transported"):
        estimate_cell_state(request, estimator=overclaiming)


def test_an_arm_without_exactly_one_guide_is_refused(
    arm_slice: ArmSlice, estimator: GSE274113ObservationEstimator
) -> None:
    request = _request(arm_slice, estimator, HELD_OUT, "GATA1")
    # CellHistory orders interventions before observations, so select by type rather than by
    # position -- slicing would silently drop the observation instead of the guide.
    from cellstate.domain.events import ObservationEvent

    without_guide = tuple(
        event for event in request.history.events if isinstance(event, ObservationEvent)
    )
    stripped = request.model_copy(
        update={"history": request.history.model_copy(update={"events": without_guide})}
    )
    with pytest.raises(CapabilityError, match="exactly one active"):
        estimate_cell_state(stripped, estimator=estimator)


# ------------------------------------------------------- capability measurements


def test_capability_measurements_are_computed_and_can_fail(arm_slice: ArmSlice) -> None:
    """The measurements exist, are grouped at the library, and are not rigged to pass.

    Rule 10: a phase gate that is a measurement is passed by *producing* the measurement.  These
    currently report negative, and that is a result rather than a defect -- what would be a defect
    is a measurement that could only ever come out favourable.
    """

    from cellstate.evaluation.gse274113_reports import (
        held_out_states,
        measure_earned_spread,
        measure_intervention_response,
        measure_nuisance_separation,
        measure_point_predictor_spread,
    )

    states = held_out_states(arm_slice)
    assert len(states) == 280

    null, non_null = measure_intervention_response(arm_slice)
    spread = measure_earned_spread(arm_slice)
    separation = measure_nuisance_separation(states, bound=0.35)

    for measurement in (null, non_null, spread, separation):
        assert measurement.unit_count == 14, "intervals must be grouped at the library"
        assert measurement.interval.lower <= measurement.interval.upper
        assert measurement.statement
        assert isinstance(measurement.passed, bool)

    # The four measured values are PINNED.  docs/roadmap.md and the model card both quote them, and
    # before this they were guarded by three assertions that restated the implementation
    # (`assert separation.passed is (separation.interval.upper <= 0.35)`) and so could not fail for
    # any input.  Editing BIOLOGY_RANK or NUISANCE_RANK in fit.py, or anything else that moves the
    # fitted subspaces, must fail here rather than leave four documents silently wrong.
    #
    # The tolerance is relative, loose enough for BLAS variation across platforms and far tighter
    # than the drift being guarded against: sweeping the nuisance rank moves S5 across 11.30 to
    # 74.52, and every value below is deterministic given the committed slice (`held_out_states`
    # takes no randomness, and the bootstrap is seeded).
    assert separation.value == pytest.approx(10.36468, rel=1e-3)
    assert separation.interval.lower == pytest.approx(6.26717, rel=1e-3)
    assert separation.interval.upper == pytest.approx(16.65935, rel=1e-3)

    # S2 is the split-half replicate on NT decided by ADR 0023, NOT the library-blind point
    # predictor that reported 0.27806 before it.  The estimand changed; the verdict did not.
    assert spread.value == pytest.approx(0.84148, rel=1e-3)
    assert spread.interval.lower == pytest.approx(0.71437, rel=1e-3)
    assert spread.interval.upper == pytest.approx(0.97593, rel=1e-3)

    assert null.value == pytest.approx(2.02578, rel=1e-3)
    assert null.interval.lower == pytest.approx(1.43838, rel=1e-3)
    assert null.interval.upper == pytest.approx(2.67070, rel=1e-3)

    assert non_null.value == pytest.approx(2.09008, rel=1e-3)
    assert non_null.interval.lower == pytest.approx(1.61546, rel=1e-3)
    assert non_null.interval.upper == pytest.approx(2.56195, rel=1e-3)

    # All four verdicts are negative on the committed evidence, and that is the recorded result.
    assert not separation.passed
    assert not spread.passed
    assert not null.passed
    assert not non_null.passed

    # The verdict is read off the interval and BOTH branches are reachable.  A bound above the
    # measured upper end passes and one below it fails, so `passed` is not a constant -- which is
    # the property the replaced tautologies were reaching for but could not demonstrate.
    assert measure_nuisance_separation(states, bound=17.0).passed is True
    assert measure_nuisance_separation(states, bound=16.0).passed is False

    # ADR 0023 decision 3: the superseded construction stays reported, so the change of estimand is
    # visible in the record rather than looking like a number that moved on its own.  Both are
    # pinned, and they must stay DISTINCT -- if a future edit collapses one into the other, the
    # diagnostic stops diagnosing anything and this fails.
    diagnostic = measure_point_predictor_spread(states)
    assert diagnostic.value == pytest.approx(0.30205, rel=1e-3)
    assert diagnostic.interval.lower == pytest.approx(0.22527, rel=1e-3)
    assert diagnostic.interval.upper == pytest.approx(0.37581, rel=1e-3)
    assert not diagnostic.passed
    assert diagnostic.value < spread.value, (
        "the library-blind predictor is handicapped twice over and must read more pessimistic "
        "than the split-half replicate; equal values mean the estimand change was not applied"
    )

    # ADR 0023 decision 2: both sides aggregate as a root-mean-square.  The superseded form paired a
    # MEAN of per-gene standard deviations against an RMS residual, which understates the numerator
    # by Jensen and reported 0.27806 for this same construction.  Pinning the RMS value is what
    # keeps a revert to the mixed aggregation from passing silently.
    assert diagnostic.value != pytest.approx(0.27806, rel=1e-3)

    # One replicate per library, so S2 resolves to 14 units rather than 280 arms.  This is the
    # scope limit ADR 0023 decision 5 declares, asserted rather than left to the model card.
    assert spread.unit_count == 14
    assert len(arm_slice.libraries) == 14


def test_the_s5_block_decomposition_is_computed_rather_than_asserted(arm_slice: ArmSlice) -> None:
    """The decomposition that overturned S5's first diagnosis, computed from the shipped path.

    ``docs/roadmap.md`` and the model card both rest their S5 reading on three numbers -- 79.27 in
    the nuisance block, 3.07 in biology, 0.257 between targets -- and until this test existed **no
    committed code computed any of them.**  They were a recorded claim, which is exactly the failure
    mode this repository keeps finding in its own work.

    The reading they support is the one that matters: the obvious diagnosis, that the nuisance basis
    fails to absorb the library, is **wrong**.  The basis absorbs roughly 134x more than leaks past
    it.  What fails is the denominator -- between-target variance of 0.109, smaller than the 0.609
    of residual library variation the biology block carries -- so the signal S5 needs is weak rather
    than the separation being broken.  Raising the nuisance rank does not repair that.

    **ADR 0022 moved both terms, and reporting the ratio alone would have hidden it.**  Under the
    old technical variance these read 79.27 / 3.07 / 0.257, ratio 25.8.  Evaluating the technical
    term at a pooled rate cut the library variation leaking into the biology block **5x**
    (3.07 -> 0.609) -- a large, real improvement in exactly the separation S5 names -- while the
    between-target signal fell **2.4x** with it (0.257 -> 0.109).  S5 itself improved only from
    19.22 to 10.36 and still fails by a factor of thirty.  A failing ratio names two suspects, and
    here the numerator was the one that got better.

    Both quantities are means-first, and that is not incidental: the across-library figure is the
    variance of the fourteen per-library mean coefficient vectors, and the between-target figure is
    the variance of the per-target means.  Comparing a means-first quantity against a per-item one
    would be comparing two different estimators and would put the ratio out by a third.
    """

    folds = {library: fit_fold(arm_slice, library) for library in arm_slice.libraries}
    biology: dict[tuple[str, str], np.ndarray] = {}
    nuisance: dict[tuple[str, str], np.ndarray] = {}

    for library in arm_slice.libraries:
        fold = folds[library]
        rank = fold.biology_basis.shape[1]
        width = fold.nuisance_basis.shape[1]
        for target in arm_slice.targets:
            composition, depth = arm_slice.log_composition(library, target)
            design = fold.design(target)
            mean, _ = posterior(
                composition,
                intercept=fold.intercept,
                design=design,
                prior_precision=fold.prior_precision(),
                observation_variance_diagonal=fold.observation_variance(depth),
            )
            biology[(library, target)] = np.asarray(mean[:rank], dtype=np.float64)
            nuisance[(library, target)] = np.asarray(mean[rank : rank + width], dtype=np.float64)

    libraries = list(arm_slice.libraries)
    targets = list(arm_slice.targets)

    def across_library(block: dict[tuple[str, str], np.ndarray]) -> float:
        means = np.stack([np.mean([block[(lib, t)] for t in targets], axis=0) for lib in libraries])
        return float(np.mean(np.var(means, axis=0)))

    biology_means = np.stack(
        [np.mean([biology[(lib, t)] for lib in libraries], axis=0) for t in targets]
    )
    between_target = float(np.mean(np.var(biology_means, axis=0)))
    nuisance_across = across_library(nuisance)
    biology_across = across_library(biology)

    assert nuisance_across == pytest.approx(81.30157, rel=1e-3)
    assert biology_across == pytest.approx(0.60868, rel=1e-3)
    assert between_target == pytest.approx(0.10910, rel=1e-3)

    # The claim the roadmap makes, restated as a relation rather than as three loose constants.
    assert nuisance_across / biology_across == pytest.approx(133.57, rel=1e-3)
    assert between_target < biology_across, (
        "the S5 denominator must be the smaller quantity; if this flips, the roadmap's "
        "'the signal is weak, not the separation broken' reading no longer follows"
    )


def test_the_three_decomposition_terms_pool_coefficients_across_fourteen_bases(
    arm_slice: ArmSlice,
) -> None:
    """The decomposition above compares coordinates from fourteen *different* fitted bases.

    ``compare_arms`` refuses two arms from different libraries, and its docstring gives the reason:
    a belief about library *L* comes from the fold that excluded *L*, so arms in different libraries
    are expressed in different fitted bases and their coordinates are not comparable.  The signature
    makes the incomparable case impossible to write.

    :func:`measure_nuisance_separation` and the decomposition above do exactly that comparison as
    their core operation -- they average and difference ``ArmState.biology`` across all fourteen
    folds.  ``gse274113_reports._biology`` says "``W`` is shared across every target, so biology
    coefficients ARE comparable between arms", which is true *within* a fold and is the only sense
    in which it is true.

    The size of the problem is measured here rather than argued, and it is not small.
    ``_canonical_signs`` fixes each column's sign by its largest-magnitude entry, but that entry's
    **identity** changes between folds: ``biology_0``'s top-loading gene is ``MPO`` in seven folds
    and ``CD79A`` in the other seven, and those are opposite poles of the same axis, so the
    convention pins the wrong thing and the whole column flips.

    Why this is recorded and not repaired: aligning the bases moves S5 from 10.36 to **9.23**
    (-11%) and the between-target signal from 0.109 to **0.129** (+18%) -- three of the four pinned
    numbers, and no verdict.  S5 still fails by a factor of twenty-six and the ledger stays 0 of 10.
    Changing how a capability is computed is an ADR's decision, not a test's, so this test states
    the defect and its magnitude and leaves the estimand where the merged ADRs put it.
    """

    import itertools

    bases = {library: fit_fold(arm_slice, library).biology_basis for library in arm_slice.libraries}
    pairs = list(itertools.combinations(arm_slice.libraries, 2))
    assert len(pairs) == 91

    # Column-wise agreement: does "biology_k" name the same direction in every fold?
    flips = []
    for axis in range(4):
        cosines = [float(bases[a][:, axis] @ bases[b][:, axis]) for a, b in pairs]
        flips.append(sum(1 for value in cosines if value < 0.0))

    assert flips[0] > 40, (
        "biology_0 reverses sign between folds in roughly half of the 91 pairs; a coefficient "
        "of +2 in one fold is -2 in another, and S5 averages them together"
    )

    # The mechanism, named concretely so a future sign convention can be checked against it.
    top_gene = {
        library: arm_slice.gene_symbols[int(np.argmax(np.abs(basis[:, 0])))]
        for library, basis in bases.items()
    }
    assert set(top_gene.values()) == {"MPO", "CD79A"}, (
        "biology_0's largest-magnitude entry is what _canonical_signs keys the sign off; it "
        "alternates between two genes at OPPOSITE poles of the axis"
    )

    # Axes 2 and 3 sit on near-degenerate singular values, so they are not individually identified.
    for axis in (2, 3):
        cosines = [abs(float(bases[a][:, axis] @ bases[b][:, axis])) for a, b in pairs]
        assert min(cosines) < 0.05, (
            f"biology_{axis} is near-orthogonal to itself across some fold pair; which direction "
            "receives this name is close to arbitrary and the loadings printed for it are too"
        )


def test_the_s2_depth_caveat_reproduces(arm_slice: ArmSlice) -> None:
    """The depth sensitivity ADR 0023 decision 4 rests on, computed instead of quoted.

    Decision 4 discharges S2's single biggest structural handicap: the split-half arms are shallower
    than a full arm, and lower depth inflates both the claimed spread and the realized error.  Its
    own standard for that is quoted in the ADR -- "a caveat that is only ever stated is a caveat
    nobody has checked" -- and neither of its two figures reproduced.

    * The shortfall is **exactly 2.00x**, because ``NT`` is the bitwise sum of ``NT_A`` and
      ``NT_B``.  The ADR says 2.06x, which is ``NT_B``'s depth alone quoted as the mean over halves.
    * Recomputing at full-arm depth gives **0.7727**, an 8.2% move.  The ADR says 0.8322, 1.1%.
      0.8322 is what a 1.1x depth multiplier returns, not the 2.0x the sentence names.

    The conclusion is unchanged and stronger: the move is in the direction that makes S2's failure
    *worse*, by seven times the stated margin.
    """

    from cellstate.backends.gse274113.likelihood import posterior
    from cellstate.evaluation.gse274113_reports import measure_earned_spread

    # The halves are exactly half the arm, which is what makes the shortfall exact rather than
    # empirical -- `scripts/gse274113_build_slice.py` splits the NT cells, it does not resample.
    for library in arm_slice.libraries:
        halves = arm_slice.counts[(library, "NT_A")] + arm_slice.counts[(library, "NT_B")]
        assert np.array_equal(arm_slice.counts[(library, NULL_TARGET)], halves)

    mean_half = float(
        np.mean(
            [
                arm_slice.counts[(library, half)].sum()
                for library in arm_slice.libraries
                for half in PLACEBO_TARGETS
            ]
        )
    )
    mean_full = float(
        np.mean([arm_slice.counts[(library, NULL_TARGET)].sum() for library in arm_slice.libraries])
    )
    assert mean_full / mean_half == pytest.approx(2.0, abs=1e-12)
    assert mean_half == pytest.approx(684370.0, rel=1e-4)

    # S2 recomputed with the technical term at each library's own full NT depth on both sides.
    ratios = []
    for library in arm_slice.libraries:
        fitted = fit_fold(arm_slice, library)
        design = fitted.design(NULL_TARGET)
        depth = float(arm_slice.counts[(library, NULL_TARGET)].sum())
        variance = fitted.observation_variance(depth)
        inferred_from, _ = arm_slice.log_composition(library, PLACEBO_TARGETS[0])
        predicted_for, _ = arm_slice.log_composition(library, PLACEBO_TARGETS[1])
        mean, covariance = posterior(
            inferred_from,
            intercept=fitted.intercept,
            design=design,
            prior_precision=fitted.prior_precision(),
            observation_variance_diagonal=variance,
        )
        predictive = np.einsum("gi,ij,gj->g", design, covariance, design) + variance
        residual = predicted_for - fitted.intercept - design @ mean
        ratios.append(float(np.sqrt(np.mean(predictive)) / np.sqrt(np.mean(residual**2))))

    shipped = measure_earned_spread(arm_slice).value
    at_full_depth = float(np.mean(ratios))
    assert shipped == pytest.approx(0.84148, rel=1e-3)
    assert at_full_depth == pytest.approx(0.77270, rel=1e-3)
    assert at_full_depth < shipped, (
        "the depth correction must move S2 DOWN; if it ever moves up, depth becomes a candidate "
        "explanation for the failure and ADR 0023 decision 4 needs revisiting"
    )


def test_the_fitted_dispersion_carries_a_measured_degrees_of_freedom_bias(
    arm_slice: ArmSlice,
) -> None:
    """``psi^2``'s scalar ``G/(G-df)`` correction assumes a homoscedasticity the panel lacks.

    ``fit.py`` fits the dispersion from an *unweighted* least-squares residual and rescales it
    by a single ``panel_size / (panel_size - degrees_of_freedom)`` factor.  That factor is exact
    only if the residual is homoscedastic.  Here ``Omega = technical + psi^2`` spans four orders of
    magnitude across the panel, and for an OLS fit ``E[sum r^2] = sum_j (1 - h_jj) omega_j`` -- so
    the scalar form is right only if leverage is uncorrelated with variance.  It is not: the design
    columns load most heavily on exactly the low-count, high-variance genes.

    Measured here rather than argued.  This is **recorded, not repaired**: correcting it moves
    ``psi^2`` up about 9% and S2 from 0.8415 to roughly 0.86, which changes a published capability
    measurement, and ADR 0022 is the precedent for how a change to this term gets made -- by an ADR,
    not by a test.  Note also that ADR 0022's repair is intact and unrelated: 0 of 14 folds reach
    ``DISPERSION_FLOOR``, so this is a bias in the estimator that replaced the clamp, not the clamp.
    """

    from cellstate.backends.gse274113.likelihood import technical_variance

    correlations: list[float] = []
    ratios: list[float] = []
    for library in arm_slice.libraries:
        fitted = fit_fold(arm_slice, library)
        assert not fitted.dispersion_is_clamped

        residual_sum = leverage_weighted_technical = leverage_weight = 0.0
        for fit_library in fitted.fit_library_ids:
            for target in arm_slice.targets:
                composition, depth = arm_slice.log_composition(fit_library, target)
                design = fitted.design(target)
                coefficients, *_ = np.linalg.lstsq(
                    design, composition - fitted.intercept, rcond=None
                )
                residual = composition - fitted.intercept - design @ coefficients
                leverage = np.einsum(
                    "gi,ij,gj->g", design, np.linalg.pinv(design.T @ design), design
                )
                technical = technical_variance(fitted.pooled_rate, depth)
                residual_sum += float((residual**2).sum())
                leverage_weighted_technical += float(((1.0 - leverage) * technical).sum())
                leverage_weight += float((1.0 - leverage).sum())
                correlations.append(float(np.corrcoef(leverage, technical)[0, 1]))

        unbiased = (residual_sum - leverage_weighted_technical) / leverage_weight
        ratios.append(fitted.biological_observation_variance / unbiased)

    # Positive in every one of the 3,640 arm fits, and averaging around 0.42.  It is the sign and
    # the consistency that create the bias, not the magnitude on any single arm.
    assert min(correlations) > 0.0
    assert float(np.mean(correlations)) > 0.3, (
        "leverage and technical variance must be correlated for this bias to exist; if they "
        "decorrelate, the scalar correction becomes adequate and this test should be retired"
    )
    assert min(ratios) > 0.87 and max(ratios) < 0.94, (
        "the shipped psi^2 sits 6-13% below the leverage-weighted moment estimator in every fold"
    )
    assert float(np.mean(ratios)) == pytest.approx(0.916, abs=0.01)


def test_a_state_is_never_estimated_from_its_own_library(arm_slice: ArmSlice) -> None:
    """Leave-one-library-out, asserted on the estimate itself rather than on the fold alone."""

    from cellstate.backends.gse274113.fit import fit_fold

    for library in arm_slice.libraries:
        assert library not in fit_fold(arm_slice, library).fit_library_ids


# ------------------------------------------------------------ the usable surface


def test_one_call_estimates_an_arm_from_the_fold_that_excludes_it() -> None:
    """``estimate_arm`` replaces roughly forty lines, and it may not replace the fold discipline.

    The two-pass fingerprint construction it hides is real -- the query commits to the model that
    answers it, so the query cannot be built before the model's fingerprint exists -- but hiding it
    must not also hide *which* fold answers.  A belief about library ``L`` comes only from the fold
    that excluded ``L``, and that is asserted here rather than trusted to the wrapper.
    """

    from cellstate.backends.gse274113 import estimate_arm
    from cellstate.backends.gse274113.usage import _fold_and_estimator

    belief = estimate_arm(HELD_OUT, "GATA1")
    assert belief.subject.subject_id == f"gse274113:{HELD_OUT}:GATA1"
    assert belief.readiness.abstention_required is True

    fold, _ = _fold_and_estimator(HELD_OUT)
    assert HELD_OUT not in fold.fit_library_ids
    assert len(fold.fit_library_ids) == 13


def test_the_convenience_surface_refuses_what_it_cannot_answer() -> None:
    """Both refusals are exercised from the side that fails; neither is a comment."""

    from cellstate.backends.gse274113 import compare_arms, estimate_arm

    with pytest.raises(ValueError, match="unknown library"):
        estimate_arm("no-such-library", "GATA1")
    with pytest.raises(ValueError, match="unknown arm"):
        estimate_arm(HELD_OUT, "NOT_A_TARGET")
    with pytest.raises(ValueError, match="against itself"):
        compare_arms(HELD_OUT, "GATA1", "GATA1")


def test_the_readout_names_axes_by_genes_and_never_by_a_label() -> None:
    """The biology block becomes readable, and stays a loading rather than an interpretation.

    ``biology_0`` loading MPO, ELANE and AZU1 against CD79A and PPBP is granulocyte biology against
    B-cell and platelet biology, and a reader can see that.  What must NOT appear is a field saying
    so: an asserted label is a claim no measurement backs, which is the defect this repository keeps
    removing from its own reports.
    """

    from cellstate.backends.gse274113 import describe_state, estimate_arm

    description = describe_state(estimate_arm(HELD_OUT, "GATA1"))
    assert description.library == HELD_OUT
    assert description.target == "GATA1"
    assert [axis.name for axis in description.axes] == [f"biology_{index}" for index in range(4)]

    # No axis carries a name for what it means -- only the genes and the coordinate.
    for axis in description.axes:
        assert not hasattr(axis, "label")
        assert len(axis.top_genes) == 6
        assert axis.standard_deviation > 0.0
        loadings = [abs(gene.loading) for gene in axis.top_genes]
        assert loadings == sorted(loadings, reverse=True), "top genes must be ranked by |loading|"

    # The abstention is reprinted, not smoothed away by the convenience layer.
    assert description.abstention_required is True
    assert description.reasons
    assert "ABSTENTION REQUIRED" in str(description)

    # Pinned: the leading axis is a real, recognisable contrast.  This fires if the fitted basis
    # drifts, which is exactly when the printed interpretation would silently stop being true.
    leading = description.axes[0]
    assert leading.top_genes[0].symbol == "MPO"
    assert {gene.symbol for gene in leading.top_genes} >= {"MPO", "ELANE", "CD79A"}


def test_the_readout_refuses_a_belief_from_another_backend() -> None:
    """A description is only meaningful in the basis the belief was scored under."""

    from cellstate.backends.gse274113 import describe_state, estimate_arm

    belief = estimate_arm(HELD_OUT, "GATA1")
    foreign = belief.model_copy(
        update={"subject": belief.subject.model_copy(update={"subject_id": "somewhere-else:x:y"})}
    )
    with pytest.raises(ValueError, match="did not come from this backend"):
        describe_state(foreign)


def test_the_unexpressed_target_ordering_does_not_replicate_across_libraries() -> None:
    """The ordering that reads as a sanity check in ``rep1`` is a property of ``rep1``.

    An earlier revision of this test asserted, in ``HELD_OUT`` alone, that GATA1's contrast exceeds
    SNAI2's and that SNAI2 -- measured at 0.6 CPM, so not expressed and incapable of being knocked
    down -- stays under its own understated noise floor.  Both hold in ``rep1``.  Neither is a
    property of the backend, and the population the assertions ranged over was a one-element
    hand-written literal.

    Measured over all fourteen libraries:

    * ``GATA1 > SNAI2`` fails in **4 of 14** (``rep3``, ``rep5``, ``rep9``, ``rep10``).  In ``rep3``
      SNAI2 produces the largest contrast of all nineteen targets.
    * ``SNAI2`` stays under its own floor in only **5 of 14**.  In ``rep5`` it clears that floor by
      3.3x.

    This is recorded as a measurement rather than deleted, because it is the readout's own account
    of why the ledger is 0 of 10: the perturbation is a measured null (mean on-target log2FC about
    -0.06), the matrix ``W`` is fitted on has the singular-value profile of the placebo contrast,
    and an ordering read off one library is reading that noise.  A test that pinned the favourable
    library would have kept saying the surface separates a real knockdown from a dead one.
    """

    from cellstate.backends.gse274113 import compare_arms, load_arm_slice

    libraries = load_arm_slice().libraries
    ordered = [
        library
        for library in libraries
        if compare_arms(library, "NT", "GATA1").distance
        > compare_arms(library, "NT", "SNAI2").distance
    ]
    under_floor = [
        library
        for library in libraries
        if (contrast := compare_arms(library, "NT", "SNAI2")).distance
        < contrast.distance_lower_bound_sd
    ]

    assert len(libraries) == 14
    assert sorted(set(libraries) - set(ordered)) == ["rep10", "rep3", "rep5", "rep9"], (
        "the GATA1 > SNAI2 ordering holds in 10 of 14 libraries; if this set moved, the "
        "ordering's dependence on which library you look at has changed"
    )
    assert len(under_floor) == 5, (
        "an unexpressed target clears its own understated noise floor in 9 of 14 libraries; "
        "this is the readout failing its own sanity property off the library it was read on"
    )

    # The rep1 reading itself, kept so the contrast between one library and fourteen is visible.
    assert HELD_OUT in ordered and HELD_OUT in under_floor
    dead = compare_arms(HELD_OUT, "NT", "SNAI2")
    master = compare_arms(HELD_OUT, "NT", "GATA1")
    assert master.axis_names == dead.axis_names == tuple(f"biology_{i}" for i in range(4))
