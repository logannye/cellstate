"""The web UI serves the shipped numbers, and its controls are not inert.

Two properties matter more than route coverage here.

**The HTTP layer must not alter a measurement.** At the fitted configuration every endpoint has to
return the same values the model card publishes, so a reader cannot be shown a different number
merely because it arrived over a socket.

**The rank control must be able to change something.** A slider that renders but does not move the
result is the same defect as an inert ``do`` operator, and ADR 0019 warns that such a thing passes
every test a repository has unless one is written to catch it.
"""

from __future__ import annotations

import os

import pytest

if os.environ.get("CELLSTATE_REQUIRE_UI") == "1":
    # CI sets this. A suite that skips itself is a lane that cannot report, so where the extra is
    # supposed to be installed, its absence has to be a failure rather than a silent pass.
    import fastapi  # noqa: F401
else:
    pytest.importorskip("fastapi", reason="the web UI needs the 'ui' extra")

from fastapi.testclient import TestClient

from cellstate.backends.gse274113.fit import BIOLOGY_RANK, NUISANCE_RANK
from cellstate.ui.server import app

client = TestClient(app)

# Every route, with the arguments a page actually sends.
ROUTES = [
    "/api/inventory",
    "/api/panel",
    "/api/spectrum",
    "/api/knockdown",
    "/api/day",
    "/api/measure",
    "/api/ranks",
    "/api/basis",
    "/api/rank-response",
    "/api/arm/rep1/GATA1",
    "/api/library/rep1/states",
    "/api/fold/rep1",
    "/api/sweep/rep1",
]


@pytest.mark.parametrize("route", ROUTES)
def test_every_route_answers(route: str) -> None:
    response = client.get(route)
    assert response.status_code == 200, response.text
    assert response.json()


def test_the_page_and_its_assets_are_served() -> None:
    """A server that answers JSON and serves no page is not a UI."""

    assert client.get("/").status_code == 200
    for asset in ("app.js", "charts.js", "style.css", "index.html"):
        assert client.get(f"/static/{asset}").status_code == 200, asset


# --------------------------------------------------------------------- the numbers are unchanged


def test_the_default_ranks_reproduce_the_published_measurements() -> None:
    """Through the API, at the fitted configuration, these are the model card's numbers."""

    body = client.get("/api/measure").json()
    assert body["ranks"]["is_default"] is True
    by_name = {m["name"]: m for m in body["measurements"]}

    s2 = next(m for n, m in by_name.items() if n.startswith("S2"))
    s5 = next(m for n, m in by_name.items() if n.startswith("S5"))
    null_half = next(m for n, m in by_name.items() if n.startswith("S4 null"))
    non_null = next(m for n, m in by_name.items() if n.startswith("S4 non-null"))

    assert s2["value"] == pytest.approx(0.84148, rel=1e-3)
    assert s5["value"] == pytest.approx(10.3647, rel=1e-3)
    assert null_half["value"] == pytest.approx(2.0258, rel=1e-3)
    assert non_null["value"] == pytest.approx(2.0901, rel=1e-3)
    assert not any(m["passed"] for m in body["measurements"]), "the ledger is 0 of 10"

    decomposition = body["decomposition"]
    assert decomposition["nuisance_across_library"] == pytest.approx(81.30157, rel=1e-3)
    assert decomposition["biology_across_library"] == pytest.approx(0.60868, rel=1e-3)
    assert decomposition["between_target"] == pytest.approx(0.10910, rel=1e-3)


def test_the_substrate_screens_serve_the_figures_the_docs_quote() -> None:
    knockdown = client.get("/api/knockdown").json()
    assert knockdown["mean_log2_fold_change"] == pytest.approx(-0.058, abs=0.001)
    assert (knockdown["wrong_signed"], knockdown["target_count"]) == (6, 19)

    day = client.get("/api/day").json()
    assert day["differentiation_over_placebo"] == pytest.approx(7.97, abs=0.01)
    assert day["tracking_gene_count"] == 92

    series = {s["name"].split(":")[0]: s for s in client.get("/api/spectrum").json()["series"]}
    assert series["perturbation"]["s1_over_s0"] == pytest.approx(0.76, abs=0.01)
    assert series["placebo"]["s1_over_s0"] == pytest.approx(0.75, abs=0.01)
    assert series["differentiation"]["s1_over_s0"] == pytest.approx(0.20, abs=0.01)


def test_the_basis_endpoint_reports_the_declared_limit() -> None:
    """The cross-fold disagreement S5 is computed under, served rather than described."""

    body = client.get("/api/basis").json()
    assert body["pair_count"] == 91
    assert body["sign_flips_by_axis"]["biology_0"] == 48
    assert set(body["anchor_gene_by_axis"]["biology_0"]) == {"MPO", "CD79A"}


# --------------------------------------------------------------------- the control is not inert


def test_the_rank_control_changes_the_measurement() -> None:
    """A slider that cannot move the result is decoration.

    This is the ``do``-operator lesson applied to a UI control: the assertion is that a *different*
    input produces a *different* output, not merely that the endpoint accepts the parameter.
    """

    def s5_at(biology: int, nuisance: int = NUISANCE_RANK) -> float:
        body = client.get(
            "/api/measure", params={"biology_rank": biology, "nuisance_rank": nuisance}
        ).json()
        return next(m["value"] for m in body["measurements"] if m["name"].startswith("S5"))

    fitted = s5_at(BIOLOGY_RANK)
    assert fitted == pytest.approx(10.3647, rel=1e-3)
    for other in (2, 3, 5, 6):
        assert s5_at(other) != pytest.approx(fitted, rel=1e-3), (
            f"biology_rank={other} returned the fitted S5; the control is not reaching the fit"
        )

    # The nuisance rank must move it too, or only half the laboratory works.
    assert s5_at(BIOLOGY_RANK, 5) != pytest.approx(fitted, rel=1e-3)


def test_an_off_default_rank_is_marked_and_carries_no_verdict() -> None:
    """A screenshot at a non-fitted rank must not be mistakable for the published number."""

    body = client.get("/api/measure", params={"biology_rank": 6}).json()
    assert body["ranks"]["is_default"] is False
    assert body["ranks"]["default_biology"] == BIOLOGY_RANK

    # S2 and S4 are defined by merged ADRs at the fitted configuration only.
    names = [m["name"] for m in body["measurements"]]
    assert not any(n.startswith(("S2", "S4")) for n in names)
    s5 = next(m for m in body["measurements"] if m["name"].startswith("S5"))
    assert "point estimate only" in s5["name"]
    assert s5["interval"]["lower"] is None, "no interval exists away from the fitted config"
    assert s5["interval"]["upper"] is None

    arm = client.get("/api/arm/rep1/GATA1", params={"biology_rank": 6}).json()
    assert arm["abstention_required"] is True
    assert any("not the fitted configuration" in r for r in arm["reasons"])
    assert len(arm["axes"]) == 6, "the belief must actually be computed at the requested rank"


def test_the_two_s5_paths_agree_at_the_fitted_configuration() -> None:
    """The sweep's point estimate and the shipped bootstrap measurement must be the same number.

    They are computed by different code, and a UI that disagreed with the model card at the very
    configuration the model card describes would be worse than no UI.
    """

    from cellstate.ui.server import _s5_value

    published = next(
        m["value"]
        for m in client.get("/api/measure").json()["measurements"]
        if m["name"].startswith("S5")
    )
    assert _s5_value(BIOLOGY_RANK, NUISANCE_RANK) == pytest.approx(published, rel=1e-9)


# --------------------------------------------------------------------- refusals


@pytest.mark.parametrize(
    ("route", "params", "status"),
    [
        ("/api/arm/rep99/GATA1", {}, 404),
        ("/api/arm/rep1/NOT_A_GENE", {}, 404),
        ("/api/fold/rep99", {}, 404),
        ("/api/sweep/rep99", {}, 404),
        ("/api/library/rep99/states", {}, 404),
        ("/api/measure", {"biology_rank": 0}, 422),
        ("/api/measure", {"nuisance_rank": 0}, 422),
        # 14 exceeds the 13 directions a fold can resolve for V -- the guard against a
        # silently-truncated basis, exercised through HTTP.
        ("/api/measure", {"nuisance_rank": 14}, 422),
        ("/api/measure", {"biology_rank": 99, "nuisance_rank": 40}, 422),
        ("/api/measure", {"bound": 0}, 422),
    ],
)
def test_bad_requests_are_refused(route: str, params: dict[str, object], status: int) -> None:
    """Each refusal is exercised from the side that fails, not asserted to exist."""

    assert client.get(route, params=params).status_code == status


def test_the_placebo_halves_are_not_answerable_as_arms() -> None:
    """NT_A and NT_B are in the counts but are not targets; asking for one is a 404, not a crash."""

    assert client.get("/api/arm/rep1/NT_A").status_code == 404


# --------------------------------------------------------------------- S6 calibration


def test_the_calibration_endpoint_serves_the_published_s6() -> None:
    """The values ADR 0024 quotes, through HTTP."""

    payload = client.get("/api/calibration").json()
    assert payload["nominal_probability"] == 0.90
    assert payload["empirical_coverage"] == pytest.approx(0.8836, abs=5e-4)
    assert payload["calibration_error_upper_bound"] == pytest.approx(0.0548, abs=5e-4)
    assert payload["maximum_calibration_error"] == 0.05
    assert payload["outcome"] == "failed"
    assert payload["unit_count"] == 14
    assert len(payload["by_library"]) == 14


def test_the_calibration_endpoint_carries_both_decompositions() -> None:
    """A single coverage number invites two wrong readings; both correctives ship with it."""

    payload = client.get("/api/calibration").json()
    assert payload["standard_deviation"] == pytest.approx(1.2848, abs=5e-4)
    assert payload["trimmed_standard_deviation"] == pytest.approx(1.0045, abs=5e-4)
    assert payload["depth_coverage_correlation"] == pytest.approx(-0.8573, abs=5e-3)
    depths = [row["depth"] for row in payload["by_library"]]
    assert min(depths) > 0 and max(depths) > 5 * min(depths)


def test_the_calibration_endpoint_takes_no_rank_arguments() -> None:
    """S6's estimand is fixed at the fitted configuration, so the route offers no knob.

    FastAPI ignores unknown query parameters, so the check that matters is that the value does not
    move -- not that the request is refused.
    """

    baseline = client.get("/api/calibration").json()["empirical_coverage"]
    off = client.get("/api/calibration", params={"biology_rank": 7}).json()["empirical_coverage"]
    assert off == baseline


def test_the_arm_abstention_names_the_calibration_failure() -> None:
    """The belief's reasons are reprinted, not summarized, so S6 reaches the page that shows it."""

    reasons = client.get("/api/arm/rep1/GATA1").json()["reasons"]
    assert any(reason.startswith("S6 calibration FAILED") for reason in reasons)
    assert any("criteria not met:" in reason for reason in reasons)
