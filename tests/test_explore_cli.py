"""The exploration CLI runs, and the numbers the guide quotes are the numbers it prints.

``docs/guides/explore-the-system.md`` and the model card quote this tool's output verbatim.  A
quoted number that no test checks is the failure mode this repository keeps finding in its own work,
so the three screens whose figures are cited elsewhere are executed here and their headline values
matched against the text.

The CLI lives in ``scripts/`` rather than in the package: it is a surface over the shipped path, not
part of it, and putting it in ``src/`` would add several hundred uncovered lines to a package with
an 85% floor.  That keeps it out of ``--cov=cellstate``, which is exactly why it needs this file.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXPLORE = ROOT / "scripts" / "explore.py"

COMMANDS = [
    ["inventory"],
    ["panel"],
    ["knockdown"],
    ["day"],
    ["spectrum"],
    ["ranks"],
    ["measure"],
    ["state", "rep1", "GATA1"],
    ["axes", "rep1"],
    ["contrast", "rep1", "NT", "GATA1"],
    ["sweep", "rep1"],
]


def _run(*arguments: str) -> str:
    completed = subprocess.run(
        [sys.executable, str(EXPLORE), *arguments],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    assert completed.returncode == 0, (
        f"{arguments} exited {completed.returncode}: {completed.stderr}"
    )
    return completed.stdout


@pytest.mark.parametrize("command", COMMANDS, ids=lambda c: "-".join(c))
def test_every_command_runs(command: list[str]) -> None:
    """Eleven commands, each exercised from the entry point a reader actually types."""

    assert _run(*command).strip()


def test_an_unknown_library_is_refused_rather_than_answered() -> None:
    completed = subprocess.run(
        [sys.executable, str(EXPLORE), "sweep", "rep99"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    assert completed.returncode != 0
    assert "unknown library" in (completed.stdout + completed.stderr)


def test_the_knockdown_screen_prints_the_figures_the_model_card_quotes() -> None:
    """The substrate verdict: mean on-target log2FC, and how many targets move the wrong way."""

    output = _run("knockdown")
    mean = float(re.search(r"mean on-target log2FC\s+([-+][\d.]+)", output).group(1))
    wrong = re.search(r"wrong-signed\s+(\d+) of (\d+)", output)
    restricted = float(re.search(r"restricted to >200 CPM\s+([-+][\d.]+)", output).group(1))

    assert mean == pytest.approx(-0.058, abs=0.001)
    assert (wrong.group(1), wrong.group(2)) == ("6", "19")
    assert restricted == pytest.approx(-0.097, abs=0.001)
    assert mean > -0.5, "a working CRISPRi screen would give roughly -1 to -2; this one is a null"


def test_the_differentiation_readout_prints_the_ratio_the_readme_quotes() -> None:
    """7.97x is the project's one positive capability figure. It is computed here."""

    output = _run("day")
    ratios = [float(value) for value in re.findall(r"([\d.]+)x", output)]
    assert max(ratios) == pytest.approx(7.97, abs=0.01)
    assert "92 of 100 panel genes track day" in output


def test_the_spectrum_screen_separates_real_biology_from_the_null() -> None:
    """The perturbation's spectral profile matches the placebo's, and differentiation's does not."""

    output = _run("spectrum")
    rows = {
        name: [float(value) for value in re.findall(r"(\d+\.\d+)", line)]
        for name, line in (
            (key, next(row for row in output.splitlines() if row.startswith(key)))
            for key in ("perturbation", "placebo", "differentiation")
        )
    }
    # The last two numbers on each row are s1/s0 and the PC1 variance share.
    perturbation, placebo, differentiation = (rows[key][-2] for key in rows)

    assert perturbation == pytest.approx(0.76, abs=0.01)
    assert placebo == pytest.approx(0.75, abs=0.01)
    assert differentiation == pytest.approx(0.20, abs=0.01)
    assert abs(perturbation - placebo) < 0.05, (
        "the matrix the biology basis is fitted on has the spectral shape of the placebo contrast; "
        "if these separate, the substrate has changed and every downstream verdict is back in play"
    )
    assert differentiation < placebo / 2, "real biology on this slice concentrates; noise does not"
