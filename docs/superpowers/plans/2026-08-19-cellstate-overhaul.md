# cellstate Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make at least one ledger criterion demonstrably passable, retract the false substrate verdict that has been explaining away every negative, and widen the represented object from a pseudobulk coordinate to a population carrying a growth field — using only bytes already on disk.

**Architecture:** Three movements, in strict order. (A) Correct facts that later decision records must cite — the perturbation modality, and the documents that still deny the repository has produced numbers. (B) Close the hole under the ledger: S5 is currently passed *exactly* by a state that ignores its evidence, and nine of ten rungs have never been observed passing on any substrate, so `0/10` carries no information. (C) Only then change the science: add the relative-growth-rate estimand that `DynamicSummary` already has slots for, fix the observation variance's sampling unit from reads to cells, and fit the biology basis in the metric the posterior actually uses.

**Tech Stack:** Python 3.11, numpy, scipy, pydantic v2, pytest, hypothesis, ruff, mypy, uv, mkdocs. No new runtime dependency before Task 15.

**Spec:** `https://claude.ai/code/artifact/f66a39e4-20cf-477d-9b67-39cc209ba238` — "The Substrate Alibi", design audit of 2026-08-19. Every task below cites a finding in it.

## Global Constraints

- **A change to phase order, to the ledger, or to a graduation gate requires a contemporaneous ADR** (`AGENTS.md:141-143`). **The commit that amends the roadmap may not also implement the work it authorizes** — so every gate change here is *two* PRs: authorization, then implementation. This matches the repo's own history (PR #29/#30 for ADR 0022, PR #31/#32 for ADR 0023).
- **Accepted ADR bodies are never rewritten.** Corrections go on the ADR's **Status line** only (`AGENTS.md:148-149`). `audits/`, `benchmarks/`, `data_manifests/` and `containers/` are frozen evidence.
- **Every reported quantity carries an interval, resampled at the independent experimental unit** — here the library, K=14. Never at the cell (`docs/validation/scientific-validation.md` rules 1–2).
- **Bounds are predeclared in a merged ADR before the number exists.** Verifiable by `git log -S"<bound literal>"` returning the ADR commit strictly before the measurement commit.
- **Cite artifacts and ADRs, never roadmap queue IDs**, in any document other than `docs/roadmap.md` (`AGENTS.md:144-147`).
- **Never push to `main`.** Branch + PR for every task. `main` has no branch protection, so this is a discipline, not an enforcement.
- **`make check` must pass before every PR** = `lint` + `lock-check` + `type` + `test` + `schemas` + `example` + `build`. Runtime ~11 min for pytest alone; CI ~29 min.
- **Coverage floor is `--cov-fail-under=85` over `source = ["cellstate"]`** (`pyproject.toml:74,80`). It excludes `scripts/`. Do not lower it before Task 18.
- **Next free ADR number is 0026.** ADRs 0001–0025 are accepted.
- **Local checkout:** `/Users/logannye/Documents/Codex/2026-08-08/i-w/outputs/cell-state`. Raw source bytes (unused by the package until Task 15): `~/Documents/Codex/2026-08-08/i-w/work/gse274113`, 3.0 GB.
- **The n=1 donor limit stands.** ADR 0018 finding 4 is not overturned by anything in this plan. No task here licenses a biological claim; every measurement added is instrument validation within one donor's culture.

---

## File Structure

| File | Responsibility | Tasks |
|---|---|---|
| `docs/adr/0026-*.md` | Authorizes the S5 degeneracy refusal and the demonstrated-pass requirement | 4 |
| `docs/adr/0027-*.md` | Authorizes the relative-growth-rate estimand and predeclares its bounds | 9 |
| `docs/adr/0028-*.md` | Authorizes the observation variance's sampling unit changing from read to cell | 11 |
| `src/cellstate/evaluation/gse274113_reports.py` | S2/S4/S5 measurements; gains the degeneracy refusal and `measure_relative_growth_rate` | 5, 10 |
| `src/cellstate/backends/gse274113/fit.py` | The fold; gains cells-based variance, averaged/weighted basis, day-cosine diagnostic | 12, 13, 14 |
| `src/cellstate/backends/gse274113/likelihood.py` | Variance functional form | 12 |
| `src/cellstate/backends/gse274113/estimator.py` | Fills `DynamicSummary` hazards | 10 |
| `src/cellstate/data/representability.py` | Census gains a required, refusable perturbation-modality field | 2 |
| `src/cellstate/data/ingest/` *(new)* | Ingestion moved inside the package, under test | 15, 16 |
| `scripts/explore.py` | `spectrum` gains a matched null; `knockdown` replaced by a modality-appropriate screen | 1, 8 |
| `tests/test_capability_positive_controls.py` *(new)* | The demonstrated-pass substrate for every ledger criterion | 5, 6, 7 |
| `tests/test_documentation_claims.py` *(new)* | Guards the "no scientific numbers" class | 3 |

---

## Phase 0 — Correct the facts every later ADR must cite

### Task 1: Retract the CRISPRi modality

**Spec finding:** #02 — the deposit is Cas9 nuclease knockout, so the on-target-mRNA control is a CRISPRi criterion applied to a different technology.

**Files:**
- Modify: `docs/roadmap.md:30`, `README.md:271` and `:315`, `AGENTS.md:104`, `src/cellstate/backends/gse274113/estimator.py:12,217`, `src/cellstate/backends/gse274113/arm_request.py:186,353`, `docs/guides/explore-the-system.md:50-54`, `docs/backends/gse274113-rna-observation-model.md:212-215`, `docs/index.md:35`, `docs/guides/estimate-a-real-cell-state.md:69`, `scripts/explore.py:365-366,412`, `src/cellstate/ui/static/index.html:217,248`, `src/cellstate/ui/static/charts.js:374`, `backends/vertical-a/gse274113-rna-obs-v1/panel.json:2`
- Test: `tests/test_no_crispri_claim.py` (new)

**Interfaces:**
- Produces: the string literal `"Cas9 nuclease knockout guide"` as the `OntologyTerm` label used by every later task that references the intervention type.

> ⚠️ `panel.json` lives under `backends/vertical-a/`, which `AGENTS.md:147` calls frozen evidence. Its `description` field is prose, not a measurement, and the panel's `gene_axis_sha256` does not cover it — **verify that before editing** (Step 3). If the hash does cover it, leave `panel.json` alone and record the discrepancy in the model card instead.

- [ ] **Step 1: Write the failing guard test**

```python
# tests/test_no_crispri_claim.py
"""GSE274113 is Cas9 nuclease knockout, not CRISPRi.

Science 10.1126/science.ads7951: "a lentiviral library of guide RNAs targeting 18 hematopoietic
master regulator transcription factors was introduced into adult CD34+ purified human stem and
progenitor cells at a low multiplicity of infection with Cas9 protein."

The distinction is not cosmetic. CRISPRi (dCas9-KRAB) represses transcription, so on-target mRNA
must fall and "-1 to -2 log2FC" is the right expectation. Nuclease cutting destroys the PROTEIN;
the transcript falls only through nonsense-mediated decay and is frequently flat or positive. The
repository's substrate verdict applied the CRISPRi expectation to a nuclease experiment.

Accepted ADR bodies are historical records and are excluded: they are allowed to contain the
claim they were written under.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = (ROOT / "docs" / "adr", ROOT / "CHANGELOG.md", ROOT / "tests")
SUFFIXES = {".py", ".md", ".html", ".js", ".json", ".toml"}
PATTERN = re.compile(r"crispri", re.IGNORECASE)


def _searchable() -> list[Path]:
    paths = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in SUFFIXES:
            continue
        if any(str(path).startswith(str(excluded)) for excluded in EXCLUDED):
            continue
        if any(
            part in {".git", ".venv", "dist", ".mypy_cache", ".ruff_cache", "node_modules"}
            for part in path.parts
        ):
            continue
        paths.append(path)
    return paths


def test_no_live_file_calls_this_deposit_a_crispri_screen() -> None:
    offenders = [
        f"{path.relative_to(ROOT)}:{n}"
        for path in _searchable()
        for n, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1)
        if PATTERN.search(line)
    ]
    assert offenders == [], (
        "GSE274113 is Cas9 nuclease knockout (Science 10.1126/science.ads7951), not CRISPRi. "
        f"Still claimed at: {offenders}"
    )


def test_the_searcher_can_find_the_claim() -> None:
    """The guard's population is computed, not asserted -- it must be able to fail."""
    assert len(_searchable()) > 50
    assert PATTERN.search("a working CRISPRi screen gives -1 to -2")
```

- [ ] **Step 2: Run it and confirm it fails, listing the real sites**

```bash
cd "/Users/logannye/Documents/Codex/2026-08-08/i-w/outputs/cell-state"
uv run pytest tests/test_no_crispri_claim.py -v
```

Expected: `test_no_live_file_calls_this_deposit_a_crispri_screen` FAILS listing ~15 `path:line` sites. `test_the_searcher_can_find_the_claim` PASSES. **Record the failure list — it is the work list for Step 4.**

- [ ] **Step 3: Check whether `panel.json`'s description is covered by its hash**

```bash
uv run python - <<'PY'
import json, hashlib
p = json.load(open("backends/vertical-a/gse274113-rna-obs-v1/panel.json"))
genes = p["genes"]
digest = hashlib.sha256("\n".join(g if isinstance(g, str) else g["symbol"] for g in genes).encode()).hexdigest()
print("recorded:", p["gene_axis_sha256"])
print("genes-only digest:", digest)
print("MATCH -> description is NOT hashed, safe to edit" if digest == p["gene_axis_sha256"] else "NO MATCH -> inspect scripts/gse274113_build_panel.py before touching panel.json")
PY
```

If it does not match, read `scripts/gse274113_build_panel.py` to find the exact hashed pre-image before deciding. Do not guess.

- [ ] **Step 4: Correct every site from the Step 2 list**

Two substitutions, applied by hand per site (not `sed` — several sites need the surrounding sentence rewritten, not just the word):

1. `CRISPRi` → `Cas9 nuclease knockout` (or `Cas9 knockout` where the line is tight).
2. Every occurrence of the expectation sentence — "A working CRISPRi screen gives roughly −1 to −2" and its variants — is **deleted**, not reworded. It is the wrong control, and there is no correct threshold to substitute; Task 8 supplies the replacement screen.

The two contract fields, which are the ones that matter most because they serialize:

```python
# src/cellstate/backends/gse274113/arm_request.py:186 and :353
kind = (OntologyTerm(label="Cas9 nuclease knockout guide"),)
...
intervention_type = (OntologyTerm(label="Cas9 nuclease knockout guide"),)
```

And the estimator's prose:

```python
# src/cellstate/backends/gse274113/estimator.py:12
* the four biology dimensions become a ``REGULATORY`` factor -- the perturbations are Cas9
  nuclease knockouts of transcription factors;
```

- [ ] **Step 5: Add the mechanism note where the retracted expectation used to live**

In `docs/backends/gse274113-rna-observation-model.md`, replace the deleted "working CRISPRi screen" limitation with:

```markdown
**On-target mRNA is not a validity control for this deposit.** The perturbation is Cas9 nuclease
knockout delivered by lentiviral sgRNA at low multiplicity with Cas9 protein
([Science 10.1126/science.ads7951](https://doi.org/10.1126/science.ads7951)). A frameshift destroys
the protein; the transcript falls only through nonsense-mediated decay, and edits that escape NMD --
or that de-repress an autoregulatory promoter, as a transcriptional repressor's own knockout does --
give zero or positive log2 fold change. The mean on-target log2FC recorded here is therefore a
measurement of NMD escape and editing mosaicism, and carries no information about whether the
perturbation worked. Controls that are valid for this modality are guide-level replication,
expression-dependence of effect size, and the cutting-versus-non-cutting contrast; they are
measured by the screen this document points to.
```

- [ ] **Step 6: Run the guard and the full check**

```bash
uv run pytest tests/test_no_crispri_claim.py -v
make check
```

Expected: both tests PASS; `make check` green. If `make schemas` produces a diff, the `OntologyTerm` label change reached an exported schema — commit the regenerated schema with the change.

- [ ] **Step 7: Commit and open the PR**

```bash
git switch -c fix/retract-the-crispri-modality
git add -A
git commit -m "Retract the CRISPRi modality: GSE274113 is Cas9 nuclease knockout

The on-target-mRNA control that produced the substrate verdict is calibrated to
dCas9-KRAB, where repression forces the transcript down. This deposit uses lentiviral
sgRNA at low MOI with Cas9 protein (Science 10.1126/science.ads7951), where a frameshift
destroys the protein and the transcript need not move. The expectation is deleted rather
than rewritten; a modality-appropriate screen replaces it separately.

Advances: no ledger row. This is a fact correction that later decision records cite."
gh pr create --fill
```

---

### Task 2: Make the census refuse a source whose perturbation modality is unrecorded

**Spec finding:** #02, structural half — 15,000+ lines of admission apparatus, and not one check asks what the perturbation technology is.

**Files:**
- Modify: `src/cellstate/data/representability.py:87-103` (add criterion), `:110-120` (evidence methods)
- Modify: `docs/data/representability/gse274113-perturb-multiome.md` (record the modality)
- Test: `tests/test_representability.py`

**Interfaces:**
- Consumes: the `"Cas9 nuclease knockout guide"` label from Task 1.
- Produces: `RepresentabilityCriterion.PERTURBATION_MODALITY_RECORDED`, consumed by Task 8's pre-download screen documentation.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_representability.py
def test_a_source_without_a_recorded_perturbation_modality_is_refused() -> None:
    """The one fact that decided every downstream interpretation was the one nothing watched.

    The census computed peak counts to four significant figures and never recorded whether the
    Cas9 was catalytically dead. Fifteen admission criteria existed and none of them asked.
    """

    assert RepresentabilityCriterion.PERTURBATION_MODALITY_RECORDED in RepresentabilityCriterion
    proof = _proof_without(RepresentabilityCriterion.PERTURBATION_MODALITY_RECORDED)
    with pytest.raises(ValueError, match="perturbation modality"):
        verify_representability(proof)


def test_a_modality_recorded_only_from_the_publication_is_still_admissible() -> None:
    """Modality is not in the bytes. PUBLICATION_METHOD is the honest evidence method for it,
    and this is the one criterion for which that is true -- which is why it must be named."""

    proof = _proof_with_modality(method=RepresentabilityEvidenceMethod.PUBLICATION_METHOD)
    assert (
        verify_representability(proof).criteria_status[
            RepresentabilityCriterion.PERTURBATION_MODALITY_RECORDED
        ]
        is RepresentabilityCriterionStatus.PASSED
    )
```

- [ ] **Step 2: Run it and confirm it fails**

```bash
uv run pytest tests/test_representability.py -k perturbation_modality -v
```

Expected: FAIL with `AttributeError: PERTURBATION_MODALITY_RECORDED`.

- [ ] **Step 3: Add the criterion**

```python
# src/cellstate/data/representability.py, inside RepresentabilityCriterion
    # Modality is not derivable from the bytes: a CRISPRi and a Cas9-nuclease deposit are
    # byte-identical in shape. It decides which positive controls are valid, so a source that
    # does not record it cannot be screened at all. Evidence method is PUBLICATION_METHOD by
    # necessity -- this is the criterion that justifies keeping that method in the enum.
    PERTURBATION_MODALITY_RECORDED = "perturbation_modality_recorded"
```

Then add it to whichever criterion set `verify_representability` requires for `DESTRUCTIVE_POPULATION` proofs. **Read the existing required-set construction first** — do not assume its shape.

- [ ] **Step 4: Record GSE274113's modality in its census document**

```markdown
## Perturbation modality

**Cas9 nuclease knockout.** Lentiviral sgRNA library at low multiplicity of infection with Cas9
protein, per [Science 10.1126/science.ads7951](https://doi.org/10.1126/science.ads7951). Recorded
under `PERTURBATION_MODALITY_RECORDED` with evidence method `PUBLICATION_METHOD`, which is the only
method available: the modality is not present in the deposited bytes and cannot be computed from
them. **Consequence for screening:** on-target mRNA log2 fold change is not a validity control for
this source.
```

- [ ] **Step 5: Run the tests, then the full check**

```bash
uv run pytest tests/test_representability.py -v && make check
```

- [ ] **Step 6: Commit and PR**

```bash
git switch -c feat/census-records-the-perturbation-modality
git add -A
git commit -m "Refuse a source whose perturbation modality is unrecorded

Fifteen admission criteria, none of which asked what the CRISPR was. Modality decides
which positive controls are valid and is not derivable from the bytes, so it needs its
own criterion and PUBLICATION_METHOD is its honest evidence method.

Advances: no ledger row. Closes the gap that produced the retracted substrate verdict."
gh pr create --fill
```

---

### Task 3: Retract the stale "no scientific numbers" claims and guard the class

**Spec finding:** #11 — the file that defines what counts as a scientific number says there are none. Fourth instance of this shape.

**Files:**
- Modify: `docs/validation/scientific-validation.md:6-7`, `docs/guides/migrate-v1-to-v2.md:4`
- Test: `tests/test_documentation_claims.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_documentation_claims.py
"""No live document may deny that the repository has produced scientific numbers.

It has, since PR #27. AGENTS.md:95 already retracts these exact sentences, and two other
documents kept their copies -- including docs/validation/scientific-validation.md, the file that
DEFINES what counts as a scientific number. No test reads any of these files, which is why the
same defect has now been found four times. This is the guard for the class, not another one-off fix.

Accepted ADR bodies and the changelog are historical records and are excluded.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = (ROOT / "docs" / "adr", ROOT / "CHANGELOG.md", ROOT / "tests")

RETRACTED = (
    "no biological backend is registered",
    "the repository has produced no scientific numbers",
    "no belief has been emitted by a biological model",
)


def _live_markdown() -> list[Path]:
    return [
        path
        for path in ROOT.rglob("*.md")
        if not any(str(path).startswith(str(e)) for e in EXCLUDED)
        and not any(p in {".git", ".venv", "dist", "node_modules"} for p in path.parts)
    ]


def test_no_live_document_denies_the_backend_exists() -> None:
    offenders = [
        f"{path.relative_to(ROOT)}:{n}: {claim}"
        for path in _live_markdown()
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        for claim in RETRACTED
        if claim in line.lower()
    ]
    assert offenders == [], f"retracted claims still live at: {offenders}"


def test_the_guard_reads_the_file_that_defines_the_standard() -> None:
    """A guard whose population misses the one file that matters is not a guard.
    See the repository's own note on hand-written guard populations."""
    assert ROOT / "docs" / "validation" / "scientific-validation.md" in _live_markdown()
    assert len(_live_markdown()) > 10
```

- [ ] **Step 2: Run it and confirm it fails at both known sites**

```bash
uv run pytest tests/test_documentation_claims.py -v
```

Expected: FAIL naming `docs/validation/scientific-validation.md:6` and `docs/guides/migrate-v1-to-v2.md:4`.

- [ ] **Step 3: Correct `docs/validation/scientific-validation.md:5-7`**

```markdown
Software correctness is necessary and insufficient. This document states the standard a biological
backend must meet, aligned to the exact `StateQuery` it claims. [`../roadmap.md`](../roadmap.md) is
the sole authority for the order in which that work happens and for what has been met so far. The
answer today: one biological backend is registered and has emitted a `CellStateBelief` from real
cells; no benchmark is scientifically admitted; and the eligibility ledger stands at 0 of 10, with
the caveat recorded in [ADR 0026](../adr/0026-a-criterion-must-be-shown-to-be-passable.md) that
nine of those ten rungs have never been observed passing on any substrate.
```

> The ADR reference resolves in Task 4. If Task 3 ships first, use the sentence without the ADR clause and add it in Task 4's PR.

- [ ] **Step 4: Correct `docs/guides/migrate-v1-to-v2.md:4`** to state that a biological backend is registered and that v1 beliefs from it exist.

- [ ] **Step 5: Run and check**

```bash
uv run pytest tests/test_documentation_claims.py -v && make check
```

- [ ] **Step 6: Commit and PR**

```bash
git switch -c fix/retract-no-scientific-numbers
git add -A
git commit -m "Guard the class: no live document may deny the backend exists

Fourth instance of this shape. The two remaining copies were in migrate-v1-to-v2 and --
worse -- in the file that defines what counts as a scientific number. No test read any of
these files, so the fix is a guard over live markdown, not another one-off correction.

Advances: no ledger row."
gh pr create --fill
```

---

## Phase 1 — Make a rung demonstrably passable

> **This is the priority of the whole plan.** Until it is done, no other result in the repository can be attributed to either the deposit or the estimator.

### Task 4: ADR 0026 — authorization only, no implementation

**Spec finding:** #01 — S5 returns `0.000000, [0, 0], passed=True` for a state that is a deterministic function of the target label; and no test supplies a substrate on which S2, S4 or S5 pass.

**Files:**
- Create: `docs/adr/0026-a-criterion-must-be-shown-to-be-passable.md`
- Modify: `docs/roadmap.md` (ledger caveat + queue entry)

> ⚠️ Per `AGENTS.md:141-143` this commit **authorizes and does not implement.** Tasks 5–7 are separate PRs. Do not add code here.

- [ ] **Step 1: Write the ADR**

```markdown
# ADR 0026: A criterion must be shown to be passable, and S5 must refuse a label-determined state

- **Status:** Proposed
- **Date:** 2026-08-19

## Context

`measure_nuisance_separation` (`src/cellstate/evaluation/gse274113_reports.py:666-712`) computes,
for each arm, the squared deviation of its biology coefficients from the mean over arms sharing its
target, divided by the across-target variance. **A state that is a deterministic function of the
target label makes every deviation identically zero.** Such an estimator scores S5 = 0.000000 with a
zero-width interval and `passed = True`. It does not read its evidence. The unique optimum of this
repository's hardest gate is therefore the maximally ignorant estimator, and the gate as written is
necessary but not sufficient.

This is the defect shape the repository has already named twice: a gate that a wrong computation
passes. It was not caught because the exploit was never attempted.

Second, and larger: **no test anywhere supplies a substrate on which S5 returns a value at or below
0.35, S2 returns a value above 1, or the S4 bands separate.** The existing reachability
demonstration at `tests/test_gse274113_observation_model.py:697-698` varies the *bound*
(`bound=17.0` passes, `bound=16.0` fails) against a fixed measured value. That establishes that the
comparison operator is not a constant. It establishes nothing about the computation. The single
exception is S6, whose `test_the_six_level_gate_can_be_passed` rescales the residuals by 1.21 and
observes all six nominal levels clear, with 1.20 and 1.22 failing on opposite sides. That test is
the template and it exists for one rung of ten.

The consequence is that the ledger's `0 of 10` cannot be attributed. A failing rung is consistent
with a null substrate and equally consistent with a criterion that nothing could pass, and the
project has no measurement that separates them. The programme's next planned action — select a new
corpus — is not decidable on this evidence.

## Decision

1. **`measure_nuisance_separation` must refuse a degenerate state.** When the total within-target
   spread of the biology coefficients is at or below a predeclared floor, the function raises rather
   than returning a passing measurement. The floor is `1e-9`, predeclared here, in the units of
   mean squared deviation of the biology block. The refusal message must name the condition.
   Rationale for a refusal rather than a failing verdict: a state that ignores its evidence is not a
   worse estimator on this axis, it is not an estimator of this quantity at all, and reporting it as
   a number invites the number to be compared.

2. **Every ledger criterion must carry a demonstrated pass before its verdict is quotable.** For
   each of S2, S4 and S5 the repository must hold a test that (a) constructs a substrate on which
   the criterion passes and observes it pass, and (b) exhibits a neighbouring substrate on which it
   fails. The substrate may be synthetic — this establishes a property of the *criterion*, never of
   the biology, and no such test may be cited as biological evidence. `tests/test_s6_calibration.py`
   is the reference implementation of this pattern.

3. **The ledger gains a per-row `demonstrated_pass` column** recording whether the criterion has
   ever been observed passing. A row reading `0/10` whose criterion has no demonstrated pass is
   reported as **uninterpretable**, not as a negative result. This does not weaken rule 10 — a
   negative result still graduates its phase — it distinguishes a measured negative from an
   unmeasurable one.

4. **This decision does not change any published value.** S2, S4, S5 and S6 keep their measured
   numbers and their failing verdicts. What changes is what may be *concluded* from them.

## Consequences

The substrate verdict recorded against `GSE274113` was already retracted on modality grounds. This
decision removes the second support under it: even had the modality been right, `0/10` could not
have discriminated the deposit from the instrument. Source selection is therefore blocked on
decision 2 rather than on a new corpus, and that is the intended ordering.

Decision 1 will not change S5's measured value on the committed slice, whose within-target spread is
far above the floor. It closes an exploit, and the test that proves the exploit existed is the
deliverable.
```

- [ ] **Step 2: Add the ledger caveat and the queue entry to `docs/roadmap.md`**

Add a `demonstrated pass` column to the ledger table at `docs/roadmap.md:43-46` with `no` for S2/S4/S5 and `yes` for S6, and a sentence directly beneath it:

```markdown
**A `fails` in a row whose `demonstrated pass` reads `no` is uninterpretable, not negative.** Under
[ADR 0026](adr/0026-a-criterion-must-be-shown-to-be-passable.md) such a row is consistent with a null
substrate and equally consistent with a criterion nothing could pass. S6 is the only row currently
free of this caveat.
```

- [ ] **Step 3: Verify the commit authorizes and does not implement**

```bash
git add -A
git diff --cached --stat
```

Expected: exactly two paths — `docs/adr/0026-*.md` and `docs/roadmap.md`. **If `src/` or `tests/` appears, remove it from the commit.**

- [ ] **Step 4: Verify the bound is predeclared before it is used**

```bash
make check
git commit -m "ADR 0026: a criterion must be shown to be passable

S5 returns 0.000000 with a zero-width interval and passed=True for a state that is a
deterministic function of the target label. Separately, no test supplies a substrate on
which S2, S4 or S5 pass -- the existing reachability test varies the bound against a fixed
value. So 0/10 cannot be attributed to the deposit or to the instrument.

Authorizes only. Implementation follows separately per AGENTS.md:141-143.

Advances: the precondition for interpreting S2, S4, S5 and every future ledger row."
gh pr create --fill
```

After merge, confirm predeclaration holds:

```bash
git log -S"1e-9" --oneline -- docs/adr/0026-a-criterion-must-be-shown-to-be-passable.md
```

Expected: this ADR's commit, and it must precede Task 5's commit.

---

### Task 5: Implement the S5 degeneracy refusal, proven by the exploit

**Files:**
- Modify: `src/cellstate/evaluation/gse274113_reports.py:666-712`
- Test: `tests/test_capability_positive_controls.py` (new)

**Interfaces:**
- Consumes: `ArmState(library: str, target: str, biology: FloatArray, realization: float, realization_sd: float, predictive_sd: FloatArray, point_residual: FloatArray)` from `gse274113_reports`; `measure_nuisance_separation(states, *, bound, seed=DEFAULT_SEED) -> CapabilityMeasurement`.
- Produces: `DEGENERATE_STATE_FLOOR = 1e-9` and the refusal, relied on by Task 6.

- [ ] **Step 1: Write the exploit test — it must PASS against current code, proving the hole**

```python
# tests/test_capability_positive_controls.py
"""Substrates on which a criterion is observed to pass, and the exploits that must not.

Every construction here is SYNTHETIC. Under ADR 0026 decision 2 these establish properties of the
CRITERIA and may never be cited as biological evidence.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellstate.evaluation.gse274113_reports import (
    ArmState,
    measure_nuisance_separation,
)

LIBRARIES = tuple(f"rep{i}" for i in range(1, 15))
TARGETS = ("GATA1", "TAL1", "MYB", "RUNX1", "IRF1")
BIOLOGY_DIM = 4


def _label_determined_states() -> tuple[ArmState, ...]:
    """A state that reads no evidence: biology is a fixed function of the target label alone."""
    rng = np.random.default_rng(0)
    by_label = {t: rng.normal(size=BIOLOGY_DIM) for t in TARGETS}
    return tuple(
        ArmState(
            library=library,
            target=target,
            biology=by_label[target].copy(),  # identical in every library
            realization=0.0,
            realization_sd=1.0,
            predictive_sd=np.ones(BIOLOGY_DIM),
            point_residual=np.zeros(BIOLOGY_DIM),
        )
        for library in LIBRARIES
        for target in TARGETS
    )


def test_the_exploit_exists_before_the_fix() -> None:
    """RUN THIS ONCE AGAINST UNPATCHED CODE. It must pass, then be replaced by the test below.

    Delete this test in the same commit that adds the refusal -- keeping both would leave a test
    asserting the defect still exists.
    """
    measurement = measure_nuisance_separation(_label_determined_states(), bound=0.35)
    assert measurement.value == pytest.approx(0.0, abs=1e-12)
    assert measurement.interval.upper == pytest.approx(0.0, abs=1e-12)
    assert measurement.passed is True
```

- [ ] **Step 2: Run it against unpatched code and confirm the exploit is real**

```bash
uv run pytest tests/test_capability_positive_controls.py::test_the_exploit_exists_before_the_fix -v
```

Expected: **PASS** — value 0.0, interval upper 0.0, `passed is True`. This is the proof the defect is real and not a reading error. Copy the printed values into the commit message.

- [ ] **Step 3: Replace the exploit test with the refusal test**

Delete `test_the_exploit_exists_before_the_fix` and add:

```python
def test_a_label_determined_state_is_refused_not_passed() -> None:
    """ADR 0026 decision 1.

    Before the refusal this construction returned value 0.000000, interval [0.0, 0.0], passed=True:
    the unique optimum of S5 was an estimator that ignores its evidence entirely. A state that is a
    deterministic function of the target label is not a worse estimator of nuisance separation, it
    is not an estimator of it, so the function refuses rather than scoring it.
    """
    with pytest.raises(ValueError, match="carries no information beyond the target label"):
        measure_nuisance_separation(_label_determined_states(), bound=0.35)


def test_a_state_just_above_the_floor_is_still_measured() -> None:
    """The refusal must not swallow a real but small within-target spread."""
    rng = np.random.default_rng(1)
    base = _label_determined_states()
    states = tuple(
        ArmState(
            library=s.library,
            target=s.target,
            biology=s.biology + rng.normal(scale=1e-3, size=BIOLOGY_DIM),
            realization=s.realization,
            realization_sd=s.realization_sd,
            predictive_sd=s.predictive_sd,
            point_residual=s.point_residual,
        )
        for s in base
    )
    measurement = measure_nuisance_separation(states, bound=0.35)
    assert measurement.value > 0.0
    assert measurement.passed is True  # a genuinely nuisance-free state passes, as it should
```

- [ ] **Step 4: Run and confirm both FAIL**

```bash
uv run pytest tests/test_capability_positive_controls.py -v
```

Expected: `test_a_label_determined_state_is_refused_not_passed` FAILS (no exception raised). The second test may already pass.

- [ ] **Step 5: Implement the refusal**

```python
# src/cellstate/evaluation/gse274113_reports.py, near DEFAULT_SEED
# ADR 0026 decision 1, predeclared before any state was scored against it. In the units of mean
# squared deviation of the biology block.
DEGENERATE_STATE_FLOOR = 1e-9
```

Inside `measure_nuisance_separation`, after `target_means` is built and before `across_target`:

```python
within_target = float(
    np.mean([np.mean((state.biology - target_means[state.target]) ** 2) for state in states])
)
if within_target <= DEGENERATE_STATE_FLOOR:
    # ADR 0026 decision 1. Without this the ratio is identically zero and S5 PASSES, so the
    # gate's unique optimum is an estimator that never reads its evidence. Refused rather than
    # scored: a number invites comparison, and there is nothing here to compare.
    raise ValueError(
        "the inferred state carries no information beyond the target label "
        f"(within-target spread {within_target:.3g} <= {DEGENERATE_STATE_FLOOR:g}); "
        "S5 is not defined for a state that does not read its evidence"
    )
```

- [ ] **Step 6: Run the tests, then confirm the published S5 is unchanged**

```bash
uv run pytest tests/test_capability_positive_controls.py -v
uv run pytest tests/test_gse274113_observation_model.py -v
```

Expected: new tests PASS. **The pinned S5 value 10.365 and its interval must be unchanged** — the committed slice's within-target spread is far above `1e-9`. If any pinned figure moves, stop: the floor is too high and the ADR needs amending on its Status line before proceeding.

- [ ] **Step 7: `make check`, commit, PR**

```bash
make check
git switch -c feat/s5-refuses-a-label-determined-state
git add -A
git commit -m "S5 refuses a state that carries no information beyond the target label

Implements ADR 0026 decision 1. Measured against unpatched code first: a state that is a
deterministic function of the target label returned value 0.000000, interval [0.0, 0.0],
passed=True -- the gate's unique optimum was an estimator that reads no evidence.

Every pinned figure on the committed slice is unchanged; the slice's within-target spread
is far above the 1e-9 floor.

Advances: S5's interpretability. The verdict on the committed evidence still fails."
gh pr create --fill
```

---

### Task 6: Demonstrate that S5 can be passed by a real estimator

**Files:**
- Modify: `tests/test_capability_positive_controls.py`

**Interfaces:**
- Consumes: `fit_fold`, `ArmSlice` from `cellstate.backends.gse274113.fit`; `held_out_states`, `measure_nuisance_separation` from `gse274113_reports`.

> The amplification factor is **not** given here. It is swept and then pinned, exactly as `tests/test_s6_calibration.py` found 1.21 by sweeping. A plan that hands you the answer would let a wrong implementation reproduce it.

- [ ] **Step 1: Write the sweep as a throwaway script and find the passing region**

```bash
uv run python - <<'PY'
import json, numpy as np
from cellstate.backends.gse274113.fit import ArmSlice
from cellstate.evaluation.gse274113_reports import held_out_states, measure_nuisance_separation

payload = json.load(open("backends/vertical-a/gse274113-rna-obs-v1/arms.json"))
base = ArmSlice.from_payload(payload)

def amplified(factor: float) -> ArmSlice:
    """Scale each arm's within-library contrast against its library's NT, leaving NT fixed.
    This adds biology without touching library structure or depth."""
    counts = dict(base.counts)
    for (lib, tgt), vec in base.counts.items():
        if tgt.startswith("NT"):
            continue
        nt = base.counts[(lib, "NT")].astype(float)
        p_t, p_n = vec / vec.sum(), nt / nt.sum()
        scaled = p_n * np.exp(factor * np.log((p_t + 1e-9) / (p_n + 1e-9)))
        scaled /= scaled.sum()
        counts[(lib, tgt)] = np.round(scaled * vec.sum()).astype(np.int64)
    return ArmSlice(base.gene_symbols, base.libraries, base.targets,
                    base.library_day, counts, base.cells)

for f in (1, 2, 3, 5, 8, 12, 20):
    s = measure_nuisance_separation(held_out_states(amplified(float(f))), bound=0.35)
    print(f"amplification {f:>3}x   S5 {s.value:8.4f}  upper {s.interval.upper:8.4f}  passed {s.passed}")
PY
```

Record the full table. Identify the **smallest** amplification whose interval upper end clears 0.35, and confirm a smaller one fails.

- [ ] **Step 2: Write the test pinning what the sweep found**

```python
def test_s5_can_be_passed_by_an_estimator_that_reads_its_evidence() -> None:
    """ADR 0026 decision 2, for S5.

    A gate that has only ever been observed failing is not yet known to be a gate. The shipped
    estimator, unmodified, passes S5 on a substrate whose biology is amplified <FACTOR>x -- so the
    0.35 bound is reachable and the failure on the committed slice is a statement about the
    evidence, not about the criterion.

    SYNTHETIC. Not biological evidence (ADR 0026 decision 2).
    """
    passing = measure_nuisance_separation(held_out_states(_amplified(FACTOR)), bound=0.35)
    assert passing.passed is True
    assert passing.interval.upper <= 0.35


def test_s5_still_fails_one_step_below_the_passing_amplification() -> None:
    """The neighbouring substrate, so the pass is a threshold and not a constant."""
    failing = measure_nuisance_separation(held_out_states(_amplified(BELOW)), bound=0.35)
    assert failing.passed is False
```

Substitute the measured `FACTOR` and `BELOW` from Step 1 and move `_amplified` into the test module with the docstring explaining that it scales the within-library contrast and leaves NT and depth untouched.

- [ ] **Step 3: Run**

```bash
uv run pytest tests/test_capability_positive_controls.py -v
```

Expected: both PASS.

- [ ] **Step 4: Flip the ledger's `demonstrated pass` for S5 to `yes`** in `docs/roadmap.md`, citing ADR 0026.

- [ ] **Step 5: `make check`, commit, PR**

```bash
make check
git switch -c test/s5-has-a-demonstrated-pass
git add -A
git commit -m "Demonstrate that S5 is passable by the shipped estimator

ADR 0026 decision 2. The unmodified pipeline clears the 0.35 bound on a substrate with
amplified biology, and fails one step below it. The 0.35 bound is therefore reachable and
S5's failure on the committed slice is a statement about the evidence.

Advances: S5's demonstrated-pass column."
gh pr create --fill
```

---

### Task 7: Demonstrate that S2 and S4 can be passed

**Files:**
- Modify: `tests/test_capability_positive_controls.py`, `docs/roadmap.md` (ledger column)

- [ ] **Step 1: Sweep S4 first — it is the more likely to be malformed**

S4 requires the perturbed band to sit above the null band. Reuse `_amplified` from Task 6: amplification moves the perturbed arms and leaves `NT_A`/`NT_B` untouched, so the two halves should separate.

```bash
uv run python - <<'PY'
# same _amplified helper as Task 6
from cellstate.evaluation.gse274113_reports import measure_null_and_non_null_contrast
for f in (1, 2, 3, 5, 8, 12, 20):
    null, non_null = measure_null_and_non_null_contrast(amplified(float(f)))
    print(f"{f:>3}x  null {null.value:7.3f} [{null.interval.lower:6.3f},{null.interval.upper:6.3f}]"
          f"  perturbed {non_null.value:7.3f} [{non_null.interval.lower:6.3f},{non_null.interval.upper:6.3f}]"
          f"  separated {non_null.interval.lower > null.interval.upper}")
PY
```

> Read the real function name and return signature from `gse274113_reports.py` before running this — the S4 pair may be returned by one function or two. Adjust the call, not the intent.

**If no amplification separates the bands, stop and report it.** That would mean S4 as posed cannot be passed by any estimator, which is a finding that outranks the rest of this plan and requires its own ADR before Phase 3 proceeds.

- [ ] **Step 2: Sweep S2** — S2 requires the posterior's spread to exceed the error realized on a held-out replicate. Amplifying biology will not help; S2 is passed by *widening the posterior*. Use the `_rescaled` monkeypatch pattern from `tests/test_s6_calibration.py:245-279` as the model, sweeping the residual scale until the ratio crosses 1.

- [ ] **Step 3: Write both tests in the Task 6 pattern** — a passing substrate and a neighbouring failing one, each with a docstring naming the measured threshold and the SYNTHETIC caveat.

- [ ] **Step 4: Run, update the ledger column, `make check`, commit, PR**

```bash
uv run pytest tests/test_capability_positive_controls.py -v && make check
git switch -c test/s2-and-s4-have-demonstrated-passes
git add -A
git commit -m "Demonstrate that S2 and S4 are passable

ADR 0026 decision 2, completing the ledger's demonstrated-pass column for every criterion
the repository currently measures. 0/10 is now an attributable negative rather than an
uninterpretable one.

Advances: S2 and S4 demonstrated-pass columns."
gh pr create --fill
```

---

## Phase 2 — Repair the pre-download screen

### Task 8: Give `spectrum` a matched null, and replace `knockdown`

**Spec finding:** #03 — `s1/s0` is not row-count invariant and the placebo is not a noise reference. The matched multinomial null is ~0.89; the observed 0.7598 sits *below* its 2.5th percentile, so the verdict inverts. The README instructs readers to run this as a pre-download gate.

**Files:**
- Modify: `scripts/explore.py:516-598` (`cmd_spectrum`), `scripts/explore.py:365-412` (`cmd_knockdown`)
- Modify: `README.md:315`, `AGENTS.md:109`, `docs/guides/explore-the-system.md`
- Test: `tests/test_explore_cli.py`

- [ ] **Step 1: Write the failing test**

```python
def test_the_spectrum_screen_compares_against_a_simulated_null_band() -> None:
    """s1/s0 is not row-count invariant, and the placebo is a contrast between two halves of a
    heterogeneous differentiating population -- not noise. The reference must be simulated at the
    observed row count and the observed per-arm depths."""
    output = _run("spectrum")
    assert "simulated null" in output
    assert "matched rows" in output


def test_a_pure_noise_matrix_lands_inside_its_own_null() -> None:
    """The screen must be able to identify noise as noise, or it is not a screen."""
    from scripts.explore import simulated_null_band

    band = simulated_null_band(n_rows=266, depths=[576903.0] * 266, rate=_pooled_rate(), seed=0)
    assert band.lower < band.median < band.upper
    assert 0.80 < band.median < 0.95
```

- [ ] **Step 2: Run and confirm both fail**

```bash
uv run pytest tests/test_explore_cli.py -k spectrum -v
```

- [ ] **Step 3: Implement `simulated_null_band` in `scripts/explore.py`**

```python
def simulated_null_band(
    *, n_rows: int, depths: list[float], rate: np.ndarray, seed: int, draws: int = 200
) -> NullBand:
    """The s1/s0 a contrast matrix of this shape shows under PURE multinomial sampling.

    This is the reference the screen lacked. Two properties make the placebo contrast unusable as
    one: ``s1/s0`` is not invariant to row count -- a 14-row matrix and a 266-row matrix are not
    comparable -- and the placebo is a contrast between two halves of a heterogeneous
    differentiating population, so it carries composition structure rather than sampling noise.

    Each simulated row is the log-composition difference of two independent multinomial draws at
    the observed depths from the pooled panel rate, which is exactly the null "this matrix is
    sampling noise" and nothing else.
    """
    rng = np.random.default_rng(seed)
    ratios = []
    for _ in range(draws):
        rows = []
        for depth in depths:
            a = rng.multinomial(int(depth), rate).astype(float)
            b = rng.multinomial(int(depth), rate).astype(float)
            rows.append(
                np.log((a + 0.5) / (a.sum() + a.size / 2))
                - np.log((b + 0.5) / (b.sum() + b.size / 2))
            )
        values = np.linalg.svd(np.vstack(rows), compute_uv=False)
        ratios.append(float(values[1] / values[0]))
    lower, median, upper = np.percentile(ratios, [2.5, 50.0, 97.5])
    return NullBand(lower=float(lower), median=float(median), upper=float(upper))
```

- [ ] **Step 4: Print the band beside each observed row in `cmd_spectrum`**, and rewrite the interpretation block. The corrected reading, from the measured numbers:

```
The perturbation matrix's s1/s0 is 0.7598 against a simulated null of 0.8895
[0.8191, 0.9719] at matched rows and depths. It sits BELOW the 2.5th percentile of
its own noise null: this matrix is MORE concentrated than sampling noise, not
indistinguishable from it. The placebo contrast (0.7518) is also below its null,
which is why the placebo was never a valid reference -- it is not noise either.
```

> Re-measure these four numbers with the implemented function rather than copying them. They came from a lens agent's independent implementation; if yours disagrees, yours is the one that ships and the discrepancy goes in the commit message.

- [ ] **Step 5: Replace `cmd_knockdown` with a modality-appropriate screen**

`knockdown` computes on-target mRNA log2FC, which Task 1 established is not a validity control for a nuclease deposit. Replace its body with the three controls that are valid regardless of modality, all computable from the committed slice plus (for the first) the guide field that arrives in Task 16:

1. **Expression-dependence** — correlation between a target's NT expression and its effect size. A guide can only act on a transcribed gene. Measured: `r = −0.60, p = 0.006`.
2. **Cutting-versus-non-cutting** — deferred to Task 16, which splits AAVS1 from NT. Print `NOT YET MEASURED — requires the guide axis (Task 16)` until then, never a placeholder number.
3. **Guide-level replication** — likewise deferred to Task 16.

Until Task 16 lands, `knockdown` prints one measured control and two typed absences. That is the repository's own doctrine: absence is typed, never imputed.

- [ ] **Step 6: Update `README.md:315`, `AGENTS.md:109` and the guide** to describe `spectrum` as "observed s1/s0 against a simulated null at matched rows and depths", and delete the claim that the perturbation matrix is spectrally indistinguishable from noise.

- [ ] **Step 7: Run, check, commit, PR**

```bash
uv run pytest tests/test_explore_cli.py -v && make explore && make check
git switch -c fix/spectrum-gets-a-matched-null
git add -A
git commit -m "Give the spectrum screen a null it can be compared against

s1/s0 is not row-count invariant and the placebo contrast is not noise, so the screen's
reference class could not support its verdict. Against a simulated multinomial null at
matched rows and depths the perturbation matrix is MORE concentrated than noise, not
indistinguishable from it -- the verdict inverts.

This matters beyond the one verdict: README and AGENTS.md instruct readers to run this as
a pre-download gate, and as written it rejects well-designed Perturb-seq deposits.

knockdown is replaced by modality-appropriate controls; two of the three are typed as not
yet measured pending the guide axis.

Advances: no ledger row. Repairs the source-selection screen before it is aimed again."
gh pr create --fill
```

---

## Phase 3 — The growth field

### Task 9: ADR 0027 — authorization only

**Spec finding:** #06 and the Evidence section — 12 of 19 targets have library-grouped intervals excluding zero on a readout the model has no slot for; `estimator.py:483-492` refuses the hazards on a reason that is false for them.

**Files:**
- Create: `docs/adr/0027-the-relative-growth-rate-is-a-measured-state-quantity.md`
- Modify: `docs/roadmap.md`

> Authorization only. No code.

- [ ] **Step 1: Write the ADR, predeclaring every bound**

The ADR must contain, before any implementation exists:

```markdown
## Decision

1. **The estimand.** For perturbation `g` in library `L` at differentiation day `d(L)`, with `N` the
   arm's cell count and `ctrl` the library's pooled control arm:

       y[L,g] = log2( N[L,g] / N[L,ctrl] ) = a[g] + lambda[g] * d(L) + eps[L]

   `lambda[g]` is the net relative growth rate in log2 per day. The library main effect cancels in
   the ratio to the internal control; what is identified is the target-by-time interaction. The
   resampling unit is the **library**, K=14, consistent with rule 1. The null is a permutation of
   the day label across libraries.

2. **What this does and does not claim.** It does **not** claim S1: no unit spans a cutoff and none
   will, and ADR 0018 finding 4's single-donor limit is untouched. `lambda` is reported as
   `division_hazard` and `death_hazard` on `DynamicSummary` because that is what those fields mean,
   and the belief's causal status for it is `identified_population_effect` **within this culture**,
   never transported.

3. **Predeclared bounds**, fixed here before any value is computed:
   - the null half of S4 on this readout passes when the control guides' mean `lambda` has a
     library-grouped 95% interval containing zero;
   - the non-null half passes when at least **one third** of targeting guides fall outside the
     control guides' 95% band;
   - a target's effect is reportable when its interval excludes zero **and** its 7-day fold change
     is outside `[0.90, 1.11]`. The effect-size floor is required because an
     interval-excludes-zero rule alone is not specific: SNAI2 is not expressed in this culture and
     its interval excludes zero.

4. **`estimator.py`'s refusal is corrected, not removed.** "No library spans a timepoint" remains
   the correct reason for `velocity`, `bifurcation_proximity` and `recovery_timescale`, which need a
   cell tracked across time. It is **not** a valid reason for a population growth-rate differential
   against an internal control across four harvest days, which is what a destructive time course
   estimates and what every pooled screen reports.
```

- [ ] **Step 2: Confirm the commit is authorization-only, `make check`, commit, PR**

```bash
git add docs/adr/0027-*.md docs/roadmap.md
git diff --cached --stat   # must be exactly these two paths
make check
git commit -m "ADR 0027: the relative growth rate is a measured state quantity

DynamicSummary already carries division_hazard and death_hazard; estimator.py refuses them
because 'no library spans a timepoint', which is correct for velocity and false for a
population growth-rate differential against an internal control across four harvest days.

Bounds predeclared here, before any value exists. Authorizes only.

Advances: S4 on a second readout; the first dynamical quantity the project has carried."
gh pr create --fill
```

Verify predeclaration after merge: `git log -S"0.90, 1.11" --oneline` must return only this ADR's commit.

---

### Task 10: Implement `measure_relative_growth_rate` and fill the hazards

**Files:**
- Modify: `src/cellstate/evaluation/gse274113_reports.py` (new measurement), `src/cellstate/backends/gse274113/estimator.py:477-492` (`_dynamics`)
- Test: `tests/test_relative_growth_rate.py` (new)

**Interfaces:**
- Consumes: `ArmSlice.cells: dict[tuple[str, str], int]`, `ArmSlice.library_day: dict[str, int]` — both already present and read by no fitting code.
- Produces: `measure_relative_growth_rate(slice_data: ArmSlice, *, seed: int = DEFAULT_SEED) -> dict[str, CapabilityMeasurement]` keyed by target, consumed by `_dynamics`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_relative_growth_rate.py
def test_the_control_arm_has_a_growth_rate_of_zero_by_construction() -> None:
    rates = measure_relative_growth_rate(arm_slice())
    assert "NT" not in rates, "the control is the denominator, not a measured arm"


def test_a_constant_cell_count_gives_a_zero_growth_rate() -> None:
    """The positive control for the computation: if no arm changes, lambda is zero everywhere.
    A wrong computation that ignores day would also return zero here, so the next test is the
    one that discriminates."""
    flat = _slice_with_constant_cells(arm_slice(), n=400)
    for measurement in measure_relative_growth_rate(flat).values():
        assert measurement.value == pytest.approx(0.0, abs=1e-9)


def test_an_injected_slope_is_recovered() -> None:
    """Discriminating control: inject a known lambda into one target and recover it.

    A computation that ignores library_day cannot pass this.
    """
    injected = _slice_with_injected_slope(arm_slice(), target="GATA1", slope=-0.20)
    recovered = measure_relative_growth_rate(injected)["GATA1"]
    assert recovered.value == pytest.approx(-0.20, rel=0.05)


def test_the_measured_rates_reproduce_from_the_committed_slice() -> None:
    """Pinned at rel=1e-3 per the repository's convention. Verified load-bearing: every
    neighbouring construction gives a different number."""
    rates = measure_relative_growth_rate(arm_slice())
    assert rates["RUNX1"].value == pytest.approx(-0.2225, rel=1e-3)
    assert rates["RUNX1"].interval.upper < 0.0
    assert rates["NFE2"].value == pytest.approx(0.0547, rel=1e-3)
    assert rates["NFE2"].interval.lower > 0.0
    reportable = [t for t, m in rates.items() if m.interval.upper < 0 or m.interval.lower > 0]
    assert len(reportable) == 12
```

- [ ] **Step 2: Run and confirm all four fail**

```bash
uv run pytest tests/test_relative_growth_rate.py -v
```

- [ ] **Step 3: Implement the measurement**

```python
def measure_relative_growth_rate(
    slice_data: ArmSlice, *, seed: int = DEFAULT_SEED
) -> dict[str, CapabilityMeasurement]:
    """ADR 0027: net relative growth rate per perturbation, in log2 per differentiation day.

    ``cells`` and ``library_day`` are both carried on ``ArmSlice`` and are read by no fitting code.
    They are the two fields in the committed slice that carry the deposit's strongest measured
    effect, and this is what reads them.

    The library main effect cancels in the ratio to the library's own control arm, so what is
    regressed on day is a within-library quantity and the resampling unit stays the library.
    """
    ...
```

Regress `log2(cells[L,g] / cells[L,"NT"])` on `library_day[L]`, resample libraries with `_interval`, and return one `CapabilityMeasurement` per target. **Arms with fewer than 5 cells on either side are excluded, not imputed** — record the exclusion count in `statement`.

- [ ] **Step 4: Correct `_dynamics`**

```python
def _dynamics(self) -> DynamicSummary:
    # ADR 0027 decision 4. "No library spans a timepoint" is the correct reason for velocity,
    # bifurcation proximity and recovery timescale: each needs a cell tracked across time.
    # It is NOT a reason for a population growth-rate differential against an internal control
    # across four harvest days, which is what a destructive time course estimates.
    no_tracked_cell = (
        "no cell in GSE274113 is observed at two times, so no velocity, bifurcation proximity "
        "or recovery timescale is identifiable from this evidence"
    )
    rate = self._relative_growth_rate()
    return DynamicSummary(
        velocity=UnavailableDistribution(reason_code="no_tracked_cell", message=no_tracked_cell),
        stability=_unavailable(no_tracked_cell),
        division_hazard=rate,
        death_hazard=rate,
        bifurcation_proximity=_unavailable(no_tracked_cell),
        recovery_timescale=_unavailable(no_tracked_cell),
    )
```

> `division_hazard` and `death_hazard` both receive the *net* rate because this readout measures their difference, not either alone. Say so in the `EvaluatedScalar.reason` — do not let two fields silently imply two measurements.

- [ ] **Step 5: Run everything**

```bash
uv run pytest tests/test_relative_growth_rate.py tests/test_gse274113_observation_model.py -v
make check
```

- [ ] **Step 6: Commit and PR**

```bash
git switch -c feat/the-relative-growth-rate
git add -A
git commit -m "Measure the relative growth rate, and fill the hazards it belongs in

Implements ADR 0027. Twelve of nineteen targets have library-grouped 95% intervals
excluding zero; RUNX1 falls to 0.34x over seven days. Both fields the estimator refused on
a reason that does not apply to them are now EVALUATED.

Uses arms[].cells and library_day, both already on ArmSlice and previously read by no
fitting code. Zero new bytes.

Scope: instrument validation within one donor's culture (ADR 0018 finding 4). Not a
biological claim.

Advances: S4 on the growth readout."
gh pr create --fill
```

---

## Phase 4 — The instrument's functional form

### Task 11: ADR 0028 — the observation variance's sampling unit is the cell

**Spec finding:** #06 — slope of log(replicate variance) on log(cells) is −1.159 against a theoretical −1; on log(reads) it is +0.335, the wrong sign. And `corr(log cells, day) = −0.428` versus `corr(log reads, day) = +0.984`, which reopens the confound ADR 0024 recorded as unidentifiable.

**Files:**
- Create: `docs/adr/0028-the-observation-variance-counts-cells.md`
- Modify: `docs/adr/0024-*.md` **Status line only** (never the body), `docs/roadmap.md`

- [ ] **Step 1: Reproduce the two slopes and put them in the ADR** — the ADR must carry the measurement that motivates it, as ADR 0022 carried its per-count-bucket table. Fit on the 14 `NT_A`/`NT_B` replicate pairs.

- [ ] **Step 2: Write the ADR**, deciding that the observation variance takes the arm's cell count, that ψ² is refitted, and that the depth/day confound is re-examined because cells is not collinear with day.

- [ ] **Step 3: Amend ADR 0024's Status line only**

```markdown
- **Status:** Accepted. **Corrected 2026-08-19 by [ADR 0028](0028-the-observation-variance-counts-cells.md):** this record states that the depth/day confound "is not identified here, and no amount of re-analysis of this deposit will identify it." That holds for read depth, which correlates with day at +0.984. It does not hold for cell count, which correlates with day at −0.428 and is carried on the committed slice. The body is unchanged per the repository's rule on historical records.
```

- [ ] **Step 4: Confirm authorization-only, `make check`, commit, PR.**

---

### Task 12: Implement the cells-based variance and re-measure S2 and S6

**Files:**
- Modify: `src/cellstate/backends/gse274113/likelihood.py:66` (`technical_variance`), `src/cellstate/backends/gse274113/fit.py:152-163` (`observation_variance`), `:383` (ψ² fit)
- Test: `tests/test_gse274113_observation_model.py`, `tests/test_s6_calibration.py`

> **Every pinned figure in the repository will move.** That is expected and is the point. Re-pin all of them in this commit at `rel=1e-3`, and record the before/after for S2, S5, S6 and the block decomposition in the commit message.

- [ ] **Step 1: Write the failing test for the variance's scaling**

```python
def test_the_observation_variance_scales_with_cells_not_reads() -> None:
    """ADR 0028. Two arms at equal read depth and different cell counts must not receive equal
    variance -- the number of independent biological draws is the cell count."""
    fold = fit_fold(arm_slice(), "rep1")
    shallow = fold.observation_variance(depth=576_903.0, cells=105)
    deep = fold.observation_variance(depth=576_903.0, cells=1_296)
    assert np.all(deep < shallow)
    ratio = float(np.mean(shallow) / np.mean(deep))
    assert 3.0 < ratio < 20.0, "roughly the 12.3x cell-count range, damped by the fitted psi^2"
```

- [ ] **Step 2: Run, confirm it fails on the signature (`observation_variance` takes no `cells`).**

- [ ] **Step 3: Implement** — thread `cells` through `FittedFold.observation_variance` and the ψ² fit. Keep the read-multinomial term (it is still real) and add the cell-sampling term; the docstring must state which term dominates at the committed slice's depths.

- [ ] **Step 4: Re-measure and re-pin.** Run S2, S5, S6 and the block decomposition; update every pinned figure; verify with a neighbour sweep that `rel=1e-3` is still load-bearing.

- [ ] **Step 5: Report the outcome honestly in the commit message, whichever way it goes.**

If S6's coverage does not move toward nominal, say so plainly — the diagnosis was wrong and ADR 0028 gets a corrective Status line. Do not tune to reach a passing number; the bounds were predeclared.

- [ ] **Step 6: `make check`, commit, PR.**

---

### Task 13: Fit the biology basis on library-averaged, precision-weighted contrasts

**Spec finding:** #04 — cross-half subspace cosines move from `0.781 / 0.689 / 0.414 / 0.118` to `0.865 / 0.800 / 0.667 / 0.309`; `biology_3` as shipped is a different direction in each half.

**Files:**
- Modify: `src/cellstate/backends/gse274113/fit.py:194` (`_leading_subspace`), `:281` (`biology_rows`)
- Test: `tests/test_gse274113_observation_model.py`

- [ ] **Step 1: Write the failing test** — split libraries 1–7 and 8–14, fit a basis on each, and assert the fourth principal cosine exceeds 0.25.
- [ ] **Step 2: Run, confirm it fails at ~0.118.**
- [ ] **Step 3: Implement** — average contrasts within target across fit libraries, weight rows by `1/sqrt(observation_variance)`, keep `_canonical_signs`. The docstring must note that `u_g` at `fit.py:305` already averages, so this removes an inconsistency inside one file rather than introducing a new convention.
- [ ] **Step 4: Re-pin every figure that moves. Report whether S5 improved, worsened, or held.**
- [ ] **Step 5: `make check`, commit, PR.**

---

### Task 14: Emit `cos(V, day)` as a fold diagnostic and rename the block

**Spec finding:** #05 — the nuisance basis captures 99.9% of the day-7-to-day-14 direction in 14 of 14 folds; `fit.py:13` states a false premise as a fact.

**Files:**
- Modify: `src/cellstate/backends/gse274113/fit.py:13,269,283` (docstring + `FittedFold` field)
- Test: `tests/test_gse274113_observation_model.py`

- [ ] **Step 1: Write the failing test**

```python
def test_every_fold_reports_how_much_of_the_day_axis_its_nuisance_basis_absorbs() -> None:
    """fit.py's premise -- "NT is the same biology in every library, so whatever moves there is
    the library" -- is false in this deposit: the fourteen libraries sit at four differentiation
    days. Measured, not asserted: V captures 0.999 of the day-7-to-day-14 NT direction in every
    fold, while pure sampling noise projects only 0.067 into the same basis.
    """
    for library in arm_slice().libraries:
        fold = fit_fold(arm_slice(), library)
        assert fold.day_axis_in_nuisance_basis == pytest.approx(0.999, abs=0.005)
```

- [ ] **Step 2: Run, confirm it fails on the missing attribute.**
- [ ] **Step 3: Implement** `FittedFold.day_axis_in_nuisance_basis`, computed inside `fit_fold` from `library_day`.
- [ ] **Step 4: Replace the false premise in the docstring** with the measured statement, keeping the existing declared-bias sentence, which was correct and is what made this findable.
- [ ] **Step 5: `make check`, commit, PR.**

---

## Phase 5 — Re-ingest at the cell

### Task 15: Move ingestion into the package, with `h5py` as a real dependency

**Spec finding:** #07 — 364 lines outside the package, outside coverage, outside CI, with a hardcoded absolute path; no `h5py` in the runtime dependency set.

**Files:**
- Create: `src/cellstate/data/ingest/__init__.py`, `src/cellstate/data/ingest/gse274113.py`
- Modify: `pyproject.toml:22-26`, `Makefile`
- Delete: `scripts/gse274113_build_slice.py`, `scripts/gse274113_build_panel.py`
- Test: `tests/test_ingest_gse274113.py` (new)

- [ ] **Step 1: Write the failing test** — ingestion from a synthetic 3-cell HDF5 fixture built in the test, asserting the emitted payload schema and that a missing source path raises a typed error rather than returning an empty slice.
- [ ] **Step 2: Run, confirm it fails on the missing module.**
- [ ] **Step 3: Add `h5py>=3.16` to `[project.dependencies]`, run `uv lock`, confirm `make lock-check` passes.**
- [ ] **Step 4: Port the two scripts into the package**, replacing the hardcoded `DATA` path with a required argument and an environment-variable fallback.
- [ ] **Step 5: Verify byte-identity** — rebuild the committed slice from the raw h5 files and confirm `git diff --exit-code backends/vertical-a/gse274113-rna-obs-v1/`. **If the bytes differ, stop**: the port changed a number, and that must be understood before it ships.
- [ ] **Step 6: `make check`, commit, PR.**

---

### Task 16: Keep the guide axis, split AAVS1 from NT, keep the cell types

**Spec finding:** #07 — 63 guides collapse to 20 labels; the NT arm merges 3 true non-targeting guides (7,304 cells) with 3 AAVS1 safe-harbour *cutting* guides (4,143 cells).

**Files:**
- Modify: `src/cellstate/data/ingest/gse274113.py`, `backends/vertical-a/gse274113-rna-obs-v1/arms.json` (regenerated)
- Modify: `scripts/explore.py` (`knockdown`'s two deferred controls from Task 8)
- Test: `tests/test_ingest_gse274113.py`, `tests/test_relative_growth_rate.py`

> ⚠️ Regenerating `arms.json` changes the committed evidence. Its fingerprint is referenced by ADRs and manifests — **find every reference before regenerating** (`grep -rn "slice_id\|panel_sha256\|arms.json"`), and add the new arms rather than replacing the existing ones if any frozen artifact binds the old digest.

- [ ] **Step 1: Write the failing tests** — the slice carries 63 guide-level arms per library; `AAVS1` and `NT` are distinct targets; `annotation_simplified` composition is present per arm.
- [ ] **Step 2: Run, confirm they fail.**
- [ ] **Step 3: Implement** the three retained columns.
- [ ] **Step 4: Measure the three-level S4** — no cut (`NT`) / cut at a safe harbour (`AAVS1`) / cut in a gene (targeting) — with library-grouped intervals. Report all three bands.
- [ ] **Step 5: Fill `knockdown`'s two deferred controls** — guide-level ICC and the cutting contrast — replacing the typed absences from Task 8.
- [ ] **Step 6: `make check`, commit, PR.**

---

## Phase 6 — Delete

### Task 17: Remove the suspended stack

**Spec finding:** #09 — 52,471 lines, 47% of the repository, governing work the roadmap declares suspended, with zero executed statements on any live path.

**Files:**
- Delete: `src/cellstate/evaluation/sciplex3_*.py`, `src/cellstate/backends/sciplex3_*.py`, `src/cellstate/training/item12_3_authorization.py`, `src/cellstate/training/execution.py`, and their tests
- Modify: `src/cellstate/backends/__init__.py` (stop the eager import)

- [ ] **Step 1: Verify each module is dead before deleting it**

```bash
for m in sciplex3_candidate sciplex3_runner sciplex3_loader item12_3_authorization; do
  echo "=== $m ==="; grep -rn "$m" src/ examples/ scripts/ --include=*.py | grep -v "^src/cellstate/.*/$m.py"
done
```

A module whose only importers are its own tests is dead for scientific purposes. **Anything with a live importer stays** and is reported instead of deleted.

- [ ] **Step 2: Check the roadmap actually declares each one suspended**, and quote it in the commit message. Do not delete on this plan's authority alone.
- [ ] **Step 3: Delete, and stop `backends/__init__.py` importing the stack eagerly.**
- [ ] **Step 4: `make check`.** Coverage will move — record the new figure. If it drops below 85, that is information: the deleted code was inflating it.
- [ ] **Step 5: Commit and PR.**

---

### Task 18: Move `explore.py` into the package and recalibrate the coverage floor

**Spec finding:** #09 — the most scientifically productive file in the repository lives in `scripts/` deliberately, to dodge an 85% coverage floor.

**Files:**
- Create: `src/cellstate/explore/`
- Modify: `pyproject.toml:74` (floor), `Makefile`, `docs/guides/explore-the-system.md`
- Delete: `scripts/explore.py`

- [ ] **Step 1: Move the module, keeping every command and its tests.**
- [ ] **Step 2: Measure the real coverage** and set the floor to what the repository actually sustains with its science inside it. **Record the old and new floor and the reason in `pyproject.toml` as a comment** — a lowered gate with no explanation is indistinguishable from a weakened one.
- [ ] **Step 3: Add `explore` to `make check`.** It was outside every gating lane.
- [ ] **Step 4: `make check`, commit, PR.**

---

## Out of scope

**Buying a second donor** (audit step 7). Not scheduled here. It becomes decidable only after Phase 1 gives at least one criterion a demonstrated pass, and it is gated on the repaired `spectrum` screen from Task 8. Attempting it earlier reproduces the failure this plan exists to correct.

**Declared-axis biology basis** (audit step 4, second half). Task 13 fixes *how* the basis is fitted; replacing the fitted basis with a priori lineage axes is a larger change that should be scored against Task 13's improved fitted basis, not against the current one. Schedule it after Task 13 reports.

---

## Self-review

**Spec coverage.** Audit findings #01→Tasks 4–7 · #02→Tasks 1–2 · #03→Task 8 · #04→Task 13 · #05→Task 14 · #06→Tasks 9–12 · #07→Tasks 15–16 · #08 (DISTRIBUTION vs MEAN)→**no task**; it is a contract-semantics correction requiring its own ADR and is deliberately deferred, since Task 16 changes what the subject *is* and the label should be corrected once, afterwards. #09→Tasks 17–18 · #10 (unfailable thresholds)→**no task**; the audit notes the OOD threshold must ship with an OOD-estimand ADR that does not exist yet, so it is not schedulable here. #11→Task 3.

**Two gaps are therefore deliberate and named**: the `DISTRIBUTION`/`MEAN` label and the unfailable thresholds. Both need ADRs whose content depends on work in this plan.

**Type consistency.** `measure_relative_growth_rate` returns `dict[str, CapabilityMeasurement]` in Tasks 10 and 16. `observation_variance(depth, cells)` gains its parameter in Task 12 and is used with it thereafter. `DEGENERATE_STATE_FLOOR` is defined in Task 5 and referenced in Task 6. `_amplified` is defined in Task 6 and reused in Task 7.

**Ordering risk.** Task 12 moves every pinned figure in the repository. Tasks 13 and 14 move some of them again. Do not run Tasks 12–14 in parallel branches — each must land before the next re-pins.
