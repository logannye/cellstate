"""GSE274113 is Cas9 nuclease knockout, not CRISPRi, and no live file may say otherwise.

*Science* 10.1126/science.ads7951 ("Perturb-multiome") states the delivery: "a lentiviral library
of guide RNAs targeting 18 hematopoietic master regulator transcription factors was introduced into
adult CD34+ purified human stem and progenitor cells at a low multiplicity of infection **with Cas9
protein**."  That is nuclease knockout.

The distinction is not cosmetic, and it is the reason this guard exists.  CRISPRi (dCas9-KRAB)
represses transcription, so on-target mRNA *must* fall and "roughly -1 to -2 log2 fold change" is
the correct expectation for it.  Nuclease cutting destroys the **protein**; the transcript falls
only through nonsense-mediated decay, and edits that escape NMD -- or that de-repress an
autoregulatory promoter, as a transcriptional repressor's own knockout does -- give zero or positive
log2 fold change.  This repository declared the deposit "a measured null" by applying the CRISPRi
expectation to a nuclease experiment, and that verdict then explained away every failing capability
measurement.

**The predicate is scoped to GSE274113, deliberately.**  Two of this repository's other registered
sources genuinely are CRISPRi -- Replogle 2022 K562 and GWCD4i / GSE314342 -- and the documents that
say so are correct.  A guard that banned the word outright would demand that four true sentences be
falsified.  Every exemption below therefore names the accession it refers to, and
``test_every_exemption_still_refers_to_a_real_line`` fails if an exemption goes stale, so the
allowlist cannot quietly grow into a way of not looking.

Accepted ADR bodies and ``CHANGELOG.md`` are historical records under ``AGENTS.md``: they are
allowed to contain the claim they were written under, and are excluded.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Historical records: an accepted ADR's body is never rewritten, only its Status line (AGENTS.md).
#
# ``docs/superpowers/`` and this file itself are excluded for a different reason: their subject IS
# the retraction, so they must be able to quote the claim in order to retract it. A guard that
# forbade that would forbid explaining itself.
HISTORICAL = (
    ROOT / "docs" / "adr",
    ROOT / "CHANGELOG.md",
    ROOT / "docs" / "superpowers",
    Path(__file__).resolve(),
)

# Frozen evidence (AGENTS.md working practices). A reviewed manifest is not editable prose.
FROZEN = (ROOT / "data_manifests", ROOT / "audits", ROOT / "benchmarks", ROOT / "containers")

SUFFIXES = {".py", ".md", ".html", ".js", ".json", ".toml", ".yml", ".yaml"}

CRISPRI = re.compile(r"crispri", re.IGNORECASE)

# CRISPRi is also a legitimate VOCABULARY term. `PerturbationModality.CRISPRI` is an enum member
# naming a real technology, and the generated JSON schema carries its value; neither asserts
# anything about GSE274113. These are syntactic forms rather than prose, so they are removed from a
# line before the claim is looked for -- which keeps the predicate about the CLAIM instead of the
# word, and means the modality vocabulary can grow without anyone editing this guard.
VOCABULARY = re.compile(
    r"PerturbationModality\.CRISPR[IA]"  # a qualified enum reference
    r"|CRISPR[IA]\s*=\s*[\"']crispr[ia][\"']"  # the enum member's own definition
    r"|[\"']crispr[ia][\"'],?",  # a bare quoted value, as in the exported schema
    re.IGNORECASE,
)

# The retracted expectation. It only ever referred to GSE274113, so it is banned outright rather
# than scoped -- there is no correct threshold to substitute, because on-target mRNA is not a
# validity control for a nuclease knockout at all.
# The U+2212 MINUS SIGN alternative is deliberate and load-bearing, not a typo: the markdown and the
# web UI write the threshold with a typographic minus, and a guard that only recognised ASCII hyphen
# would have passed over every prose site while looking like it had checked them.
RETRACTED_EXPECTATION = re.compile(
    r"-1\s*to\s*-2|−1\s*to\s*−2|working CRISPRi screen",  # noqa: RUF001
    re.IGNORECASE,
)

# A retraction has to be able to name what it retracts, so each check exempts the lines that DENY
# its claim rather than assert it. The markers differ per check, and that difference is the point:
#
#   * a modality line is a retraction when it names the real modality ("Cas9") or announces the
#     withdrawal;
#   * an expectation line is a retraction when it attributes the -1 to -2 threshold to CRISPRi, or
#     announces the withdrawal. "Cas9" is deliberately NOT a marker here -- otherwise
#     "for Cas9 knockout a working screen still gives -1 to -2" would smuggle the threshold back
#     in under the very word that retracts it.
#
# Markers are looked for in a bounded window around the offending line, not in the whole file.
# Prose wraps at 100 characters, so a retraction sentence routinely spans two lines and neither
# half carries both halves of the thought; a strictly per-line rule made the two checks chase each
# other around a line break. One line of context each way is enough for a wrapped sentence and far
# too little for a re-assertion in a later paragraph, which is exactly the discrimination wanted.
#
# Both sets also carry a small vocabulary of past-reference phrases. A sentence that says what a
# field "previously read" is describing superseded text, not asserting it, and the repository needs
# to keep such sentences: a correction that erases what it corrected teaches nobody why. These are
# phrases rather than bare words on purpose -- "previously" alone would exempt "the CRISPRi arm
# previously showed no effect", which is an assertion wearing a past tense.
PAST_REFERENCE = ("previously read", "previously said", "used to read", "no longer", "once ")
MARKER_CONTEXT_LINES = 1
MODALITY_RETRACTION_MARKERS = ("cas9", "withdrawn", *PAST_REFERENCE)
EXPECTATION_RETRACTION_MARKERS = ("crispri", "withdrawn", *PAST_REFERENCE)

# Lines that legitimately say CRISPRi, each with the accession it refers to. A line matching one of
# these substrings in the named file is exempt.
EXEMPTIONS: tuple[tuple[str, str, str], ...] = (
    (
        "docs/data/representability/replogle-2022-k562.md",
        "CRISPRi",
        "Replogle 2022 K562 is a genuine genome-scale CRISPRi Perturb-seq screen.",
    ),
    (
        "docs/data/evidence-inventory.md",
        "CRISPRi",
        "GWCD4i / GSE314342 is a genuine CRISPRi screen in primary CD4 T cells.",
    ),
)


def _tracked_paths() -> list[Path]:
    """Every file git tracks, which is the honest definition of "in this repository".

    An earlier version walked the tree with a skip-list. That was wrong in a way worth recording:
    ``mkdocs build`` writes ``site/``, and the guard then scanned the RENDERED html of the Replogle
    and evidence-inventory pages, whose paths no longer matched the source-path exemptions. It
    failed the whole gate on two documents that were correct. Any generated directory would have
    done the same, so the skip-list was a population that had to be maintained by hand -- the exact
    shape this repository has been caught by before. Asking git removes the class.
    """

    result = subprocess.run(
        # --others --exclude-standard adds files that are new but not ignored. Without them the
        # guard cannot fail in the commit that INTRODUCES an offence, only in the next run: a new
        # file is invisible until it is tracked, which is exactly when review is over. Ignored
        # paths stay out, so generated trees like mkdocs' site/ are still excluded.
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    return [ROOT / name for name in result.stdout.split("\0") if name]


def _live_files() -> list[Path]:
    """The tracked, text-bearing files this guard is responsible for."""

    files: list[Path] = []
    for path in _tracked_paths():
        if not path.is_file() or path.suffix not in SUFFIXES:
            continue
        if any(str(path).startswith(str(excluded)) for excluded in (*HISTORICAL, *FROZEN)):
            continue
        files.append(path)
    return files


def _is_exempt(
    path: Path, line: str, markers: tuple[str, ...], context: tuple[str, ...] = ()
) -> bool:
    """Whether this line denies the claim rather than asserting it.

    ``context`` is the small window of neighbouring lines a wrapped sentence may have spilled its
    marker into. It defaults to empty so a caller testing one line in isolation gets the strict
    per-line rule.
    """

    window = " ".join((line, *context)).lower()
    if any(marker in window for marker in markers):
        return True
    relative = str(path.relative_to(ROOT))
    return any(relative == exempt_path and needle in line for exempt_path, needle, _ in EXEMPTIONS)


def _offences(pattern: re.Pattern[str], markers: tuple[str, ...]) -> list[str]:
    offences: list[str] = []
    for path in _live_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:  # pragma: no cover - binary that slipped the suffix filter
            continue
        lines = text.splitlines()
        for index, line in enumerate(lines):
            # Vocabulary is removed BEFORE the claim is looked for, so a line that only names the
            # technology -- an enum member, its exported schema value -- carries no claim to find.
            if not pattern.search(VOCABULARY.sub("", line)):
                continue
            low = max(0, index - MARKER_CONTEXT_LINES)
            high = min(len(lines), index + MARKER_CONTEXT_LINES + 1)
            context = tuple(lines[low:index] + lines[index + 1 : high])
            if not _is_exempt(path, line, markers, context):
                offences.append(f"{path.relative_to(ROOT)}:{index + 1}: {line.strip()[:100]}")
    return offences


def test_no_live_file_calls_gse274113_a_crispri_screen() -> None:
    """The claim, at every site that is not a historical record or another dataset's document."""

    offences = _offences(CRISPRI, MODALITY_RETRACTION_MARKERS)
    assert offences == [], (
        "GSE274113 is Cas9 nuclease knockout (Science 10.1126/science.ads7951), not CRISPRi. "
        "Still claimed at:\n  " + "\n  ".join(offences)
    )


def test_no_live_file_carries_the_retracted_on_target_expectation() -> None:
    """ "A working CRISPRi screen gives roughly -1 to -2" is a dCas9-KRAB number.

    Deleted rather than rewritten: for a nuclease knockout there is no threshold that would be
    correct in its place, because the transcript is not what the perturbation acts on.
    """

    offences = _offences(RETRACTED_EXPECTATION, EXPECTATION_RETRACTION_MARKERS)
    assert offences == [], (
        "on-target mRNA is not a validity control for a Cas9 nuclease knockout; the retracted "
        "expectation survives at:\n  " + "\n  ".join(offences)
    )


def test_vocabulary_uses_are_not_claims_but_prose_still_is() -> None:
    """Naming the technology is not asserting this deposit used it.

    The distinction has to hold in both directions, or the exemption becomes a way to write the
    claim: a quoted enum value is vocabulary, and a sentence about GSE274113 is not, even when the
    two appear in the same file.
    """

    assert VOCABULARY.sub("", "PerturbationModality.CRISPRI,") == ","
    assert VOCABULARY.sub("", '    "crispri",') == "    "
    assert VOCABULARY.sub("", '    CRISPRI = "crispri"') == "    "

    # Prose survives the strip, so the claim is still caught.
    prose = "GSE274113's CRISPRi arm is a measured null."
    assert CRISPRI.search(VOCABULARY.sub("", prose))


def test_the_guard_looks_at_a_real_population() -> None:
    """A guard whose population is empty or tiny passes for the wrong reason.

    This repository has found a guard whose predicate was right and whose population was a
    hand-written literal before; the population here is walked, and these assertions are what
    would notice if the walk stopped returning anything.
    """

    files = _live_files()
    assert len(files) > 100, f"only {len(files)} files walked; the population collapsed"

    # A generated directory must never enter the population. mkdocs writes `site/` containing the
    # rendered copy of every doc, including the two that legitimately say CRISPRi; scanning it
    # failed the gate on correct documents. Nothing untracked is this guard's business.
    assert not any("site/" in str(path.relative_to(ROOT)) for path in files)

    relative = {str(path.relative_to(ROOT)) for path in files}
    # The three documents most likely to carry the claim must be inside the population.
    assert "README.md" in relative
    assert "AGENTS.md" in relative
    assert "docs/backends/gse274113-rna-observation-model.md" in relative
    # Historical and frozen records must be outside it.
    assert not any(name.startswith("docs/adr/") for name in relative)
    assert not any(name.startswith("data_manifests/") for name in relative)


def test_the_patterns_match_what_they_claim_to_match() -> None:
    """The predicates, exercised directly. A regex that matches nothing is not a guard."""

    assert CRISPRI.search("the deposit's CRISPRi arm is a measured null")
    assert CRISPRI.search('kind=OntologyTerm(label="CRISPRi guide")')
    assert not CRISPRI.search("Cas9 nuclease knockout guide")

    assert RETRACTED_EXPECTATION.search("A working CRISPRi screen gives roughly -1 to -2.")
    assert RETRACTED_EXPECTATION.search("gives about -1 to -2")
    assert not RETRACTED_EXPECTATION.search("mean on-target log2FC is -0.058")

    # The typographic-minus branch, exercised. Every prose site in this repository wrote the
    # threshold with U+2212, so an ASCII-only pattern would have reported a clean sweep of the
    # markdown while matching none of it. These assertions are what make the `noqa: RUF001` honest.
    typographic = "gives roughly −1 to −2."  # noqa: RUF001 - U+2212, as every prose site writes it
    assert RETRACTED_EXPECTATION.search(typographic), "the U+2212 alternative must fire"
    ascii_only = re.compile(r"-1\s*to\s*-2", re.IGNORECASE)
    assert not ascii_only.search(typographic), (
        "an ASCII-only pattern misses every prose site, which is why the alternative exists"
    )


def test_a_re_assertion_is_caught_even_inside_a_document_that_retracts_it() -> None:
    """The exemption is per LINE, so a retraction elsewhere in the file buys no cover.

    This is the failure mode a file-scoped guard would have: one honest paragraph at the top would
    license the claim everywhere below it.
    """

    readme = ROOT / "README.md"
    retraction = "GSE274113 is Cas9 nuclease knockout, not CRISPRi."
    re_assertion = "GSE274113's CRISPRi arm is a measured null."

    # A re-assertion far enough from the retraction to be a separate thought is still caught: the
    # window is one line, so a later paragraph buys nothing.
    distant = tuple([""] * MARKER_CONTEXT_LINES)
    assert not _is_exempt(readme, re_assertion, MODALITY_RETRACTION_MARKERS, distant)

    assert _is_exempt(readme, retraction, MODALITY_RETRACTION_MARKERS), (
        "a line naming the real modality is a retraction"
    )
    assert not _is_exempt(readme, re_assertion, MODALITY_RETRACTION_MARKERS), (
        "a bare re-assertion must never be exempt"
    )


def test_the_retraction_markers_do_not_exempt_the_retracted_expectation() -> None:
    """ "Cas9" must not become a password that smuggles the -1 to -2 threshold back in.

    A sentence may say "GSE274113 is Cas9 nuclease knockout" and may explain that -1 to -2 was a
    CRISPRi number. It may not state -1 to -2 as this deposit's expectation. The expectation check
    therefore ignores the markers entirely.
    """

    smuggled = "For Cas9 knockout a working screen still gives roughly -1 to -2."
    assert RETRACTED_EXPECTATION.search(smuggled), "the regex must match the smuggled line"
    assert not _is_exempt(ROOT / "README.md", smuggled, EXPECTATION_RETRACTION_MARKERS), (
        "naming Cas9 must not exempt a line that states -1 to -2 as this deposit's expectation"
    )
    # And the marker set that WOULD have exempted it, to show the two sets genuinely differ.
    assert _is_exempt(ROOT / "README.md", smuggled, MODALITY_RETRACTION_MARKERS)


def test_a_past_tense_assertion_is_not_a_retraction() -> None:
    """The past-reference vocabulary must not become a way to keep asserting the claim.

    "This field previously read X" describes superseded text. "The CRISPRi arm previously showed
    no effect" asserts the claim in the past tense. The phrases are chosen so the first is exempt
    and the second is not.
    """

    readme = ROOT / "README.md"
    describing = 'This field previously read "transcriptional repression", the CRISPRi mechanism.'
    asserting = "GSE274113's CRISPRi arm previously showed no perturbation effect."

    assert _is_exempt(readme, describing, MODALITY_RETRACTION_MARKERS)
    assert not _is_exempt(readme, asserting, MODALITY_RETRACTION_MARKERS), (
        "a bare past tense must not exempt an assertion; only an explicit reference to superseded "
        "text does"
    )


def test_every_exemption_still_refers_to_a_real_line() -> None:
    """A stale exemption is a hole that nobody is looking through.

    If one of these documents is rewritten and no longer says CRISPRi, the exemption must be
    deleted rather than left standing as permission nobody needs.
    """

    for exempt_path, needle, reason in EXEMPTIONS:
        path = ROOT / exempt_path
        assert path.exists(), f"exemption points at a missing file: {exempt_path} ({reason})"
        assert needle in path.read_text(encoding="utf-8"), (
            f"exemption for {exempt_path} is stale -- the file no longer contains {needle!r}. "
            f"Delete the exemption. Reason it was granted: {reason}"
        )
