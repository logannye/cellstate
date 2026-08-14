"""Enforce the roadmap's own program rules 1 and 3.

Rule 1 (the purpose test) requires every implementation-queue item to name the state-capability
ledger entries it advances. Rule 2 exists because a rule that nothing checks is a guard that cannot
fire; these tests are that check for rules 1 and 3.

Rule 3 also says queue IDs are **ordinals, not stable identifiers**, which makes every prose
reference to one a reference that a reorder silently invalidates.  A citation that dangles is
harmless -- a reader notices ``Q12`` in a nine-item queue.  A citation that still *resolves*, to a
different item than the author meant, is the dangerous case: after ADR 0019 and ADR 0020 the
Phase 1 graduation gate still said the observational floor was measured at ``Q5``, and ``Q5`` by
then held the observation model, which measures no floor.  The gate read as satisfiable and pointed
at an item that could never satisfy it.

So the cross-reference tests below do not check that a cited ID exists.  They check that it names
the item the sentence is *about*, by pinning each citation to a title regex.  The pin is the part a
renumber cannot quietly follow.

**The population these tests cover is ``docs/roadmap.md`` only.**  Rule 3's other half -- that
documents *outside* this file cite artifacts and ADRs rather than queue IDs -- is not enforced,
because every ADR from 0015 onward cites queue IDs and is read through a mapping table instead.
That convention is stated in rule 4 and is carried by the ADRs themselves; nothing checks it.  A
reader should not mistake a green run here for the whole of rule 3.

Rules 2 and 4 are not enforced here. They need a diff-aware pull-request job, which does not exist,
and this module deliberately does not pretend otherwise.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROADMAP = Path(__file__).resolve().parents[1] / "docs" / "roadmap.md"

LEDGER_HEADING = "## The state-capability ledger"
QUEUE_HEADING = "## Implementation queue"

_LEDGER_ROW = re.compile(r"^\|\s*(S\d+)\s*\|")
_QUEUE_ITEM = re.compile(r"^(\d+)\.\s+\*\*`(Q\d+)`")
_CAPABILITY_TAG = re.compile(r"\[((?:S\d+)(?:,\s*S\d+)*)\]")
_QUEUE_ID = re.compile(r"`(Q\d+)`")


def _section(heading: str) -> str:
    """Return the text between ``heading`` and the next level-two heading."""

    text = ROADMAP.read_text(encoding="utf-8")
    start = text.index(heading) + len(heading)
    remainder = text[start:]
    end = remainder.find("\n## ")
    return remainder if end == -1 else remainder[:end]


@pytest.fixture(scope="module")
def ledger_ids() -> frozenset[str]:
    ids = {
        match.group(1)
        for line in _section(LEDGER_HEADING).splitlines()
        if (match := _LEDGER_ROW.match(line.strip()))
    }
    return frozenset(ids)


@pytest.fixture(scope="module")
def queue_items() -> list[tuple[int, str, str]]:
    """Return ``(ordinal, queue_id, body)`` for each implementation-queue item."""

    items: list[tuple[int, str, list[str]]] = []
    for line in _section(QUEUE_HEADING).splitlines():
        match = _QUEUE_ITEM.match(line)
        if match:
            items.append((int(match.group(1)), match.group(2), [line]))
        elif items and line.startswith("   "):
            items[-1][2].append(line)
    return [(ordinal, queue_id, "\n".join(body)) for ordinal, queue_id, body in items]


def test_ledger_defines_a_contiguous_capability_set(ledger_ids: frozenset[str]) -> None:
    assert ledger_ids, "the state-capability ledger defines no capabilities"
    expected = {f"S{index}" for index in range(1, len(ledger_ids) + 1)}
    assert ledger_ids == expected, "ledger IDs must be S1..Sn with no gaps"


def test_queue_is_a_single_ordered_list(queue_items: list[tuple[int, str, str]]) -> None:
    assert queue_items, "the implementation queue contains no items"
    ordinals = [ordinal for ordinal, _, _ in queue_items]
    assert ordinals == list(range(1, len(ordinals) + 1)), (
        "rule 3: the queue must be one contiguous ordered list"
    )


def test_queue_ids_are_unique_and_ordered(queue_items: list[tuple[int, str, str]]) -> None:
    queue_ids = [queue_id for _, queue_id, _ in queue_items]
    assert len(set(queue_ids)) == len(queue_ids), "duplicate queue ID"
    assert queue_ids == sorted(queue_ids, key=lambda value: int(value[1:])), (
        "queue IDs must ascend with their ordinals"
    )


def test_every_queue_item_names_a_ledger_capability(
    queue_items: list[tuple[int, str, str]],
    ledger_ids: frozenset[str],
) -> None:
    """Rule 1: an item that advances no ledger capability is not scheduled."""

    for _, queue_id, body in queue_items:
        tags = _CAPABILITY_TAG.search(body)
        assert tags is not None, (
            f"{queue_id} names no state capability; rule 1 forbids scheduling it"
        )
        claimed = {token.strip() for token in tags.group(1).split(",")}
        unknown = claimed - ledger_ids
        assert not unknown, (
            f"{queue_id} cites capabilities absent from the ledger: {sorted(unknown)}"
        )


def _normalized(text: str) -> str:
    """Collapse wrapping so a citation split across lines still matches as one phrase."""

    return re.sub(r"\s+", " ", text)


@pytest.fixture(scope="module")
def queue_titles() -> dict[str, str]:
    """Return ``{queue_id: title}`` for each item, where the title is its bolded heading line."""

    titles: dict[str, str] = {}
    for line in _section(QUEUE_HEADING).splitlines():
        if match := _QUEUE_ITEM.match(line):
            titles[match.group(2)] = _normalized(line)
    return titles


class CrossReference:
    """A prose citation of a queue ID, pinned to the item the sentence is about.

    ``citation`` locates the reference and captures the ID actually written.  ``anchor`` identifies
    the intended item by a phrase from its title, which a renumber does not change.  When the two
    disagree the roadmap is pointing a commitment at the wrong item.
    """

    def __init__(self, *, what: str, citation: str, anchor: str) -> None:
        self.what = what
        self.citation = re.compile(citation)
        self.anchor = re.compile(anchor)


CROSS_REFERENCES = (
    CrossReference(
        what="the starting-work pointer names the next item",
        citation=r"the next action is item `(Q\d+)`",
        # Moved from the observation model to the modality test when Q5 was delivered.  The pin is
        # updated deliberately, which is the whole point: the guard refused to let the pointer
        # drift silently onto a completed item.
        anchor=r"run the held-out-modality test",
    ),
    CrossReference(
        what="the starting-work pointer names the blocked estimand freeze",
        citation=r"`(Q\d+)`, the estimand freeze, is blocked",
        anchor=r"freeze the state-bearing estimand",
    ),
    CrossReference(
        what="the delivered-with-negatives note names the observation-model item",
        citation=(
            r"\*\*`(Q\d+)` and `Q\d+` are\s+delivered, and their measurements"
            r" are negative\*\*"
        ),
        anchor=r"fit the observation model",
    ),
    CrossReference(
        what="the delivered-with-negatives note names the first-belief item",
        citation=(
            r"\*\*`Q\d+` and `(Q\d+)` are\s+delivered, and their measurements"
            r" are negative\*\*"
        ),
        anchor=r"emit the first biological belief",
    ),
    CrossReference(
        what="Phase 1's gate names the item at which the specification-only constraint binds",
        citation=r"a constraint that first binds at `(Q\d+)`",
        anchor=r"freeze the state-bearing estimand",
    ),
    CrossReference(
        what="Phase 1's gate names the item that measures the observational floor",
        citation=r"The floor itself is measured at `(Q\d+)`",
        anchor=r"measure its floor",
    ),
    CrossReference(
        what="the observation-model item names where the modality test went",
        citation=r"held-out-modality test therefore moves to `(Q\d+)`",
        anchor=r"run the held-out-modality test",
    ),
    CrossReference(
        what="the modality-test item names the item it was carved out of",
        citation=r"Carries the test moved out of `(Q\d+)` by",
        anchor=r"fit the observation model",
    ),
    CrossReference(
        what="the source-review item names the item that uses the library nuisance axis",
        citation=r"retained under rule 7 for `(Q\d+)`, where a measured library nuisance axis",
        anchor=r"fit the observation model",
    ),
)


def test_every_cited_queue_id_resolves_to_an_item(queue_titles: dict[str, str]) -> None:
    """A reference to a retired or not-yet-created ordinal is a dangling citation."""

    cited = set(_QUEUE_ID.findall(ROADMAP.read_text(encoding="utf-8")))
    dangling = cited - set(queue_titles)
    assert not dangling, f"roadmap cites queue IDs that no item carries: {sorted(dangling)}"


@pytest.mark.parametrize("reference", CROSS_REFERENCES, ids=lambda ref: ref.what)
def test_cross_references_name_the_item_they_mean(
    reference: CrossReference,
    queue_titles: dict[str, str],
) -> None:
    """Rule 3: queue IDs are ordinals, so a reorder can leave a citation on the wrong item."""

    intended = [
        queue_id for queue_id, title in queue_titles.items() if reference.anchor.search(title)
    ]
    assert len(intended) == 1, (
        f"the anchor for {reference.what!r} matches {len(intended)} queue items ({intended}); "
        "it must identify exactly one, or the pin cannot detect a renumber"
    )

    found = reference.citation.findall(_normalized(ROADMAP.read_text(encoding="utf-8")))
    assert found, (
        f"the citation for {reference.what!r} is gone from the roadmap; "
        "delete this cross-reference or restore the sentence, but do not leave it unchecked"
    )
    for cited in found:
        assert cited == intended[0], (
            f"{reference.what}: the roadmap cites `{cited}`, but the item it means is "
            f"`{intended[0]}` ({queue_titles[intended[0]]}). A reorder moved the item and the "
            "citation stayed put."
        )


def test_queue_ids_do_not_reuse_historical_item_numbers() -> None:
    """Historical Item 1-12 numbering is bound into content-addressed manifests."""

    section = _section(QUEUE_HEADING)
    assert not re.search(r"^\d+\.\s+\*\*(?!`Q)", section, flags=re.MULTILINE), (
        "queue items must be identified by a Q-prefixed ID, never a bare historical item number"
    )
