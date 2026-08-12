"""Enforce the roadmap's own program rules 1 and 3.

Rule 1 (the purpose test) requires every implementation-queue item to name the state-capability
ledger entries it advances. Rule 2 exists because a rule that nothing checks is a guard that cannot
fire; these tests are that check for rules 1 and 3.

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


def test_queue_ids_do_not_reuse_historical_item_numbers() -> None:
    """Historical Item 1-12 numbering is bound into content-addressed manifests."""

    section = _section(QUEUE_HEADING)
    assert not re.search(r"^\d+\.\s+\*\*(?!`Q)", section, flags=re.MULTILINE), (
        "queue items must be identified by a Q-prefixed ID, never a bare historical item number"
    )
