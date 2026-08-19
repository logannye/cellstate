"""The perturbation modality is recorded per source, and refused when it is not.

The defect this closes: ``GSE274113`` was screened with an on-target-mRNA control calibrated to
CRISPRi and declared a measured null. It is Cas9 nuclease knockout. Fifteen thousand lines of
admission, manifest and representability machinery, and not one check asked what the perturbation
technology was -- because modality is not in the bytes, and every check computed from the bytes.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest

from cellstate.data import (
    AssignmentMechanism,
    DatasetManifest,
    InterventionCapability,
    PerturbationModality,
)
from cellstate.data.modality_registry import (
    REGISTERED_MODALITIES,
    UnrecordedModalityError,
    on_target_expression_is_a_validity_control,
    perturbation_modality_for,
)

# The two properties the whole module exists to express. Named here so a change to the enum has to
# come past this table rather than past a reader's attention.
ACTS_ON_TRANSCRIPTION = {
    PerturbationModality.CRISPRI,
    PerturbationModality.CRISPRA,
    PerturbationModality.RNAI,
    PerturbationModality.OVEREXPRESSION,
}


def test_an_unrecorded_source_is_refused_and_never_defaulted() -> None:
    """The refusal is the point. A default is what produced the original defect."""

    with pytest.raises(UnrecordedModalityError, match="not recorded"):
        perturbation_modality_for("GSE999999")

    with pytest.raises(UnrecordedModalityError):
        on_target_expression_is_a_validity_control("some-unregistered-source")


def test_gse274113_is_recorded_as_a_nuclease_knockout() -> None:
    """Science 10.1126/science.ads7951: lentiviral sgRNA at low MOI *with Cas9 protein*."""

    assert perturbation_modality_for("GSE274113") is PerturbationModality.CAS9_NUCLEASE_KNOCKOUT
    assert on_target_expression_is_a_validity_control("GSE274113") is False


def test_a_transcription_acting_source_may_still_be_screened_on_expression() -> None:
    """The check discriminates. A guard that refused every source would also 'pass' this file."""

    assert perturbation_modality_for("replogle-2022-k562-essential") is PerturbationModality.CRISPRI
    assert on_target_expression_is_a_validity_control("replogle-2022-k562-essential") is True


def test_lookup_is_case_insensitive_on_the_source_key() -> None:
    """A caller holding a lowercased slice identifier must not be told the source is unrecorded."""

    assert perturbation_modality_for("gse274113") is PerturbationModality.CAS9_NUCLEASE_KNOCKOUT


@pytest.mark.parametrize("modality", list(PerturbationModality))
def test_every_modality_declares_whether_it_acts_on_transcription(
    modality: PerturbationModality,
) -> None:
    """Every enum member is classified, so a new modality cannot be added and left unclassified.

    This is the branch that would otherwise go unfired: a modality added without a decision would
    fall to whichever side the property's default happened to be.
    """

    assert modality.acts_on_transcription is (modality in ACTS_ON_TRANSCRIPTION)


def test_the_registry_entries_are_self_consistent() -> None:
    """Keys are the upper-cased source key, and every entry cites where the modality was read."""

    for key, entry in REGISTERED_MODALITIES.items():
        assert key == entry.source_key.upper(), f"{key} does not match its source_key"
        assert entry.citation.strip(), f"{key} records no citation"
        assert entry.note.strip(), f"{key} records no note"


def test_a_dataset_without_interventions_cannot_declare_a_modality() -> None:
    """The manifest field mirrors the rule already enforced for every other intervention field."""

    with pytest.raises(ValueError, match="without interventions"):
        InterventionCapability(
            assignment=AssignmentMechanism.NONE,
            perturbation_modality=PerturbationModality.CRISPRI,
        )


def test_an_unset_modality_is_absent_from_the_canonical_payload() -> None:
    """This is what lets the three reviewed 0.3 manifests keep their content-addressed digests.

    They are frozen evidence and predate the field. Had it been serialized as an explicit null,
    every one of their fingerprints -- and the proofs bound to them -- would have moved. Absent
    means absent, which is the same incremental-migration rule the 0.3-compatible identity and
    permission fields already use.

    Making the field *required* would have invalidated all three, which is why the registry above,
    and not this field, is what actually gates the screens.
    """

    root = Path(__file__).resolve().parents[1]
    reviewed = sorted((root / "data_manifests" / "reviewed").glob("*.json"))
    assert len(reviewed) == 3, f"expected three reviewed manifests, found {len(reviewed)}"

    for path in reviewed:
        manifest = DatasetManifest.model_validate_json(path.read_text(encoding="utf-8"))
        interventions = manifest.canonical_payload["capabilities"]["interventions"]
        assert "perturbation_modality" not in interventions, (
            f"{path.name} would have its fingerprint moved by the new field"
        )
        # The digest is self-consistent, which is the property the frozen proofs are bound to.
        assert sha256(manifest.canonical_json_bytes).hexdigest() == manifest.fingerprint


def test_a_set_modality_does_enter_the_canonical_payload() -> None:
    """The pop must be conditional, not unconditional.

    If it dropped the field unconditionally the migration would be silent in the other direction:
    a manifest that recorded its modality would hash as though it had not, and the record would
    stop being content-addressed for the one field this task added.
    """

    root = Path(__file__).resolve().parents[1]
    path = root / "data_manifests" / "reviewed" / "replogle-2022-k562-essential.json"
    manifest = DatasetManifest.model_validate_json(path.read_text(encoding="utf-8"))

    updated = manifest.capabilities.interventions.model_copy(
        update={"perturbation_modality": PerturbationModality.CRISPRI}
    )
    with_modality = manifest.model_copy(
        update={"capabilities": manifest.capabilities.model_copy(update={"interventions": updated})}
    )
    interventions = with_modality.canonical_payload["capabilities"]["interventions"]
    assert interventions["perturbation_modality"] == "crispri"
    assert with_modality.fingerprint != manifest.fingerprint
