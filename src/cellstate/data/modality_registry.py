"""Which perturbation technology each registered biological source used, and what may be screened.

This module exists because of a specific failure, and it is scoped to prevent that failure rather
than to be general.  ``GSE274113`` was screened with an on-target-mRNA control -- "a working screen
gives roughly -1 to -2" -- and declared a measured null.  That threshold belongs to CRISPRi, where
dCas9-KRAB represses transcription and the transcript must fall.  The deposit is **Cas9 nuclease
knockout**: cutting destroys the protein, the transcript falls only through nonsense-mediated decay,
and edits that escape NMD, or that de-repress an autoregulatory promoter, leave it flat or raise it.
The screen therefore reported a failure on a criterion the assay does not answer to, and that
verdict was then cited as the reason every capability measurement came back negative.

**Why a registry and not a manifest field.**  ``InterventionCapability.perturbation_modality`` now
exists and new manifests should carry it, but it could not close this hole on its own: the three
reviewed manifests are frozen evidence and cannot be rewritten, and ``GSE274113`` -- the source that
motivated all of this -- has no reviewed manifest at all.  A field that the one relevant source
cannot populate is a claim recorded and never checked, which is the defect being repaired, not a
repair of it.  So the registry is keyed by source accession, is required for any source a screen
runs against, and refuses rather than defaulting.

Modality is not derivable from deposited bytes: a CRISPRi and a Cas9-nuclease deposit are
byte-identical in shape.  It is read from the methods, recorded here with its citation, and
checked by a test.
"""

from __future__ import annotations

from dataclasses import dataclass

from .manifests import PerturbationModality

__all__ = [
    "REGISTERED_MODALITIES",
    "RegisteredModality",
    "UnrecordedModalityError",
    "on_target_expression_is_a_validity_control",
    "perturbation_modality_for",
]


class UnrecordedModalityError(LookupError):
    """Raised when a screen asks about a source whose perturbation modality is not recorded.

    Deliberately an error and never a default.  A default is what produced the original defect:
    the pipeline behaved as though it knew the modality, and nothing in fifteen thousand lines of
    admission machinery asked whether it did.
    """


@dataclass(frozen=True)
class RegisteredModality:
    """One source's perturbation technology, with the citation it was read from.

    ``source_key`` is this repository's own identifier for the source, not a GEO accession: not
    every registered source has one. Replogle 2022 is distributed through Figshare and is keyed by
    its manifest ``dataset_id``, while ``GSE274113`` is keyed by the accession the repository uses
    for it everywhere. ``citation`` is where the modality was read from, which is always a
    publication -- never the bytes.
    """

    source_key: str
    modality: PerturbationModality
    citation: str
    note: str


REGISTERED_MODALITIES: dict[str, RegisteredModality] = {
    "GSE274113": RegisteredModality(
        source_key="GSE274113",
        modality=PerturbationModality.CAS9_NUCLEASE_KNOCKOUT,
        citation="Science 10.1126/science.ads7951 (Perturb-multiome)",
        note=(
            "A lentiviral library of guide RNAs targeting 18 haematopoietic master regulator "
            "transcription factors was introduced into adult CD34+ HSPCs at a low multiplicity of "
            "infection WITH CAS9 PROTEIN. Nuclease knockout: the target transcript is not what the "
            "perturbation acts on, so its fold change is not a validity control."
        ),
    ),
    "REPLOGLE-2022-K562-ESSENTIAL": RegisteredModality(
        source_key="replogle-2022-k562-essential",
        modality=PerturbationModality.CRISPRI,
        citation="Replogle et al. 2022, Cell, doi:10.1016/j.cell.2022.05.013",
        note=(
            "dCas9-KRAB CRISPRi in K562. Transcriptional repression, so on-target mRNA knockdown "
            "IS a validity control for this source and roughly -1 to -2 is the right expectation."
        ),
    ),
}


def perturbation_modality_for(source_key: str) -> PerturbationModality:
    """The recorded modality for ``accession``, or a refusal.

    ``source_key`` is matched case-insensitively so a caller passing a lowercased slice identifier
    does not silently miss a recorded source and get told it is unrecorded.
    """

    entry = REGISTERED_MODALITIES.get(source_key.strip().upper())
    if entry is None:
        raise UnrecordedModalityError(
            f"the perturbation modality of {source_key!r} is not recorded, so no "
            "modality-dependent screen may run against it. Modality cannot be computed from the "
            "deposited bytes; read it from the source publication's methods and add it to "
            "REGISTERED_MODALITIES."
        )
    return entry.modality


def on_target_expression_is_a_validity_control(source_key: str) -> bool:
    """Whether a change in the target's own mRNA is evidence that this source's screen worked.

    True only for modalities that act on transcription.  For a nuclease knockout, a base or prime
    edit, a small molecule or a degron, the target transcript is not the thing being perturbed, and
    a screen that reads its fold change as a pass/fail verdict is answering a question the assay
    was never asked.
    """

    return perturbation_modality_for(source_key).acts_on_transcription
