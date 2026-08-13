#!/usr/bin/env python3
"""Freeze the GSE274113 RNA observation panel.

The panel is declared from biology and from the experiment's own design -- the nineteen CRISPRi
target transcription factors, lineage markers spanning the haematopoietic tree this culture
differentiates through, and housekeeping depth anchors.  **It is never chosen from the response.**
Selecting genes by how they behave across arms would leak the very contrast the estimator is fitted
to, and no fold discipline downstream can undo that.

Run from the repository root with the GSE274113 bytes present.  Not run in CI: CI reads the frozen
artifact and checks its SHA-256.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py

DATA = Path("/Users/logannye/Documents/Codex/2026-08-08/i-w/work/gse274113")
OUT = Path("backends/vertical-a/gse274113-rna-obs-v1/panel.json")
REPS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 16]

# The nineteen perturbed transcription factors.  NT is the twentieth arm and is not a gene.
TARGET_TFS = (
    "ATF4 BCL11A FOSL1 GATA1 GATA2 GFI1B IRF1 IRF9 KLF1 LDB1 "
    "LMO2 MYB NFE2 PRDM16 RUNX1 SNAI2 SOX12 SPI1 TAL1"
).split()

MARKERS: dict[str, list[str]] = {
    "erythroid": "HBB HBG1 HBG2 HBA1 HBA2 ALAS2 GYPA GYPB SLC4A1 EPOR AHSP CA1 TFRC KEL".split(),
    "megakaryocyte": "PF4 PPBP ITGA2B VWF GP9 GP1BA THBS1 TUBB1".split(),
    "myeloid": "ELANE MPO PRTN3 LYZ S100A8 S100A9 CTSG AZU1 CEBPA CSF1R".split(),
    "eobasomast": "PRG2 CLC MS4A2 CPA3 HDC GATA3 IL5RA".split(),
    "progenitor": "CD34 AVP CRHBP HLF MECOM SPINK2 PROM1 KIT MPL".split(),
    "lymphoid": "DNTT VPREB1 IGLL1 IL7R CD79A".split(),
    "cell_cycle": "MKI67 TOP2A PCNA CCNB1 TYMS".split(),
    "housekeeping": (
        "ACTB GAPDH B2M TUBB RPL13A RPS18 PPIA TBP HPRT1 UBC SDHA YWHAZ EEF1A1 "
        "PGK1 RPLP0 PSMB2 VPS29 CHMP2A EMC7 GPI REEP5 SNRPD3 VCP"
    ).split(),
}


def gene_axis(rep: int) -> tuple[list[str], list[str]]:
    """Return (symbols, ensembl_ids) for the Gene Expression block of one library."""

    path = DATA / f"GSE274113_rep{rep}_filtered_feature_bc_matrix.h5"
    with h5py.File(path, "r") as handle:
        kinds = handle["matrix/features/feature_type"][:]
        names = handle["matrix/features/name"][:]
        ids = handle["matrix/features/id"][:]
    keep = kinds == b"Gene Expression"
    return (
        [value.decode() for value in names[keep]],
        [value.decode() for value in ids[keep]],
    )


def main() -> int:
    symbols, ensembl = gene_axis(REPS[0])

    # The gene axis is byte-identical across libraries (established in the representability
    # ledger), but assert it here rather than inherit it: this artifact pins row indices.
    for rep in REPS[1:]:
        other_symbols, other_ids = gene_axis(rep)
        if other_symbols != symbols or other_ids != ensembl:
            raise SystemExit(f"rep{rep} gene axis differs from rep{REPS[0]}")

    index_of: dict[str, int] = {}
    ambiguous: set[str] = set()
    for row, symbol in enumerate(symbols):
        if symbol in index_of:
            ambiguous.add(symbol)
        else:
            index_of[symbol] = row

    entries = []
    seen: set[str] = set()
    for category, members in [("target_tf", TARGET_TFS), *MARKERS.items()]:
        for symbol in members:
            if symbol in ambiguous:
                raise SystemExit(f"{symbol} maps to multiple rows; refusing an ambiguous panel gene")
            if symbol not in index_of:
                raise SystemExit(f"{symbol} is absent from the gene axis")
            if symbol in seen:
                raise SystemExit(f"{symbol} appears in the panel twice")
            seen.add(symbol)
            row = index_of[symbol]
            entries.append(
                {
                    "symbol": symbol,
                    "ensembl_id": ensembl[row],
                    "row_index": row,
                    "category": category,
                }
            )

    entries.sort(key=lambda entry: entry["row_index"])
    payload = {
        "panel_id": "gse274113-rna-panel-v1",
        "description": (
            "Declared RNA panel for the GSE274113 observation model: the 19 CRISPRi target "
            "transcription factors, haematopoietic lineage markers, and housekeeping depth "
            "anchors. Chosen from biology and from the experiment's design, never from the "
            "response."
        ),
        "gene_axis_sha256": hashlib.sha256(
            b"\x00".join(value.encode() for value in ensembl)
        ).hexdigest(),
        "gene_count": len(entries),
        "genes": entries,
    }

    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT}  genes={len(entries)}  sha256={hashlib.sha256(text.encode()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
