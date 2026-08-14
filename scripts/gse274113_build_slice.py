#!/usr/bin/env python3
"""Reduce the GSE274113 bytes to a checked-in panel slice: one count vector per arm.

An arm is one ``(library, target)`` population.  Fourteen libraries times twenty targets is 280
arms, and the census established that all 280 are populated.

Three things this does that are easy to get wrong, and are the reason it exists as a committed
runner rather than a notebook:

* **The annotated metadata carries blank rows.**  Every other row is empty.  Reading it without
  filtering doubles the cell count, and the doubling is silent.
* **The h5 carries more barcodes than the metadata annotates** -- 10,678 versus 6,611 in rep13.
  The unannotated barcodes are real cells at day 14, not empty droplets, so subsetting to the
  annotated set is a selection that has to be recorded, not assumed away.
* **The zero-panel doctrine.**  A *cell* whose panel total is zero is a missing observation, not a
  zero measurement; it is dropped and counted.  An *arm* whose panel total is zero is emitted with
  a null count vector so that no belief can be built for it.  Neither is imputed.

Not run in CI.  CI reads the frozen slice and checks its SHA-256.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

DATA = Path("/Users/logannye/Documents/Codex/2026-08-08/i-w/work/gse274113")
PANEL = Path("backends/vertical-a/gse274113-rna-obs-v1/panel.json")
OUT = Path("backends/vertical-a/gse274113-rna-obs-v1/arms.json")
REPS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 16]


def placebo_half(library: str, barcode: str) -> str:
    """Split NT cells deterministically so the S4 null half has a real, non-degenerate contrast.

    ADR 0019 warns that an inert ``do`` operator passes every test this repository has.  If the NT
    direction were structurally zero the null half could not fail.  Splitting NT into two halves
    gives it a genuine estimated direction whose expected magnitude is zero.
    """

    digest = hashlib.blake2b(f"placebo/{library}/{barcode}".encode(), digest_size=8).digest()
    return "NT_A" if digest[0] & 1 else "NT_B"


def annotation() -> tuple[dict[tuple[str, str], str], dict[str, int], int, int]:
    cell_target: dict[tuple[str, str], str] = {}
    day: dict[str, int] = {}
    rows = blank = 0
    with gzip.open(DATA / "GSE274113_annotated_metadata.csv.gz", "rt") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            library = row["replicate"]
            if not library:
                blank += 1
                continue
            cell_target[(library, row[""].split("_", 1)[1])] = row["target"]
            day[library] = int(row["Timepoint"].split()[1])
    return cell_target, day, rows, blank


def main() -> int:
    panel = json.loads(PANEL.read_text(encoding="utf-8"))
    panel_sha = hashlib.sha256(PANEL.read_bytes()).hexdigest()
    rows_wanted = np.array([gene["row_index"] for gene in panel["genes"]], dtype=np.int64)
    symbols = [gene["symbol"] for gene in panel["genes"]]

    cell_target, day, csv_rows, csv_blank = annotation()
    print(f"metadata rows={csv_rows:,} blank={csv_blank:,} annotated={len(cell_target):,}")

    counts: dict[tuple[str, str], np.ndarray] = defaultdict(
        lambda: np.zeros(len(rows_wanted), dtype=np.int64)
    )
    cells: dict[tuple[str, str], int] = defaultdict(int)
    dropped: dict[tuple[str, str], int] = defaultdict(int)

    row_slot = np.full(36601, -1, dtype=np.int64)
    row_slot[rows_wanted] = np.arange(len(rows_wanted))

    for rep in REPS:
        library = f"rep{rep}"
        path = DATA / f"GSE274113_{library}_filtered_feature_bc_matrix.h5"
        with h5py.File(path, "r") as handle:
            barcodes = [value.decode() for value in handle["matrix/barcodes"][:]]
            kinds = handle["matrix/features/feature_type"][:]
            gene_rows = np.flatnonzero(kinds == b"Gene Expression")
            gmin = int(gene_rows.min())
            indptr = handle["matrix/indptr"][:]
            indices = handle["matrix/indices"][:]
            values = handle["matrix/data"][:]

        for column, barcode in enumerate(barcodes):
            target = cell_target.get((library, barcode))
            if target is None:
                continue
            lo, hi = int(indptr[column]), int(indptr[column + 1])
            rows = indices[lo:hi].astype(np.int64) - gmin
            keep = (rows >= 0) & (rows < 36601)
            slots = row_slot[rows[keep]]
            on_panel = slots >= 0
            vector = np.zeros(len(rows_wanted), dtype=np.int64)
            if on_panel.any():
                np.add.at(vector, slots[on_panel], values[lo:hi][keep][on_panel].astype(np.int64))

            arms = [(library, target)]
            if target == "NT":
                arms.append((library, placebo_half(library, barcode)))
            for arm in arms:
                if vector.sum() == 0:
                    dropped[arm] += 1
                    continue
                counts[arm] += vector
                cells[arm] += 1
        print(f"  {library} done", flush=True)

    targets = sorted({target for (_, target) in counts if not target.startswith("NT_")})
    arms_out = []
    for library in (f"rep{rep}" for rep in REPS):
        for target in [*targets, "NT_A", "NT_B"]:
            key = (library, target)
            vector = counts.get(key)
            total = int(vector.sum()) if vector is not None else 0
            arms_out.append(
                {
                    "library": library,
                    "target": target,
                    "day": day[library],
                    "cells": int(cells.get(key, 0)),
                    "zero_panel_cells_dropped": int(dropped.get(key, 0)),
                    "panel_total": total,
                    # A zero-total arm is NOT MEASURED, never a vector of zeros.
                    "counts": [int(value) for value in vector] if total > 0 else None,
                }
            )

    payload = {
        "slice_id": "gse274113-rna-panel-arms-v1",
        "panel_id": panel["panel_id"],
        "panel_sha256": panel_sha,
        "gene_symbols": symbols,
        "libraries": [f"rep{rep}" for rep in REPS],
        "targets": targets,
        "placebo_targets": ["NT_A", "NT_B"],
        "library_day": {f"rep{rep}": day[f"rep{rep}"] for rep in REPS},
        "annotated_cells": len(cell_target),
        "metadata_rows_total": csv_rows,
        "metadata_rows_blank": csv_blank,
        "arms": arms_out,
    }
    text = json.dumps(payload, indent=1, sort_keys=True) + "\n"
    OUT.write_text(text, encoding="utf-8")

    real = [a for a in arms_out if not a["target"].startswith("NT_")]
    unmeasured = [a for a in real if a["counts"] is None]
    print(
        f"wrote {OUT}\n  arms={len(arms_out)}"
        f" (real={len(real)}, placebo={len(arms_out) - len(real)})"
        f"\n  cells retained={sum(a['cells'] for a in real):,}"
        f"  zero-panel cells dropped={sum(a['zero_panel_cells_dropped'] for a in real):,}"
        f"\n  NOT_MEASURED arms={len(unmeasured)}"
        f"\n  sha256={hashlib.sha256(text.encode()).hexdigest()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
