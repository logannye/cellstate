#!/usr/bin/env python3
"""Build the p1-only sci-Plex3 loader contract and Item 11 software fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from cellstate.domain.common import canonical_json_bytes
from cellstate.evaluation.sciplex3_baselines import (
    NO_ACTION,
    RNG_ALGORITHM,
    SCIPLEX3_BASELINE_IMPLEMENTATION_VERSION,
    SCIPLEX3_BASELINE_SAMPLES_PER_CASE_PER_SEED,
    SCIPLEX3_BASELINE_SEEDS,
    SCIPLEX3_FEATURE_COUNT,
    BaselineSampleRequest,
    CompoundDose,
    ImmutableCSRCounts,
    P1TrainingData,
    P1WellCounts,
    PredictionTarget,
    TargetCondition,
    fit_sciplex3_baseline_suite,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PREPARATION_DIRECTORY = REPOSITORY_ROOT / "benchmarks/artifacts/sciplex3-k562-24h-v1"
BENCHMARK_DIRECTORY = REPOSITORY_ROOT / "benchmarks/vertical-a/sciplex3-k562-24h-v1"
SUPPORT_DIRECTORY = BENCHMARK_DIRECTORY / "support"
P1_LOADER_CONTRACT_PATH = SUPPORT_DIRECTORY / "p1-loader-contract.json"
BASELINE_GOLDEN_FIXTURE_PATH = SUPPORT_DIRECTORY / "baseline-golden-fixtures.json"
MANIFEST_PATH = REPOSITORY_ROOT / "data_manifests/reviewed/sciplex3-k562-24h.json"
QUERY_PATH = BENCHMARK_DIRECTORY / "state-query.json"
BENCHMARK_PATH = BENCHMARK_DIRECTORY / "benchmark-artifact.json"
TARGET_VALUE_SCHEMA_PATH = SUPPORT_DIRECTORY / "target-value-schema.json"
SCORING_TRANSFORM_PATH = SUPPORT_DIRECTORY / "scoring-transform.json"


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_reference(
    path: Path,
    *,
    relative_path: str,
    encoding: str,
) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "byte_count": len(payload),
        "encoding": encoding,
        "relative_path": relative_path,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def build_p1_loader_contract() -> dict[str, Any]:
    source_verification_path = PREPARATION_DIRECTORY / "source-verification.json"
    feature_panel_path = PREPARATION_DIRECTORY / "feature-panel.json"
    partitions_path = PREPARATION_DIRECTORY / "partitions.json"
    source_verification = _load_object(source_verification_path)
    feature_panel = _load_object(feature_panel_path)
    partitions = _load_object(partitions_path)
    source = source_verification["source"]
    h5ad_structure = source_verification["h5ad_structure"]
    train = next(item for item in partitions["partitions"] if item["partition_role"] == "train")
    memberships = train["membership_artifacts"]
    for name, declared in memberships.items():
        path = PREPARATION_DIRECTORY / declared["relative_path"]
        observed = _artifact_reference(
            path,
            relative_path=declared["relative_path"],
            encoding=declared["encoding"],
        )
        expected = {
            key: declared[key] for key in ("byte_count", "encoding", "relative_path", "sha256")
        }
        if observed != expected:
            raise ValueError(f"p1 membership declaration drifted: {name}")

    source_fields = (
        "filename",
        "byte_count",
        "md5",
        "sha256",
    )
    structure_fields = (
        "matrix_encoding",
        "matrix_shape",
        "matrix_nonzero_count",
        "matrix_value_dtype",
        "required_observation_fields",
        "source_feature_axis_sha256",
    )
    feature_reference = _artifact_reference(
        feature_panel_path,
        relative_path=feature_panel_path.relative_to(REPOSITORY_ROOT).as_posix(),
        encoding="canonical_json_utf8_object_v1",
    )
    source_reference = _artifact_reference(
        source_verification_path,
        relative_path=source_verification_path.relative_to(REPOSITORY_ROOT).as_posix(),
        encoding="canonical_json_utf8_object_v1",
    )
    return {
        "artifact_schema": "sciplex3-k562-p1-loader-contract",
        "artifact_schema_version": "1.0.0",
        "artifacts": {
            "feature_panel": feature_reference,
            "source_verification": source_reference,
            **{
                name: {
                    key: declared[key]
                    for key in ("byte_count", "encoding", "relative_path", "sha256")
                }
                for name, declared in sorted(memberships.items())
            },
        },
        "bindings": {
            "benchmark_sha256": _sha256(BENCHMARK_PATH),
            "dataset_manifest_sha256": _sha256(MANIFEST_PATH),
            "query_sha256": _sha256(QUERY_PATH),
            "scoring_transform_sha256": _sha256(SCORING_TRANSFORM_PATH),
            "target_value_schema_sha256": _sha256(TARGET_VALUE_SCHEMA_PATH),
        },
        "feature_panel": {
            "feature_count": feature_panel["feature_count"],
            "ordered_feature_keys_sha256": feature_panel["ordered_feature_keys_sha256"],
            "source_feature_axis_sha256": h5ad_structure["source_feature_axis_sha256"],
        },
        "h5ad_structure": {key: h5ad_structure[key] for key in structure_fields},
        "heldout_memberships_referenced": False,
        "loader_outputs_can_mint_lifecycle_evidence": False,
        "partition": {
            "access_purpose": "train_parameters",
            "artifact_role": "train",
            "partition_id": "p1-train",
            "record_count": train["record_count"],
            "selector": {
                "cell_line": "K562",
                "plates": train["selector"]["plate"],
                "replicate": train["selector"]["replicate"],
            },
            "well_count": train["well_count"],
        },
        "scientifically_admissible_without_trusted_workflow_receipt": False,
        "source": {key: source[key] for key in source_fields},
    }


def _synthetic_counts(*rows: tuple[int, ...]) -> np.ndarray[Any, np.dtype[np.int64]]:
    matrix = np.zeros((len(rows), SCIPLEX3_FEATURE_COUNT), dtype=np.int64)
    for row_index, row in enumerate(rows):
        matrix[row_index, : len(row)] = row
    return matrix


def _synthetic_well(
    well_id: str,
    plate_id: str,
    condition: CompoundDose | None,
    *rows: tuple[int, ...],
) -> P1WellCounts:
    source_base = int(hashlib.sha256(well_id.encode()).hexdigest()[:12], 16) * 1_000
    return P1WellCounts(
        well_id=well_id,
        plate_id=plate_id,
        condition=condition,
        counts=ImmutableCSRCounts.from_dense(_synthetic_counts(*rows)),
        record_ids=tuple(f"{well_id}-record-{index:04d}" for index in range(len(rows))),
        source_row_indices=tuple(source_base + index for index in range(len(rows))),
    )


def _synthetic_training_data() -> tuple[P1TrainingData, list[dict[str, Any]]]:
    a10 = CompoundDose("compound-a", 10)
    a100 = CompoundDose("compound-a", 100)
    b10 = CompoundDose("compound-b", 10)
    b100 = CompoundDose("compound-b", 100)
    rows: tuple[tuple[str, str, CompoundDose | None, tuple[tuple[int, ...], ...]], ...] = (
        ("vehicle-a-1", "plate-a", None, ((1, 2, 1), (2, 1, 1), (1, 1, 2))),
        ("vehicle-a-2", "plate-a", None, ((30, 1, 1), (31, 2, 1))),
        ("a-10", "plate-a", a10, ((7, 2, 1), (9, 3, 2), (8, 4, 1))),
        ("a-100", "plate-a", a100, ((13, 3, 2), (15, 4, 3), (14, 5, 2))),
        ("vehicle-b", "plate-b", None, ((2, 3, 1), (3, 2, 1), (2, 2, 2))),
        ("b-10", "plate-b", b10, ((4, 9, 1), (5, 8, 2), (6, 10, 1))),
        ("b-100", "plate-b", b100, ((5, 15, 2), (7, 16, 2), (6, 14, 3))),
    )
    wells = tuple(
        _synthetic_well(well_id, plate_id, condition, *counts)
        for well_id, plate_id, condition, counts in rows
    )
    feature_keys = tuple(f"golden-feature-{index:04d}" for index in range(SCIPLEX3_FEATURE_COUNT))
    training = P1TrainingData(ordered_feature_keys=feature_keys, wells=wells)
    fixture = [
        {
            "condition": (
                {"kind": "no_action"}
                if condition is None
                else {
                    "compound": condition.compound,
                    "dose_nm": condition.dose_nm,
                    "kind": "compound_dose",
                }
            ),
            "plate_id": plate_id,
            "rows_first_coordinates": [list(row) for row in counts],
            "well_id": well_id,
        }
        for well_id, plate_id, condition, counts in rows
    ]
    return training, fixture


def _condition_payload(condition: TargetCondition) -> dict[str, Any]:
    if not isinstance(condition, CompoundDose):
        return {"kind": "no_action"}
    return {
        "compound": condition.compound,
        "dose_nm": condition.dose_nm,
        "kind": "compound_dose",
    }


def _sample_digest(samples: np.ndarray[Any, np.dtype[np.int64]]) -> str:
    canonical = np.asarray(samples, dtype="<i8", order="C")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def build_baseline_golden_fixture() -> dict[str, Any]:
    """Execute every baseline on a small software-only fixture and bind exact results."""

    training, fixture = _synthetic_training_data()
    baselines = fit_sciplex3_baseline_suite(training, low_rank=1)
    conditions: tuple[TargetCondition, ...] = (CompoundDose("compound-a", 10), NO_ACTION)
    results: list[dict[str, Any]] = []
    for baseline_id, baseline in sorted(baselines.items()):
        manifest_payload = canonical_json_bytes(baseline.fitted_state_manifest())
        samples: list[dict[str, Any]] = []
        for index, condition in enumerate(conditions):
            for seed in SCIPLEX3_BASELINE_SEEDS:
                request = BaselineSampleRequest(
                    target=PredictionTarget(
                        case_id=f"golden-case-{index}",
                        target_well_id=f"golden-target-{index}",
                        plate_id="golden-evaluation-plate",
                        partition_id="golden-software-fixture",
                        condition=condition,
                    ),
                    sample_count=SCIPLEX3_BASELINE_SAMPLES_PER_CASE_PER_SEED,
                    seed=seed,
                )
                output = baseline.sample(request)
                samples.append(
                    {
                        "condition": _condition_payload(condition),
                        "dtype": "little-endian-int64",
                        "panel_totals_sha256": hashlib.sha256(
                            np.asarray(
                                np.sum(output.samples, axis=1, dtype=np.int64),
                                dtype="<i8",
                            ).tobytes(order="C")
                        ).hexdigest(),
                        "sample_count": request.sample_count,
                        "samples_sha256": _sample_digest(output.samples),
                        "seed": request.seed,
                        "shape": list(output.samples.shape),
                    }
                )
        results.append(
            {
                "baseline_id": baseline_id,
                "fitted_state_manifest_byte_count": len(manifest_payload),
                "fitted_state_manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
                "samples": samples,
            }
        )
    return {
        "artifact_schema": "sciplex3-k562-baseline-golden-fixtures",
        "artifact_schema_version": "1.0.0",
        "biological_performance_evidence": False,
        "feature_count": SCIPLEX3_FEATURE_COUNT,
        "fixture": {
            "feature_key_template": "golden-feature-{zero_padded_index_4}",
            "purpose": "deterministic software conformance only",
            "wells": fixture,
        },
        "implementation_version": SCIPLEX3_BASELINE_IMPLEMENTATION_VERSION,
        "low_rank": 1,
        "production_sampling_contract": {
            "rng_algorithm": RNG_ALGORITHM,
            "samples_per_case_per_seed": SCIPLEX3_BASELINE_SAMPLES_PER_CASE_PER_SEED,
            "seeds": list(SCIPLEX3_BASELINE_SEEDS),
        },
        "results": results,
        "scientific_admission_authorized": False,
    }


def _emit(path: Path, payload: bytes, *, check: bool) -> None:
    if check:
        if not path.exists() or path.read_bytes() != payload:
            raise SystemExit(f"generated Item 11 artifact is stale: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    loader_payload = canonical_json_bytes(build_p1_loader_contract())
    golden_payload = canonical_json_bytes(build_baseline_golden_fixture())
    _emit(P1_LOADER_CONTRACT_PATH, loader_payload, check=args.check)
    _emit(BASELINE_GOLDEN_FIXTURE_PATH, golden_payload, check=args.check)
    print(f"p1_loader_contract_sha256 {hashlib.sha256(loader_payload).hexdigest()}")
    print(f"baseline_golden_fixture_sha256 {hashlib.sha256(golden_payload).hexdigest()}")


if __name__ == "__main__":
    main()
