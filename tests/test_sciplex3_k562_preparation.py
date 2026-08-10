from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

ARTIFACT_ROOT = Path(__file__).parents[1] / "benchmarks" / "artifacts" / "sciplex3-k562-24h-v1"
SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "prepare_sciplex3_k562.py"


def load_preparation_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("cellstate_sciplex3_preparation", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load sci-Plex3 preparation script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


preparation = load_preparation_module()


class RecordingVector:
    def __init__(self, values: list[int]) -> None:
        self.values = np.asarray(values, dtype=np.int64)
        self.requests: list[tuple[int | None, int | None]] = []

    def __getitem__(self, key: slice) -> np.ndarray:
        self.requests.append((key.start, key.stop))
        return self.values[key]


def load_artifact(filename: str) -> dict[str, object]:
    return json.loads((ARTIFACT_ROOT / filename).read_text(encoding="utf-8"))


def test_typed_composite_well_identity_is_canonical_json() -> None:
    identity = preparation.composite_well_id("plate1", "A01")

    assert identity == '["plate1","A01"]'
    assert json.loads(identity) == ["plate1", "A01"]
    with pytest.raises(ValueError, match="canonical"):
        preparation.composite_well_id(" plate1", "A01")


def test_partition_rules_cover_only_the_frozen_plate_design() -> None:
    assert preparation.partition_role("rep1", "plate1") == "train"
    assert preparation.partition_role("rep1", "plate8") == "train"
    assert preparation.partition_role("rep2", "plate25") == "calibration"
    assert preparation.partition_role("rep2", "plate28") == "model_selection_validation"
    assert preparation.partition_role("rep2", "plate32") == "untouched_test"

    with pytest.raises(ValueError, match="does not resolve"):
        preparation.partition_role("rep2", "plate24")
    with pytest.raises(ValueError, match="does not resolve"):
        preparation.partition_role("rep1", "plate25")


def test_membership_hash_uses_sorted_canonical_json_and_rejects_duplicates() -> None:
    digest, byte_count = preparation.hash_string_array(["well-b", "well-a"])
    encoded = b'["well-a","well-b"]'

    assert digest == hashlib.sha256(encoded).hexdigest()
    assert byte_count == len(encoded)
    with pytest.raises(ValueError, match="unique"):
        preparation.hash_string_array(["well-a", "well-a"])


def test_train_statistics_never_read_a_heldout_csr_interval() -> None:
    # Rows 1 and 3 are held out and deliberately contain large sentinel counts.
    data = RecordingVector([1, 2, 900, 3, 4, 700])
    indices = RecordingVector([0, 2, 1, 0, 1, 2])
    indptr = np.asarray([0, 2, 3, 5, 6], dtype=np.int64)

    statistics = preparation.compute_train_feature_statistics(
        data=data,
        indices=indices,
        indptr=indptr,
        train_rows=(0, 2),
        n_features=3,
        expected_train_library_sizes=np.asarray([3, 7], dtype=np.float64),
        row_batch_size=1,
    )

    assert data.requests == [(0, 2), (3, 5)]
    assert indices.requests == [(0, 2), (3, 5)]
    assert statistics.accessed_rows == (0, 2)
    assert statistics.accessed_count_total == 10
    assert statistics.detection_count.tolist() == [2, 1, 1]
    assert statistics.raw_count_total.tolist() == [4, 4, 2]


def test_feature_panel_selection_is_ordered_and_excludes_technical_or_ambiguous_ids() -> None:
    statistics = preparation.FeatureStatistics(
        detection_count=np.asarray([4, 4, 4, 4, 4, 4, 1, 4, 4], dtype=np.int64),
        raw_count_total=np.asarray([5, 5, 5, 5, 5, 5, 1, 10, 6], dtype=np.int64),
        log_sum=np.asarray([4, 4, 4, 4, 4, 4, 1, 8, 2], dtype=np.float64),
        log_sum_squares=np.asarray([6, 6, 6, 6, 6, 6, 1, 18, 3], dtype=np.float64),
        accessed_rows=(0, 1, 2, 3),
        accessed_library_count=4,
        accessed_count_total=47,
    )
    ensembl_ids = [
        "ENSG000001",
        "ENSG000002",
        "ENSG000003",
        "ENSG000004",
        "ENSG000004",
        "NOT_ENSEMBL",
        "ENSG000006",
        "ENSG000007",
        "ENSG000008",
    ]
    symbols = ["A", "MT-A", "RPL3", "DUP1", "DUP2", "E", "LOW", "C", "D"]

    panel, summary = preparation.select_train_feature_panel(
        statistics=statistics,
        ensembl_ids=ensembl_ids,
        gene_symbols=symbols,
        train_cell_count=4,
        panel_size=2,
        minimum_detection_fraction=0.5,
        mean_bin_count=1,
    )

    assert [(item["rank"], item["gene_symbol"]) for item in panel] == [(1, "D"), (2, "A")]
    assert summary == {
        "below_train_detection_threshold": 1,
        "duplicate_ensembl_id": 2,
        "eligible_feature_count": 3,
        "minimum_train_detection_count": 2,
        "mitochondrial_or_ribosomal_symbol": 2,
        "nonhuman_or_noncanonical_ensembl_id": 1,
        "selected_feature_count": 2,
    }


def test_checked_in_artifacts_are_content_addressed_and_matrix_free() -> None:
    index = load_artifact("artifact-index.json")
    records = index["artifacts"]
    assert isinstance(records, list)
    assert {record["relative_path"] for record in records} == set(preparation.OUTPUT_FILENAMES)
    assert index["source_sha256"] == preparation.EXPECTED_SOURCE_SHA256
    assert index["contains_source_matrix"] is False
    assert index["contains_normalized_matrix"] is False
    assert index["contains_materialized_membership_arrays"] is True
    assert index["membership_artifacts_are_canonical_json_bytes"] is True
    assert index["admission_decision"] == "not_made"

    script_bytes = SCRIPT_PATH.read_bytes()
    assert index["generator_script_sha256"] == hashlib.sha256(script_bytes).hexdigest()
    for record in records:
        path = ARTIFACT_ROOT / record["relative_path"]
        content = path.read_bytes()
        assert record["byte_count"] == len(content)
        assert record["sha256"] == hashlib.sha256(content).hexdigest()
        assert "/Volumes/" not in content.decode("utf-8")


def test_checked_in_universe_partitions_and_feature_panel_match_frozen_counts() -> None:
    source_verification = load_artifact("source-verification.json")
    assert source_verification["h5ad_structure"]["source_features_with_human_ensembl_id"] == (
        58_347
    )
    assert source_verification["h5ad_structure"]["source_features_with_mouse_ensembl_id"] == (
        52_636
    )

    universe = load_artifact("k562-universe.json")
    assert universe["record_count"] == 173_652
    assert universe["composite_well_count"] == 1_536
    assert universe["treated_well_count"] == 1_504
    assert universe["control_well_count"] == 32
    assert universe["source_perturbation_count"] == 188
    assert universe["condition_group_count"] == 753
    assert universe["treated_condition_group_count"] == 752
    assert universe["treated_wells_per_condition"] == 2
    assert universe["treated_condition_replicate_values"] == ["rep1", "rep2"]
    assert universe["composite_well_identity"] == {
        "definition": "compact JSON string array of the obs field values [plate, well]",
        "encoding": "canonical_json_utf8_string_array_v1",
        "kind": "COMPOSITE_SOURCE_FIELDS",
        "source_fields": ["plate", "well"],
    }
    universe_memberships = universe["membership_artifacts"]
    assert set(universe_memberships) == {
        "plate_ids",
        "record_ids",
        "record_to_well",
        "well_ids",
        "well_to_condition",
    }
    for reference in universe_memberships.values():
        content = (ARTIFACT_ROOT / reference["relative_path"]).read_bytes()
        assert reference["byte_count"] == len(content)
        assert reference["sha256"] == hashlib.sha256(content).hexdigest()

    universe_record_ids = json.loads(
        (ARTIFACT_ROOT / universe_memberships["record_ids"]["relative_path"]).read_text(
            encoding="utf-8"
        )
    )
    universe_well_ids = json.loads(
        (ARTIFACT_ROOT / universe_memberships["well_ids"]["relative_path"]).read_text(
            encoding="utf-8"
        )
    )
    record_to_well = json.loads(
        (ARTIFACT_ROOT / universe_memberships["record_to_well"]["relative_path"]).read_text(
            encoding="utf-8"
        )
    )
    assert len(universe_record_ids) == len(set(universe_record_ids)) == 173_652
    assert len(universe_well_ids) == len(set(universe_well_ids)) == 1_536
    assert len(record_to_well) == 173_652
    assert {pair[0] for pair in record_to_well} == set(universe_record_ids)
    assert {pair[1] for pair in record_to_well} == set(universe_well_ids)

    partitions = load_artifact("partitions.json")
    assert partitions["assignment_unit"] == "plate"
    assert partitions["record_unit"] == "_index"
    assert partitions["protected_parent_unit"] == "plate"
    assert partitions["metric_evaluation_unit"]["source_fields"] == ["plate", "well"]
    observed = {
        item["partition_role"]: {
            key: item[key]
            for key in (
                "record_count",
                "well_count",
                "treated_well_count",
                "control_well_count",
                "compound_count",
            )
        }
        for item in partitions["partitions"]
    }
    assert observed == preparation.EXPECTED_PARTITION_COUNTS
    assert partitions["cross_partition_checks"] == {
        "all_evaluation_conditions_have_rep1_train_counterparts": True,
        "protected_plate_overlap_count": 0,
        "record_overlap_count": 0,
        "rep2_compound_overlap_count": 0,
        "rep2_compound_union_count": 188,
        "well_overlap_count": 0,
    }
    for item in partitions["partitions"]:
        membership_artifacts = item["membership_artifacts"]
        record_ids = json.loads(
            (ARTIFACT_ROOT / membership_artifacts["record_ids"]["relative_path"]).read_text(
                encoding="utf-8"
            )
        )
        well_ids = json.loads(
            (ARTIFACT_ROOT / membership_artifacts["well_ids"]["relative_path"]).read_text(
                encoding="utf-8"
            )
        )
        plate_ids = json.loads(
            (ARTIFACT_ROOT / membership_artifacts["plate_ids"]["relative_path"]).read_text(
                encoding="utf-8"
            )
        )
        descendant_mapping = json.loads(
            (ARTIFACT_ROOT / membership_artifacts["record_to_well"]["relative_path"]).read_text(
                encoding="utf-8"
            )
        )
        condition_mapping = json.loads(
            (ARTIFACT_ROOT / membership_artifacts["well_to_condition"]["relative_path"]).read_text(
                encoding="utf-8"
            )
        )
        assert len(record_ids) == item["record_count"]
        assert len(well_ids) == item["well_count"]
        assert len(plate_ids) == len(item["selector"]["plate"])
        assert len(descendant_mapping) == item["record_count"]
        assert len(condition_mapping) == item["well_count"]
        assert {pair[0] for pair in descendant_mapping} == set(record_ids)
        assert {pair[1] for pair in descendant_mapping} == set(well_ids)
        assert {pair[0] for pair in condition_mapping} == set(well_ids)
        assert len({pair[1] for pair in condition_mapping}) == item["condition_group_count"]

    groups = load_artifact("well-groups.json")
    assert groups["group_count"] == len(groups["groups"]) == 1_536
    group_ids = [group["composite_well_id"] for group in groups["groups"]]
    assert len(group_ids) == len(set(group_ids))
    for group in groups["groups"]:
        assert group["composite_well_id"] == preparation.composite_well_id(
            group["plate"], group["well"]
        )

    intervention_labels = load_artifact("intervention-labels.json")
    mapping_reference = intervention_labels["mapping_artifact"]
    label_pairs = json.loads(
        (ARTIFACT_ROOT / mapping_reference["relative_path"]).read_text(encoding="utf-8")
    )
    assert len(label_pairs) == 189
    assert len({pair[0] for pair in label_pairs}) == 189
    assert len({pair[1] for pair in label_pairs}) == 189
    assert sum(source != normalized for source, normalized in label_pairs) == 11
    assert all(source.strip() == normalized for source, normalized in label_pairs)
    assert intervention_labels["normalization"]["ontology_mapping"] is False
    assert intervention_labels["normalization"]["chemical_name_resolution"] is False

    feature_panel = load_artifact("feature-panel.json")
    assert feature_panel["feature_count"] == len(feature_panel["features"]) == 2_000
    assert [item["rank"] for item in feature_panel["features"]] == list(range(1, 2_001))
    assert len({item["ensembl_id"] for item in feature_panel["features"]}) == 2_000
    assert feature_panel["feature_selection"]["count_accessed_partition_roles"] == ["train"]
    assert feature_panel["feature_selection"]["heldout_count_rows_accessed"] == 0
    assert feature_panel["feature_selection"]["accessed_source_row_count"] == 94_785
    assert feature_panel["transformation"]["formula"] == ("log1p(10000 * count / library_size)")
    assert feature_panel["transformation"]["preparation_or_fit_requires_heldout_counts"] is False
    assert (
        feature_panel["transformation"]["evaluation_requires_counts_for_each_evaluated_record"]
        is True
    )
