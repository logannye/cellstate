"""Trusted-shape, non-admissible execution boundary for sci-Plex3 baselines.

This module connects four deliberately narrow surfaces:

* the permanently ``p1-train``-bound immutable H5AD loader;
* an exact CSR-native well assembler with no caller-supplied biological metadata;
* the frozen action domain and outcome-free p4 evaluation-case design; and
* content-addressed fitted-state receipts and bounded predictive-sample shards.

It is not a locked evaluator.  It never opens p2, p3, or p4 outcome rows, never computes a
benchmark metric, never constructs a public biological response, and never issues lifecycle or
scientific-admission evidence.  A completed prediction manifest only proves deterministic software
materialization against exact content identities; it cannot make a ``BaselineRun`` pass.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Literal, cast

import numpy as np
import numpy.typing as npt

import cellstate.evaluation.sciplex3_baselines as _loaded_baselines_module
from cellstate.backends.sciplex3_k562 import (
    SCIPLEX3_K562_BENCHMARK_SHA256,
    SCIPLEX3_K562_MANIFEST_SHA256,
    SCIPLEX3_K562_QUERY_SHA256,
    SCIPLEX3_K562_TARGET_VALUE_SCHEMA_SHA256,
    PopulationComponentAccessPurpose,
)
from cellstate.backends.sciplex3_loader import (
    SCIPLEX3_FEATURE_PANEL_ARTIFACT_SHA256,
    SCIPLEX3_ORDERED_FEATURE_KEYS_SHA256,
    SCIPLEX3_P1_LOADER_CONTRACT_SHA256,
    SCIPLEX3_SCORING_TRANSFORM_SHA256,
    SCIPLEX3_SOURCE_BYTE_COUNT,
    SCIPLEX3_SOURCE_MD5,
    SCIPLEX3_SOURCE_SHA256,
    SciPlex3P1FinalizedCountScanReceipt,
    SciPlex3P1SourceScanReceipt,
    SciPlex3PartitionDescriptor,
    SciPlex3SparseCountBatch,
    SciPlex3TrainingDataLoader,
)
from cellstate.data.benchmarks import (
    BenchmarkArtifact,
    BenchmarkEvaluationCase,
    BenchmarkPartitionRole,
    EvaluationCaseRole,
)
from cellstate.domain.common import canonical_json_bytes
from cellstate.domain.query import StateQuery
from cellstate.errors import ContractViolationError
from cellstate.evaluation.sciplex3_baselines import (
    DEFAULT_LOW_RANK,
    NO_ACTION,
    RNG_ALGORITHM,
    SCIPLEX3_BASELINE_IMPLEMENTATION_VERSION,
    SCIPLEX3_BASELINE_IMPLEMENTATIONS,
    SCIPLEX3_BASELINE_SAMPLES_PER_CASE_PER_SEED,
    SCIPLEX3_BASELINE_SEEDS,
    SCIPLEX3_FEATURE_COUNT,
    BaselineSampleRequest,
    CompoundDose,
    ImmutableCSRCounts,
    LowRankCompoundDoseResponse,
    NoAction,
    P1TrainingData,
    P1WellCounts,
    PredictionTarget,
    PredictiveRawCountSamples,
    SciPlex3RawCountBaseline,
    TargetCondition,
)

SCIPLEX3_RUNNER_IMPLEMENTATION_VERSION: Final = "1.0.0"
SCIPLEX3_P1_RECORD_COUNT: Final = 94_785
SCIPLEX3_P1_WELL_COUNT: Final = 768
SCIPLEX3_P1_TREATED_WELL_COUNT: Final = 752
SCIPLEX3_P1_CONTROL_WELL_COUNT: Final = 16
SCIPLEX3_ACTION_ENTRY_COUNT: Final = 752
SCIPLEX3_P4_CASE_COUNT: Final = 384
SCIPLEX3_P4_TREATED_CASE_COUNT: Final = 376
SCIPLEX3_P4_CONTROL_CASE_COUNT: Final = 8
SCIPLEX3_PREDICTION_SHARD_DRAW_COUNT: Final = 128

SCIPLEX3_P1_RECORD_IDS_SHA256: Final = (
    "a57e083023e818f5e9dabcef1f9a45af23b270ed0e198049a970867fb28fffcd"
)
SCIPLEX3_P1_RECORD_TO_WELL_SHA256: Final = (
    "d01406c592fc67a0730fdddde912a9ca7801cd1d356eb13586c35251fad36d2b"
)
SCIPLEX3_P1_WELL_IDS_SHA256: Final = (
    "6bc034fed88548b25002d3d273cc6744e1d7875acb11c42d5b066599cb473c46"
)
SCIPLEX3_P1_WELL_TO_CONDITION_SHA256: Final = (
    "4bb1bc08e0fb9c819f9512fc34e28beb6ad8cc3c19d853851884a89517939743"
)
SCIPLEX3_ACTION_DOMAIN_SHA256: Final = (
    "1c42475bc740b53b0bd8e5e4c28e4f121dcd5adfef0873e2487eaff5baa65aa9"
)
SCIPLEX3_EVALUATION_CASES_SHA256: Final = (
    "3aebe598f80427cb1fb581f34115d57d20b62c5b5ecb793d6d7d527eecd4d7c0"
)
SCIPLEX3_BASELINE_CODE_SHA256: Final = (
    "5c078218f5af53b7815bb98b83e77a26f742e42a5853da711034a10c14330c58"
)
SCIPLEX3_BASELINE_GOLDEN_FIXTURE_SHA256: Final = (
    "59fd7410df297ce8a63e37068fc7d5727ebd12268526a5f43be6bded553dde49"
)
SCIPLEX3_P4_PREDICTION_TARGETS_SHA256: Final = (
    "383cc780e2f714819ca11841e098074f32634e3c25843ebcd66bcf406096c708"
)
SCIPLEX3_LOADER_CODE_SHA256: Final = (
    "bed3b56f7a91f1bb60f799ea2e28dc31505f196579cb8cb4ff386df5364a979d"
)
SCIPLEX3_EXECUTABLE_PYTHON: Final = (3, 11)
SCIPLEX3_EXECUTABLE_NUMPY_VERSION: Final = "2.4.6"

if _loaded_baselines_module.__file__ is None:  # pragma: no cover - fail closed at import
    raise ImportError("loaded sci-Plex3 baseline module has no source path")
_IMPORTED_BASELINE_CODE_PATH: Final = Path(_loaded_baselines_module.__file__).resolve()
_IMPORTED_BASELINE_CODE_SHA256: Final = hashlib.sha256(
    _IMPORTED_BASELINE_CODE_PATH.read_bytes()
).hexdigest()
_IMPORTED_RUNNER_CODE_PATH: Final = Path(__file__).resolve()
_IMPORTED_RUNNER_CODE_SHA256: Final = hashlib.sha256(
    _IMPORTED_RUNNER_CODE_PATH.read_bytes()
).hexdigest()

_BENCHMARK_DIRECTORY = Path("benchmarks/vertical-a/sciplex3-k562-24h-v1")
_PREPARATION_DIRECTORY = Path("benchmarks/artifacts/sciplex3-k562-24h-v1")
_LOADER_CONTRACT_PATH = _BENCHMARK_DIRECTORY / "support/p1-loader-contract.json"
_QUERY_PATH = _BENCHMARK_DIRECTORY / "state-query.json"
_BENCHMARK_PATH = _BENCHMARK_DIRECTORY / "benchmark-artifact.json"
_ACTION_DOMAIN_PATH = _BENCHMARK_DIRECTORY / "support/action-domain-mapping.json"
_EVALUATION_CASES_PATH = _BENCHMARK_DIRECTORY / "support/evaluation-cases.json"
_SCORING_TRANSFORM_PATH = _BENCHMARK_DIRECTORY / "support/scoring-transform.json"
_BASELINE_SUITE_SPEC_PATH = _BENCHMARK_DIRECTORY / "support/baseline-suite-spec.json"
_BASELINE_GOLDEN_FIXTURE_PATH = _BENCHMARK_DIRECTORY / "support/baseline-golden-fixtures.json"
_FEATURE_PANEL_PATH = _PREPARATION_DIRECTORY / "feature-panel.json"
_LOADER_CODE_PATH = Path("src/cellstate/backends/sciplex3_loader.py")
_BASELINE_CODE_PATH = Path("src/cellstate/evaluation/sciplex3_baselines.py")
_RUNNER_CODE_PATH = Path("src/cellstate/evaluation/sciplex3_runner.py")
_CONTROL_CONDITION_ID = "source-control@0nM"
_P1_PARTITION_ID = "p1-train"
_P4_PARTITION_ID = "p4-untouched-test"
_TARGET_OUTPUT_KEY = "cellstate:sciplex3-k562-24h-train-2000-raw-umi-distribution"
_HORIZON_NAME = "24h-endpoint"

_EXPECTED_ACTION_KEYS = frozenset(
    {
        "dose_nm",
        "duration_seconds",
        "intervention_kind_key",
        "normalized_perturbation_label",
        "query_spec_id",
        "source_perturbation_labels",
        "source_scoped_condition_id",
    }
)


class SciPlex3RunnerError(ContractViolationError):
    """Raised when the trusted-shape runner cannot prove its exact frozen closure."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SciPlex3RunnerError("value is not canonical-JSON-compatible") from error


def _read_bytes(path: Path, *, name: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise SciPlex3RunnerError(f"missing {name}: {path}") from error


def _read_exact(path: Path, expected_sha256: str, *, name: str) -> bytes:
    payload = _read_bytes(path, name=name)
    actual = _sha256(payload)
    if actual != expected_sha256:
        raise SciPlex3RunnerError(
            f"{name} SHA-256 drift: expected {expected_sha256}, observed {actual}"
        )
    return payload


def _json_value(payload: bytes, *, name: str) -> object:
    try:
        return json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SciPlex3RunnerError(f"invalid JSON for {name}") from error


def _json_object(payload: bytes, *, name: str) -> dict[str, object]:
    value = _json_value(payload, name=name)
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise SciPlex3RunnerError(f"{name} must be a JSON object")
    return cast(dict[str, object], value)


def _as_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise SciPlex3RunnerError(f"{name} must be an object")
    return cast(Mapping[str, object], value)


def _as_list(value: object, *, name: str) -> list[object]:
    if not isinstance(value, list):
        raise SciPlex3RunnerError(f"{name} must be an array")
    return cast(list[object], value)


def _canonical_model_bytes(value: object) -> bytes:
    return canonical_json_bytes(cast(Any, value))


def _artifact_reference_occurs(
    value: object,
    *,
    sha256: str,
    byte_count: int,
) -> bool:
    if isinstance(value, dict):
        if value.get("sha256") == sha256 and value.get("byte_count") == byte_count:
            return True
        return any(
            _artifact_reference_occurs(item, sha256=sha256, byte_count=byte_count)
            for item in value.values()
        )
    if isinstance(value, list):
        return any(
            _artifact_reference_occurs(item, sha256=sha256, byte_count=byte_count) for item in value
        )
    return False


def _named_scalar_binding_count(value: object, *, name: str, expected: str) -> int:
    if isinstance(value, dict):
        own = int(value.get("name") == name and value.get("value") == expected)
        return own + sum(
            _named_scalar_binding_count(item, name=name, expected=expected)
            for item in value.values()
        )
    if isinstance(value, list):
        return sum(
            _named_scalar_binding_count(item, name=name, expected=expected) for item in value
        )
    return 0


def _parse_composite_well(value: str) -> tuple[str, str]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise SciPlex3RunnerError("composite well ID is not canonical JSON") from error
    if (
        not isinstance(parsed, list)
        or len(parsed) != 2
        or any(not isinstance(item, str) or not item for item in parsed)
        or value != _canonical_json(parsed).decode("utf-8")
    ):
        raise SciPlex3RunnerError("composite well ID must be an exact [plate,well] string pair")
    return cast(tuple[str, str], tuple(parsed))


def _exact_text(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise SciPlex3RunnerError(f"{name} must be an exact nonblank trimmed string")
    return value


def _exact_sha256(value: object, *, name: str) -> str:
    digest = _exact_text(value, name=name)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise SciPlex3RunnerError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _deep_freeze_json(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType(
            {cast(str, key): _deep_freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_deep_freeze_json(item) for item in value)
    return value


def _mutable_json_value(value: object) -> object:
    """Return a fresh JSON-compatible tree from an internally deep-frozen value."""

    if isinstance(value, Mapping):
        return {str(key): _mutable_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_mutable_json_value(item) for item in value]
    return value


def _immutable_json_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    # JSON round-tripping proves compatibility and severs all caller-owned nested containers;
    # recursive freezing then makes the retained fitted binding genuinely deeply immutable.
    mutable_copy = _mutable_json_value(value)
    frozen_copy = _json_value(_canonical_json(mutable_copy), name="immutable manifest")
    assert isinstance(frozen_copy, dict)
    deep_frozen = _deep_freeze_json(frozen_copy)
    assert isinstance(deep_frozen, Mapping)
    return cast(Mapping[str, object], deep_frozen)


@dataclass(frozen=True, slots=True)
class SciPlex3ActionBinding:
    """One exact source condition to frozen query-action mapping."""

    source_condition_id: str
    query_spec_id: str
    compound: str
    dose_nm: int
    intervention_kind_key: str

    def __post_init__(self) -> None:
        source_condition_id = _exact_text(
            self.source_condition_id, name="action source condition ID"
        )
        _exact_text(self.query_spec_id, name="action query spec ID")
        compound = _exact_text(self.compound, name="action compound")
        intervention_kind_key = _exact_text(
            self.intervention_kind_key, name="action intervention kind key"
        )
        if type(self.dose_nm) is not int or self.dose_nm not in {10, 100, 1_000, 10_000}:
            raise SciPlex3RunnerError(
                "action dose must be an exact supported integer nanomolar dose"
            )
        if source_condition_id != f"source-label:{compound}@{self.dose_nm}nM":
            raise SciPlex3RunnerError("action source condition ID differs from compound/dose")
        if intervention_kind_key != intervention_kind_key.casefold():
            raise SciPlex3RunnerError("action intervention kind key must be case-folded")


@dataclass(frozen=True, slots=True)
class SciPlex3P1DesignBindings:
    """P1-safe query, action, and scoring identities (no held-out case membership)."""

    query_sha256: str
    query_fingerprint: str
    benchmark_sha256: str
    action_domain_sha256: str
    scoring_transform_sha256: str
    target_value_schema_sha256: str
    ordered_feature_keys_sha256: str
    actions_by_source_condition: Mapping[str, SciPlex3ActionBinding] = field(repr=False)
    actions_by_query_spec: Mapping[str, SciPlex3ActionBinding] = field(repr=False)
    can_mint_lifecycle_evidence: Literal[False] = False
    scientifically_admissible: Literal[False] = False

    def __post_init__(self) -> None:
        for name in (
            "query_sha256",
            "query_fingerprint",
            "benchmark_sha256",
            "action_domain_sha256",
            "scoring_transform_sha256",
            "target_value_schema_sha256",
            "ordered_feature_keys_sha256",
        ):
            _exact_sha256(getattr(self, name), name=name)
        if (
            self.can_mint_lifecycle_evidence is not False
            or self.scientifically_admissible is not False
        ):
            raise SciPlex3RunnerError("p1 design authority flags must be exactly false")
        try:
            source_snapshot = dict(self.actions_by_source_condition)
            query_snapshot = dict(self.actions_by_query_spec)
        except (TypeError, ValueError) as error:
            raise SciPlex3RunnerError("frozen action bindings must be exact mappings") from error
        source_bindings: dict[str, SciPlex3ActionBinding] = {}
        for raw_key, binding in source_snapshot.items():
            key = _exact_text(raw_key, name="source-condition action key")
            if type(binding) is not SciPlex3ActionBinding or key != binding.source_condition_id:
                raise SciPlex3RunnerError("source-condition action mapping is not exact")
            source_bindings[key] = binding
        query_bindings: dict[str, SciPlex3ActionBinding] = {}
        for raw_key, binding in query_snapshot.items():
            key = _exact_text(raw_key, name="query-spec action key")
            if type(binding) is not SciPlex3ActionBinding or key != binding.query_spec_id:
                raise SciPlex3RunnerError("query-spec action mapping is not exact")
            query_bindings[key] = binding
        object.__setattr__(
            self,
            "actions_by_source_condition",
            MappingProxyType({key: source_bindings[key] for key in sorted(source_bindings)}),
        )
        object.__setattr__(
            self,
            "actions_by_query_spec",
            MappingProxyType({key: query_bindings[key] for key in sorted(query_bindings)}),
        )
        if len(source_bindings) != SCIPLEX3_ACTION_ENTRY_COUNT:
            raise SciPlex3RunnerError("frozen action domain must contain exactly 752 entries")
        if len(query_bindings) != SCIPLEX3_ACTION_ENTRY_COUNT:
            raise SciPlex3RunnerError("frozen query action IDs must contain exactly 752 entries")
        if {
            binding.query_spec_id: binding for binding in source_bindings.values()
        } != query_bindings:
            raise SciPlex3RunnerError("source-condition and query-spec action mappings disagree")


@dataclass(frozen=True, slots=True)
class SciPlex3P4PredictionDesign:
    """Post-fit, outcome-free p4 case design; never an evaluation-outcome grant."""

    query_sha256: str
    benchmark_sha256: str
    action_domain_sha256: str
    evaluation_cases_sha256: str
    scoring_transform_sha256: str
    target_value_schema_sha256: str
    ordered_feature_keys_sha256: str
    fitted_state_artifact_sha256: str
    baseline_suite_specification_sha256: str
    preparation_fingerprint: str
    prediction_targets_sha256: str
    p4_targets: tuple[PredictionTarget, ...]
    heldout_outcomes_read: Literal[False] = False
    can_mint_lifecycle_evidence: Literal[False] = False
    scientifically_admissible: Literal[False] = False

    def __post_init__(self) -> None:
        for name in (
            "query_sha256",
            "benchmark_sha256",
            "action_domain_sha256",
            "evaluation_cases_sha256",
            "scoring_transform_sha256",
            "target_value_schema_sha256",
            "ordered_feature_keys_sha256",
            "fitted_state_artifact_sha256",
            "baseline_suite_specification_sha256",
            "preparation_fingerprint",
            "prediction_targets_sha256",
        ):
            _exact_sha256(getattr(self, name), name=name)
        p4_targets = tuple(self.p4_targets)
        if any(type(target) is not PredictionTarget for target in p4_targets):
            raise SciPlex3RunnerError("p4 design targets must be exact immutable PredictionTargets")
        object.__setattr__(self, "p4_targets", p4_targets)
        if (
            self.heldout_outcomes_read is not False
            or self.can_mint_lifecycle_evidence is not False
            or self.scientifically_admissible is not False
        ):
            raise SciPlex3RunnerError("p4 design authority/outcome flags must be exactly false")
        if len(self.p4_targets) != SCIPLEX3_P4_CASE_COUNT:
            raise SciPlex3RunnerError("frozen p4 design must contain exactly 384 cases")
        case_ids = tuple(target.case_id for target in self.p4_targets)
        if len(set(case_ids)) != len(case_ids) or case_ids != tuple(sorted(case_ids)):
            raise SciPlex3RunnerError(
                "frozen p4 target cases must be unique and canonically ordered"
            )
        actual = _sha256(
            _canonical_json([_prediction_target_value(target) for target in self.p4_targets])
        )
        if (
            self.prediction_targets_sha256 != SCIPLEX3_P4_PREDICTION_TARGETS_SHA256
            or actual != SCIPLEX3_P4_PREDICTION_TARGETS_SHA256
        ):
            raise SciPlex3RunnerError("frozen p4 target adapter identity drifted")


@dataclass(frozen=True, slots=True)
class SciPlex3P1AssemblyReceipt:
    """Exact software/data-shape receipt; never a trusted workflow or admission receipt."""

    loader_source_scan_fingerprint: str
    finalized_count_scan_fingerprint: str
    loader_implementation_sha256: str
    loader_contract_sha256: str
    source_sha256: str
    record_count: int
    well_count: int
    treated_well_count: int
    control_well_count: int
    record_ids_sha256: str
    record_to_well_sha256: str
    well_ids_sha256: str
    well_to_condition_sha256: str
    source_row_indices_sha256: str
    emitted_source_row_indices_sha256: str
    ordered_record_source_well_condition_sha256: str
    runner_panel_count_stream_sha256: str
    loader_panel_count_stream_sha256: str
    panel_nonzero_count: int
    zero_panel_record_count: int
    panel_umi_total: int
    full_source_umi_total: int
    batch_count: int
    ordered_feature_keys_sha256: str
    feature_panel_artifact_sha256: str
    action_domain_sha256: str
    query_sha256: str
    benchmark_sha256: str
    scoring_transform_sha256: str
    target_value_schema_sha256: str
    partition_id: Literal["p1-train"] = "p1-train"
    access_purpose: Literal["train_parameters"] = "train_parameters"
    exact_record_coverage: Literal[True] = True
    count_scan_complete: Literal[True] = True
    close_reverification_completed: Literal[True] = True
    heldout_memberships_read: Literal[False] = False
    heldout_outcomes_read: Literal[False] = False
    can_mint_lifecycle_evidence: Literal[False] = False
    scientifically_admissible: Literal[False] = False

    def __post_init__(self) -> None:
        for name in (
            "loader_source_scan_fingerprint",
            "finalized_count_scan_fingerprint",
            "loader_implementation_sha256",
            "loader_contract_sha256",
            "source_sha256",
            "record_ids_sha256",
            "record_to_well_sha256",
            "well_ids_sha256",
            "well_to_condition_sha256",
            "source_row_indices_sha256",
            "emitted_source_row_indices_sha256",
            "ordered_record_source_well_condition_sha256",
            "runner_panel_count_stream_sha256",
            "loader_panel_count_stream_sha256",
            "ordered_feature_keys_sha256",
            "feature_panel_artifact_sha256",
            "action_domain_sha256",
            "query_sha256",
            "benchmark_sha256",
            "scoring_transform_sha256",
            "target_value_schema_sha256",
        ):
            _exact_sha256(getattr(self, name), name=name)
        for name in (
            "record_count",
            "well_count",
            "treated_well_count",
            "control_well_count",
            "panel_nonzero_count",
            "zero_panel_record_count",
            "panel_umi_total",
            "full_source_umi_total",
            "batch_count",
        ):
            if type(getattr(self, name)) is not int:
                raise SciPlex3RunnerError(f"{name} must be an exact integer")
        _exact_text(self.partition_id, name="assembly partition ID")
        _exact_text(self.access_purpose, name="assembly access purpose")
        for name in (
            "exact_record_coverage",
            "count_scan_complete",
            "close_reverification_completed",
        ):
            if getattr(self, name) is not True:
                raise SciPlex3RunnerError(f"{name} must be exactly true")
        for name in (
            "heldout_memberships_read",
            "heldout_outcomes_read",
            "can_mint_lifecycle_evidence",
            "scientifically_admissible",
        ):
            if getattr(self, name) is not False:
                raise SciPlex3RunnerError(f"{name} must be exactly false")

    @property
    def fingerprint(self) -> str:
        return _sha256(_canonical_json(_json_ready_dataclass(self)))


@dataclass(frozen=True, slots=True)
class SciPlex3BaselinePreparation:
    """CSR-native p1 training surface plus its exact outcome-free design bindings."""

    training_data: P1TrainingData = field(repr=False)
    receipt: SciPlex3P1AssemblyReceipt
    finalized_count_scan_receipt: SciPlex3P1FinalizedCountScanReceipt = field(repr=False)
    design: SciPlex3P1DesignBindings
    repository_root: Path = field(repr=False)
    can_mint_lifecycle_evidence: Literal[False] = False
    scientifically_admissible: Literal[False] = False

    def __post_init__(self) -> None:
        if type(self.training_data) is not P1TrainingData:
            raise SciPlex3RunnerError("preparation training data must be exact P1TrainingData")
        if type(self.receipt) is not SciPlex3P1AssemblyReceipt:
            raise SciPlex3RunnerError("preparation assembly receipt must use the exact type")
        if type(self.finalized_count_scan_receipt) is not SciPlex3P1FinalizedCountScanReceipt:
            raise SciPlex3RunnerError("preparation finalized receipt must use the exact type")
        if type(self.design) is not SciPlex3P1DesignBindings:
            raise SciPlex3RunnerError("preparation design must use the exact type")
        if (
            self.can_mint_lifecycle_evidence is not False
            or self.scientifically_admissible is not False
        ):
            raise SciPlex3RunnerError("preparation authority flags must be exactly false")
        object.__setattr__(self, "repository_root", Path(self.repository_root).resolve())
        if len(self.training_data.wells) != SCIPLEX3_P1_WELL_COUNT:
            raise SciPlex3RunnerError("baseline preparation lacks the exact 768-well p1 closure")
        if self.receipt.record_count != SCIPLEX3_P1_RECORD_COUNT:
            raise SciPlex3RunnerError("baseline preparation lacks the exact 94,785-record closure")
        if (
            self.finalized_count_scan_receipt.fingerprint
            != self.receipt.finalized_count_scan_fingerprint
        ):
            raise SciPlex3RunnerError("preparation finalized-scan receipt fingerprint drifted")

    def finalized_count_scan_manifest(self) -> dict[str, object]:
        """Return the exact canonical-JSON-ready, explicitly non-admissible scan receipt."""

        return _json_ready_dataclass(self.finalized_count_scan_receipt)


@dataclass(frozen=True, slots=True)
class LocalContentAddressedArtifact:
    """One local, immutable-by-contract content-addressed output."""

    path: Path
    sha256: str
    byte_count: int
    media_type: str
    can_mint_lifecycle_evidence: Literal[False] = False
    scientifically_admissible: Literal[False] = False

    def __post_init__(self) -> None:
        try:
            path = Path(self.path).resolve()
        except (OSError, TypeError, ValueError) as error:
            raise SciPlex3RunnerError("local artifact path must be path-like") from error
        object.__setattr__(self, "path", path)
        _exact_sha256(self.sha256, name="local artifact SHA-256")
        if type(self.byte_count) is not int or self.byte_count <= 0:
            raise SciPlex3RunnerError("local artifact byte count must be a positive exact integer")
        _exact_text(self.media_type, name="local artifact media type")
        if (
            self.can_mint_lifecycle_evidence is not False
            or self.scientifically_admissible is not False
        ):
            raise SciPlex3RunnerError("local artifact authority flags must be exactly false")


@dataclass(frozen=True, slots=True)
class FittedSciPlex3Baseline:
    """In-memory fitted baseline bound to a re-read p1 fitted-state identity artifact."""

    baseline: SciPlex3RawCountBaseline = field(repr=False)
    artifact: LocalContentAddressedArtifact
    artifact_manifest: Mapping[str, object] = field(repr=False)
    preparation_fingerprint: str
    can_mint_lifecycle_evidence: Literal[False] = False
    scientifically_admissible: Literal[False] = False

    def __post_init__(self) -> None:
        if type(self.artifact) is not LocalContentAddressedArtifact:
            raise SciPlex3RunnerError("fitted artifact must use the exact local artifact type")
        _exact_sha256(self.preparation_fingerprint, name="preparation fingerprint")
        if (
            self.can_mint_lifecycle_evidence is not False
            or self.scientifically_admissible is not False
        ):
            raise SciPlex3RunnerError("fitted baseline authority flags must be exactly false")
        object.__setattr__(
            self, "artifact_manifest", _immutable_json_mapping(self.artifact_manifest)
        )


def _json_ready_dataclass(value: object) -> dict[str, object]:
    fields = getattr(value, "__dataclass_fields__", None)
    if not isinstance(fields, dict):
        raise SciPlex3RunnerError("internal receipt is not a dataclass")
    output: dict[str, object] = {}
    for name in fields:
        item = getattr(value, name)
        if isinstance(item, Path):
            output[name] = str(item)
        elif isinstance(item, tuple):
            output[name] = list(item)
        else:
            output[name] = cast(object, item)
    return output


@dataclass(slots=True)
class _WellBuilder:
    record_ids: list[str] = field(default_factory=list)
    source_rows: list[int] = field(default_factory=list)
    feature_indices: list[npt.NDArray[np.int64]] = field(default_factory=list)
    counts: list[npt.NDArray[np.int64]] = field(default_factory=list)

    def append(
        self,
        *,
        record_id: str,
        source_row: int,
        feature_indices: npt.NDArray[np.int64],
        counts: npt.NDArray[np.int64],
    ) -> None:
        self.record_ids.append(record_id)
        self.source_rows.append(source_row)
        self.feature_indices.append(np.asarray(feature_indices, dtype=np.int64).copy())
        self.counts.append(np.asarray(counts, dtype=np.int64).copy())

    def freeze(self) -> ImmutableCSRCounts:
        row_sizes = np.asarray([len(value) for value in self.feature_indices], dtype=np.int64)
        indptr = np.concatenate((np.asarray([0], dtype=np.int64), np.cumsum(row_sizes)))
        if not self.feature_indices:
            raise SciPlex3RunnerError("p1 well builder contains no rows")
        indices = np.concatenate(self.feature_indices)
        values = np.concatenate(self.counts)
        return ImmutableCSRCounts(
            indptr=indptr,
            feature_indices=indices,
            values=values,
            row_count=len(self.record_ids),
        )


@dataclass(frozen=True, slots=True)
class _P1ArtifactClosure:
    contract_sha256: str
    record_ids: tuple[str, ...]
    record_to_well: tuple[tuple[str, str], ...]
    well_ids: tuple[str, ...]
    well_to_condition: tuple[tuple[str, str], ...]
    feature_keys: tuple[str, ...]


def _string_array(payload: bytes, *, name: str) -> tuple[str, ...]:
    values = _as_list(_json_value(payload, name=name), name=name)
    if any(not isinstance(value, str) or not value for value in values):
        raise SciPlex3RunnerError(f"{name} must contain only nonblank strings")
    strings = cast(tuple[str, ...], tuple(values))
    if strings != tuple(sorted(strings)) or len(strings) != len(set(strings)):
        raise SciPlex3RunnerError(f"{name} must be unique and sorted")
    return strings


def _string_pairs(payload: bytes, *, name: str) -> tuple[tuple[str, str], ...]:
    values = _as_list(_json_value(payload, name=name), name=name)
    pairs: list[tuple[str, str]] = []
    for value in values:
        if (
            not isinstance(value, list)
            or len(value) != 2
            or any(not isinstance(item, str) or not item for item in value)
        ):
            raise SciPlex3RunnerError(f"{name} must contain nonblank string pairs")
        pairs.append((cast(str, value[0]), cast(str, value[1])))
    result = tuple(pairs)
    if result != tuple(sorted(result)) or len({left for left, _ in result}) != len(result):
        raise SciPlex3RunnerError(f"{name} must have a unique sorted domain")
    return result


def _resolve_contract_artifact(
    repository_root: Path,
    reference: Mapping[str, object],
    *,
    name: str,
) -> bytes:
    relative = reference.get("relative_path")
    digest = reference.get("sha256")
    byte_count = reference.get("byte_count")
    if (
        not isinstance(relative, str)
        or not relative
        or not isinstance(digest, str)
        or len(digest) != 64
        or isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count < 0
    ):
        raise SciPlex3RunnerError(f"malformed p1 loader-contract reference: {name}")
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise SciPlex3RunnerError("p1 loader-contract artifact path escapes its trusted root")
    path = (
        repository_root / relative_path
        if relative_path.parts and relative_path.parts[0] == "benchmarks"
        else repository_root / _PREPARATION_DIRECTORY / relative_path
    )
    payload = _read_exact(path, digest, name=f"p1 {name} artifact")
    if len(payload) != byte_count:
        raise SciPlex3RunnerError(f"p1 {name} artifact byte count drifted")
    return payload


def _load_p1_artifact_closure(
    repository_root: Path,
    descriptor: SciPlex3PartitionDescriptor,
    panel_keys: tuple[str, ...],
) -> _P1ArtifactClosure:
    contract_payload = _read_exact(
        repository_root / _LOADER_CONTRACT_PATH,
        descriptor.loader_contract_sha256,
        name="p1 loader contract",
    )
    contract = _json_object(contract_payload, name="p1 loader contract")
    if contract_payload != _canonical_json(contract):
        raise SciPlex3RunnerError("p1 loader contract must be canonical JSON")
    if (
        contract.get("artifact_schema") != "sciplex3-k562-p1-loader-contract"
        or contract.get("artifact_schema_version") != "1.0.0"
        or contract.get("heldout_memberships_referenced") is not False
        or contract.get("loader_outputs_can_mint_lifecycle_evidence") is not False
        or contract.get("scientifically_admissible_without_trusted_workflow_receipt") is not False
    ):
        raise SciPlex3RunnerError("p1 loader contract crosses its non-admissible boundary")
    partition = _as_mapping(contract.get("partition"), name="p1 contract partition")
    if (
        partition.get("partition_id") != _P1_PARTITION_ID
        or partition.get("artifact_role") != "train"
        or partition.get("access_purpose") != "train_parameters"
        or partition.get("record_count") != SCIPLEX3_P1_RECORD_COUNT
        or partition.get("well_count") != SCIPLEX3_P1_WELL_COUNT
    ):
        raise SciPlex3RunnerError("p1 loader contract partition drifted")
    bindings = _as_mapping(contract.get("bindings"), name="p1 contract bindings")
    expected_bindings = {
        "benchmark_sha256": descriptor.benchmark_sha256,
        "dataset_manifest_sha256": descriptor.dataset_manifest_sha256,
        "query_sha256": descriptor.query_sha256,
        "scoring_transform_sha256": descriptor.scoring_transform_sha256,
        "target_value_schema_sha256": descriptor.target_value_schema_sha256,
    }
    if dict(bindings) != expected_bindings:
        raise SciPlex3RunnerError("p1 loader contract semantic bindings differ from descriptor")
    artifacts = _as_mapping(contract.get("artifacts"), name="p1 contract artifacts")
    expected_artifact_names = {
        "feature_panel",
        "plate_ids",
        "record_ids",
        "record_to_well",
        "source_verification",
        "well_ids",
        "well_to_condition",
    }
    if set(artifacts) != expected_artifact_names:
        raise SciPlex3RunnerError("p1 loader contract artifact closure is not exact")
    payloads = {
        name: _resolve_contract_artifact(
            repository_root,
            _as_mapping(reference, name=f"p1 {name} reference"),
            name=name,
        )
        for name, reference in artifacts.items()
    }
    exact_digests = {
        "record_ids": SCIPLEX3_P1_RECORD_IDS_SHA256,
        "record_to_well": SCIPLEX3_P1_RECORD_TO_WELL_SHA256,
        "well_ids": SCIPLEX3_P1_WELL_IDS_SHA256,
        "well_to_condition": SCIPLEX3_P1_WELL_TO_CONDITION_SHA256,
        "feature_panel": SCIPLEX3_FEATURE_PANEL_ARTIFACT_SHA256,
    }
    for name, digest in exact_digests.items():
        if _sha256(payloads[name]) != digest:
            raise SciPlex3RunnerError(f"authoritative p1 {name} identity drifted")
    record_ids = _string_array(payloads["record_ids"], name="p1 record IDs")
    record_to_well = _string_pairs(payloads["record_to_well"], name="p1 record-to-well")
    well_ids = _string_array(payloads["well_ids"], name="p1 well IDs")
    well_to_condition = _string_pairs(payloads["well_to_condition"], name="p1 well-to-condition")
    if (
        len(record_ids) != SCIPLEX3_P1_RECORD_COUNT
        or len(record_to_well) != SCIPLEX3_P1_RECORD_COUNT
        or len(well_ids) != SCIPLEX3_P1_WELL_COUNT
        or len(well_to_condition) != SCIPLEX3_P1_WELL_COUNT
        or tuple(left for left, _ in record_to_well) != record_ids
        or tuple(left for left, _ in well_to_condition) != well_ids
    ):
        raise SciPlex3RunnerError("authoritative p1 memberships do not form an exact closure")
    feature_panel = _json_object(payloads["feature_panel"], name="feature panel")
    if (
        feature_panel.get("feature_count") != SCIPLEX3_FEATURE_COUNT
        or feature_panel.get("ordered_feature_keys_sha256") != SCIPLEX3_ORDERED_FEATURE_KEYS_SHA256
    ):
        raise SciPlex3RunnerError("feature-panel summary drifted")
    features = _as_list(feature_panel.get("features"), name="feature panel rows")
    derived_keys: list[str] = []
    for rank, raw in enumerate(features, start=1):
        row = _as_mapping(raw, name="feature-panel row")
        ensembl = row.get("ensembl_id")
        symbol = row.get("gene_symbol")
        if row.get("rank") != rank or not isinstance(ensembl, str) or not isinstance(symbol, str):
            raise SciPlex3RunnerError("feature-panel row is malformed or out of order")
        derived_keys.append(f"{ensembl}|{symbol}")
    if tuple(derived_keys) != panel_keys:
        raise SciPlex3RunnerError("loader feature keys differ from the authoritative panel")
    return _P1ArtifactClosure(
        contract_sha256=_sha256(contract_payload),
        record_ids=record_ids,
        record_to_well=record_to_well,
        well_ids=well_ids,
        well_to_condition=well_to_condition,
        feature_keys=tuple(derived_keys),
    )


def _load_actions(
    payload: bytes,
    query: StateQuery,
) -> tuple[Mapping[str, SciPlex3ActionBinding], Mapping[str, SciPlex3ActionBinding]]:
    action_document = _json_object(payload, name="action-domain mapping")
    if (
        action_document.get("artifact_schema") != "sciplex3-k562-query-action-domain"
        or action_document.get("artifact_schema_version") != "1.0.0"
        or action_document.get("entry_count") != SCIPLEX3_ACTION_ENTRY_COUNT
        or action_document.get("combination_order") != 1
        or action_document.get("dose_values_nm") != [10, 100, 1_000, 10_000]
    ):
        raise SciPlex3RunnerError("action-domain header drifted")
    raw_entries = _as_list(action_document.get("entries"), name="action-domain entries")
    if len(raw_entries) != SCIPLEX3_ACTION_ENTRY_COUNT:
        raise SciPlex3RunnerError("action-domain entry count drifted")
    query_document = query.model_dump(mode="json")
    query_interventions = _as_list(
        query_document.get("intervention_space"), name="query intervention space"
    )
    if len(query_interventions) != SCIPLEX3_ACTION_ENTRY_COUNT:
        raise SciPlex3RunnerError("query intervention-space size differs from action domain")
    query_by_spec: dict[str, Mapping[str, object]] = {}
    for raw in query_interventions:
        intervention = _as_mapping(raw, name="query intervention")
        spec_id = intervention.get("spec_id")
        if not isinstance(spec_id, str) or not spec_id or spec_id in query_by_spec:
            raise SciPlex3RunnerError("query intervention IDs are invalid or duplicated")
        query_by_spec[spec_id] = intervention

    by_condition: dict[str, SciPlex3ActionBinding] = {}
    by_spec: dict[str, SciPlex3ActionBinding] = {}
    for raw in raw_entries:
        entry = _as_mapping(raw, name="action-domain entry")
        if set(entry) != _EXPECTED_ACTION_KEYS:
            raise SciPlex3RunnerError("action-domain entry fields drifted")
        condition_id = entry.get("source_scoped_condition_id")
        spec_id = entry.get("query_spec_id")
        compound = entry.get("normalized_perturbation_label")
        dose_nm = entry.get("dose_nm")
        duration = entry.get("duration_seconds")
        kind_key = entry.get("intervention_kind_key")
        source_labels = entry.get("source_perturbation_labels")
        if (
            not isinstance(condition_id, str)
            or not condition_id
            or not isinstance(spec_id, str)
            or not spec_id
            or not isinstance(compound, str)
            or not compound
            or isinstance(dose_nm, bool)
            or not isinstance(dose_nm, int)
            or dose_nm not in {10, 100, 1_000, 10_000}
            or duration != 86_400
            or not isinstance(kind_key, str)
            or not kind_key
            or not isinstance(source_labels, list)
            or not source_labels
            or any(
                not isinstance(label, str) or label.strip() != compound for label in source_labels
            )
            or condition_id != f"source-label:{compound}@{dose_nm}nM"
            or condition_id in by_condition
            or spec_id in by_spec
        ):
            raise SciPlex3RunnerError("action-domain entry violates exact source/query semantics")
        try:
            query_entry = query_by_spec[spec_id]
        except KeyError as error:
            raise SciPlex3RunnerError(
                "action-domain entry is absent from the frozen query"
            ) from error
        dose_domain = _as_mapping(query_entry.get("dose_domain"), name="query dose domain")
        duration_domain = _as_mapping(
            query_entry.get("duration_seconds"), name="query duration domain"
        )
        kind = _as_mapping(query_entry.get("kind"), name="query intervention kind")
        namespace = kind.get("namespace")
        identifier = kind.get("identifier")
        query_kind_key = ""
        if isinstance(namespace, str) and isinstance(identifier, str):
            query_kind_key = (
                identifier.casefold()
                if ":" in identifier
                else f"{namespace}:{identifier}".casefold()
            )
        if (
            dose_domain.get("kind") != "numeric"
            or dose_domain.get("units") != "nM"
            or dose_domain.get("minimum") != float(dose_nm)
            or dose_domain.get("maximum") != float(dose_nm)
            or duration_domain.get("minimum") != 86_400.0
            or duration_domain.get("maximum") != 86_400.0
            or query_kind_key != kind_key.casefold()
        ):
            raise SciPlex3RunnerError("action-domain entry differs from its exact query action")
        binding = SciPlex3ActionBinding(
            source_condition_id=condition_id,
            query_spec_id=spec_id,
            compound=compound,
            dose_nm=dose_nm,
            intervention_kind_key=kind_key.casefold(),
        )
        by_condition[condition_id] = binding
        by_spec[spec_id] = binding
    if set(by_spec) != set(query_by_spec):
        raise SciPlex3RunnerError("action domain does not cover the exact query intervention space")
    return MappingProxyType(by_condition), MappingProxyType(by_spec)


def _case_to_target(
    case: BenchmarkEvaluationCase,
    actions_by_spec: Mapping[str, SciPlex3ActionBinding],
) -> PredictionTarget:
    if case.partition_id != _P4_PARTITION_ID:
        raise SciPlex3RunnerError("case adapter accepts only p4 design cases")
    plate, _ = _parse_composite_well(case.evaluation_unit_id)
    if plate != case.matching_stratum_id:
        raise SciPlex3RunnerError("p4 case plate and matching stratum disagree")
    if case.horizon_name != _HORIZON_NAME or case.target_output_keys != (_TARGET_OUTPUT_KEY,):
        raise SciPlex3RunnerError("p4 case horizon or target output drifted")
    condition: TargetCondition
    if case.role is EvaluationCaseRole.MATCHED_NO_ACTION_CONTROL:
        if case.intervention_spec_ids or case.matched_control_evaluation_unit_ids:
            raise SciPlex3RunnerError("p4 no-action case contains treated-case metadata")
        condition = NO_ACTION
    elif case.role is EvaluationCaseRole.TREATED:
        if len(case.intervention_spec_ids) != 1:
            raise SciPlex3RunnerError("p4 treated case must bind exactly one action")
        try:
            action = actions_by_spec[case.intervention_spec_ids[0]]
        except KeyError as error:
            raise SciPlex3RunnerError("p4 case references an unknown frozen action") from error
        condition = CompoundDose(action.compound, action.dose_nm)
    else:  # pragma: no cover - enum schema makes this unreachable, retained fail-closed.
        raise SciPlex3RunnerError("unsupported p4 evaluation-case role")
    return PredictionTarget(
        case_id=case.case_id,
        target_well_id=case.evaluation_unit_id,
        plate_id=plate,
        partition_id=_P4_PARTITION_ID,
        condition=condition,
    )


def _prediction_target_value(target: PredictionTarget) -> list[object]:
    if type(target) is not PredictionTarget:
        raise SciPlex3RunnerError("prediction target must be an exact immutable PredictionTarget")
    condition: dict[str, object]
    if type(target.condition) is NoAction:
        condition = {"kind": "no_action"}
    elif type(target.condition) is CompoundDose:
        condition = {
            "compound": target.condition.compound,
            "dose_nm": target.condition.dose_nm,
            "kind": "compound_dose",
        }
    else:  # pragma: no cover - PredictionTarget already rejects this, retained fail-closed.
        raise SciPlex3RunnerError("prediction target condition is not an exact immutable condition")
    return [
        target.case_id,
        target.target_well_id,
        target.plate_id,
        target.partition_id,
        condition,
    ]


def _validate_prediction_design_structure(design: SciPlex3P4PredictionDesign) -> None:
    if type(design) is not SciPlex3P4PredictionDesign:
        raise SciPlex3RunnerError("p4 design must be the exact registered immutable design type")
    for name in (
        "query_sha256",
        "benchmark_sha256",
        "action_domain_sha256",
        "evaluation_cases_sha256",
        "scoring_transform_sha256",
        "target_value_schema_sha256",
        "ordered_feature_keys_sha256",
        "fitted_state_artifact_sha256",
        "baseline_suite_specification_sha256",
        "preparation_fingerprint",
        "prediction_targets_sha256",
    ):
        _exact_sha256(getattr(design, name), name=name)
    targets = design.p4_targets
    if type(targets) is not tuple or any(
        type(target) is not PredictionTarget for target in targets
    ):
        raise SciPlex3RunnerError("p4 design targets are not exact immutable PredictionTargets")
    case_ids = tuple(target.case_id for target in targets)
    treated_count = sum(type(target.condition) is CompoundDose for target in targets)
    control_count = sum(type(target.condition) is NoAction for target in targets)
    actual_targets_sha256 = _sha256(
        _canonical_json([_prediction_target_value(target) for target in targets])
    )
    if (
        len(targets) != SCIPLEX3_P4_CASE_COUNT
        or treated_count != SCIPLEX3_P4_TREATED_CASE_COUNT
        or control_count != SCIPLEX3_P4_CONTROL_CASE_COUNT
        or len(set(case_ids)) != len(case_ids)
        or case_ids != tuple(sorted(case_ids))
        or design.action_domain_sha256 != SCIPLEX3_ACTION_DOMAIN_SHA256
        or design.evaluation_cases_sha256 != SCIPLEX3_EVALUATION_CASES_SHA256
        or design.scoring_transform_sha256 != SCIPLEX3_SCORING_TRANSFORM_SHA256
        or design.ordered_feature_keys_sha256 != SCIPLEX3_ORDERED_FEATURE_KEYS_SHA256
        or design.prediction_targets_sha256 != SCIPLEX3_P4_PREDICTION_TARGETS_SHA256
        or actual_targets_sha256 != SCIPLEX3_P4_PREDICTION_TARGETS_SHA256
        or design.heldout_outcomes_read is not False
        or design.can_mint_lifecycle_evidence is not False
        or design.scientifically_admissible is not False
    ):
        raise SciPlex3RunnerError("p4 prediction design identity or safety boundary drifted")


def _load_p1_design_bindings(
    repository_root: Path,
    descriptor: SciPlex3PartitionDescriptor,
) -> SciPlex3P1DesignBindings:
    query_payload = _read_exact(
        repository_root / _QUERY_PATH,
        descriptor.query_sha256,
        name="frozen state query",
    )
    action_payload = _read_exact(
        repository_root / _ACTION_DOMAIN_PATH,
        SCIPLEX3_ACTION_DOMAIN_SHA256,
        name="action-domain mapping",
    )
    scoring_payload = _read_exact(
        repository_root / _SCORING_TRANSFORM_PATH,
        SCIPLEX3_SCORING_TRANSFORM_SHA256,
        name="scoring transform",
    )
    if descriptor.scoring_transform_sha256 != SCIPLEX3_SCORING_TRANSFORM_SHA256:
        raise SciPlex3RunnerError(
            "loader descriptor does not bind the authoritative scoring transform"
        )
    try:
        query = StateQuery.model_validate_json(query_payload)
    except ValueError as error:
        raise SciPlex3RunnerError("query failed schema validation") from error
    if query_payload != _canonical_model_bytes(query.model_dump(mode="json")):
        raise SciPlex3RunnerError("frozen query is not canonical JSON")
    scoring = _json_object(scoring_payload, name="scoring transform")
    declared_panel = _as_mapping(scoring.get("declared_panel"), name="scoring panel")
    input_contract = _as_mapping(scoring.get("input_contract"), name="scoring input")
    validation = _as_mapping(scoring.get("validation_policy"), name="scoring validation")
    if (
        scoring.get("artifact_schema") != "sciplex3-k562-panel-only-scoring-transform"
        or scoring.get("artifact_schema_version") != "1.0.0"
        or declared_panel.get("feature_count") != SCIPLEX3_FEATURE_COUNT
        or declared_panel.get("ordered_feature_keys_sha256") != SCIPLEX3_ORDERED_FEATURE_KEYS_SHA256
        or input_contract.get("full_source_axis_input_allowed") is not False
        or input_contract.get("feature_count") != SCIPLEX3_FEATURE_COUNT
        or validation.get("zero_panel_total_policy")
        != "error_fail_evaluation_no_exclusion_or_imputation"
    ):
        raise SciPlex3RunnerError("scoring-transform scope or panel binding drifted")
    actions_by_condition, actions_by_spec = _load_actions(action_payload, query)
    return SciPlex3P1DesignBindings(
        query_sha256=descriptor.query_sha256,
        query_fingerprint=query.fingerprint,
        benchmark_sha256=descriptor.benchmark_sha256,
        action_domain_sha256=SCIPLEX3_ACTION_DOMAIN_SHA256,
        scoring_transform_sha256=SCIPLEX3_SCORING_TRANSFORM_SHA256,
        target_value_schema_sha256=descriptor.target_value_schema_sha256,
        ordered_feature_keys_sha256=SCIPLEX3_ORDERED_FEATURE_KEYS_SHA256,
        actions_by_source_condition=actions_by_condition,
        actions_by_query_spec=actions_by_spec,
    )


def open_sciplex3_p4_prediction_design(
    preparation: SciPlex3BaselinePreparation,
    fitted: FittedSciPlex3Baseline,
) -> SciPlex3P4PredictionDesign:
    """Open outcome-free p4 case design only after a p1 fitted artifact is re-authenticated."""

    _validate_preparation(preparation)
    fitted_manifest = _verify_fitted_artifact_binding(preparation, fitted)
    repository_root = preparation.repository_root
    benchmark_payload = _read_exact(
        repository_root / _BENCHMARK_PATH,
        preparation.design.benchmark_sha256,
        name="frozen benchmark",
    )
    cases_payload = _read_exact(
        repository_root / _EVALUATION_CASES_PATH,
        SCIPLEX3_EVALUATION_CASES_SHA256,
        name="evaluation cases",
    )
    try:
        benchmark = BenchmarkArtifact.model_validate_json(benchmark_payload)
    except ValueError as error:
        raise SciPlex3RunnerError("benchmark failed schema validation") from error
    if benchmark_payload != _canonical_model_bytes(benchmark.model_dump(mode="json")):
        raise SciPlex3RunnerError("frozen benchmark is not canonical JSON")
    embedded_query_payload = _canonical_model_bytes(
        benchmark.definition.query.state_query.model_dump(mode="json")
    )
    if (
        _sha256(embedded_query_payload) != preparation.design.query_sha256
        or benchmark.definition.query.query_fingerprint != preparation.design.query_fingerprint
    ):
        raise SciPlex3RunnerError("benchmark does not bind the exact pre-fit query")
    case_set = benchmark.definition.evaluation_case_set
    if case_set is None:
        raise SciPlex3RunnerError("benchmark lacks the exact evaluation-case set")
    _as_list(_json_value(cases_payload, name="evaluation cases"), name="evaluation cases")
    cases = case_set.cases
    if cases_payload != _canonical_json([case.model_dump(mode="json") for case in cases]):
        raise SciPlex3RunnerError("evaluation-case artifact is not exact canonical JSON")
    if (
        case_set.case_artifact.sha256 != SCIPLEX3_EVALUATION_CASES_SHA256
        or case_set.case_artifact.byte_count != len(cases_payload)
    ):
        raise SciPlex3RunnerError("benchmark case-artifact binding drifted")
    benchmark_dump = benchmark.model_dump(mode="json")
    action_payload = _read_exact(
        repository_root / _ACTION_DOMAIN_PATH,
        SCIPLEX3_ACTION_DOMAIN_SHA256,
        name="action-domain mapping",
    )
    for digest, payload, name in (
        (SCIPLEX3_ACTION_DOMAIN_SHA256, action_payload, "action domain"),
        (SCIPLEX3_EVALUATION_CASES_SHA256, cases_payload, "evaluation cases"),
    ):
        if not _artifact_reference_occurs(benchmark_dump, sha256=digest, byte_count=len(payload)):
            raise SciPlex3RunnerError(f"benchmark lacks an exact {name} artifact binding")
    executable_binding = _as_mapping(
        fitted_manifest.get("executable_binding"), name="fitted executable binding"
    )
    suite_reference = _as_mapping(
        executable_binding.get("baseline_suite_specification"),
        name="fitted baseline-suite reference",
    )
    suite_sha256 = suite_reference.get("sha256")
    suite_byte_count = suite_reference.get("byte_count")
    if (
        not isinstance(suite_sha256, str)
        or isinstance(suite_byte_count, bool)
        or not isinstance(suite_byte_count, int)
        or not _artifact_reference_occurs(
            benchmark_dump, sha256=suite_sha256, byte_count=suite_byte_count
        )
    ):
        raise SciPlex3RunnerError("benchmark does not bind the pre-fit executable baseline suite")
    if _named_scalar_binding_count(
        benchmark_dump,
        name="scoring_transform_sha256",
        expected=SCIPLEX3_SCORING_TRANSFORM_SHA256,
    ) != len(benchmark.definition.metrics):
        raise SciPlex3RunnerError("benchmark metrics do not all bind the scoring transform")
    p4_cases = tuple(case for case in cases if case.partition_id == _P4_PARTITION_ID)
    p4_targets = tuple(
        _case_to_target(case, preparation.design.actions_by_query_spec) for case in p4_cases
    )
    treated_count = sum(not isinstance(target.condition, type(NO_ACTION)) for target in p4_targets)
    control_count = len(p4_targets) - treated_count
    if (
        len(p4_targets) != SCIPLEX3_P4_CASE_COUNT
        or treated_count != SCIPLEX3_P4_TREATED_CASE_COUNT
        or control_count != SCIPLEX3_P4_CONTROL_CASE_COUNT
        or tuple(target.case_id for target in p4_targets)
        != tuple(sorted(target.case_id for target in p4_targets))
    ):
        raise SciPlex3RunnerError("p4 treated/no-action case closure drifted")
    return SciPlex3P4PredictionDesign(
        query_sha256=preparation.design.query_sha256,
        benchmark_sha256=preparation.design.benchmark_sha256,
        action_domain_sha256=SCIPLEX3_ACTION_DOMAIN_SHA256,
        evaluation_cases_sha256=SCIPLEX3_EVALUATION_CASES_SHA256,
        scoring_transform_sha256=SCIPLEX3_SCORING_TRANSFORM_SHA256,
        target_value_schema_sha256=preparation.design.target_value_schema_sha256,
        ordered_feature_keys_sha256=SCIPLEX3_ORDERED_FEATURE_KEYS_SHA256,
        fitted_state_artifact_sha256=fitted.artifact.sha256,
        baseline_suite_specification_sha256=suite_sha256,
        preparation_fingerprint=preparation.receipt.fingerprint,
        prediction_targets_sha256=SCIPLEX3_P4_PREDICTION_TARGETS_SHA256,
        p4_targets=p4_targets,
    )


def _validate_descriptor(descriptor: SciPlex3PartitionDescriptor) -> None:
    if (
        type(descriptor) is not SciPlex3PartitionDescriptor
        or descriptor.partition_id != _P1_PARTITION_ID
        or descriptor.artifact_role != "train"
        or descriptor.benchmark_role is not BenchmarkPartitionRole.TRAIN
        or descriptor.access_purpose is not PopulationComponentAccessPurpose.TRAIN_PARAMETERS
        or descriptor.record_count != SCIPLEX3_P1_RECORD_COUNT
        or descriptor.well_count != SCIPLEX3_P1_WELL_COUNT
        or descriptor.record_ids_sha256 != SCIPLEX3_P1_RECORD_IDS_SHA256
        or descriptor.record_to_well_sha256 != SCIPLEX3_P1_RECORD_TO_WELL_SHA256
        or descriptor.source_sha256 != SCIPLEX3_SOURCE_SHA256
        or descriptor.feature_panel_artifact_sha256 != SCIPLEX3_FEATURE_PANEL_ARTIFACT_SHA256
        or descriptor.ordered_feature_keys_sha256 != SCIPLEX3_ORDERED_FEATURE_KEYS_SHA256
        or descriptor.scoring_transform_sha256 != SCIPLEX3_SCORING_TRANSFORM_SHA256
        or descriptor.count_access_sealed
        or descriptor.can_mint_lifecycle_evidence
        or descriptor.scientifically_admissible
    ):
        raise SciPlex3RunnerError("loader descriptor is not the exact non-admissible p1 surface")


def _validate_source_receipt(
    receipt: SciPlex3P1SourceScanReceipt,
    descriptor: SciPlex3PartitionDescriptor,
) -> None:
    if (
        type(receipt) is not SciPlex3P1SourceScanReceipt
        or receipt.source_sha256 != descriptor.source_sha256
        or receipt.p1_loader_contract_sha256 != descriptor.loader_contract_sha256
        or receipt.feature_panel_artifact_sha256 != descriptor.feature_panel_artifact_sha256
        or receipt.ordered_feature_keys_sha256 != descriptor.ordered_feature_keys_sha256
        or receipt.record_ids_sha256 != descriptor.record_ids_sha256
        or receipt.record_to_well_sha256 != descriptor.record_to_well_sha256
        or receipt.dataset_manifest_sha256 != descriptor.dataset_manifest_sha256
        or receipt.query_sha256 != descriptor.query_sha256
        or receipt.benchmark_sha256 != descriptor.benchmark_sha256
        or receipt.target_value_schema_sha256 != descriptor.target_value_schema_sha256
        or receipt.scoring_transform_sha256 != descriptor.scoring_transform_sha256
        or receipt.partition_id != _P1_PARTITION_ID
        or receipt.access_purpose != "train_parameters"
        or receipt.heldout_memberships_parsed
        or receipt.heldout_outcome_values_parsed
        or receipt.lifecycle_evidence_issued
        or receipt.scientifically_admissible
        or receipt.trusted_workflow_receipt_present
        or not receipt.close_reverification_required
        or receipt.count_scan_complete
        or receipt.close_reverification_completed
        or receipt.count_records_consumed != 0
        or receipt.count_batches_consumed != 0
    ):
        raise SciPlex3RunnerError("loader source-scan receipt crosses or mismatches p1 scope")


def _validate_finalized_count_receipt(
    receipt: SciPlex3P1FinalizedCountScanReceipt,
    *,
    initial: SciPlex3P1SourceScanReceipt,
    descriptor: SciPlex3PartitionDescriptor,
    emitted_source_rows_sha256: str,
    ordered_binding_sha256: str,
    panel_count_stream_sha256: str,
    batch_count: int,
    panel_nonzero_count: int,
    zero_panel_record_count: int,
    panel_umi_total: int,
) -> None:
    expected_encoding = (
        "canonical_json_utf8_array_of_[record_id,source_row_index,composite_well_id,"
        "condition_id,[[panel_feature_index,count],...],panel_total]_v1"
    )
    if (
        type(receipt) is not SciPlex3P1FinalizedCountScanReceipt
        or receipt.initial_source_authentication_fingerprint != initial.fingerprint
        or receipt.source_sha256 != descriptor.source_sha256
        or receipt.p1_loader_contract_sha256 != descriptor.loader_contract_sha256
        or receipt.dataset_manifest_sha256 != descriptor.dataset_manifest_sha256
        or receipt.query_sha256 != descriptor.query_sha256
        or receipt.benchmark_sha256 != descriptor.benchmark_sha256
        or receipt.target_value_schema_sha256 != descriptor.target_value_schema_sha256
        or receipt.scoring_transform_sha256 != descriptor.scoring_transform_sha256
        or receipt.feature_panel_artifact_sha256 != descriptor.feature_panel_artifact_sha256
        or receipt.ordered_feature_keys_sha256 != descriptor.ordered_feature_keys_sha256
        or receipt.record_ids_sha256 != descriptor.record_ids_sha256
        or receipt.record_to_well_sha256 != descriptor.record_to_well_sha256
        or receipt.emitted_source_row_indices_sha256 != emitted_source_rows_sha256
        or receipt.ordered_record_source_well_condition_sha256 != ordered_binding_sha256
        or receipt.count_stream_encoding != expected_encoding
        or receipt.panel_count_stream_sha256 != panel_count_stream_sha256
        or receipt.record_count != SCIPLEX3_P1_RECORD_COUNT
        or receipt.well_count != SCIPLEX3_P1_WELL_COUNT
        or receipt.treated_well_count != SCIPLEX3_P1_TREATED_WELL_COUNT
        or receipt.control_well_count != SCIPLEX3_P1_CONTROL_WELL_COUNT
        or receipt.batch_count != batch_count
        or receipt.panel_nonzero_count != panel_nonzero_count
        or receipt.zero_panel_record_count != zero_panel_record_count
        or receipt.zero_panel_record_count < 0
        or receipt.zero_panel_record_count > receipt.record_count
        or receipt.panel_umi_total != panel_umi_total
        or receipt.full_source_umi_total < panel_umi_total
        or receipt.source_descriptor_identity_before != receipt.source_descriptor_identity_after
        or receipt.loader_implementation_sha256 != SCIPLEX3_LOADER_CODE_SHA256
        or receipt.partition_id != _P1_PARTITION_ID
        or receipt.access_purpose != "train_parameters"
        or receipt.accessed_partition_roles != (_P1_PARTITION_ID,)
        or receipt.accessed_count_datasets != ("X.data", "X.indices", "X.indptr", "obs.ncounts")
        or not receipt.exact_record_coverage
        or not receipt.count_scan_complete
        or not receipt.source_descriptor_reverified
        or not receipt.close_reverification_completed
        or not receipt.finalized
        or receipt.heldout_memberships_parsed
        or receipt.heldout_outcome_values_parsed
        or receipt.trusted_workflow_receipt_present
        or receipt.lifecycle_evidence_issued
        or receipt.scientifically_admissible
    ):
        raise SciPlex3RunnerError(
            "finalized loader receipt does not prove exact p1 scan plus close reauthentication"
        )


def _validate_batch_structure(
    batch: SciPlex3SparseCountBatch,
    descriptor: SciPlex3PartitionDescriptor,
    expected_batch_index: int,
) -> None:
    if batch.partition != descriptor or batch.batch_index != expected_batch_index:
        raise SciPlex3RunnerError("p1 batch descriptor or sequence index drifted")
    row_count = len(batch.record_ids)
    if row_count == 0 or batch.shape != (row_count, SCIPLEX3_FEATURE_COUNT):
        raise SciPlex3RunnerError("p1 batch has an invalid row/feature shape")
    if not (
        len(batch.composite_well_ids)
        == len(batch.condition_ids)
        == len(batch.source_row_indices)
        == len(batch.panel_totals)
        == row_count
    ):
        raise SciPlex3RunnerError("p1 batch metadata are misaligned")
    if (
        batch.indptr.shape != (row_count + 1,)
        or int(batch.indptr[0]) != 0
        or bool(np.any(np.diff(batch.indptr) < 0))
        or int(batch.indptr[-1]) != len(batch.feature_indices)
        or len(batch.feature_indices) != len(batch.counts)
        or batch.counts.dtype.kind not in {"i", "u"}
        or batch.feature_indices.dtype.kind not in {"i", "u"}
        or bool(np.any(batch.counts <= 0))
        or bool(np.any(batch.panel_totals < 0))
    ):
        raise SciPlex3RunnerError("p1 batch CSR structure or raw-count domain is invalid")


def assemble_sciplex3_p1_training_data(
    loader: SciPlex3TrainingDataLoader,
    repository_root: Path,
    *,
    batch_size: int = 512,
) -> SciPlex3BaselinePreparation:
    """Assemble the exact full p1 loader stream into immutable well-local CSR counts.

    Biological metadata are never accepted as parameters.  Every record, well, condition, action,
    feature, query, action, and scoring identity is obtained from the loader's authenticated
    descriptor/receipt and the exact checked-in content-addressed closure.  Benchmark and held-out
    case bytes are deliberately not opened by this pre-fit operation.
    """

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise SciPlex3RunnerError("batch_size must be a positive integer")
    if not isinstance(loader, SciPlex3TrainingDataLoader):
        raise SciPlex3RunnerError("loader does not implement the sci-Plex3 training interface")
    if loader.access_purpose is not PopulationComponentAccessPurpose.TRAIN_PARAMETERS:
        raise SciPlex3RunnerError("baseline assembly accepts only a p1 parameter-training session")
    descriptor = loader.describe_partition()
    _validate_descriptor(descriptor)
    source_receipt = loader.source_scan_receipt
    _validate_source_receipt(source_receipt, descriptor)
    panel = loader.feature_panel
    panel_keys = tuple(panel.ordered_feature_keys)
    if (
        panel.ordered_feature_keys_sha256 != SCIPLEX3_ORDERED_FEATURE_KEYS_SHA256
        or _sha256(_canonical_json(list(panel_keys))) != SCIPLEX3_ORDERED_FEATURE_KEYS_SHA256
        or len(panel_keys) != SCIPLEX3_FEATURE_COUNT
    ):
        raise SciPlex3RunnerError("loader panel does not match the exact ordered feature surface")
    root = Path(repository_root).resolve()
    closure = _load_p1_artifact_closure(root, descriptor, panel_keys)
    design = _load_p1_design_bindings(root, descriptor)
    expected_well_by_record = dict(closure.record_to_well)
    expected_condition_by_well = dict(closure.well_to_condition)
    builders = {well_id: _WellBuilder() for well_id in closure.well_ids}
    source_rows: list[int] = []
    binding_rows: list[list[object]] = []
    runner_panel_stream = hashlib.sha256(b"[")
    panel_nonzero_count = 0
    zero_panel_record_count = 0
    panel_umi_total = 0
    observed_position = 0
    batch_count = 0
    for expected_batch_index, batch in enumerate(
        loader.iter_parameter_training_batches(
            batch_size=batch_size,
            partition_id=_P1_PARTITION_ID,
        )
    ):
        if not isinstance(batch, SciPlex3SparseCountBatch):
            raise SciPlex3RunnerError("training loader yielded a non-sci-Plex3 batch")
        _validate_batch_structure(batch, descriptor, expected_batch_index)
        batch_count += 1
        for row_index, (record_id, well_id, condition_id) in enumerate(
            zip(
                batch.record_ids,
                batch.composite_well_ids,
                batch.condition_ids,
                strict=True,
            )
        ):
            if observed_position >= SCIPLEX3_P1_RECORD_COUNT:
                raise SciPlex3RunnerError("loader yielded more than 94,785 p1 records")
            expected_record = closure.record_ids[observed_position]
            if record_id != expected_record:
                raise SciPlex3RunnerError("loader p1 record order differs from exact membership")
            if expected_well_by_record.get(record_id) != well_id:
                raise SciPlex3RunnerError("loader record-to-well mapping differs from p1 closure")
            if expected_condition_by_well.get(well_id) != condition_id:
                raise SciPlex3RunnerError("loader well condition differs from p1 closure")
            source_row = int(batch.source_row_indices[row_index])
            if source_row < 0:
                raise SciPlex3RunnerError("loader yielded a negative source row")
            start = int(batch.indptr[row_index])
            stop = int(batch.indptr[row_index + 1])
            row_indices = np.asarray(batch.feature_indices[start:stop], dtype=np.int64)
            row_counts = np.asarray(batch.counts[start:stop], dtype=np.int64)
            if (
                bool(np.any(np.diff(row_indices) <= 0))
                or bool(np.any(row_indices < 0))
                or bool(np.any(row_indices >= SCIPLEX3_FEATURE_COUNT))
                or int(np.sum(row_counts, dtype=np.int64)) != int(batch.panel_totals[row_index])
            ):
                raise SciPlex3RunnerError("loader yielded an invalid CSR row")
            try:
                builder = builders[well_id]
            except KeyError as error:
                raise SciPlex3RunnerError("loader yielded a well outside exact p1") from error
            builder.append(
                record_id=record_id,
                source_row=source_row,
                feature_indices=row_indices,
                counts=row_counts,
            )
            source_rows.append(source_row)
            binding_rows.append([record_id, source_row, well_id, condition_id])
            panel_pairs = [
                [int(index), int(count)]
                for index, count in zip(row_indices, row_counts, strict=True)
            ]
            if observed_position:
                runner_panel_stream.update(b",")
            runner_panel_stream.update(
                _canonical_json(
                    [
                        record_id,
                        source_row,
                        well_id,
                        condition_id,
                        panel_pairs,
                        int(batch.panel_totals[row_index]),
                    ]
                )
            )
            panel_nonzero_count += len(panel_pairs)
            zero_panel_record_count += int(batch.panel_totals[row_index]) == 0
            panel_umi_total += int(batch.panel_totals[row_index])
            observed_position += 1
    if batch_count == 0 or observed_position != SCIPLEX3_P1_RECORD_COUNT:
        raise SciPlex3RunnerError("loader did not yield the exact 94,785-record p1 closure")
    if len(set(source_rows)) != SCIPLEX3_P1_RECORD_COUNT:
        raise SciPlex3RunnerError("loader p1 source-row indices are not globally unique")
    runner_panel_stream.update(b"]")
    runner_panel_stream_digest = runner_panel_stream.hexdigest()
    record_ids_digest = _sha256(_canonical_json(list(closure.record_ids)))
    record_to_well_digest = _sha256(
        _canonical_json([list(pair) for pair in closure.record_to_well])
    )
    well_ids_digest = _sha256(_canonical_json(list(builders)))
    observed_well_conditions = [
        [well_id, expected_condition_by_well[well_id]] for well_id in builders
    ]
    well_to_condition_digest = _sha256(_canonical_json(observed_well_conditions))
    source_rows_digest = _sha256(_canonical_json([str(row) for row in sorted(source_rows)]))
    emitted_source_rows_digest = _sha256(_canonical_json(source_rows))
    ordered_binding_digest = _sha256(_canonical_json(binding_rows))
    finalized_count_receipt = loader.finalize_parameter_training_count_scan()
    _validate_finalized_count_receipt(
        finalized_count_receipt,
        initial=source_receipt,
        descriptor=descriptor,
        emitted_source_rows_sha256=emitted_source_rows_digest,
        ordered_binding_sha256=ordered_binding_digest,
        panel_count_stream_sha256=runner_panel_stream_digest,
        batch_count=batch_count,
        panel_nonzero_count=panel_nonzero_count,
        zero_panel_record_count=zero_panel_record_count,
        panel_umi_total=panel_umi_total,
    )
    if (
        record_ids_digest != SCIPLEX3_P1_RECORD_IDS_SHA256
        or record_ids_digest != descriptor.record_ids_sha256
        or record_to_well_digest != SCIPLEX3_P1_RECORD_TO_WELL_SHA256
        or record_to_well_digest != descriptor.record_to_well_sha256
        or well_ids_digest != SCIPLEX3_P1_WELL_IDS_SHA256
        or source_rows_digest != source_receipt.source_row_indices_sha256
        or ordered_binding_digest != source_receipt.ordered_record_source_well_condition_sha256
    ):
        raise SciPlex3RunnerError("assembled p1 hashes differ from authoritative loader closure")

    wells: list[P1WellCounts] = []
    treated_conditions: set[CompoundDose] = set()
    control_count = 0
    for well_id, builder in builders.items():
        plate_id, _ = _parse_composite_well(well_id)
        condition_id = expected_condition_by_well[well_id]
        if condition_id == _CONTROL_CONDITION_ID:
            condition = None
            control_count += 1
        else:
            try:
                action = design.actions_by_source_condition[condition_id]
            except KeyError as error:
                raise SciPlex3RunnerError(
                    "p1 treated well lacks exact action-domain mapping"
                ) from error
            condition = CompoundDose(action.compound, action.dose_nm)
            if condition in treated_conditions:
                raise SciPlex3RunnerError("p1 contains duplicate support for one action condition")
            treated_conditions.add(condition)
        wells.append(
            P1WellCounts(
                well_id=well_id,
                plate_id=plate_id,
                condition=condition,
                counts=builder.freeze(),
                record_ids=tuple(builder.record_ids),
                source_row_indices=tuple(builder.source_rows),
            )
        )
    if (
        len(wells) != SCIPLEX3_P1_WELL_COUNT
        or len(treated_conditions) != SCIPLEX3_P1_TREATED_WELL_COUNT
        or control_count != SCIPLEX3_P1_CONTROL_WELL_COUNT
        or treated_conditions
        != {
            CompoundDose(action.compound, action.dose_nm)
            for action in design.actions_by_source_condition.values()
        }
    ):
        raise SciPlex3RunnerError("assembled p1 treated/control action coverage drifted")
    training = P1TrainingData(ordered_feature_keys=panel_keys, wells=tuple(wells))
    receipt = SciPlex3P1AssemblyReceipt(
        loader_source_scan_fingerprint=source_receipt.fingerprint,
        finalized_count_scan_fingerprint=finalized_count_receipt.fingerprint,
        loader_implementation_sha256=finalized_count_receipt.loader_implementation_sha256,
        loader_contract_sha256=closure.contract_sha256,
        source_sha256=descriptor.source_sha256,
        record_count=SCIPLEX3_P1_RECORD_COUNT,
        well_count=SCIPLEX3_P1_WELL_COUNT,
        treated_well_count=SCIPLEX3_P1_TREATED_WELL_COUNT,
        control_well_count=SCIPLEX3_P1_CONTROL_WELL_COUNT,
        record_ids_sha256=record_ids_digest,
        record_to_well_sha256=record_to_well_digest,
        well_ids_sha256=well_ids_digest,
        well_to_condition_sha256=well_to_condition_digest,
        source_row_indices_sha256=source_rows_digest,
        emitted_source_row_indices_sha256=emitted_source_rows_digest,
        ordered_record_source_well_condition_sha256=ordered_binding_digest,
        runner_panel_count_stream_sha256=runner_panel_stream_digest,
        loader_panel_count_stream_sha256=finalized_count_receipt.panel_count_stream_sha256,
        panel_nonzero_count=panel_nonzero_count,
        zero_panel_record_count=zero_panel_record_count,
        panel_umi_total=panel_umi_total,
        full_source_umi_total=finalized_count_receipt.full_source_umi_total,
        batch_count=batch_count,
        ordered_feature_keys_sha256=SCIPLEX3_ORDERED_FEATURE_KEYS_SHA256,
        feature_panel_artifact_sha256=descriptor.feature_panel_artifact_sha256,
        action_domain_sha256=design.action_domain_sha256,
        query_sha256=design.query_sha256,
        benchmark_sha256=design.benchmark_sha256,
        scoring_transform_sha256=design.scoring_transform_sha256,
        target_value_schema_sha256=design.target_value_schema_sha256,
    )
    return SciPlex3BaselinePreparation(
        training_data=training,
        receipt=receipt,
        finalized_count_scan_receipt=finalized_count_receipt,
        design=design,
        repository_root=root,
    )


def _code_identity(
    repository_root: Path, relative_path: Path, *, entrypoint: str
) -> dict[str, object]:
    repository_path = (repository_root / relative_path).resolve()
    if relative_path == _BASELINE_CODE_PATH:
        imported_path = _IMPORTED_BASELINE_CODE_PATH
        imported_sha256 = _IMPORTED_BASELINE_CODE_SHA256
        expected_sha256: str | None = SCIPLEX3_BASELINE_CODE_SHA256
    elif relative_path == _RUNNER_CODE_PATH:
        imported_path = _IMPORTED_RUNNER_CODE_PATH
        imported_sha256 = _IMPORTED_RUNNER_CODE_SHA256
        expected_sha256 = None
    else:  # pragma: no cover - this helper has a deliberately closed code domain
        raise SciPlex3RunnerError("implementation code path is outside the loaded runner closure")
    if repository_path != imported_path:
        raise SciPlex3RunnerError(
            f"loaded implementation path differs from repository closure: {relative_path}"
        )
    payload = _read_bytes(repository_path, name=f"implementation code {relative_path}")
    observed_sha256 = _sha256(payload)
    if observed_sha256 != imported_sha256:
        raise SciPlex3RunnerError(f"loaded implementation changed since import: {relative_path}")
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        raise SciPlex3RunnerError(
            f"loaded implementation differs from frozen executable: {relative_path}"
        )
    return {
        "byte_count": len(payload),
        "entrypoint": entrypoint,
        "relative_path": relative_path.as_posix(),
        "sha256": observed_sha256,
    }


def _runtime_identity() -> dict[str, object]:
    return {
        "byte_order": sys.byteorder,
        "numpy_version": np.__version__,
        "platform_machine": platform.machine(),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "rng_algorithm": RNG_ALGORITHM,
    }


def _require_golden_runtime() -> None:
    if (
        sys.version_info[:2] != SCIPLEX3_EXECUTABLE_PYTHON
        or np.__version__ != SCIPLEX3_EXECUTABLE_NUMPY_VERSION
    ):
        raise SciPlex3RunnerError(
            "baseline executable binding requires Python 3.11 and NumPy 2.4.6; "
            "this runtime may not emit fitted or prediction artifacts"
        )


def _authenticate_executable_baseline_binding(
    repository_root: Path,
    baseline_id: str,
) -> dict[str, object]:
    _require_golden_runtime()
    if baseline_id not in SCIPLEX3_BASELINE_IMPLEMENTATIONS:
        raise SciPlex3RunnerError(f"unknown frozen sci-Plex3 baseline: {baseline_id!r}")
    specification_payload = _read_bytes(
        repository_root / _BASELINE_SUITE_SPEC_PATH,
        name="baseline-suite specification",
    )
    specification = _json_object(specification_payload, name="baseline-suite specification")
    code = _as_mapping(specification.get("code"), name="baseline-suite code binding")
    golden = _as_mapping(specification.get("golden_fixture"), name="baseline-suite golden binding")
    input_contract = _as_mapping(
        specification.get("input_contract"), name="baseline-suite input contract"
    )
    prediction_contract = _as_mapping(
        specification.get("prediction_contract"),
        name="baseline-suite prediction contract",
    )
    if (
        specification.get("artifact_schema") != "sciplex3-k562-frozen-baseline-suite"
        or specification.get("artifact_schema_version") != "1.1.0"
        or code.get("path") != _BASELINE_CODE_PATH.as_posix()
        or code.get("implementation_version") != SCIPLEX3_BASELINE_IMPLEMENTATION_VERSION
        or input_contract.get("fit_partition") != f"{_P1_PARTITION_ID} only"
        or input_contract.get("heldout_outcomes_allowed") is not False
        or prediction_contract.get("rng_algorithm") != RNG_ALGORITHM
        or prediction_contract.get("samples_per_case_per_seed")
        != SCIPLEX3_BASELINE_SAMPLES_PER_CASE_PER_SEED
        or prediction_contract.get("seeds") != list(SCIPLEX3_BASELINE_SEEDS)
    ):
        raise SciPlex3RunnerError("baseline-suite executable contract drifted")
    raw_baselines = _as_list(specification.get("baselines"), name="baseline-suite IDs")
    if baseline_id not in raw_baselines:
        raise SciPlex3RunnerError("requested baseline is absent from frozen suite")
    baseline_code_identity = _code_identity(
        repository_root,
        _BASELINE_CODE_PATH,
        entrypoint="cellstate.evaluation.sciplex3_baselines:SCIPLEX3_BASELINE_IMPLEMENTATIONS",
    )
    runner_code_identity = _code_identity(
        repository_root,
        _RUNNER_CODE_PATH,
        entrypoint="cellstate.evaluation.sciplex3_runner",
    )
    code_sha256 = baseline_code_identity["sha256"]
    if (
        code_sha256 != SCIPLEX3_BASELINE_CODE_SHA256
        or code.get("sha256") != SCIPLEX3_BASELINE_CODE_SHA256
    ):
        raise SciPlex3RunnerError("baseline implementation differs from executable binding")
    golden_path = golden.get("path")
    golden_sha256 = golden.get("sha256")
    if (
        golden_path != _BASELINE_GOLDEN_FIXTURE_PATH.as_posix()
        or golden_sha256 != SCIPLEX3_BASELINE_GOLDEN_FIXTURE_SHA256
    ):
        raise SciPlex3RunnerError("baseline golden-fixture binding drifted")
    golden_payload = _read_exact(
        repository_root / _BASELINE_GOLDEN_FIXTURE_PATH,
        golden_sha256,
        name="baseline golden fixtures",
    )
    golden_document = _json_object(golden_payload, name="baseline golden fixtures")
    production = _as_mapping(
        golden_document.get("production_sampling_contract"),
        name="golden production sampling contract",
    )
    results = _as_list(golden_document.get("results"), name="golden results")
    result_ids = {
        result.get("baseline_id")
        for raw in results
        for result in [_as_mapping(raw, name="golden baseline result")]
    }
    if (
        golden_document.get("artifact_schema") != "sciplex3-k562-baseline-golden-fixtures"
        or golden_document.get("implementation_version") != SCIPLEX3_BASELINE_IMPLEMENTATION_VERSION
        or golden_document.get("feature_count") != SCIPLEX3_FEATURE_COUNT
        or golden_document.get("biological_performance_evidence") is not False
        or golden_document.get("scientific_admission_authorized") is not False
        or production.get("rng_algorithm") != RNG_ALGORITHM
        or production.get("samples_per_case_per_seed")
        != SCIPLEX3_BASELINE_SAMPLES_PER_CASE_PER_SEED
        or production.get("seeds") != list(SCIPLEX3_BASELINE_SEEDS)
        or baseline_id not in result_ids
    ):
        raise SciPlex3RunnerError("baseline golden-fixture contract drifted")
    return {
        "baseline_suite_specification": {
            "byte_count": len(specification_payload),
            "relative_path": _BASELINE_SUITE_SPEC_PATH.as_posix(),
            "sha256": _sha256(specification_payload),
        },
        "golden_fixture": {
            "byte_count": len(golden_payload),
            "relative_path": _BASELINE_GOLDEN_FIXTURE_PATH.as_posix(),
            "sha256": golden_sha256,
        },
        "implementation_code": baseline_code_identity,
        "runner_code": runner_code_identity,
    }


def _exclusive_directory(path: Path) -> Path:
    output = Path(path)
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise SciPlex3RunnerError(
            f"refusing to overwrite existing artifact directory: {output}"
        ) from error
    except OSError as error:
        raise SciPlex3RunnerError(f"cannot create artifact directory: {output}") from error
    return output


def _write_exclusive(path: Path, payload: bytes) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            written = stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as error:
        raise SciPlex3RunnerError(f"cannot write content-addressed artifact: {path}") from error
    if written != len(payload):
        raise SciPlex3RunnerError(f"short write for content-addressed artifact: {path}")


def _verify_local_artifact(
    path: Path,
    *,
    expected_payload: bytes,
    media_type: str,
) -> LocalContentAddressedArtifact:
    observed = _read_bytes(path, name="newly written local artifact")
    if observed != expected_payload:
        raise SciPlex3RunnerError("newly written artifact differs on immediate re-read")
    return LocalContentAddressedArtifact(
        path=path.resolve(),
        sha256=_sha256(observed),
        byte_count=len(observed),
        media_type=media_type,
    )


@dataclass(frozen=True, slots=True)
class _InMemoryP1Identity:
    record_count: int
    well_count: int
    treated_well_count: int
    control_well_count: int
    record_ids_sha256: str
    record_to_well_sha256: str
    well_ids_sha256: str
    well_to_condition_sha256: str
    source_row_indices_sha256: str
    emitted_source_row_indices_sha256: str
    ordered_record_source_well_condition_sha256: str
    panel_count_stream_sha256: str
    panel_nonzero_count: int
    zero_panel_record_count: int
    panel_umi_total: int
    ordered_feature_keys_sha256: str


def _recompute_in_memory_p1_identity(
    preparation: SciPlex3BaselinePreparation,
) -> _InMemoryP1Identity:
    """Rehash the fit-ready CSR surface so a valid receipt cannot be paired with other counts."""

    condition_id_by_target: dict[CompoundDose, str] = {}
    for condition_id, action in preparation.design.actions_by_source_condition.items():
        target = CompoundDose(action.compound, action.dose_nm)
        if target in condition_id_by_target:
            raise SciPlex3RunnerError("p1 action domain has duplicate fitted target semantics")
        condition_id_by_target[target] = condition_id
    ordered_rows: list[tuple[str, int, str, str, list[list[int]], int]] = []
    well_ids: list[str] = []
    well_conditions: list[list[str]] = []
    treated_conditions: set[CompoundDose] = set()
    control_well_count = 0
    for well in preparation.training_data.wells:
        plate_id, _ = _parse_composite_well(well.well_id)
        if well.plate_id != plate_id:
            raise SciPlex3RunnerError("in-memory p1 well plate differs from its composite ID")
        if well.condition is None:
            condition_id = _CONTROL_CONDITION_ID
            control_well_count += 1
        else:
            try:
                condition_id = condition_id_by_target[well.condition]
            except KeyError as error:
                raise SciPlex3RunnerError(
                    "in-memory p1 well condition is outside the exact action domain"
                ) from error
            if well.condition in treated_conditions:
                raise SciPlex3RunnerError("in-memory p1 repeats one treated action condition")
            treated_conditions.add(well.condition)
        well_ids.append(well.well_id)
        well_conditions.append([well.well_id, condition_id])
        for row_index, (record_id, source_row) in enumerate(
            zip(well.record_ids, well.source_row_indices, strict=True)
        ):
            start = int(well.counts.indptr[row_index])
            stop = int(well.counts.indptr[row_index + 1])
            indices = well.counts.feature_indices[start:stop]
            values = well.counts.values[start:stop]
            pairs = [[int(index), int(count)] for index, count in zip(indices, values, strict=True)]
            panel_total = int(np.sum(values, dtype=np.int64))
            ordered_rows.append(
                (record_id, source_row, well.well_id, condition_id, pairs, panel_total)
            )
    ordered_rows.sort(key=lambda item: item[0])
    record_ids = [row[0] for row in ordered_rows]
    record_to_well = [[row[0], row[2]] for row in ordered_rows]
    source_rows = [row[1] for row in ordered_rows]
    binding_rows = [[row[0], row[1], row[2], row[3]] for row in ordered_rows]
    panel_stream = hashlib.sha256(b"[")
    panel_nonzero_count = 0
    zero_panel_record_count = 0
    panel_umi_total = 0
    for position, row in enumerate(ordered_rows):
        if position:
            panel_stream.update(b",")
        panel_stream.update(_canonical_json(list(row)))
        panel_nonzero_count += len(row[4])
        zero_panel_record_count += row[5] == 0
        panel_umi_total += row[5]
    panel_stream.update(b"]")
    return _InMemoryP1Identity(
        record_count=len(ordered_rows),
        well_count=len(well_ids),
        treated_well_count=len(treated_conditions),
        control_well_count=control_well_count,
        record_ids_sha256=_sha256(_canonical_json(record_ids)),
        record_to_well_sha256=_sha256(_canonical_json(record_to_well)),
        well_ids_sha256=_sha256(_canonical_json(well_ids)),
        well_to_condition_sha256=_sha256(_canonical_json(well_conditions)),
        source_row_indices_sha256=_sha256(
            _canonical_json([str(row) for row in sorted(source_rows)])
        ),
        emitted_source_row_indices_sha256=_sha256(_canonical_json(source_rows)),
        ordered_record_source_well_condition_sha256=_sha256(_canonical_json(binding_rows)),
        panel_count_stream_sha256=panel_stream.hexdigest(),
        panel_nonzero_count=panel_nonzero_count,
        zero_panel_record_count=zero_panel_record_count,
        panel_umi_total=panel_umi_total,
        ordered_feature_keys_sha256=_sha256(
            _canonical_json(list(preparation.training_data.ordered_feature_keys))
        ),
    )


def _exact_p1_descriptor() -> SciPlex3PartitionDescriptor:
    return SciPlex3PartitionDescriptor(
        partition_id=_P1_PARTITION_ID,
        artifact_role="train",
        benchmark_role=BenchmarkPartitionRole.TRAIN,
        access_purpose=PopulationComponentAccessPurpose.TRAIN_PARAMETERS,
        record_count=SCIPLEX3_P1_RECORD_COUNT,
        well_count=SCIPLEX3_P1_WELL_COUNT,
        record_ids_sha256=SCIPLEX3_P1_RECORD_IDS_SHA256,
        record_to_well_sha256=SCIPLEX3_P1_RECORD_TO_WELL_SHA256,
        source_sha256=SCIPLEX3_SOURCE_SHA256,
        loader_contract_sha256=SCIPLEX3_P1_LOADER_CONTRACT_SHA256,
        dataset_manifest_sha256=SCIPLEX3_K562_MANIFEST_SHA256,
        query_sha256=SCIPLEX3_K562_QUERY_SHA256,
        benchmark_sha256=SCIPLEX3_K562_BENCHMARK_SHA256,
        target_value_schema_sha256=SCIPLEX3_K562_TARGET_VALUE_SCHEMA_SHA256,
        scoring_transform_sha256=SCIPLEX3_SCORING_TRANSFORM_SHA256,
        feature_panel_artifact_sha256=SCIPLEX3_FEATURE_PANEL_ARTIFACT_SHA256,
        ordered_feature_keys_sha256=SCIPLEX3_ORDERED_FEATURE_KEYS_SHA256,
        count_access_sealed=False,
    )


def _validate_finalized_receipt_for_fit(
    preparation: SciPlex3BaselinePreparation,
    in_memory: _InMemoryP1Identity,
) -> None:
    receipt = preparation.receipt
    finalized = preparation.finalized_count_scan_receipt
    descriptor_identity = finalized.source_descriptor_identity_before
    runtime_values = (
        finalized.python_version,
        finalized.python_implementation,
        finalized.numpy_version,
        finalized.h5py_version,
        finalized.hdf5_version,
    )
    expected_encoding = (
        "canonical_json_utf8_array_of_[record_id,source_row_index,composite_well_id,"
        "condition_id,[[panel_feature_index,count],...],panel_total]_v1"
    )
    if (
        finalized.artifact_schema != "sciplex3-k562-p1-finalized-count-scan-receipt"
        or finalized.artifact_schema_version != "1.0.0"
        or finalized.loader_interface_id != "cellstate.sciplex3-training-data-loader.v1"
        or finalized.initial_source_authentication_fingerprint
        != receipt.loader_source_scan_fingerprint
        or len(finalized.initial_source_authentication_fingerprint) != 64
        or finalized.source_sha256 != SCIPLEX3_SOURCE_SHA256
        or finalized.source_sha256 != receipt.source_sha256
        or finalized.source_md5 != SCIPLEX3_SOURCE_MD5
        or finalized.source_byte_count != SCIPLEX3_SOURCE_BYTE_COUNT
        or finalized.source_descriptor_identity_after != descriptor_identity
        or len(descriptor_identity) != 5
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in descriptor_identity
        )
        or descriptor_identity[2] != SCIPLEX3_SOURCE_BYTE_COUNT
        or finalized.loader_implementation_sha256 != SCIPLEX3_LOADER_CODE_SHA256
        or finalized.loader_implementation_sha256 != receipt.loader_implementation_sha256
        or finalized.p1_loader_contract_sha256 != SCIPLEX3_P1_LOADER_CONTRACT_SHA256
        or finalized.p1_loader_contract_sha256 != receipt.loader_contract_sha256
        or finalized.dataset_manifest_sha256 != SCIPLEX3_K562_MANIFEST_SHA256
        or finalized.query_sha256 != SCIPLEX3_K562_QUERY_SHA256
        or finalized.query_sha256 != receipt.query_sha256
        or finalized.benchmark_sha256 != SCIPLEX3_K562_BENCHMARK_SHA256
        or finalized.benchmark_sha256 != receipt.benchmark_sha256
        or finalized.target_value_schema_sha256 != SCIPLEX3_K562_TARGET_VALUE_SCHEMA_SHA256
        or finalized.target_value_schema_sha256 != receipt.target_value_schema_sha256
        or finalized.scoring_transform_sha256 != SCIPLEX3_SCORING_TRANSFORM_SHA256
        or finalized.scoring_transform_sha256 != receipt.scoring_transform_sha256
        or finalized.feature_panel_artifact_sha256 != SCIPLEX3_FEATURE_PANEL_ARTIFACT_SHA256
        or finalized.feature_panel_artifact_sha256 != receipt.feature_panel_artifact_sha256
        or finalized.ordered_feature_keys_sha256 != SCIPLEX3_ORDERED_FEATURE_KEYS_SHA256
        or finalized.ordered_feature_keys_sha256 != receipt.ordered_feature_keys_sha256
        or finalized.record_ids_sha256 != SCIPLEX3_P1_RECORD_IDS_SHA256
        or finalized.record_ids_sha256 != receipt.record_ids_sha256
        or finalized.record_to_well_sha256 != SCIPLEX3_P1_RECORD_TO_WELL_SHA256
        or finalized.record_to_well_sha256 != receipt.record_to_well_sha256
        or finalized.emitted_source_row_indices_sha256
        != in_memory.emitted_source_row_indices_sha256
        or finalized.emitted_source_row_indices_sha256 != receipt.emitted_source_row_indices_sha256
        or finalized.ordered_record_source_well_condition_sha256
        != in_memory.ordered_record_source_well_condition_sha256
        or finalized.count_stream_encoding != expected_encoding
        or finalized.panel_count_stream_sha256 != in_memory.panel_count_stream_sha256
        or finalized.panel_count_stream_sha256 != receipt.runner_panel_count_stream_sha256
        or finalized.panel_count_stream_sha256 != receipt.loader_panel_count_stream_sha256
        or finalized.record_count != SCIPLEX3_P1_RECORD_COUNT
        or finalized.record_count != receipt.record_count
        or finalized.well_count != SCIPLEX3_P1_WELL_COUNT
        or finalized.well_count != receipt.well_count
        or finalized.treated_well_count != SCIPLEX3_P1_TREATED_WELL_COUNT
        or finalized.treated_well_count != receipt.treated_well_count
        or finalized.control_well_count != SCIPLEX3_P1_CONTROL_WELL_COUNT
        or finalized.control_well_count != receipt.control_well_count
        or finalized.batch_count != receipt.batch_count
        or finalized.batch_count <= 0
        or finalized.panel_nonzero_count != in_memory.panel_nonzero_count
        or finalized.panel_nonzero_count != receipt.panel_nonzero_count
        or finalized.zero_panel_record_count != in_memory.zero_panel_record_count
        or finalized.zero_panel_record_count != receipt.zero_panel_record_count
        or finalized.panel_umi_total != in_memory.panel_umi_total
        or finalized.panel_umi_total != receipt.panel_umi_total
        or finalized.full_source_umi_total != receipt.full_source_umi_total
        or finalized.full_source_umi_total < finalized.panel_umi_total
        or any(not isinstance(value, str) or not value for value in runtime_values)
        or finalized.partition_id != _P1_PARTITION_ID
        or finalized.access_purpose != "train_parameters"
        or finalized.accessed_partition_roles != (_P1_PARTITION_ID,)
        or finalized.accessed_count_datasets != ("X.data", "X.indices", "X.indptr", "obs.ncounts")
        or not finalized.exact_record_coverage
        or not finalized.count_scan_complete
        or not finalized.source_descriptor_reverified
        or not finalized.close_reverification_completed
        or not finalized.finalized
        or finalized.heldout_memberships_parsed
        or finalized.heldout_outcome_values_parsed
        or finalized.trusted_workflow_receipt_present
        or finalized.lifecycle_evidence_issued
        or finalized.scientifically_admissible
        or finalized.fingerprint != receipt.finalized_count_scan_fingerprint
    ):
        raise SciPlex3RunnerError(
            "baseline preparation finalized receipt is not the exact closed p1 count scan"
        )


def _validate_preparation(preparation: SciPlex3BaselinePreparation) -> None:
    receipt = preparation.receipt
    finalized = preparation.finalized_count_scan_receipt
    loader_code = _read_exact(
        preparation.repository_root / _LOADER_CODE_PATH,
        SCIPLEX3_LOADER_CODE_SHA256,
        name="p1 loader implementation",
    )
    if _sha256(loader_code) != finalized.loader_implementation_sha256:
        raise SciPlex3RunnerError("finalized scan does not bind the checked-in loader code")
    if (
        receipt.record_count != SCIPLEX3_P1_RECORD_COUNT
        or receipt.well_count != SCIPLEX3_P1_WELL_COUNT
        or receipt.treated_well_count != SCIPLEX3_P1_TREATED_WELL_COUNT
        or receipt.control_well_count != SCIPLEX3_P1_CONTROL_WELL_COUNT
        or receipt.record_ids_sha256 != SCIPLEX3_P1_RECORD_IDS_SHA256
        or receipt.record_to_well_sha256 != SCIPLEX3_P1_RECORD_TO_WELL_SHA256
        or receipt.well_ids_sha256 != SCIPLEX3_P1_WELL_IDS_SHA256
        or receipt.well_to_condition_sha256 != SCIPLEX3_P1_WELL_TO_CONDITION_SHA256
        or receipt.action_domain_sha256 != SCIPLEX3_ACTION_DOMAIN_SHA256
        or receipt.scoring_transform_sha256 != SCIPLEX3_SCORING_TRANSFORM_SHA256
        or receipt.ordered_feature_keys_sha256 != SCIPLEX3_ORDERED_FEATURE_KEYS_SHA256
        or len(receipt.finalized_count_scan_fingerprint) != 64
        or receipt.loader_implementation_sha256 != SCIPLEX3_LOADER_CODE_SHA256
        or len(receipt.runner_panel_count_stream_sha256) != 64
        or len(receipt.loader_panel_count_stream_sha256) != 64
        or receipt.batch_count <= 0
        or receipt.panel_nonzero_count < 0
        or receipt.zero_panel_record_count < 0
        or receipt.zero_panel_record_count > SCIPLEX3_P1_RECORD_COUNT
        or receipt.panel_umi_total <= 0
        or receipt.full_source_umi_total < receipt.panel_umi_total
        or not receipt.exact_record_coverage
        or not receipt.count_scan_complete
        or not receipt.close_reverification_completed
        or receipt.heldout_memberships_read
        or receipt.heldout_outcomes_read
        or receipt.can_mint_lifecycle_evidence
        or receipt.scientifically_admissible
        or preparation.can_mint_lifecycle_evidence
        or preparation.scientifically_admissible
        or finalized.fingerprint != receipt.finalized_count_scan_fingerprint
        or finalized.loader_implementation_sha256 != SCIPLEX3_LOADER_CODE_SHA256
        or finalized.p1_loader_contract_sha256 != receipt.loader_contract_sha256
        or finalized.panel_count_stream_sha256 != receipt.runner_panel_count_stream_sha256
        or finalized.panel_count_stream_sha256 != receipt.loader_panel_count_stream_sha256
        or finalized.panel_nonzero_count != receipt.panel_nonzero_count
        or finalized.zero_panel_record_count != receipt.zero_panel_record_count
        or finalized.panel_umi_total != receipt.panel_umi_total
        or finalized.full_source_umi_total != receipt.full_source_umi_total
        or finalized.batch_count != receipt.batch_count
        or not finalized.count_scan_complete
        or not finalized.close_reverification_completed
        or not finalized.finalized
        or finalized.heldout_memberships_parsed
        or finalized.heldout_outcome_values_parsed
        or finalized.lifecycle_evidence_issued
        or finalized.scientifically_admissible
    ):
        raise SciPlex3RunnerError("baseline preparation is not the exact non-admissible p1 closure")
    if (
        len(preparation.training_data.wells) != SCIPLEX3_P1_WELL_COUNT
        or sum(well.counts.row_count for well in preparation.training_data.wells)
        != SCIPLEX3_P1_RECORD_COUNT
    ):
        raise SciPlex3RunnerError("baseline preparation in-memory p1 counts are incomplete")
    descriptor = _exact_p1_descriptor()
    exact_design = _load_p1_design_bindings(preparation.repository_root, descriptor)
    if preparation.design != exact_design:
        raise SciPlex3RunnerError(
            "baseline preparation design differs from the exact p1-safe "
            "query/action/scoring closure"
        )
    in_memory = _recompute_in_memory_p1_identity(preparation)
    if (
        in_memory.record_count != receipt.record_count
        or in_memory.well_count != receipt.well_count
        or in_memory.treated_well_count != receipt.treated_well_count
        or in_memory.control_well_count != receipt.control_well_count
        or in_memory.record_ids_sha256 != receipt.record_ids_sha256
        or in_memory.record_to_well_sha256 != receipt.record_to_well_sha256
        or in_memory.well_ids_sha256 != receipt.well_ids_sha256
        or in_memory.well_to_condition_sha256 != receipt.well_to_condition_sha256
        or in_memory.source_row_indices_sha256 != receipt.source_row_indices_sha256
        or in_memory.emitted_source_row_indices_sha256 != receipt.emitted_source_row_indices_sha256
        or in_memory.ordered_record_source_well_condition_sha256
        != receipt.ordered_record_source_well_condition_sha256
        or in_memory.panel_count_stream_sha256 != receipt.runner_panel_count_stream_sha256
        or in_memory.panel_count_stream_sha256 != receipt.loader_panel_count_stream_sha256
        or in_memory.panel_nonzero_count != receipt.panel_nonzero_count
        or in_memory.zero_panel_record_count != receipt.zero_panel_record_count
        or in_memory.panel_umi_total != receipt.panel_umi_total
        or in_memory.ordered_feature_keys_sha256 != receipt.ordered_feature_keys_sha256
    ):
        raise SciPlex3RunnerError(
            "baseline preparation in-memory CSR differs from its authenticated p1 count stream"
        )
    _validate_finalized_receipt_for_fit(preparation, in_memory)


def _require_exact_fitted_implementation(fitted: FittedSciPlex3Baseline) -> str:
    if type(fitted) is not FittedSciPlex3Baseline:
        raise SciPlex3RunnerError("fitted baseline binding is not the exact registered type")
    if type(fitted.artifact) is not LocalContentAddressedArtifact:
        raise SciPlex3RunnerError("fitted artifact binding is not the exact registered type")
    baseline_id = fitted.baseline.baseline_id
    expected = SCIPLEX3_BASELINE_IMPLEMENTATIONS.get(baseline_id)
    if expected is None or type(fitted.baseline) is not expected:
        raise SciPlex3RunnerError(
            "fitted baseline object is not the exact registered executable implementation"
        )
    return baseline_id


def _verify_fitted_artifact_binding(
    preparation: SciPlex3BaselinePreparation,
    fitted: FittedSciPlex3Baseline,
) -> Mapping[str, object]:
    exact_baseline_id = _require_exact_fitted_implementation(fitted)
    if (
        fitted.preparation_fingerprint != preparation.receipt.fingerprint
        or fitted.can_mint_lifecycle_evidence is not False
        or fitted.scientifically_admissible is not False
        or fitted.artifact.can_mint_lifecycle_evidence is not False
        or fitted.artifact.scientifically_admissible is not False
    ):
        raise SciPlex3RunnerError("fitted baseline is not bound to this exact p1 preparation")
    fit_payload = _read_exact(
        fitted.artifact.path,
        fitted.artifact.sha256,
        name="p1 fitted-state artifact",
    )
    if len(fit_payload) != fitted.artifact.byte_count:
        raise SciPlex3RunnerError("p1 fitted-state artifact byte count drifted")
    parsed_fit = _json_object(fit_payload, name="p1 fitted-state artifact")
    artifact_manifest = _mutable_json_value(fitted.artifact_manifest)
    if parsed_fit != artifact_manifest or fit_payload != _canonical_json(parsed_fit):
        raise SciPlex3RunnerError("p1 fitted-state artifact differs from in-memory binding")
    if parsed_fit.get("finalized_count_scan") != preparation.finalized_count_scan_manifest():
        raise SciPlex3RunnerError("fitted artifact differs from finalized loader scan receipt")
    baseline_section = _as_mapping(parsed_fit.get("baseline"), name="fitted baseline binding")
    baseline_id = baseline_section.get("baseline_id")
    if not isinstance(baseline_id, str) or baseline_id != exact_baseline_id:
        raise SciPlex3RunnerError("fitted artifact baseline identity differs from implementation")
    code_section = _as_mapping(parsed_fit.get("code"), name="fitted code binding")
    baseline_code = _as_mapping(code_section.get("baseline"), name="fitted baseline code binding")
    expected_entrypoint = (
        f"{type(fitted.baseline).__module__}:{type(fitted.baseline).__qualname__}.fit/sample"
    )
    if (
        baseline_code.get("entrypoint") != expected_entrypoint
        or baseline_code.get("sha256") != SCIPLEX3_BASELINE_CODE_SHA256
    ):
        raise SciPlex3RunnerError("fitted artifact does not bind the exact registered class code")
    current_state_payload = _canonical_json(fitted.baseline.fitted_state_manifest())
    current_state = _json_object(current_state_payload, name="current fitted state")
    if parsed_fit.get("fitted_state") != current_state or parsed_fit.get(
        "fitted_state_sha256"
    ) != _sha256(current_state_payload):
        raise SciPlex3RunnerError("in-memory fitted state differs from re-read fitted artifact")
    current_binding = _authenticate_executable_baseline_binding(
        preparation.repository_root, baseline_id
    )
    if parsed_fit.get("executable_binding") != current_binding:
        raise SciPlex3RunnerError("fitted artifact executable/golden binding is no longer current")
    return MappingProxyType(parsed_fit)


def _verify_prediction_execution_binding(
    preparation: SciPlex3BaselinePreparation,
    fitted: FittedSciPlex3Baseline,
    prediction_design: SciPlex3P4PredictionDesign,
) -> Mapping[str, object]:
    """Reauthenticate every behavior-bearing post-fit input at the point of use."""

    _validate_prediction_design_structure(prediction_design)
    if (
        prediction_design.preparation_fingerprint != preparation.receipt.fingerprint
        or prediction_design.fitted_state_artifact_sha256 != fitted.artifact.sha256
        or prediction_design.query_sha256 != preparation.design.query_sha256
        or prediction_design.benchmark_sha256 != preparation.design.benchmark_sha256
        or prediction_design.action_domain_sha256 != preparation.design.action_domain_sha256
        or prediction_design.scoring_transform_sha256 != preparation.design.scoring_transform_sha256
        or prediction_design.target_value_schema_sha256
        != preparation.design.target_value_schema_sha256
        or prediction_design.ordered_feature_keys_sha256
        != preparation.design.ordered_feature_keys_sha256
    ):
        raise SciPlex3RunnerError("p4 design is not bound to this exact post-fit state")
    fitted_manifest = _verify_fitted_artifact_binding(preparation, fitted)
    executable_binding = _as_mapping(
        fitted_manifest.get("executable_binding"), name="fitted executable binding"
    )
    suite_binding = _as_mapping(
        executable_binding.get("baseline_suite_specification"),
        name="fitted suite binding",
    )
    if prediction_design.baseline_suite_specification_sha256 != suite_binding.get("sha256"):
        raise SciPlex3RunnerError("p4 design differs from fitted executable suite")
    return fitted_manifest


def fit_and_write_sciplex3_baseline(
    preparation: SciPlex3BaselinePreparation,
    baseline_id: str,
    output_directory: Path,
    *,
    low_rank: int = DEFAULT_LOW_RANK,
) -> FittedSciPlex3Baseline:
    """Fit one registered p1-only baseline and emit its content-addressed state identity.

    The artifact stores exact input/code/runtime identities plus the implementation's canonical
    fitted-state manifest (including parameter or empirical-pool hashes).  It is intentionally not
    a lifecycle receipt and contains no held-out outcomes or metric result.
    """

    _validate_preparation(preparation)
    executable_binding = _authenticate_executable_baseline_binding(
        preparation.repository_root, baseline_id
    )
    implementation = SCIPLEX3_BASELINE_IMPLEMENTATIONS.get(baseline_id)
    if implementation is None:
        raise SciPlex3RunnerError(f"unknown frozen sci-Plex3 baseline: {baseline_id!r}")
    baseline: SciPlex3RawCountBaseline
    if implementation is LowRankCompoundDoseResponse:
        if low_rank != DEFAULT_LOW_RANK:
            raise SciPlex3RunnerError("frozen low-rank baseline requires rank eight")
        baseline = LowRankCompoundDoseResponse.fit(preparation.training_data, rank=DEFAULT_LOW_RANK)
    else:
        baseline = implementation.fit(preparation.training_data)
    if not isinstance(baseline, SciPlex3RawCountBaseline):
        raise SciPlex3RunnerError("registered baseline does not implement the frozen protocol")
    fitted_state = baseline.fitted_state_manifest()
    fitted_state_payload = _canonical_json(fitted_state)
    fitted_state_copy = _json_object(fitted_state_payload, name="baseline fitted-state manifest")
    if fitted_state_copy.get("baseline_id") != baseline_id:
        raise SciPlex3RunnerError("baseline fitted-state manifest has the wrong identifier")
    preparation_fingerprint = preparation.receipt.fingerprint
    manifest: dict[str, object] = {
        "artifact_schema": "sciplex3-k562-p1-baseline-fitted-state",
        "artifact_schema_version": "1.0.0",
        "baseline": {
            "baseline_id": baseline_id,
            "implementation_version": SCIPLEX3_BASELINE_IMPLEMENTATION_VERSION,
        },
        "code": {
            "baseline": _code_identity(
                preparation.repository_root,
                _BASELINE_CODE_PATH,
                entrypoint=(
                    f"{type(baseline).__module__}:{type(baseline).__qualname__}.fit/sample"
                ),
            ),
            "runner": _code_identity(
                preparation.repository_root,
                _RUNNER_CODE_PATH,
                entrypoint=("cellstate.evaluation.sciplex3_runner:fit_and_write_sciplex3_baseline"),
            ),
        },
        "fit_partition": {
            "access_purpose": "train_parameters",
            "control_well_count": SCIPLEX3_P1_CONTROL_WELL_COUNT,
            "partition_id": _P1_PARTITION_ID,
            "record_count": SCIPLEX3_P1_RECORD_COUNT,
            "treated_well_count": SCIPLEX3_P1_TREATED_WELL_COUNT,
            "well_count": SCIPLEX3_P1_WELL_COUNT,
        },
        "fitted_state": fitted_state_copy,
        "fitted_state_sha256": _sha256(fitted_state_payload),
        "executable_binding": executable_binding,
        "finalized_count_scan": preparation.finalized_count_scan_manifest(),
        "input_bindings": _json_ready_dataclass(preparation.receipt),
        "preparation_fingerprint": preparation_fingerprint,
        "runtime": _runtime_identity(),
        "safety_boundary": {
            "baseline_run_status_issued": False,
            "can_mint_lifecycle_evidence": False,
            "heldout_memberships_read": False,
            "heldout_outcomes_read": False,
            "metric_results_issued": False,
            "scientifically_admissible": False,
            "trusted_workflow_receipt_issued": False,
        },
    }
    payload = _canonical_json(manifest)
    output = _exclusive_directory(output_directory)
    path = output / "fitted-state-manifest.json"
    _write_exclusive(path, payload)
    artifact = _verify_local_artifact(
        path,
        expected_payload=payload,
        media_type="application/json",
    )
    return FittedSciPlex3Baseline(
        baseline=baseline,
        artifact=artifact,
        artifact_manifest=manifest,
        preparation_fingerprint=preparation_fingerprint,
    )


@dataclass(frozen=True, slots=True)
class PredictionShardEntry:
    relative_path: str
    sha256: str
    byte_count: int
    baseline_id: str
    case_id: str
    target_well_id: str
    partition_id: Literal["p4-untouched-test"]
    seed: int
    rng_algorithm: Literal["numpy-pcg64dxsm-v1"]
    draw_start: int
    draw_stop_exclusive: int
    shape: tuple[int, int]
    dtype: Literal["<i8"]
    ordered_feature_keys_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "relative_path",
            "baseline_id",
            "case_id",
            "target_well_id",
            "partition_id",
            "rng_algorithm",
            "dtype",
        ):
            _exact_text(getattr(self, name), name=name)
        _exact_sha256(self.sha256, name="prediction shard SHA-256")
        _exact_sha256(
            self.ordered_feature_keys_sha256,
            name="prediction shard ordered-feature SHA-256",
        )
        for name in ("byte_count", "seed", "draw_start", "draw_stop_exclusive"):
            if type(getattr(self, name)) is not int:
                raise SciPlex3RunnerError(f"{name} must be an exact integer")
        shape = tuple(self.shape)
        if len(shape) != 2 or any(type(value) is not int or value <= 0 for value in shape):
            raise SciPlex3RunnerError("prediction shard shape must be an exact positive int pair")
        object.__setattr__(self, "shape", shape)
        if (
            self.byte_count <= 0
            or self.seed not in SCIPLEX3_BASELINE_SEEDS
            or self.draw_start < 0
            or self.draw_stop_exclusive <= self.draw_start
            or self.partition_id != _P4_PARTITION_ID
            or self.rng_algorithm != RNG_ALGORITHM
            or self.dtype != "<i8"
        ):
            raise SciPlex3RunnerError("prediction shard scalar contract is invalid")


class SciPlex3PredictionArtifactWriter:
    """Incremental writer for the exact frozen p4 prediction schedule.

    The only public write path asks the bound fitted baseline to generate an exact scheduled
    request; callers cannot supply a sample object.  There is no outcome/truth parameter.  A
    final manifest is withheld until every one of 384 cases and five seeds has exactly 512 draws.
    Evaluation outcome access and benchmark scoring remain sealed elsewhere.
    """

    def __init__(
        self,
        preparation: SciPlex3BaselinePreparation,
        fitted: FittedSciPlex3Baseline,
        prediction_design: SciPlex3P4PredictionDesign,
        output_directory: Path,
    ) -> None:
        _validate_preparation(preparation)
        _verify_prediction_execution_binding(preparation, fitted, prediction_design)
        self._preparation = preparation
        self._fitted = fitted
        self._prediction_design = prediction_design
        self._output = _exclusive_directory(output_directory)
        self._targets = MappingProxyType(
            {target.case_id: target for target in prediction_design.p4_targets}
        )
        self._case_ordinals = MappingProxyType(
            {target.case_id: index for index, target in enumerate(prediction_design.p4_targets)}
        )
        self._written: set[tuple[str, int]] = set()
        self._entries: list[PredictionShardEntry] = []
        self._finalized = False
        self._generation_token = object()

    def _reauthenticate_execution_boundary(self) -> None:
        _verify_prediction_execution_binding(
            self._preparation,
            self._fitted,
            self._prediction_design,
        )

    @property
    def shard_entries(self) -> tuple[PredictionShardEntry, ...]:
        return tuple(self._entries)

    @property
    def completed_case_seed_count(self) -> int:
        return len(self._written)

    def _write_authenticated_prediction(
        self,
        prediction: PredictiveRawCountSamples,
        *,
        generation_token: object,
    ) -> tuple[PredictionShardEntry, ...]:
        """Split output returned inside this writer's bound baseline sampling call."""

        self._reauthenticate_execution_boundary()
        if self._finalized:
            raise SciPlex3RunnerError("prediction artifact is already finalized")
        if generation_token is not self._generation_token:
            raise SciPlex3RunnerError("prediction shards require authenticated baseline generation")
        baseline = self._fitted.baseline
        if prediction.baseline_id != baseline.baseline_id:
            raise SciPlex3RunnerError("prediction baseline differs from fitted-state artifact")
        try:
            expected_target = self._targets[prediction.target.case_id]
        except KeyError as error:
            raise SciPlex3RunnerError(
                "prediction case is outside the exact frozen p4 design"
            ) from error
        if prediction.target != expected_target:
            raise SciPlex3RunnerError("prediction target metadata differ from exact p4 case")
        if prediction.seed not in SCIPLEX3_BASELINE_SEEDS:
            raise SciPlex3RunnerError("prediction seed is outside the frozen 0..4 schedule")
        key = (prediction.target.case_id, prediction.seed)
        if key in self._written:
            raise SciPlex3RunnerError("duplicate prediction for one p4 case/seed")
        if (
            prediction.rng_algorithm != RNG_ALGORITHM
            or prediction.ordered_feature_keys
            != self._preparation.training_data.ordered_feature_keys
            or _sha256(_canonical_json(list(prediction.ordered_feature_keys)))
            != SCIPLEX3_ORDERED_FEATURE_KEYS_SHA256
            or prediction.samples.shape
            != (SCIPLEX3_BASELINE_SAMPLES_PER_CASE_PER_SEED, SCIPLEX3_FEATURE_COUNT)
            or prediction.samples.dtype != np.dtype(np.int64)
            or bool(np.any(prediction.samples < 0))
            or bool(np.any(np.sum(prediction.samples, axis=1, dtype=np.int64) <= 0))
        ):
            raise SciPlex3RunnerError(
                "prediction samples violate frozen shape/value/feature contract"
            )
        case_ordinal = self._case_ordinals[prediction.target.case_id]
        new_entries: list[PredictionShardEntry] = []
        for draw_start in range(
            0,
            SCIPLEX3_BASELINE_SAMPLES_PER_CASE_PER_SEED,
            SCIPLEX3_PREDICTION_SHARD_DRAW_COUNT,
        ):
            draw_stop = min(
                draw_start + SCIPLEX3_PREDICTION_SHARD_DRAW_COUNT,
                SCIPLEX3_BASELINE_SAMPLES_PER_CASE_PER_SEED,
            )
            shard = np.ascontiguousarray(
                prediction.samples[draw_start:draw_stop], dtype=np.dtype("<i8")
            )
            payload = shard.tobytes(order="C")
            relative = Path(
                "shards",
                f"case-{case_ordinal:04d}",
                f"seed-{prediction.seed}",
                f"draws-{draw_start:03d}-{draw_stop - 1:03d}.i64le",
            )
            path = self._output / relative
            _write_exclusive(path, payload)
            verified = _read_exact(path, _sha256(payload), name="prediction shard")
            if verified != payload:
                raise SciPlex3RunnerError("prediction shard differs on immediate re-read")
            entry = PredictionShardEntry(
                relative_path=relative.as_posix(),
                sha256=_sha256(payload),
                byte_count=len(payload),
                baseline_id=baseline.baseline_id,
                case_id=prediction.target.case_id,
                target_well_id=prediction.target.target_well_id,
                partition_id="p4-untouched-test",
                seed=prediction.seed,
                rng_algorithm=RNG_ALGORITHM,
                draw_start=draw_start,
                draw_stop_exclusive=draw_stop,
                shape=shard.shape,
                dtype="<i8",
                ordered_feature_keys_sha256=SCIPLEX3_ORDERED_FEATURE_KEYS_SHA256,
            )
            self._entries.append(entry)
            new_entries.append(entry)
        self._written.add(key)
        return tuple(new_entries)

    def sample_and_write(
        self,
        target: PredictionTarget,
        seed: int,
    ) -> tuple[PredictionShardEntry, ...]:
        """Generate the one fixed request; no observed outcome can enter this call."""

        self._reauthenticate_execution_boundary()
        _code_identity(
            self._preparation.repository_root,
            _BASELINE_CODE_PATH,
            entrypoint=(
                f"{type(self._fitted.baseline).__module__}:"
                f"{type(self._fitted.baseline).__qualname__}.sample"
            ),
        )
        _code_identity(
            self._preparation.repository_root,
            _RUNNER_CODE_PATH,
            entrypoint=(
                "cellstate.evaluation.sciplex3_runner:"
                "SciPlex3PredictionArtifactWriter.sample_and_write"
            ),
        )
        if type(target) is not PredictionTarget:
            raise SciPlex3RunnerError("target must be an exact immutable PredictionTarget")
        try:
            exact_target = self._targets[target.case_id]
        except KeyError as error:
            raise SciPlex3RunnerError("target is outside the exact frozen p4 design") from error
        if target != exact_target or seed not in SCIPLEX3_BASELINE_SEEDS:
            raise SciPlex3RunnerError("target or seed differs from the exact frozen schedule")
        prediction = self._fitted.baseline.sample(
            BaselineSampleRequest(
                target=exact_target,
                sample_count=SCIPLEX3_BASELINE_SAMPLES_PER_CASE_PER_SEED,
                seed=seed,
            )
        )
        return self._write_authenticated_prediction(
            prediction,
            generation_token=self._generation_token,
        )

    def _reauthenticate_shards(self, entries: Sequence[PredictionShardEntry]) -> None:
        for entry in entries:
            shard_path = self._output / entry.relative_path
            shard_payload = _read_exact(
                shard_path,
                entry.sha256,
                name="prediction shard at finalization",
            )
            expected_byte_count = entry.shape[0] * entry.shape[1] * np.dtype("<i8").itemsize
            if len(shard_payload) != entry.byte_count or len(shard_payload) != expected_byte_count:
                raise SciPlex3RunnerError("prediction shard byte count changed before finalization")

    def finalize(self) -> LocalContentAddressedArtifact:
        """Emit a canonical manifest only after the complete 384 x 5 schedule exists."""

        self._reauthenticate_execution_boundary()
        if self._finalized:
            raise SciPlex3RunnerError("prediction artifact is already finalized")
        expected = {
            (target.case_id, seed)
            for target in self._prediction_design.p4_targets
            for seed in SCIPLEX3_BASELINE_SEEDS
        }
        if self._written != expected:
            missing = len(expected - self._written)
            extra = len(self._written - expected)
            raise SciPlex3RunnerError(
                f"prediction manifest remains sealed: {missing} case/seeds missing, {extra} extra"
            )
        sorted_entries = sorted(
            self._entries,
            key=lambda item: (
                self._case_ordinals[item.case_id],
                item.seed,
                item.draw_start,
            ),
        )
        expected_shards = (
            SCIPLEX3_P4_CASE_COUNT
            * len(SCIPLEX3_BASELINE_SEEDS)
            * (SCIPLEX3_BASELINE_SAMPLES_PER_CASE_PER_SEED // SCIPLEX3_PREDICTION_SHARD_DRAW_COUNT)
        )
        if len(sorted_entries) != expected_shards:
            raise SciPlex3RunnerError("prediction shard count differs from frozen schedule")
        self._reauthenticate_shards(sorted_entries)
        self._reauthenticate_execution_boundary()
        manifest: dict[str, object] = {
            "artifact_schema": "sciplex3-k562-p4-baseline-predictive-samples",
            "artifact_schema_version": "1.0.0",
            "baseline": {
                "baseline_id": self._fitted.baseline.baseline_id,
                "implementation_version": SCIPLEX3_BASELINE_IMPLEMENTATION_VERSION,
            },
            "bindings": {
                "action_domain_sha256": self._preparation.design.action_domain_sha256,
                "baseline_suite_specification_sha256": (
                    self._prediction_design.baseline_suite_specification_sha256
                ),
                "benchmark_sha256": self._preparation.design.benchmark_sha256,
                "evaluation_cases_sha256": self._prediction_design.evaluation_cases_sha256,
                "fitted_state_artifact": {
                    "byte_count": self._fitted.artifact.byte_count,
                    "sha256": self._fitted.artifact.sha256,
                },
                "loader_contract_sha256": self._preparation.receipt.loader_contract_sha256,
                "ordered_feature_keys_sha256": SCIPLEX3_ORDERED_FEATURE_KEYS_SHA256,
                "p1_record_ids_sha256": SCIPLEX3_P1_RECORD_IDS_SHA256,
                "p1_record_to_well_sha256": SCIPLEX3_P1_RECORD_TO_WELL_SHA256,
                "prediction_targets_sha256": (self._prediction_design.prediction_targets_sha256),
                "query_sha256": self._preparation.design.query_sha256,
                "scoring_transform_sha256": self._preparation.design.scoring_transform_sha256,
                "source_sha256": self._preparation.receipt.source_sha256,
                "target_value_schema_sha256": (self._preparation.design.target_value_schema_sha256),
            },
            "code": {
                "baseline": _code_identity(
                    self._preparation.repository_root,
                    _BASELINE_CODE_PATH,
                    entrypoint=(
                        f"{type(self._fitted.baseline).__module__}:"
                        f"{type(self._fitted.baseline).__qualname__}.sample"
                    ),
                ),
                "runner": _code_identity(
                    self._preparation.repository_root,
                    _RUNNER_CODE_PATH,
                    entrypoint=(
                        "cellstate.evaluation.sciplex3_runner:SciPlex3PredictionArtifactWriter"
                    ),
                ),
            },
            "runtime": _runtime_identity(),
            "safety_boundary": {
                "baseline_run_status_issued": False,
                "can_mint_lifecycle_evidence": False,
                "heldout_outcomes_read": False,
                "metric_results_issued": False,
                "scientifically_admissible": False,
                "trusted_evaluation_receipt_issued": False,
            },
            "schedule": {
                "case_count": SCIPLEX3_P4_CASE_COUNT,
                "draws_per_case_per_seed": SCIPLEX3_BASELINE_SAMPLES_PER_CASE_PER_SEED,
                "partition_id": _P4_PARTITION_ID,
                "rng_algorithm": RNG_ALGORITHM,
                "seeds": list(SCIPLEX3_BASELINE_SEEDS),
                "shard_draw_count": SCIPLEX3_PREDICTION_SHARD_DRAW_COUNT,
            },
            "shards": [_json_ready_dataclass(entry) for entry in sorted_entries],
        }
        payload = _canonical_json(manifest)
        path = self._output / "prediction-manifest.json"
        _write_exclusive(path, payload)
        artifact = _verify_local_artifact(
            path,
            expected_payload=payload,
            media_type="application/json",
        )
        self._finalized = True
        return artifact


def run_sciplex3_baseline_predictions(
    preparation: SciPlex3BaselinePreparation,
    fitted: FittedSciPlex3Baseline,
    prediction_design: SciPlex3P4PredictionDesign,
    output_directory: Path,
) -> LocalContentAddressedArtifact:
    """Stream the exact complete outcome-free p4 prediction schedule to bounded shards.

    This function is intentionally not called by benchmark builders.  A future locked evaluator
    may consume the resulting samples only after independently authenticating the manifest and
    acquiring its own held-out-outcome grant.
    """

    writer = SciPlex3PredictionArtifactWriter(
        preparation, fitted, prediction_design, output_directory
    )
    for target in prediction_design.p4_targets:
        for seed in SCIPLEX3_BASELINE_SEEDS:
            writer.sample_and_write(target, seed)
    return writer.finalize()


__all__ = [
    "SCIPLEX3_ACTION_DOMAIN_SHA256",
    "SCIPLEX3_ACTION_ENTRY_COUNT",
    "SCIPLEX3_BASELINE_CODE_SHA256",
    "SCIPLEX3_BASELINE_GOLDEN_FIXTURE_SHA256",
    "SCIPLEX3_EVALUATION_CASES_SHA256",
    "SCIPLEX3_EXECUTABLE_NUMPY_VERSION",
    "SCIPLEX3_EXECUTABLE_PYTHON",
    "SCIPLEX3_LOADER_CODE_SHA256",
    "SCIPLEX3_P1_CONTROL_WELL_COUNT",
    "SCIPLEX3_P1_RECORD_COUNT",
    "SCIPLEX3_P1_RECORD_IDS_SHA256",
    "SCIPLEX3_P1_RECORD_TO_WELL_SHA256",
    "SCIPLEX3_P1_TREATED_WELL_COUNT",
    "SCIPLEX3_P1_WELL_COUNT",
    "SCIPLEX3_P1_WELL_IDS_SHA256",
    "SCIPLEX3_P1_WELL_TO_CONDITION_SHA256",
    "SCIPLEX3_P4_CASE_COUNT",
    "SCIPLEX3_P4_CONTROL_CASE_COUNT",
    "SCIPLEX3_P4_PREDICTION_TARGETS_SHA256",
    "SCIPLEX3_P4_TREATED_CASE_COUNT",
    "SCIPLEX3_PREDICTION_SHARD_DRAW_COUNT",
    "SCIPLEX3_RUNNER_IMPLEMENTATION_VERSION",
    "FittedSciPlex3Baseline",
    "LocalContentAddressedArtifact",
    "PredictionShardEntry",
    "SciPlex3ActionBinding",
    "SciPlex3BaselinePreparation",
    "SciPlex3P1AssemblyReceipt",
    "SciPlex3P1DesignBindings",
    "SciPlex3P4PredictionDesign",
    "SciPlex3PredictionArtifactWriter",
    "SciPlex3RunnerError",
    "assemble_sciplex3_p1_training_data",
    "fit_and_write_sciplex3_baseline",
    "open_sciplex3_p4_prediction_design",
    "run_sciplex3_baseline_predictions",
]
