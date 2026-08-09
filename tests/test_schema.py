from __future__ import annotations

import json
from pathlib import Path

from cellstate.data import DatasetManifest
from cellstate.domain import (
    CellHistory,
    CellStateBelief,
    EstimateCellStateRequest,
    InterventionObjective,
    InterventionPlan,
    StateForecast,
    StateQuery,
)


def test_public_models_generate_json_schema() -> None:
    for model in (
        DatasetManifest,
        StateQuery,
        CellHistory,
        EstimateCellStateRequest,
        CellStateBelief,
        StateForecast,
        InterventionObjective,
        InterventionPlan,
    ):
        schema = model.model_json_schema()
        assert schema["type"] == "object"
        assert "$defs" in schema


def test_checked_in_experimental_dataset_schema_is_current_and_warns_consumers() -> None:
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "schemas"
        / "experimental"
        / "dataset-manifest.schema.json"
    )
    checked_in = json.loads(schema_path.read_text(encoding="utf-8"))

    assert checked_in == DatasetManifest.model_json_schema()
    assert "Python model" in checked_in["$comment"]
