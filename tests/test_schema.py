from __future__ import annotations

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
