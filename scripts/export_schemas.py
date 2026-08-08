"""Regenerate checked-in JSON Schemas for public boundary models."""

from __future__ import annotations

import json
from pathlib import Path

from cellstate.domain import (
    CellHistory,
    CellStateBelief,
    EstimateCellStateRequest,
    EvolutionScenario,
    InterventionObjective,
    InterventionPlan,
    StateForecast,
    StateQuery,
)

OUTPUT_DIRECTORY = Path(__file__).resolve().parents[1] / "schemas" / "v1"
MODELS = {
    "state-query.schema.json": StateQuery,
    "cell-history.schema.json": CellHistory,
    "estimate-cell-state-request.schema.json": EstimateCellStateRequest,
    "cell-state-belief.schema.json": CellStateBelief,
    "evolution-scenario.schema.json": EvolutionScenario,
    "state-forecast.schema.json": StateForecast,
    "intervention-objective.schema.json": InterventionObjective,
    "intervention-plan.schema.json": InterventionPlan,
}


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for filename, model in MODELS.items():
        destination = OUTPUT_DIRECTORY / filename
        content = json.dumps(model.model_json_schema(), indent=2, sort_keys=True) + "\n"
        destination.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
