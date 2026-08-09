"""Regenerate checked-in JSON Schemas for public boundary models."""

from __future__ import annotations

import json
from pathlib import Path

from cellstate.data import DatasetManifest
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

OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "schemas"
MODELS = {
    Path("experimental/dataset-manifest.schema.json"): DatasetManifest,
    Path("v1/state-query.schema.json"): StateQuery,
    Path("v1/cell-history.schema.json"): CellHistory,
    Path("v1/estimate-cell-state-request.schema.json"): EstimateCellStateRequest,
    Path("v1/cell-state-belief.schema.json"): CellStateBelief,
    Path("v1/evolution-scenario.schema.json"): EvolutionScenario,
    Path("v1/state-forecast.schema.json"): StateForecast,
    Path("v1/intervention-objective.schema.json"): InterventionObjective,
    Path("v1/intervention-plan.schema.json"): InterventionPlan,
}


def main() -> None:
    for relative_path, model in MODELS.items():
        destination = OUTPUT_ROOT / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(model.model_json_schema(), indent=2, sort_keys=True) + "\n"
        destination.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
