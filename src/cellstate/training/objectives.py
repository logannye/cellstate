"""Composable names and weights for intervention-focused model training."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, field_validator

from cellstate.domain.common import SchemaModel, require_finite


class LossKind(StrEnum):
    MULTI_HORIZON_FUTURE = "multi_horizon_future"
    FUNCTIONAL_OUTCOME = "functional_outcome"
    INTERVENTION_EFFECT = "intervention_effect"
    LINEAGE_TRANSITION = "lineage_transition"
    HELD_OUT_MODALITY = "held_out_modality"
    MECHANISTIC_CONSISTENCY = "mechanistic_consistency"
    UNCERTAINTY_CALIBRATION = "uncertainty_calibration"
    STATE_COMPLEXITY = "state_complexity"


class WeightedLoss(SchemaModel):
    kind: LossKind
    weight: float = Field(gt=0)


class TrainingObjective(SchemaModel):
    losses: tuple[WeightedLoss, ...] = Field(min_length=1)

    @field_validator("losses")
    @classmethod
    def unique_losses(cls, losses: tuple[WeightedLoss, ...]) -> tuple[WeightedLoss, ...]:
        kinds = [loss.kind for loss in losses]
        if len(kinds) != len(set(kinds)):
            raise ValueError("each loss kind may appear only once")
        return losses

    def combine(self, values: dict[LossKind, float]) -> float:
        missing = {loss.kind for loss in self.losses} - set(values)
        if missing:
            raise ValueError(f"missing training loss values: {sorted(missing)}")
        total = 0.0
        for loss in self.losses:
            total += loss.weight * require_finite(values[loss.kind], name=f"{loss.kind} loss")
        return total


def default_training_objective() -> TrainingObjective:
    """A documented starting point, not a universally optimal biological objective."""

    return TrainingObjective(
        losses=(
            WeightedLoss(kind=LossKind.MULTI_HORIZON_FUTURE, weight=1.0),
            WeightedLoss(kind=LossKind.FUNCTIONAL_OUTCOME, weight=2.0),
            WeightedLoss(kind=LossKind.INTERVENTION_EFFECT, weight=2.0),
            WeightedLoss(kind=LossKind.LINEAGE_TRANSITION, weight=1.0),
            WeightedLoss(kind=LossKind.HELD_OUT_MODALITY, weight=0.25),
            WeightedLoss(kind=LossKind.MECHANISTIC_CONSISTENCY, weight=0.25),
            WeightedLoss(kind=LossKind.UNCERTAINTY_CALIBRATION, weight=1.0),
            WeightedLoss(kind=LossKind.STATE_COMPLEXITY, weight=0.1),
        )
    )
