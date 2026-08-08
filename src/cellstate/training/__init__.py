"""Training-objective contracts; concrete trainers belong to model backends."""

from .objectives import (
    LossKind,
    TrainingObjective,
    WeightedLoss,
    default_training_objective,
)

__all__ = ["LossKind", "TrainingObjective", "WeightedLoss", "default_training_objective"]
