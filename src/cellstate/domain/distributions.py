"""Serializable posterior-distribution contracts shared by v2 outputs."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated, Literal

import numpy as np
from pydantic import Field, JsonValue, model_validator

from .common import ArtifactRef, SchemaModel, require_finite


class DistributionSupport(StrEnum):
    REAL = "real"
    NONNEGATIVE = "nonnegative"
    UNIT_INTERVAL = "unit_interval"
    SIMPLEX = "simplex"
    DISCRETE = "discrete"
    MIXED = "mixed"


class ParametricDistribution(SchemaModel):
    kind: Literal["parametric"] = "parametric"
    family: str = Field(min_length=1)
    dimensions: tuple[str, ...] = Field(min_length=1)
    support: DistributionSupport = DistributionSupport.REAL
    mean: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    parameters: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def compatible_shapes(self) -> ParametricDistribution:
        size = len(self.dimensions)
        if len(set(self.dimensions)) != size:
            raise ValueError("distribution dimensions must be unique")
        if len(self.mean) != size or len(self.covariance) != size:
            raise ValueError("distribution mean/covariance shape must match dimensions")
        for row in self.covariance:
            if len(row) != size:
                raise ValueError("distribution covariance must be square")
        for value in (*self.mean, *(entry for row in self.covariance for entry in row)):
            require_finite(value, name="distribution parameter")
        for row_index in range(size):
            if self.covariance[row_index][row_index] < 0:
                raise ValueError("covariance diagonal must be nonnegative")
            for column_index in range(size):
                if not math.isclose(
                    self.covariance[row_index][column_index],
                    self.covariance[column_index][row_index],
                    rel_tol=1e-8,
                    abs_tol=1e-10,
                ):
                    raise ValueError("covariance must be symmetric")
        if np.linalg.eigvalsh(np.asarray(self.covariance, dtype=float)).min() < -1e-10:
            raise ValueError("covariance must be positive semidefinite")
        return self


class SampleDistribution(SchemaModel):
    kind: Literal["samples"] = "samples"
    dimensions: tuple[str, ...] = Field(min_length=1)
    support: DistributionSupport = DistributionSupport.REAL
    samples: ArtifactRef
    sample_count: int = Field(gt=0)
    weights: ArtifactRef | None = None

    @model_validator(mode="after")
    def artifacts_match_declared_samples(self) -> SampleDistribution:
        if len(self.dimensions) != len(set(self.dimensions)):
            raise ValueError("sample-distribution dimensions must be unique")
        expected_shape = (self.sample_count, len(self.dimensions))
        if self.samples.shape != expected_shape:
            raise ValueError(
                "posterior sample artifact shape must be (sample_count, state_dimension_count)"
            )
        if self.samples.dimensions != ("sample", "state_dimension"):
            raise ValueError("posterior sample artifact axes must be ('sample', 'state_dimension')")
        if self.weights is not None and (
            self.weights.shape != (self.sample_count,) or self.weights.dimensions != ("sample",)
        ):
            raise ValueError("posterior weight artifact must align one weight per sample")
        return self


class UnavailableDistribution(SchemaModel):
    kind: Literal["unavailable"] = "unavailable"
    dimensions: tuple[str, ...] = ()
    reason_code: str = Field(min_length=1)
    message: str = Field(min_length=1)


StateDistribution = Annotated[
    ParametricDistribution | SampleDistribution | UnavailableDistribution,
    Field(discriminator="kind"),
]


__all__ = [
    "DistributionSupport",
    "ParametricDistribution",
    "SampleDistribution",
    "StateDistribution",
    "UnavailableDistribution",
]
