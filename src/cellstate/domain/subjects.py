"""Typed biological subjects and target-aggregation semantics for schema v2."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, field_validator, model_validator

from .common import OntologyTerm, SchemaModel, require_finite


class SubjectKind(StrEnum):
    """The biological estimand represented by a belief."""

    INDIVIDUAL_CELL = "individual_cell"
    CLONE_LINEAGE = "clone_lineage"
    POPULATION = "population"
    SPATIAL_NICHE = "spatial_niche"


class IdentityBasis(StrEnum):
    """Observed or explicitly inferred basis for subject identity or linkage."""

    OBSERVED_IDENTITY = "observed_identity"
    DIRECT_TRACKING = "direct_tracking"
    VIABILITY_PRESERVING_SAMPLING = "viability_preserving_sampling"
    OBSERVED_PARENTAGE = "observed_parentage"
    HERITABLE_BARCODE = "heritable_barcode"
    PHYLOGENY = "phylogeny"
    PROBABILISTIC_LINEAGE = "probabilistic_lineage"
    DECLARED_MEMBERSHIP = "declared_membership"
    EXPERIMENTAL_UNIT = "experimental_unit"
    PROBABILISTIC_MEMBERSHIP = "probabilistic_membership"
    SPATIAL_REGION = "spatial_region"
    OBSERVED_NEIGHBORHOOD_GRAPH = "observed_neighborhood_graph"
    MATCHED_EXPERIMENTAL_DESIGN = "matched_experimental_design"
    SPATIAL_PROXIMITY = "spatial_proximity"
    TRANSPORT_ASSUMPTION = "transport_assumption"
    EXTERNAL_REFERENCE = "external_reference"


class AggregationStatistic(StrEnum):
    """How a target is aggregated over its typed biological subject."""

    INDIVIDUAL = "individual"
    JOINT_DISTRIBUTION = "joint_distribution"
    DISTRIBUTION = "distribution"
    MEAN = "mean"
    SUM = "sum"
    FRACTION = "fraction"
    RATE = "rate"
    HAZARD = "hazard"


class TargetAggregation(SchemaModel):
    """The estimand level and summary represented by a query output."""

    subject_kind: SubjectKind
    statistic: AggregationStatistic
    experimental_unit: str = Field(min_length=1)

    @model_validator(mode="after")
    def statistic_matches_subject(self) -> TargetAggregation:
        if self.subject_kind is SubjectKind.INDIVIDUAL_CELL:
            if self.statistic is not AggregationStatistic.INDIVIDUAL:
                raise ValueError("an individual-cell target must use individual aggregation")
        elif self.statistic is AggregationStatistic.INDIVIDUAL:
            raise ValueError("individual aggregation requires an individual-cell subject")
        return self


_BASIS_BY_KIND: dict[SubjectKind, frozenset[IdentityBasis]] = {
    SubjectKind.INDIVIDUAL_CELL: frozenset(
        {
            IdentityBasis.OBSERVED_IDENTITY,
            IdentityBasis.DIRECT_TRACKING,
            IdentityBasis.VIABILITY_PRESERVING_SAMPLING,
        }
    ),
    SubjectKind.CLONE_LINEAGE: frozenset(
        {
            IdentityBasis.OBSERVED_PARENTAGE,
            IdentityBasis.HERITABLE_BARCODE,
            IdentityBasis.PHYLOGENY,
            IdentityBasis.PROBABILISTIC_LINEAGE,
        }
    ),
    SubjectKind.POPULATION: frozenset(
        {
            IdentityBasis.DECLARED_MEMBERSHIP,
            IdentityBasis.EXPERIMENTAL_UNIT,
            IdentityBasis.PROBABILISTIC_MEMBERSHIP,
        }
    ),
    SubjectKind.SPATIAL_NICHE: frozenset(
        {
            IdentityBasis.SPATIAL_REGION,
            IdentityBasis.OBSERVED_NEIGHBORHOOD_GRAPH,
            IdentityBasis.PROBABILISTIC_MEMBERSHIP,
        }
    ),
}

_INFERRED_BASES = frozenset(
    {
        IdentityBasis.PHYLOGENY,
        IdentityBasis.PROBABILISTIC_LINEAGE,
        IdentityBasis.PROBABILISTIC_MEMBERSHIP,
    }
)


class SubjectSpecification(SchemaModel):
    """Query-level requirements for the concrete subject being estimated."""

    kind: SubjectKind
    biological_system: OntologyTerm
    membership_semantics: str = Field(min_length=1)
    experimental_unit_kind: str = Field(min_length=1)
    allowed_identity_bases: tuple[IdentityBasis, ...] = Field(min_length=1)

    @field_validator("allowed_identity_bases")
    @classmethod
    def identity_bases_are_unique(
        cls, bases: tuple[IdentityBasis, ...]
    ) -> tuple[IdentityBasis, ...]:
        if len(bases) != len(set(bases)):
            raise ValueError("allowed identity bases must be unique")
        return bases

    @model_validator(mode="after")
    def identity_bases_match_kind(self) -> SubjectSpecification:
        invalid = set(self.allowed_identity_bases) - _BASIS_BY_KIND[self.kind]
        if invalid:
            raise ValueError(
                f"identity bases are incompatible with {self.kind.value}: "
                f"{sorted(item.value for item in invalid)}"
            )
        return self

    def supports(self, subject: BeliefSubject) -> bool:
        """Return whether a concrete subject satisfies this query-level specification."""

        return (
            subject.kind is self.kind
            and subject.biological_system.key == self.biological_system.key
            and subject.membership_semantics == self.membership_semantics
            and subject.experimental_unit_kind == self.experimental_unit_kind
            and subject.identity_basis in self.allowed_identity_bases
        )


class BeliefSubject(SchemaModel):
    """A concrete, identity-qualified biological subject of a belief or history."""

    subject_id: str = Field(min_length=1)
    kind: SubjectKind
    biological_system: OntologyTerm
    membership_semantics: str = Field(min_length=1)
    experimental_unit_kind: str = Field(min_length=1)
    experimental_unit_id: str = Field(min_length=1)
    identity_basis: IdentityBasis
    identity_confidence: float | None = Field(default=None, gt=0, le=1)
    member_ids: tuple[str, ...] = ()

    @field_validator("identity_confidence")
    @classmethod
    def finite_confidence(cls, value: float | None) -> float | None:
        if value is not None:
            return require_finite(value, name="subject identity confidence")
        return value

    @field_validator("member_ids")
    @classmethod
    def members_are_unique(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value for value in values):
            raise ValueError("subject member IDs must be nonempty")
        if len(values) != len(set(values)):
            raise ValueError("subject member IDs must be unique")
        return values

    @model_validator(mode="after")
    def identity_is_scientifically_coherent(self) -> BeliefSubject:
        if self.identity_basis not in _BASIS_BY_KIND[self.kind]:
            raise ValueError(
                f"identity basis {self.identity_basis.value!r} is incompatible with "
                f"{self.kind.value!r}"
            )
        if self.identity_basis in _INFERRED_BASES and self.identity_confidence is None:
            raise ValueError("inferred identity or membership requires explicit confidence")
        if self.kind is SubjectKind.INDIVIDUAL_CELL and self.member_ids:
            raise ValueError("an individual-cell subject cannot contain member IDs")
        if self.subject_id in self.member_ids:
            raise ValueError("an aggregate subject cannot list itself as a member")
        return self

    def is_compatible_with(self, specification: SubjectSpecification) -> bool:
        return specification.supports(self)
