# ADR 0002: Inputs are time-stamped events

Status: accepted.

Observations, interventions, environment, lineage, and contacts are represented as an ordered event
history with explicit completeness. This prevents bags of measurements from erasing causal order,
exposure duration, or missingness semantics.
