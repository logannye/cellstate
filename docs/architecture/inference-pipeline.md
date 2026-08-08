# Inference pipeline

A full backend should implement these conceptual stages, jointly where probability semantics demand
it:

1. Compile the query into a task-specific state specification.
2. Canonicalize the causal event history, units, identifiers, provenance, and missingness.
3. Convert each observed modality through its own likelihood/noise model.
4. Propagate a previous belief or construct a context-, population-, and lineage-conditioned prior.
5. Fuse evidence probabilistically while preserving shared and modality-private information.
6. Decode structured factors across fast, intermediate, and slow timescales.
7. Incorporate inherited and sibling evidence.
8. Couple the cell to environment, transport, contacts, and neighborhood state.
9. Apply mechanistic knowledge as weighted, auditable, overridable constraints.
10. Characterize continuous dynamics and discrete event hazards/jumps.
11. Decompose uncertainty, observability, identifiability, and OOD support.
12. Evaluate predictive sufficiency when held-out future evidence actually permits it.
13. Rank candidate measurements by decision-relevant uncertainty reduction per cost.
14. Return a self-auditing belief with model/data/code provenance.

The stage boundaries are extension points, not claims that the biological updates are independent.
