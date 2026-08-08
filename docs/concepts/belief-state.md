# Belief state

Cells are partially observed, stochastic, history-dependent dynamical systems. `cellstate` therefore
returns a distribution over dynamic hidden variables and relatively stable cell parameters,
conditioned on observations, interventions, environment, lineage, and static context.

A posterior mean is a summary, not the state itself. Backends must preserve a joint distribution as
parameters, samples, particles, or another explicit distribution representation. Unsupported or
unidentifiable quantities remain explicit; they are never filled with zeros.

State is structured into eight causal blocks: stable identity, slow memory, regulation, signaling,
metabolism, physical structure, damage/stress, and functional capacity. Factor marginals do not imply
that those blocks are independent; the joint posterior remains authoritative.
