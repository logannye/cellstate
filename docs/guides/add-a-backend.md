# Add a model backend

Implement `CellStateEstimator` first. Advertise capabilities before expensive inference, reject
unsupported observed modalities, and never silently ignore recorded interventions or environments.
Return each of the eight factor blocks, using an unavailable distribution where necessary.
Never fingerprint an input as conditioned evidence while ignoring it: either model it, marginalize
over it with propagated uncertainty, or reject/abstain explicitly.

If the backend supports controlled propagation, also implement `StateEvolutionModel`. Preserve the
joint posterior through evolution, accept explicit seeds, model discrete events where claimed, and
propagate uncertainty. Planning backends implement `InterventionPlanner` over explicit candidates and
objectives; they must report unsupported candidate/objective combinations. A planner must score the
typed query-target predictions at the objective's named horizon rather than treating latent state
dimensions as functional outcomes.

Every returned artifact must preserve exact input identity and provenance. Factor marginals must
agree with the authoritative joint distribution where that comparison is defined, and sample-backed
posteriors must document their sample and event axes.

Before calling a biological backend usable, publish its model card, training-support manifest,
prospective calibration evidence, held-out intervention/environment performance, OOD behavior, and
state-vs-history sufficiency results.
