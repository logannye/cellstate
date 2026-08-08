# Scientific validation contract

Software correctness is necessary and insufficient. Validation must be prospective where possible
and aligned to the exact `StateQuery`.

Required evidence for a biological backend includes:

- multi-horizon future functional and molecular prediction;
- interventions, environments, donors/genotypes, and combinations outside the training support;
- randomized perturbations, matched controls, distributional comparisons, or lineage/sister designs
  for intervention effects;
- division timing, offspring distributions, inheritance, and sibling divergence when lineage is in
  scope;
- missing-modality robustness without missing-as-zero behavior;
- credible-interval calibration (for example, nominal 90% intervals near 90% prospective coverage);
- explicit OOD abstention or uncertainty growth;
- mechanistic residuals for constraints the backend claims to represent;
- the state-only vs. state-plus-history future-prediction comparison.

Random cell-level train/test splits, cluster coherence, current-state reconstruction, and
interpolation among familiar perturbations are not sufficient validation.
