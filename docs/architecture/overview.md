# Architecture overview

The framework uses a functional core and replaceable model backends.

- `domain/` owns strict, frozen-top-level, schema-versioned boundary objects. Nested mappings are
  treated as read-only and revalidated at process/storage boundaries.
- `engine/` owns deterministic history/registry mechanics and no biology.
- `ports/` defines estimation, transition, planning, observation, fusion, constraint, diagnostic,
  inheritance, and measurement-policy interfaces.
- `reference/` contains an opt-in linear-Gaussian vertical slice for contract testing.
- `evaluation/` contains backend-independent calibration and sufficiency diagnostics.
- `training/` names the composable intervention-focused training objective.

The three top-level operations are deliberately distinct:

```python
belief = estimate_cell_state(request, estimator=model)
forecast = evolve_cell_state(belief, scenario=scenario, evolution_model=model)
plan = choose_intervention(belief, objective=objective, candidates=candidates, planner=planner)
```

Every stochastic operation receives an explicit seed. Large modality arrays and posterior samples
are referenced through content-addressed artifacts rather than embedded in manifests.

`StateForecast` contains both the propagated joint latent posterior and typed predictive
distributions for every query target at one named query horizon. A backend must return an explicit
unavailable prediction when it lacks an output decoder. Planning objectives name that horizon and
query targets, and planners evaluate those output distributions—not backend-private latent
coordinates.

Recursive estimation, evolution, and planning require compatible model ID/version, complete model
configuration fingerprint, posterior-schema identifier, and training-support identity. Matching
dimension labels alone do not establish compatible latent semantics.

Public API boundaries revalidate backend results and bind them to their request, parent belief,
query, scenario, objective, candidates, seed, and backend descriptor. Third-party adapters cannot
silently return a valid-looking artifact for a different computation.

The reference backend does not label generic covariance reduction as assay value of information.
Decision-relevant measurement selection must integrate uncertainty over the query's future targets,
interventions, environments, and horizons.

The reference backend requires complete past intervention, environment, lineage, and
neighborhood/contact records. It rejects input features its synthetic likelihood/dynamics do not
consume, including population or lineage context, assay confounder metadata, regional environment,
discrete division/contact events, and target-specific controls. Ongoing interventions and the current
environment are carried in the belief context; every scenario explicitly chooses whether they
persist. A production backend may support additional features only by advertising and implementing
their semantics. The reference prior has a configured time origin; record presence never determines
the prior epoch.
