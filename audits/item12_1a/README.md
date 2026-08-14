# Item 12.1a frozen audit lineage

**Historical.** The rank-16 continuous admixture candidate family characterized here is retired for
the state path by [ADR 0013](../../docs/adr/0013-state-first-roadmap-reordering.md): it is a
condition-level mean model whose latent is indexed at the well, with a free parameter per observed
action that cannot generalize to an unseen action in principle. Nothing in this directory is live
work. What was salvaged from it — the corrected objective mathematics, the equal-unit normalization
constant, and the effective-context diagnostic — is carried into the Phase 4 model guidance in
[`docs/roadmap.md`](../../docs/roadmap.md); the lifecycle scaffolding around it is not.

This directory makes the completed, source-free Item 12.1a software audit reproducible from the
canonical checkout. The Python and JSON files are historical, content-addressed evidence. They are
not a supported runtime API and do not authorize opening any sci-Plex3 source or held-out
partition.

## Exact files

| File | SHA-256 | Role |
| --- | --- | --- |
| `item12_1_local_map_plate_context_diagnostic.py` | `f4e6b76847bd926952995d66233389768f091135699fb60a38d7d9762bb03ff1` | Frozen harness for the historical Item 12.1b plan, produced by Item 12.1a |
| `test_item12_1_local_map_plate_context_diagnostic.py` | `8989618e259fb4aed0e0798bc010e40092c45e6bd30234bb3a7b534cdc562903` | Twenty-six source-free characterization and containment tests |
| `item12_v4_nonissuing_trajectory.py` | `795c59296f5cefb1b6dd78a021ea0eb8e795217eda5226becf6c5bf909f6623a` | Exact parent driver required by the frozen test and harness lineage |
| `item12_v4_nonissuing_report.json` | `4677fc8ef1a458bf3616abc507250572c2da7a8d53c1c8a7a03d4b097f3d4877` | Infrastructure-invalid first v4 report |
| `item12_v4_nonissuing_report_attempt2.json` | `66e9debc1a402e7aa68cbc934f7c5f641529eea3187ec15606364c912af8faa8` | Valid nonissuing v4 failure report |

Verify the source-free software from the repository root with:

```console
uv run --no-editable pytest -q --no-cov audits/item12_1a/test_item12_1_local_map_plate_context_diagnostic.py
```

The exact test also verifies the hashes of its sibling harness and parent driver. Normal `pytest`
collection includes this directory. The frozen Python bytes are excluded from repository-wide Ruff
rewrites so `make format` cannot silently change their recorded identities.

## Execution boundary

Invoking the harness with only its filename performs an argument-failure check and does not access a
dataset. A real-source invocation has the command shape
`item12_1_local_map_plate_context_diagnostic.py REPOSITORY_ROOT SOURCE_H5AD`, but must not be run.
Item 12.1b was retired before execution after a separate source-code audit confirmed that v4's
dose-block objective, gradient, and Hessian omit the equal-well factor `N/W = 94785/768` used by
the corresponding action likelihood in the tracked ELBO. Item 12.1a opened no source, and this
canonical lineage grants no real-source authority.

The harness arms a 3,600-second `ITIMER_REAL` deadline. Its 4-GiB RSS check is post-hoc rather than
hard memory containment. Any future real-source execution therefore also needs an externally
enforced worker/container memory limit and a recorded runtime-image digest. The historical reports
must not be interpreted as model, calibration, validation, performance, or lifecycle evidence.

**Historical.** The source-free v5 objective and M-step redesign this file named as its next
milestone was completed, and the whole rank-16 continuous admixture candidate family was then
retired for the state path by
[ADR 0013](../../docs/adr/0013-state-first-roadmap-reordering.md). The real-`p1` authorization it was
a prerequisite for is suspended: no proposal may be approved and no protected execution dispatched.
The retained v4 harness is reproducibility evidence and names no pending work.
