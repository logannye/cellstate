"""A fitted RNA observation model for GSE274113, and the first belief from real cells.

Authorized by ADR 0021 (the artifact kind), ADR 0019 (build on held evidence) and ADR 0020 (RNA
first).  What this backend does *not* reach is as load-bearing as what it does: the series has no
library spanning a timepoint, so S1, S3 and S7 are structurally unavailable and nothing produced
here is a sufficiency result or a faithfulness verdict.

Start at :func:`estimate_arm` and :func:`describe_state` in :mod:`.usage`; between them they take
the committed slice to a belief and print that belief's biology block in gene terms.
"""

from .usage import (
    ArmContrast,
    AxisReadout,
    GeneLoading,
    StateDescription,
    artifact_directory,
    available_arms,
    compare_arms,
    describe_state,
    estimate_arm,
    load_arm_slice,
)

__all__ = [
    "ArmContrast",
    "AxisReadout",
    "GeneLoading",
    "StateDescription",
    "artifact_directory",
    "available_arms",
    "compare_arms",
    "describe_state",
    "estimate_arm",
    "load_arm_slice",
]
