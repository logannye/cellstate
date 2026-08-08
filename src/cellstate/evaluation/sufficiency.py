"""Operational tests for approximate predictive/Markov sufficiency."""

from __future__ import annotations

from cellstate.domain.belief import SufficiencyReport
from cellstate.domain.common import SupportStatus, require_finite


def evaluate_history_information_gain(
    *,
    state_only_loss: float,
    state_plus_history_loss: float,
    tolerance: float,
    metric: str,
) -> SufficiencyReport:
    """Compare future prediction from state alone against state plus raw history.

    Lower loss is assumed to be better. This computes the diagnostic; experimental design and
    held-out predictions determine whether the comparison is scientifically meaningful.
    """

    require_finite(state_only_loss, name="state-only loss")
    require_finite(state_plus_history_loss, name="state-plus-history loss")
    require_finite(tolerance, name="sufficiency tolerance")
    if tolerance < 0:
        raise ValueError("sufficiency tolerance must be nonnegative")
    gain = state_only_loss - state_plus_history_loss
    note = (
        "Approximate sufficiency supported at the configured tolerance."
        if gain <= tolerance
        else "Raw history materially improves prediction; the state is incomplete."
    )
    return SufficiencyReport(
        status=SupportStatus.SUPPORTED,
        state_only_loss=state_only_loss,
        state_plus_history_loss=state_plus_history_loss,
        history_information_gain=gain,
        metric=metric,
        notes=(note,),
    )
