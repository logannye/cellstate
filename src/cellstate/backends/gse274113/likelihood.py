"""The observation model's closed-form posterior over a population's latent state.

An arm's evidence is a panel count vector ``y`` from ``n`` total panel counts.  Working on the
log-composition rather than on counts makes the posterior conjugate and therefore exact, which
matters more here than likelihood fidelity: a closed form is deterministic, has no sampler to
converge, and cannot silently return a mode dressed as a posterior.

    c_j  = log((y_j + 1/2) / (n + G/2))              observed log-composition
    c    = alpha + A u + eps,      eps ~ N(0, Omega)
    u    ~ N(0, Lambda^-1)
    Sigma = (A' Omega^-1 A + Lambda)^-1
    u_hat = Sigma A' Omega^-1 (c - alpha)

``Omega`` is diagonal and carries **two** separable variance sources, which is what lets the
belief's uncertainty breakdown be computed rather than declared:

    omega_j = lambda_j/(lambda_j + 1/2)^2 - 1/(n + G/2)    technical, at lambda_j = n p_j
            + psi^2                                        biological, fitted across libraries

The technical term shrinks as ``1/n``.  With ``n`` in the millions it becomes negligible, and a
model carrying only that term would report a posterior of absurd confidence -- the failure mode
where more sequencing depth is mistaken for more knowledge about the biology.  ``psi^2`` is what
stops that, and per ADR 0022 it is now genuinely fitted rather than pinned at its clamp: the rate
``p_j`` is pooled over the fold's fit libraries instead of read off the arm's own count, which is
what made the fitted dispersion negative in all fourteen folds.  How many folds still reach the
clamp is measured and reported in the model card, not assumed to be none.

The delta-method Gaussian is a genuine approximation and it is poor at very low counts.  That cost
is accepted deliberately, and the panel's realized counts per arm are recorded so a reader can
judge it rather than take it on faith.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

__all__ = [
    "log_composition",
    "observation_variance",
    "posterior",
    "stabilize",
    "technical_variance",
]


def log_composition(counts: IntArray) -> tuple[FloatArray, float]:
    """Return the Haldane-corrected log-composition and the arm's total panel depth.

    The 1/2 correction is what keeps a zero *entry* representable.  It is not an imputation of a
    missing measurement: a zero count in a measured panel is an observation of zero, whereas an arm
    whose whole panel totals zero was never measured and is refused upstream rather than corrected
    here.
    """

    total = float(counts.sum())
    if total <= 0.0:
        raise ValueError("an arm with zero panel total is not measured and has no log-composition")
    size = counts.shape[0]
    return np.log((counts.astype(np.float64) + 0.5) / (total + size / 2.0)), total


def technical_variance(pooled_rate: FloatArray, depth: float) -> FloatArray:
    """Sampling variance of the *Haldane-corrected* log-composition, per gene.

    The statistic actually formed is ``log((y + 1/2) / (n + G/2))``, whose delta-method variance is
    ``lambda / (lambda + 1/2)^2 - 1/(n + G/2)`` at an expected count ``lambda = n * p``.

    **The rate is pooled, never the arm's own count, and that is the whole point of this function.**
    An earlier form evaluated the same expression at ``y`` by way of the plug-in ``1/(y + 1/2)``.
    That plug-in approaches the expression above only for large ``lambda``; at ``y = 0`` it returns
    **2.0** whatever the true rate is, while a gene whose expected count is near zero is in fact
    nearly deterministic -- it is observed as zero almost every time.  So it claimed its largest
    variance exactly where the data carry their least.

    Measured against the observed residual mean-square, per count bucket (ADR 0022):

    ==========  ==================  =================  =====
    count       observed residual2  claimed technical  ratio
    ==========  ==================  =================  =====
    ``y = 0``   0.3113              **2.0000**         6.42
    1-4         0.3581              0.4708             1.31
    5-19        0.2401              0.1074             0.45
    >= 100      0.0243              0.0017             0.07
    ==========  ==================  =================  =====

    Zero counts were 11.1% of panel entries and **79.8% of the claimed technical mass**, which
    drove ``mean(residual^2) - mean(technical)`` negative and pinned ``psi^2`` at its clamp in all
    fourteen folds.  Above a few counts the data carry *more* spread than sampling explains, and
    absorbing that excess is what ``psi^2`` is for; the zero bucket drowned it.

    This is the second time the same defect has been repaired here.  The first repair fixed the
    Haldane correction and left the plug-in, so the failure survived in a smaller form and was not
    re-measured.  It is measured now, per fold, and the count of clamped folds is reported.

    ``pooled_rate`` is a composition over the panel, taken from the fold's fit libraries only, so a
    held-out library contributes nothing to the variance its own arms are scored under.
    """

    if depth <= 0.0:
        raise ValueError("depth must be positive")
    if pooled_rate.ndim != 1:
        raise ValueError("pooled rate must be a one-dimensional composition")
    if not np.all(pooled_rate >= 0.0):
        raise ValueError("pooled rate must be non-negative")
    size = pooled_rate.shape[0]
    expected = depth * np.asarray(pooled_rate, dtype=np.float64)
    variance = expected / (expected + 0.5) ** 2 - 1.0 / (depth + size / 2.0)
    # A gene whose pooled rate vanishes leaves the multinomial correction dominant; the sampling
    # variance of a statistic that never moves is zero, not negative.
    return np.asarray(np.clip(variance, 0.0, None), dtype=np.float64)


def observation_variance(technical: FloatArray, psi_squared: float) -> FloatArray:
    """Total per-gene observation variance: technical plus fitted biological."""

    if psi_squared < 0.0:
        raise ValueError("biological variance must be non-negative")
    return technical + psi_squared


def stabilize(covariance: FloatArray) -> FloatArray:
    """Symmetrize and clip to the positive semi-definite cone.

    ``SchemaModel`` forbids inf and NaN, so a covariance that drifts non-finite is a validation
    failure rather than a sentinel that flows downstream.  Clipping here keeps the failure at the
    numerics rather than at the contract.
    """

    symmetric = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    clipped = np.clip(eigenvalues, 0.0, None)
    return np.asarray(eigenvectors @ np.diag(clipped) @ eigenvectors.T, dtype=np.float64)


def posterior(
    log_composition_observed: FloatArray,
    *,
    intercept: FloatArray,
    design: FloatArray,
    prior_precision: FloatArray,
    observation_variance_diagonal: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Return ``(mean, covariance)`` of the latent state given one arm's log-composition."""

    if design.shape[0] != log_composition_observed.shape[0]:
        raise ValueError("design and observation disagree on the number of panel genes")
    if observation_variance_diagonal.shape[0] != design.shape[0]:
        raise ValueError("observation variance and design disagree on the number of panel genes")
    if not np.all(observation_variance_diagonal > 0.0):
        raise ValueError("observation variance must be strictly positive")

    weights = 1.0 / observation_variance_diagonal
    weighted_design = design * weights[:, None]
    precision = design.T @ weighted_design + prior_precision
    covariance = stabilize(np.asarray(np.linalg.inv(precision), dtype=np.float64))
    residual = log_composition_observed - intercept
    mean = covariance @ (weighted_design.T @ residual)
    return np.asarray(mean, dtype=np.float64), covariance
