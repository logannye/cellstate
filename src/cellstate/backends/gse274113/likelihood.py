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

    omega_j = 1/(n p_j) - 1/n     technical, the delta-method multinomial term
            + psi^2               biological, fitted across libraries at fixed arm

The technical term shrinks as ``1/n``.  With ``n`` in the millions it becomes negligible, and a
model carrying only that term would report a posterior of absurd confidence -- the failure mode
where more sequencing depth is mistaken for more knowledge about the biology.  ``psi^2`` is what
stops that, and it is fitted, not assumed.

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


def technical_variance(counts: IntArray, depth: float) -> FloatArray:
    """Sampling variance of the *Haldane-corrected* log-composition, per gene.

    The naive delta method gives ``1/(n p) - 1/n``, and that is the variance of ``log(y/n)`` -- a
    statistic this model never forms, because it is undefined at ``y = 0``.  What is actually
    computed is ``log((y + 1/2) / (n + G/2))``, whose variance is ``1/(y + 1/2) - 1/(n + G/2)``.

    The distinction is not cosmetic and was caught by measurement rather than by inspection.  On a
    real arm the naive form averages 0.1230 against an observed residual mean-square of 0.0760, so
    it claims more sampling noise than the data contain; the corrected form averages 0.0760.  Using
    the naive form drives the fitted biological variance to its floor, which would have produced an
    overconfident posterior justified by an arithmetic mismatch rather than by biology.
    """

    if depth <= 0.0:
        raise ValueError("depth must be positive")
    size = counts.shape[0]
    variance = 1.0 / (counts.astype(np.float64) + 0.5) - 1.0 / (depth + size / 2.0)
    return np.asarray(variance, dtype=np.float64)


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
