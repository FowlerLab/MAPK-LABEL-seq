"""Simplified HSP90/HSP70 triage model for LABEL-seq abundance fitting.

Implements the model spec at ``docs/hsp90_triage_model.md``::

    A_pi,DMSO   = S_p,DMSO   * (1 + K_i L_p) /
                  (1 + K_i + K_i L_p + D_i)

    A_pi,HSP90i = S_p,HSP90i * 1 /
                  (1 + K_i + D_i)

with::

    K_i = K_p exp(b * ddG_pi)
    D_i = D_p exp(d * ddG_pi)
    h = 0,  omega_H = 0   (no residual HSP90i activity)
    d > b > 0             (triage slope steeper than metastability slope)

Per protein p we fit ``K_p, L_p, D_p, S_p,DMSO, S_p,HSP90i``; globally
we fit ``b, d``. The triage state ``D_i`` is the key addition vs. the
3-state equilibrium model in ``equilibrium_model.py`` — it provides the
irreversible-degradation pathway that creates the lower-left horizontal
floor in the DMSO-vs-HSP90i scatter geometry (high-ddG variants degraded
in both conditions, but at different rates because HSP90 buffers in DMSO).

Implementation notes:

* Forward prediction is in log space with ``logsumexp`` to avoid overflow
  for the K_i and D_i terms at high ddG. Per spec §"Numerically stable
  prediction".
* Parameter transforms are unconstrained → biologically meaningful via
  ``log_K``, ``log_L``, ``log_D``, ``log_S_*``. Global slopes via
  ``b = softplus(u_b)``, ``d = b + softplus(u_d_minus_b)`` to enforce
  ``d > b > 0`` smoothly.
* Three model variants supported via flags:
  * Model A (no L): ``fit_L = False`` ⇒ pure-scaling baseline (no HSP90
    buffering); per spec §"Model variants — Model A".
  * Model B (no D): ``fit_D = False`` ⇒ no triage; should overpredict
    HSP90i abundance for highly destabilised variants. Per spec §"Model B".
  * Model C (full): both ``fit_L`` and ``fit_D`` True. Recommended main
    model. Per spec §"Model C".

The module is pure (no I/O); ``scripts/fit_hsp90_triage_model.py`` wraps
it for the actual fitting workflow (per-protein init → joint fit → multi-
start → diagnostics).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

# Same f64 sentinel as equilibrium_model.py — abundance scales span ~3
# decades and Huber-on-log loss benefits from extra dynamic range during
# line search.
jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Parameter packing / unpacking
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TriageDims:
    """Dimensions for the triage-model parameter vector.

    Attributes:
        n_proteins: Number of proteins (e.g., 9 for the standard panel).
        fit_L: If False, L_p is fixed at 0 for every protein (Model A,
            no HSP90 buffering); only triage and base-state degradation
            differentiate the conditions.
        fit_D: If False, D_p is fixed at 0 for every protein (Model B,
            no triage). Reduces to the equilibrium model with h=0, omega_H=0.
    """

    n_proteins: int
    fit_L: bool = True
    fit_D: bool = True


class _Params(NamedTuple):
    """Natural-parameter struct used inside the JIT'd predict / loss.

    All fields are JAX arrays in the natural (positive) parameter space.
    Conversion from the unconstrained optimisation vector happens in
    :func:`unpack`.
    """

    log_K: jnp.ndarray            # (n_proteins,) log K_p
    log_L: jnp.ndarray            # (n_proteins,) log L_p (-inf if not fit)
    log_D: jnp.ndarray            # (n_proteins,) log D_p (-inf if not fit)
    log_S_dmso: jnp.ndarray       # (n_proteins,) log S_p,DMSO
    log_S_hsp90i: jnp.ndarray     # (n_proteins,) log S_p,HSP90i
    b: jnp.ndarray                # () global ddG-to-metastability slope
    d: jnp.ndarray                # () global ddG-to-triage slope, d > b


def n_unconstrained(dims: TriageDims) -> int:
    """Length of the flat unconstrained optimisation vector for given dims."""
    n = dims.n_proteins              # log_K
    if dims.fit_L:
        n += dims.n_proteins         # log_L
    if dims.fit_D:
        n += dims.n_proteins         # log_D
    n += dims.n_proteins             # log_S_DMSO
    n += dims.n_proteins             # log_S_HSP90i
    n += 1                           # u_b  (b = softplus(u_b))
    n += 1                           # u_d_minus_b (d = b + softplus(...))
    return n


def _softplus(x: jnp.ndarray) -> jnp.ndarray:
    """Numerically stable softplus = log(1 + exp(x)).

    Avoids overflow for large positive x; pure JAX implementation so
    autodiff is clean.
    """
    return jnp.log1p(jnp.exp(-jnp.abs(x))) + jnp.maximum(x, 0.0)


def unpack(theta: jnp.ndarray, dims: TriageDims) -> _Params:
    """Convert a flat unconstrained vector into natural model parameters.

    Layout::

        theta = [log_K_p, (log_L_p), (log_D_p),
                 log_S_DMSO, log_S_HSP90i,
                 u_b, u_d_minus_b]

    where parenthesised blocks are present only when the corresponding
    fit flag is True. When ``fit_L`` (resp. ``fit_D``) is False, the
    natural parameter ``log_L`` (resp. ``log_D``) is set to ``-inf`` so
    ``exp(log_L) == 0`` exactly in the predict function and the term
    drops out of the partition function.

    Args:
        theta: Flat unconstrained parameter vector of length
            :func:`n_unconstrained`.
        dims: TriageDims describing which parameter groups are present.

    Returns:
        _Params NamedTuple in natural parameter space.
    """
    n_p = dims.n_proteins
    i = 0
    log_K = theta[i:i + n_p]
    i += n_p
    if dims.fit_L:
        log_L = theta[i:i + n_p]
        i += n_p
    else:
        log_L = jnp.full((n_p,), -jnp.inf)
    if dims.fit_D:
        log_D = theta[i:i + n_p]
        i += n_p
    else:
        log_D = jnp.full((n_p,), -jnp.inf)
    log_S_dmso = theta[i:i + n_p]
    i += n_p
    log_S_hsp90i = theta[i:i + n_p]
    i += n_p

    u_b = theta[i]
    i += 1
    u_d_minus_b = theta[i]
    i += 1
    b = _softplus(u_b)
    d = b + _softplus(u_d_minus_b)

    return _Params(log_K=log_K, log_L=log_L, log_D=log_D,
                   log_S_dmso=log_S_dmso, log_S_hsp90i=log_S_hsp90i,
                   b=b, d=d)


# ---------------------------------------------------------------------------
# Forward model (log-space, logsumexp)
# ---------------------------------------------------------------------------

def predict_log_abundance(
    ddG: jnp.ndarray,
    protein_idx: jnp.ndarray,
    is_hsp90i: jnp.ndarray,
    params: _Params,
) -> jnp.ndarray:
    """Predict log standardized abundance for a vector of variant observations.

    Computed in log space with ``logsumexp`` to avoid overflow at high ddG
    where K_i and D_i become large.

    DMSO::

        log A = log_S_DMSO + log(1 + K_i L_p) - log(1 + K_i + K_i L_p + D_i)

    HSP90i::

        log A = log_S_HSP90i - log(1 + K_i + D_i)

    Args:
        ddG: (n_obs,) SPURS ddG, kcal/mol.
        protein_idx: (n_obs,) integer index into per-protein arrays.
        is_hsp90i: (n_obs,) bool/0-1; True for HSP90i observations.
        params: Natural parameters from :func:`unpack`.

    Returns:
        (n_obs,) log predicted abundance, same scale as log(observed).
    """
    log_K_p = params.log_K[protein_idx]
    log_L_p = params.log_L[protein_idx]
    log_D_p = params.log_D[protein_idx]
    log_S = jnp.where(is_hsp90i,
                      params.log_S_hsp90i[protein_idx],
                      params.log_S_dmso[protein_idx])

    log_K_i = log_K_p + params.b * ddG
    log_D_i = log_D_p + params.d * ddG
    log_KL_i = log_K_p + log_L_p + params.b * ddG

    # Numerator (DMSO): logsumexp(0, log_KL_i)  -> log(1 + K_i L_p)
    # Numerator (HSP90i): 0  -> log(1) = 0
    n_obs = ddG.shape[0]
    zeros = jnp.zeros_like(log_K_i)
    num_stack_dmso = jnp.stack([zeros, log_KL_i], axis=0)
    log_num_dmso = logsumexp(num_stack_dmso, axis=0)
    log_num = jnp.where(is_hsp90i, jnp.zeros(n_obs), log_num_dmso)

    # Denominator: DMSO has 4 terms (1, K_i, K_i L_p, D_i);
    # HSP90i has 3 terms (1, K_i, D_i). We stack 4 terms and zero out
    # the K_i L_p contribution (set to -inf) under HSP90i so logsumexp
    # ignores it.
    log_KL_masked = jnp.where(is_hsp90i,
                              jnp.full((n_obs,), -jnp.inf),
                              log_KL_i)
    den_stack = jnp.stack([zeros, log_K_i, log_KL_masked, log_D_i], axis=0)
    log_den = logsumexp(den_stack, axis=0)

    return log_S + log_num - log_den


def predict_abundance(
    ddG: jnp.ndarray,
    protein_idx: jnp.ndarray,
    is_hsp90i: jnp.ndarray,
    params: _Params,
) -> jnp.ndarray:
    """Convenience wrapper: returns A on the linear scale.

    Equal to ``exp(predict_log_abundance(...))``. Use the log version
    inside the loss function; this wrapper is for diagnostic plots that
    want abundance in linear units.
    """
    return jnp.exp(predict_log_abundance(ddG, protein_idx, is_hsp90i, params))


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def _huber(resid: jnp.ndarray, delta: float = 1.0) -> jnp.ndarray:
    """Element-wise Huber loss; quadratic inside |resid|<=delta, linear outside.
    Robust to a small number of large residuals."""
    abs_r = jnp.abs(resid)
    quad = 0.5 * resid ** 2
    lin = delta * (abs_r - 0.5 * delta)
    return jnp.where(abs_r <= delta, quad, lin)


def make_loss_fn(
    ddG: np.ndarray,
    protein_idx: np.ndarray,
    is_hsp90i: np.ndarray,
    log_abundance: np.ndarray,
    dims: TriageDims,
    *,
    huber_delta: float = 1.0,
    log_K_prior: tuple[float, float] | None = (np.log(0.1), 2.0),
    log_L_prior: tuple[float, float] | None = (0.0, 2.0),
    log_D_prior: tuple[float, float] | None = (np.log(0.03), 2.0),
    log_S_prior_sigma: float | None = 1.0,
    log_S_ratio_prior_sigma: float | None = None,
    log_b_prior: tuple[float, float] | None = (np.log(0.5), 1.0),
    log_d_minus_b_prior: tuple[float, float] | None = (np.log(1.0), 1.0),
) -> tuple[Callable[[np.ndarray], float],
           Callable[[np.ndarray], np.ndarray]]:
    """Build a JIT'd (loss, grad) pair for the triage model.

    Loss is Huber on ``log A_obs - log A_pred`` (no epsilon added — the
    caller is expected to log-transform abundance and pass it in directly,
    so log_abundance is finite and well-defined). Optional weak Gaussian
    priors on each parameter group per spec §"Regularization / priors".

    Args:
        ddG: (n_obs,) variant ddG values.
        protein_idx: (n_obs,) integer per-row protein index.
        is_hsp90i: (n_obs,) 0-1 condition indicator.
        log_abundance: (n_obs,) log of observed standardized abundance.
        dims: TriageDims for parameter unpacking.
        huber_delta: Huber threshold for the variant residuals (log space).
        log_K_prior, log_L_prior, log_D_prior: Each is a (mean, sigma)
            tuple for a Gaussian prior on the per-protein log parameter.
            Defaults match spec §"Regularization / priors".
        log_S_prior_sigma: Stdev for a Gaussian prior on each log S_p,c
            centred at the WT log abundance. Set to None to disable. The
            mean is anchored externally via the loss closure (caller can
            inject WT-centred priors by setting log_S_dmso/hsp90i means).
        log_S_ratio_prior_sigma: Optional stdev for a Gaussian prior on
            ``log S_p,HSP90i - log S_p,DMSO`` centred at 0. Set to a
            small value (e.g. 0.3) to bias the optimiser toward
            attributing the WT shift to the equilibrium machinery (K, L,
            D) rather than to a per-protein scale ratio. Set to None
            (default) to disable.
        log_b_prior, log_d_minus_b_prior: (mean, sigma) for log-normal
            priors on the global slope and the slope gap.

    Returns:
        loss_fn(theta) -> scalar, grad_fn(theta) -> array. grad_fn output
        is np.ndarray for direct use with scipy.optimize.minimize.
    """
    ddG_j = jnp.asarray(ddG, dtype=jnp.float64)
    protein_idx_j = jnp.asarray(protein_idx, dtype=jnp.int32)
    is_hsp90i_j = jnp.asarray(is_hsp90i, dtype=jnp.float64)
    log_abund_j = jnp.asarray(log_abundance, dtype=jnp.float64)

    def _loss(theta):
        params = unpack(theta, dims)
        log_a_hat = predict_log_abundance(ddG_j, protein_idx_j,
                                          is_hsp90i_j, params)
        resid = log_abund_j - log_a_hat
        total = jnp.mean(_huber(resid, huber_delta))

        if log_K_prior is not None:
            mu, sig = log_K_prior
            total = total + 0.5 * jnp.mean(((params.log_K - mu) / sig) ** 2)
        if dims.fit_L and log_L_prior is not None:
            mu, sig = log_L_prior
            total = total + 0.5 * jnp.mean(((params.log_L - mu) / sig) ** 2)
        if dims.fit_D and log_D_prior is not None:
            mu, sig = log_D_prior
            total = total + 0.5 * jnp.mean(((params.log_D - mu) / sig) ** 2)
        if log_S_prior_sigma is not None and log_S_prior_sigma > 0:
            # No prior mean injected here — the WT-anchored mean is
            # passed via the optimiser's initialization. This term is a
            # mild ridge keeping log S from drifting off into space.
            total = total + 0.5 * jnp.mean(
                (params.log_S_dmso / (5.0 * log_S_prior_sigma)) ** 2
                + (params.log_S_hsp90i / (5.0 * log_S_prior_sigma)) ** 2
            )
        if log_S_ratio_prior_sigma is not None and log_S_ratio_prior_sigma > 0:
            log_S_ratio = params.log_S_hsp90i - params.log_S_dmso
            total = total + 0.5 * jnp.mean(
                (log_S_ratio / log_S_ratio_prior_sigma) ** 2
            )
        if log_b_prior is not None:
            mu, sig = log_b_prior
            total = total + 0.5 * ((jnp.log(params.b) - mu) / sig) ** 2
        if log_d_minus_b_prior is not None:
            mu, sig = log_d_minus_b_prior
            d_minus_b = params.d - params.b
            total = total + 0.5 * (
                (jnp.log(jnp.maximum(d_minus_b, 1e-9)) - mu) / sig
            ) ** 2
        return total

    loss_jit = jax.jit(_loss)
    grad_jit = jax.jit(jax.grad(_loss))

    def loss_fn(theta: np.ndarray) -> float:
        return float(loss_jit(jnp.asarray(theta, dtype=jnp.float64)))

    def grad_fn(theta: np.ndarray) -> np.ndarray:
        g = grad_jit(jnp.asarray(theta, dtype=jnp.float64))
        return np.ascontiguousarray(np.array(g, dtype=np.float64))

    return loss_fn, grad_fn


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def initial_theta(
    dims: TriageDims,
    *,
    log_S_dmso_init: np.ndarray | None = None,
    log_S_hsp90i_init: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    jitter: float = 0.5,
) -> np.ndarray:
    """Construct a heuristic initialisation per spec §"Initialization".

    * log_K = log(0.1)
    * log_L = log(1)
    * log_D = log(0.03)
    * log_S_DMSO, log_S_HSP90i: from caller (median per-condition log
      abundance, or WT log abundance), or 0 if not provided.
    * b = 0.5  =>  u_b = log(exp(0.5) - 1) ≈ -0.433 (softplus inverse)
    * d - b = 1.0  =>  u_d_minus_b ≈ 0.541

    A small Gaussian jitter is added to break ties on multi-restart fits.

    Args:
        dims: Model dimensions.
        log_S_dmso_init: (n_proteins,) initial log S_p,DMSO. Default 0.
        log_S_hsp90i_init: (n_proteins,) initial log S_p,HSP90i. Default 0.
        rng: numpy Generator for jitter.
        jitter: stdev of Gaussian noise in unconstrained space.

    Returns:
        np.ndarray of length :func:`n_unconstrained`.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n_p = dims.n_proteins

    parts = [np.full(n_p, np.log(0.1))]   # log_K
    if dims.fit_L:
        parts.append(np.full(n_p, np.log(1.0)))    # log_L
    if dims.fit_D:
        parts.append(np.full(n_p, np.log(0.03)))   # log_D

    log_S_d = (np.zeros(n_p) if log_S_dmso_init is None
               else np.asarray(log_S_dmso_init, dtype=np.float64))
    log_S_h = (np.zeros(n_p) if log_S_hsp90i_init is None
               else np.asarray(log_S_hsp90i_init, dtype=np.float64))
    parts.append(log_S_d)
    parts.append(log_S_h)

    # Softplus inverse: y = log(1 + exp(x))  =>  x = log(exp(y) - 1)
    u_b = float(np.log(np.expm1(0.5)))
    u_d_minus_b = float(np.log(np.expm1(1.0)))
    parts.append(np.array([u_b]))
    parts.append(np.array([u_d_minus_b]))

    theta = np.concatenate(parts)
    theta = theta + jitter * rng.standard_normal(theta.shape)
    return theta
