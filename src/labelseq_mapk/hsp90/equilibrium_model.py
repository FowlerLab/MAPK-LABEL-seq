"""Three-state thermodynamic equilibrium model for HSP90 chaperone buffering.

Implements the model in ``hsp90_labelseq_equilibrium_model.md``::

    A_pic = S_pc * (1 + omega_c * K_p * exp(b * ddG) * Lambda_pc)
                 / (1 + K_p * exp(b * ddG) * (1 + Lambda_pc))

with::

    Lambda_p,DMSO  = L_p
    Lambda_p,HSP90i = h * L_p
    omega_DMSO     = 1
    omega_HSP90i   = omega_H

Per protein p we fit ``K_p, L_p, S_p,DMSO, S_p,HSP90i``; globally we fit
``b, h, omega_H``. The variant-level covariate is the SPURS predicted
ddG. The biological mapping is

* ``K_p`` is the WT folded-to-metastable equilibrium constant — high in
  proteins where WT spends a lot of time in the exposed state.
* ``L_p`` is the HSP90 capture strength — high in clients where the
  metastable state is efficiently sequestered into the HSP90 complex.
* ``S_pc`` is the per-(protein, condition) scale factor, absorbing the
  cross-protein offset of the standard-adjusted abundance scale.
* ``b`` is the global ddG → metastability slope (kcal/mol)^-1.
* ``h`` is the residual active HSP90 fraction under pimitespib
  (0 ≤ h ≤ 1, typically << 1).
* ``omega_H`` is the contribution of the HSP90-complexed state under
  HSP90i (0 ≤ omega_H ≤ 1, often ≈ 0).

Reparameterisation (used internally for optimisation):

The data identifies ``K_p L_p`` (the WT bound-to-folded ratio, "clientness")
and ``K_p`` (metastable propensity) more cleanly than ``K_p, L_p`` separately.
We therefore expose user-facing parameters as ``log_K``, ``log_L`` for
interpretability but optimise in the ``(kappa_p = log(K_p L_p),
mu_p = log K_p)`` basis. Conversion::

    L_p = exp(kappa_p - mu_p)
    K_p = exp(mu_p)

The translation is invertible and orthogonalises the per-protein subspace
in our data regime.

The module is pure (no I/O); ``scripts/fit_hsp90_equilibrium_model.py``
wraps it for actual fitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

# Sentinel: keep f64 — abundance scales span ~3 orders of magnitude (MET ~0.05
# to MEK1 ~5), and Huber-on-log loss benefits from the extra dynamic range
# during line search.
jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Parameter packing / unpacking
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelDims:
    """Dimensions for the equilibrium-model parameter vector.

    Attributes:
        n_proteins: Number of proteins (e.g., 9 for the standard panel).
        fit_omega_h: If True, omega_H is a free global parameter.
            If False, omega_H is fixed at 0 and the HSP90-bound state is
            assumed to contribute nothing to abundance under HSP90i.
        fit_S: If True, S_p,DMSO and S_p,HSP90i are free per protein
            (Model 2). If False (Model 1), S_pc cancels because we fit
            WT-normalised ratios; the parameter vector omits the S terms.
        h_max: Upper bound on the residual HSP90 fraction h. h is mapped
            from the unconstrained h_raw via h_max * sigmoid(h_raw); the
            default 0.2 reflects pimitespib's clinical residual activity
            (~5-15%) and keeps the optimiser from fleeing to the
            non-mechanistic h=1 corner.
        omega_h_max: Upper bound on the HSP90-bound contribution under
            HSP90i, omega_H. Pimitespib blocks HSP90 ATPase, after which
            engaged clients are released and routed to proteasomal
            degradation (Schulte 1995 PMID 7559543; Bagatell & Whitesell
            2004 PMID 15485890; Ohkubo 2015 PMID 25612623). At the
            steady state the LABEL-seq abundance assay measures, the
            HSP90-bound-but-non-cycling state has already been cleared,
            so the biological default is omega_H = 0 (set fit_omega_h
            to False; see :class:`ModelDims`).

            Allowing omega_H > 0 lets the optimiser explore the regime
            where the bound state contributes to abundance under HSP90i
            (one possible explanation for strong-client uniform
            collapse), but this regime is at odds with the literature
            and is provided only for sensitivity analysis. omega_h_max
            of 0.3 is a generous upper bound for that purpose.
    """

    n_proteins: int
    fit_omega_h: bool = True
    fit_S: bool = True
    per_protein_b: bool = False
    h_max: float = 0.2
    omega_h_max: float = 1.0


class _Params(NamedTuple):
    """Unpacked parameter struct used inside the JIT'd predict / loss.

    All fields are JAX arrays in the natural (positive / bounded) parameter
    space. Conversion from the unconstrained optimisation vector happens in
    :func:`unpack`.
    """

    K: jnp.ndarray            # (n_proteins,) WT metastable / folded ratio
    L: jnp.ndarray            # (n_proteins,) HSP90 capture strength
    S_dmso: jnp.ndarray       # (n_proteins,) DMSO scale (unused if fit_S=False)
    S_hsp90i: jnp.ndarray     # (n_proteins,) HSP90i scale
    b: jnp.ndarray            # (n_proteins,) per-protein ddG slope, >= 0.
                              # If per_protein_b=False, all entries are
                              # equal to the global b.
    h: jnp.ndarray            # () residual HSP90 fraction in HSP90i, in (0, 1)
    omega_h: jnp.ndarray      # () bound-state contribution under HSP90i


def n_unconstrained(dims: ModelDims) -> int:
    """Length of the flat unconstrained optimisation vector for given dims."""
    n = dims.n_proteins              # mu_p (= log K_p)
    n += dims.n_proteins             # kappa_p (= log K_p L_p)
    if dims.fit_S:
        n += dims.n_proteins         # log S_p,DMSO
        n += dims.n_proteins         # log S_p,HSP90i
    n += 1                           # log b (global)
    if dims.per_protein_b:
        n += dims.n_proteins         # log_b_p_dev (per-protein deviations
                                     #  from log b, regularised toward 0)
    n += 1                           # h_raw (logit h)
    if dims.fit_omega_h:
        n += 1                       # omega_h_raw (logit omega_H)
    return n


def unpack(theta: jnp.ndarray, dims: ModelDims) -> _Params:
    """Convert a flat unconstrained vector into natural model parameters.

    The optimisation basis is::

        theta = [mu_p, kappa_p, log_S_dmso, log_S_hsp90i, log_b, h_raw, omega_h_raw]

    where ``mu_p = log K_p`` and ``kappa_p = log(K_p L_p)`` (clientness). When
    ``dims.fit_S`` is False the ``log_S_*`` blocks are omitted and the natural
    parameters are returned with the scale fields set to ``1`` so callers can
    use the same predict function for both Model 1 and Model 2.

    Args:
        theta: Flat unconstrained parameter vector of length
            :func:`n_unconstrained`.
        dims: ModelDims describing which parameter groups are present.

    Returns:
        _Params NamedTuple in natural parameter space.
    """
    n_p = dims.n_proteins
    i = 0
    mu = theta[i:i + n_p]
    i += n_p
    kappa = theta[i:i + n_p]
    i += n_p
    K = jnp.exp(mu)
    L = jnp.exp(kappa - mu)

    if dims.fit_S:
        log_S_dmso = theta[i:i + n_p]
        i += n_p
        log_S_hsp90i = theta[i:i + n_p]
        i += n_p
        S_dmso = jnp.exp(log_S_dmso)
        S_hsp90i = jnp.exp(log_S_hsp90i)
    else:
        S_dmso = jnp.ones(n_p)
        S_hsp90i = jnp.ones(n_p)

    log_b_global = theta[i]
    i += 1
    if dims.per_protein_b:
        log_b_dev = theta[i:i + n_p]
        i += n_p
        b = jnp.exp(log_b_global + log_b_dev)
    else:
        b = jnp.full((n_p,), jnp.exp(log_b_global))
    h_raw = theta[i]
    i += 1
    h = dims.h_max * jax.nn.sigmoid(h_raw)

    if dims.fit_omega_h:
        omega_h_raw = theta[i]
        i += 1
        omega_h = dims.omega_h_max * jax.nn.sigmoid(omega_h_raw)
    else:
        omega_h = jnp.array(0.0)

    return _Params(K=K, L=L, S_dmso=S_dmso, S_hsp90i=S_hsp90i,
                   b=b, h=h, omega_h=omega_h)


# ---------------------------------------------------------------------------
# Forward model
# ---------------------------------------------------------------------------

def predict_abundance(
    ddG: jnp.ndarray,
    protein_idx: jnp.ndarray,
    is_hsp90i: jnp.ndarray,
    params: _Params,
) -> jnp.ndarray:
    """Predict standardized abundance for a vector of variant observations.

    Args:
        ddG: (n_obs,) SPURS ddG, kcal/mol.
        protein_idx: (n_obs,) integer index into the per-protein parameter
            arrays, in [0, n_proteins).
        is_hsp90i: (n_obs,) bool/0-1; True where the observation is the
            HSP90i condition, False/0 for DMSO.
        params: Natural parameters from :func:`unpack`.

    Returns:
        (n_obs,) predicted abundance on the same scale as the data.
    """
    K_p = params.K[protein_idx]
    L_p = params.L[protein_idx]
    b_p = params.b[protein_idx]
    S_pc = jnp.where(is_hsp90i,
                     params.S_hsp90i[protein_idx],
                     params.S_dmso[protein_idx])
    Lambda = jnp.where(is_hsp90i, params.h * L_p, L_p)
    omega = jnp.where(is_hsp90i, params.omega_h, 1.0)

    K_i = K_p * jnp.exp(b_p * ddG)

    numer = 1.0 + omega * K_i * Lambda
    denom = 1.0 + K_i + K_i * Lambda
    return S_pc * numer / denom


def predict_wt_normalised(
    ddG: jnp.ndarray,
    protein_idx: jnp.ndarray,
    is_hsp90i: jnp.ndarray,
    params: _Params,
) -> jnp.ndarray:
    """Predict WT-normalised abundance ratio (Model 1 shape-only target).

    Computes ``A_pic / A_pWTc`` so that S_pc cancels and only the
    K_p, L_p, b, h, omega_H parameters drive the prediction. The data input
    must therefore also be WT-normalised per (protein, condition).

    Args:
        ddG: (n_obs,) ddG values.
        protein_idx: (n_obs,) protein index.
        is_hsp90i: (n_obs,) bool/0-1.
        params: Natural parameters; the S fields are ignored.

    Returns:
        (n_obs,) predicted WT-normalised abundance.
    """
    K_p = params.K[protein_idx]
    L_p = params.L[protein_idx]
    b_p = params.b[protein_idx]
    Lambda = jnp.where(is_hsp90i, params.h * L_p, L_p)
    omega = jnp.where(is_hsp90i, params.omega_h, 1.0)

    K_i = K_p * jnp.exp(b_p * ddG)
    numer_var = 1.0 + omega * K_i * Lambda
    denom_var = 1.0 + K_i + K_i * Lambda

    # WT factor (ddG = 0 -> K_i = K_p)
    K_wt = K_p
    numer_wt = 1.0 + omega * K_wt * Lambda
    denom_wt = 1.0 + K_wt + K_wt * Lambda

    return (numer_var / denom_var) / (numer_wt / denom_wt)


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def _huber(resid: jnp.ndarray, delta: float = 1.0) -> jnp.ndarray:
    """Element-wise Huber loss; quadratic inside |resid|<=delta, linear outside.
    Robust to a small number of large residuals (e.g. truncation outliers)."""
    abs_r = jnp.abs(resid)
    quad = 0.5 * resid ** 2
    lin = delta * (abs_r - 0.5 * delta)
    return jnp.where(abs_r <= delta, quad, lin)


def make_loss_fn(
    ddG: np.ndarray,
    protein_idx: np.ndarray,
    is_hsp90i: np.ndarray,
    abundance: np.ndarray,
    dims: ModelDims,
    *,
    wt_anchor: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    huber_delta: float = 1.0,
    log_eps: float = 1e-3,
    wt_weight: float = 1.0,
    log_L_prior_sigma: float = 0.0,
    log_S_ratio_prior_sigma: float = 0.0,
    log_b_dev_prior_sigma: float = 0.0,
) -> tuple[Callable[[np.ndarray], float],
           Callable[[np.ndarray], np.ndarray]]:
    """Build a JIT'd (loss, grad) pair for the equilibrium model.

    Loss is the Huber loss on ``log(A_obs + eps) - log(A_hat + eps)``. If
    ``wt_anchor`` is given it adds a Huber penalty on the per-protein WT
    log-residual so the WT shift drives the S_p,c parameters cleanly.

    Args:
        ddG: (n_obs,) variant ddG values.
        protein_idx: (n_obs,) integer per-row protein index.
        is_hsp90i: (n_obs,) 0-1 condition indicator.
        abundance: (n_obs,) observed standard-adjusted abundance.
        dims: ModelDims for parameter unpacking.
        wt_anchor: Optional tuple (wt_protein_idx, wt_is_hsp90i, wt_abund).
            wt_protein_idx is (n_proteins * n_conditions,), the same shape
            as wt_is_hsp90i. Each entry's predicted abundance at ddG=0 is
            compared to the corresponding observed WT abundance and a Huber
            penalty is added.
        huber_delta: Huber threshold for the variant residuals (log-space).
        log_eps: Floor added to abundance before the log to avoid -inf on
            very small values; should be small relative to the smallest
            real abundance (~0.02).
        wt_weight: Multiplicative weight on the WT-anchor term relative to
            the variant-residual term.
        log_L_prior_sigma: If > 0, adds a Gaussian prior penalty on log L_p
            with mean 0 and stdev log_L_prior_sigma. Recommended for fits
            where h is small or fixed near 0, because in that regime L_p
            drifts in directions that don't affect the predicted abundance
            and the optimiser leaves it at runaway values.
        log_S_ratio_prior_sigma: If > 0, adds a Gaussian prior penalty on
            log(S_p,HSP90i / S_p,DMSO) with mean 0 and stdev given. This
            is the principled fix for the WT-shift identifiability gap:
            without this prior, strong-client WT shifts can be absorbed
            either by the equilibrium machinery (large K_p L_p) or by
            S_ratio < 1 (proteome-wide stress); both fit equally well.
            With a non-zero prior centred at 0, the optimiser is biased
            toward attributing the WT shift to the equilibrium machinery,
            recovering the K_p L_p ordering against literature LUMIER
            client-affinity ranking. Set to ~0.3 to allow non-clients
            their HSF1 bystander rise (~30%) while penalising
            unconstrained S_ratio drift.

    Returns:
        loss_fn(theta) -> scalar, grad_fn(theta) -> array. Both accept
        either np.ndarray or jnp.ndarray; output of grad_fn is np.ndarray
        for direct use with scipy.optimize.minimize.
    """
    ddG_j = jnp.asarray(ddG, dtype=jnp.float64)
    protein_idx_j = jnp.asarray(protein_idx, dtype=jnp.int32)
    is_hsp90i_j = jnp.asarray(is_hsp90i, dtype=jnp.float64)
    abundance_j = jnp.asarray(abundance, dtype=jnp.float64)

    if wt_anchor is not None:
        wt_pidx_j = jnp.asarray(wt_anchor[0], dtype=jnp.int32)
        wt_is_hsp90i_j = jnp.asarray(wt_anchor[1], dtype=jnp.float64)
        wt_abund_j = jnp.asarray(wt_anchor[2], dtype=jnp.float64)
        wt_ddG_j = jnp.zeros_like(wt_abund_j)
    else:
        wt_pidx_j = None
        wt_is_hsp90i_j = None
        wt_abund_j = None
        wt_ddG_j = None

    def _loss(theta):
        params = unpack(theta, dims)
        a_hat = predict_abundance(ddG_j, protein_idx_j, is_hsp90i_j, params)
        resid = jnp.log(abundance_j + log_eps) - jnp.log(a_hat + log_eps)
        total = jnp.mean(_huber(resid, huber_delta))
        if wt_anchor is not None:
            wt_hat = predict_abundance(wt_ddG_j, wt_pidx_j,
                                       wt_is_hsp90i_j, params)
            wt_resid = jnp.log(wt_abund_j + log_eps) - jnp.log(wt_hat + log_eps)
            wt_term = jnp.mean(_huber(wt_resid, huber_delta))
            total = total + wt_weight * wt_term
        if log_L_prior_sigma > 0.0:
            log_L = jnp.log(params.L)
            total = total + 0.5 * jnp.mean((log_L / log_L_prior_sigma) ** 2)
        if log_S_ratio_prior_sigma > 0.0:
            log_S_ratio = jnp.log(params.S_hsp90i) - jnp.log(params.S_dmso)
            total = total + 0.5 * jnp.mean(
                (log_S_ratio / log_S_ratio_prior_sigma) ** 2
            )
        if log_b_dev_prior_sigma > 0.0:
            # Penalty on deviation of per-protein log_b from the geometric
            # mean across proteins (i.e., the implicit "global b"). This
            # keeps per-protein b_p from drifting without data support
            # while still allowing a few proteins to genuinely have
            # different ddG slopes.
            log_b_arr = jnp.log(params.b)
            log_b_mean = jnp.mean(log_b_arr)
            total = total + 0.5 * jnp.mean(
                ((log_b_arr - log_b_mean) / log_b_dev_prior_sigma) ** 2
            )
        return total

    loss_jit = jax.jit(_loss)
    grad_jit = jax.jit(jax.grad(_loss))

    def loss_fn(theta: np.ndarray) -> float:
        return float(loss_jit(jnp.asarray(theta, dtype=jnp.float64)))

    def grad_fn(theta: np.ndarray) -> np.ndarray:
        g = grad_jit(jnp.asarray(theta, dtype=jnp.float64))
        # L-BFGS-B's Fortran wrapper requires a contiguous, writable f64
        # array; np.asarray on a JAX device array returns a read-only view
        # with non-standard stride flags, so copy to a fresh contiguous one.
        return np.ascontiguousarray(np.array(g, dtype=np.float64))

    return loss_fn, grad_fn


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def initial_theta(
    dims: ModelDims,
    *,
    wt_ratio: np.ndarray | None = None,
    wt_ctrl: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    jitter: float = 0.5,
) -> np.ndarray:
    """Construct a reasonable initialisation for the optimiser.

    Heuristic:
    * mu_p = 0  (K_p = 1, i.e. WT half in folded, half in metastable; bland)
    * kappa_p chosen so the WT bound-to-folded ratio matches the fractional
      WT shift ``1 - wt_ratio``: ``K_p L_p ≈ (1 - wt_ratio) / wt_ratio``
      after assuming most of the shift is loss of bound-state contribution.
      If ``wt_ratio`` is None, defaults kappa_p = 0 (L_p = 1).
    * S_p,DMSO = wt_ctrl[p], S_p,HSP90i = wt_ratio[p] * wt_ctrl[p] (so the
      WT prediction matches the observed WT at ddG=0 in both conditions).
    * log_b = log(0.5)  (modest ddG slope, kcal/mol -> metastability)
    * h = 0.05         (5% residual HSP90 activity under pimitespib)
    * omega_h = 0.0    (bound state contributes nothing under HSP90i;
                       sigmoid output ~0.5 at h_raw=0, so we start at
                       a logit value of -3 to bias toward 0)

    A small Gaussian jitter is added to break ties on multi-restart fits.

    Args:
        dims: Model dimensions.
        wt_ratio: (n_proteins,) wt_hsp90i / wt_ctrl ratio per protein,
            used to seed kappa_p.
        wt_ctrl: (n_proteins,) wt_ctrl_std per protein, used to seed
            S_p,DMSO and S_p,HSP90i = wt_ratio * wt_ctrl.
        rng: numpy Generator for jitter.
        jitter: stdev of Gaussian noise in unconstrained space.

    Returns:
        np.ndarray of length n_unconstrained(dims).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n_p = dims.n_proteins

    mu = np.zeros(n_p)

    if wt_ratio is not None:
        # WT bound-to-folded ratio implied by observed WT shift; clip to
        # avoid log(0) or negative for the rare WT-rises-under-HSP90i cases
        # (MEK1 in our data). Start a non-client at very low kappa.
        ratio = np.clip(wt_ratio, 0.05, 0.99)
        boundness = (1.0 - ratio) / ratio   # K_p L_p heuristic
        kappa = np.log(np.maximum(boundness, 0.05))
    else:
        kappa = np.zeros(n_p)

    parts = [mu, kappa]

    if dims.fit_S:
        if wt_ctrl is None:
            log_S_dmso = np.zeros(n_p)
            log_S_hsp90i = np.zeros(n_p)
        else:
            log_S_dmso = np.log(np.maximum(wt_ctrl, 1e-3))
            ratio = np.ones(n_p) if wt_ratio is None else np.clip(wt_ratio, 0.05, 5.0)
            log_S_hsp90i = np.log(np.maximum(ratio * wt_ctrl, 1e-3))
        parts.extend([log_S_dmso, log_S_hsp90i])

    parts.append(np.array([np.log(0.5)]))   # log_b (global)
    if dims.per_protein_b:
        parts.append(np.zeros(n_p))          # log_b_dev (start at 0 = no deviation)
    # h_raw such that h_max * sigmoid(h_raw) = 0.05 (so 0.05 = h_max / (1+e^-r))
    h_raw_init = float(np.log(0.05 / max(dims.h_max - 0.05, 1e-3)))
    parts.append(np.array([h_raw_init]))
    if dims.fit_omega_h:
        # omega_h_raw such that omega_h_max * sigmoid(omega_h_raw) = 0.02
        omega_h_init = float(np.log(0.02 / max(dims.omega_h_max - 0.02, 1e-3)))
        parts.append(np.array([omega_h_init]))

    theta = np.concatenate(parts)
    theta = theta + jitter * rng.standard_normal(theta.shape)
    return theta
