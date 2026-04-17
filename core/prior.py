"""
prior.py
========
Constructs the Gaussian prior distribution over the parameter vector b
for a given model order N, and estimates the maximum non-dimensional
impulse response horizon t_max used for signal preparation.

Parameter ordering within b:
    b = [n_1, gamma_1, beta_1, n_2, gamma_2, beta_2, ...]

Prior structure (all pulses share the same hyperparameters):
    p(n_i)     = N(mu_n,     sig_n^2)
    p(gamma_i) = N(mu_gamma, sig_gamma^2)
    p(beta_i)  = N(mu_beta,  sig_beta^2)

The t_max estimate uses the Fenton-Wilkinson (FW) log-normal approximation
to bound the 99th percentile of the sum of N log-normally distributed
delay gaps, plus the 99th percentile of the largest probable pulse width.
This ensures a fair, order-dependent support for Bayesian model comparison
across different values of N.

References:
    Yoko & Polifke (2026), Sections 3.1.1, 4.1, and 4.4
    Fenton (1960), 'The sum of log-normal probability distributions in
        scatter transmission systems', IRE Trans. Commun. Syst.
"""

import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Default prior hyperparameters  (Table / eqs. 30-32 in Yoko & Polifke 2026)
# ---------------------------------------------------------------------------

@dataclass
class PriorConfig:
    """
    Hyperparameters defining the Gaussian prior in parameter space.

    Attributes
    ----------
    mu_n : float
        Prior mean for amplitude n_i.
    sig_n : float
        Prior std for amplitude n_i.
    mu_gamma : float
        Prior mean for log-delay-gap gamma_i.
    sig_gamma : float
        Prior std for log-delay-gap gamma_i.
    mu_beta : float
        Prior mean for log-width beta_i.
    sig_beta : float
        Prior std for log-width beta_i.
    T_h : float or None
        Fixed impulse response support [s].
        If None, t_max is estimated from the prior via Fenton-Wilkinson.
    LFL : float or None
        Low-frequency limit (gain constraint).  None = unconstrained.
    LFL_sigma : float
        Std of the soft Gaussian prior on sum(n_i) - LFL.
    """
    mu_n     : float = 0.0
    sig_n    : float = 1.0
    mu_gamma : float = 0.0
    sig_gamma: float = 0.5
    mu_beta  : float = -1.8
    sig_beta : float = 0.5
    T_h      : Optional[float] = None
    LFL      : Optional[float] = None
    LFL_sigma: float = 1e-3


# ---------------------------------------------------------------------------
# Fenton-Wilkinson t_max estimation
# ---------------------------------------------------------------------------

def _fenton_wilkinson_99(N: int,
                          mu_gamma: float,
                          var_gamma: float) -> float:
    """
    Estimate the 99th percentile of the sum of N i.i.d. log-normal variables
    using the Fenton-Wilkinson moment-matching approximation.

    Each gap alpha_i = exp(gamma_i) is log-normal with:
        E[alpha_i]   = exp(mu_gamma + 0.5 * var_gamma)
        Var[alpha_i] = (exp(var_gamma) - 1) * exp(2*mu_gamma + var_gamma)

    The sum S = sum_{i=1}^{N} alpha_i is approximated as log-normal
    by matching E[S] and Var[S].

    Parameters
    ----------
    N         : int   - Number of delay gaps (model order).
    mu_gamma  : float - Prior mean of gamma_i.
    var_gamma : float - Prior variance of gamma_i.

    Returns
    -------
    S99 : float
        99th percentile of the sum, in non-dimensional units.
    """
    n99 = 2.326   # standard normal quantile for 99th percentile

    # Moments of the sum (non-dimensional, T_c factored out)
    ES  = N * np.exp(mu_gamma + 0.5 * var_gamma)
    VS  = N * (np.exp(var_gamma) - 1.0) * np.exp(2.0 * mu_gamma + var_gamma)

    # Fenton-Wilkinson log-normal parameters for the sum
    sig_FW = np.sqrt(np.log(1.0 + VS / ES ** 2))
    mu_FW  = np.log(ES) - 0.5 * sig_FW ** 2

    # 99th percentile
    S99 = np.exp(mu_FW + n99 * sig_FW)
    return S99


def estimate_t_max(N: int, prior_cfg: PriorConfig) -> float:
    """
    Estimate the maximum non-dimensional impulse response horizon t_max
    for a model of order N, given the prior hyperparameters.

    t_max is chosen such that 99% of impulse responses drawn from the
    prior lie within [0, t_max * T_c].  This ensures a fair, order-
    dependent support for Bayesian model comparison (Section 4.4).

    Parameters
    ----------
    N         : int         - Model order (number of Gaussian pulses).
    prior_cfg : PriorConfig - Prior hyperparameters.

    Returns
    -------
    t_max : float
        Maximum non-dimensional time horizon (multiply by T_c to get seconds).
    """
    if prior_cfg.T_h is not None:
        raise ValueError(
            "T_h is fixed in prior_cfg; call estimate_t_max only when "
            "T_h is None (automatic support selection)."
        )

    n99       = 2.326
    var_gamma = prior_cfg.sig_gamma ** 2
    var_beta  = prior_cfg.sig_beta  ** 2

    # 99th percentile of the total non-dimensional delay (sum of N gaps)
    S99 = _fenton_wilkinson_99(N, prior_cfg.mu_gamma, var_gamma)

    # 99th percentile of the largest probable non-dimensional width
    sig_max = np.exp(prior_cfg.mu_beta + n99 * np.sqrt(var_beta))

    # Conservative horizon: latest delay + spread of the last pulse
    t_max = S99 + n99 * sig_max
    return float(t_max)


# ---------------------------------------------------------------------------
# Prior construction
# ---------------------------------------------------------------------------

def generate_prior(N: int,
                   T_c: float,
                   prior_cfg: PriorConfig
                   ) -> tuple:
    """
    Build the Gaussian prior mean and covariance for a model of order N.

    The prior is diagonal in parameter space: all cross-parameter
    correlations are set to zero (Section 3.1.1).

    Parameters
    ----------
    N         : int         - Number of Gaussian pulses (model order).
    T_c       : float       - Convective timescale [s].
    prior_cfg : PriorConfig - Prior hyperparameters.

    Returns
    -------
    bp     : jnp.ndarray, shape (3N,)
        Prior mean vector in interleaved order [n, gamma, beta, n, gamma, ...].
    Cp     : jnp.ndarray, shape (3N, 3N)
        Prior covariance matrix (diagonal).
    names  : list of str, length 3N
        Human-readable parameter names for reporting / plotting.
    t_max  : float
        Maximum non-dimensional impulse response horizon.
        If prior_cfg.T_h is set, returns T_h / T_c directly.
        Otherwise uses the Fenton-Wilkinson estimate.
    """
    # --- Prior mean: tile [mu_n, mu_gamma, mu_beta] N times ---
    mu_single = np.array([prior_cfg.mu_n,
                           prior_cfg.mu_gamma,
                           prior_cfg.mu_beta])          # (3,)
    bp = np.tile(mu_single, N)                          # (3N,)

    # --- Prior std: tile [sig_n, sig_gamma, sig_beta] N times ---
    sig_single = np.array([prior_cfg.sig_n,
                            prior_cfg.sig_gamma,
                            prior_cfg.sig_beta])         # (3,)
    sig_vec = np.tile(sig_single, N)                    # (3N,)
    Cp = np.diag(sig_vec ** 2)                          # (3N, 3N)

    # --- Convert to JAX arrays ---
    bp = jnp.array(bp, dtype=jnp.float32)
    Cp = jnp.array(Cp, dtype=jnp.float32)

    # --- Parameter names ---
    names = []
    for i in range(1, N + 1):
        names += [f'n_{i}', f'gamma_{i}', f'beta_{i}']

    # --- t_max ---
    if prior_cfg.T_h is not None:
        t_max = prior_cfg.T_h / T_c
    else:
        t_max = estimate_t_max(N, prior_cfg)

    return bp, Cp, names, t_max


# ---------------------------------------------------------------------------
# Prior log-probability  (used inside calculate_cost and ELBO)
# ---------------------------------------------------------------------------

def log_prior(b: jnp.ndarray,
              bp: jnp.ndarray,
              Cp: jnp.ndarray,
              prior_cfg: PriorConfig) -> jnp.ndarray:
    """
    Evaluate the log prior probability of parameter vector b.

    Includes the optional soft low-frequency limit (LFL) constraint:
        p(sum(n_i) = LFL) ~ N(LFL, LFL_sigma^2)

    Parameters
    ----------
    b         : jnp.ndarray, shape (3N,)
    bp        : jnp.ndarray, shape (3N,)  - Prior mean.
    Cp        : jnp.ndarray, shape (3N,)  - Prior covariance (diagonal).
    prior_cfg : PriorConfig

    Returns
    -------
    lp : jnp.ndarray, scalar
        Log prior (up to an additive constant).
    """
    inv_diag = 1.0 / jnp.diag(Cp)                     # (3N,)
    diff     = b - bp                                  # (3N,)

    # Gaussian prior: -0.5 * (b - bp)^T * Cp^{-1} * (b - bp)
    lp = -0.5 * jnp.dot(diff, inv_diag * diff)

    # Optional LFL soft constraint
    if prior_cfg.LFL is not None:
        n_vals  = b[0::3]                              # amplitudes only
        err     = jnp.sum(n_vals) - prior_cfg.LFL
        lp      = lp - 0.5 * err ** 2 / prior_cfg.LFL_sigma ** 2

    return lp


# ---------------------------------------------------------------------------
# Convenience: extract nonlinear prior (gamma, beta only)
# ---------------------------------------------------------------------------

def extract_nonlinear_prior(bp: jnp.ndarray,
                             Cp: jnp.ndarray,
                             N: int) -> tuple:
    """
    Extract the prior over the nonlinear parameters x = [gamma, beta, ...]
    from the full prior over b = [n, gamma, beta, ...].

    Used by the variable-projection optimizer, which only iterates over x.

    Parameters
    ----------
    bp : jnp.ndarray, shape (3N,)
    Cp : jnp.ndarray, shape (3N, 3N)
    N  : int

    Returns
    -------
    xp     : jnp.ndarray, shape (2N,)  - Prior mean over x.
    Cx_diag: jnp.ndarray, shape (2N,)  - Prior variance over x (diagonal).
    """
    # Indices of gamma and beta in b
    idx_gamma = jnp.arange(N) * 3 + 1    # 1, 4, 7, ...
    idx_beta  = jnp.arange(N) * 3 + 2    # 2, 5, 8, ...

    # Interleave: x = [gamma_1, beta_1, gamma_2, beta_2, ...]
    idx_x = jnp.stack([idx_gamma, idx_beta], axis=1).ravel()   # (2N,)

    xp      = bp[idx_x]
    Cx_diag = jnp.diag(Cp)[idx_x]

    return xp, Cx_diag