"""
cost.py
=======
Evaluates the negative log-posterior cost function J, its gradient dJ,
and the Gauss-Newton Hessian approximation d2J, for both the full
parameter vector b and the variable-projection (VarPro) formulation
over nonlinear parameters x = [gamma, beta, ...] only.

Two cost functions are provided:

    calculate_cost(signals, Ce, b, bp, Cp, T_c, prior_cfg)
        Full cost over b = [n, gamma, beta, ...].
        Used for Laplace covariance computation and final model scoring.

    calculate_cost_varpro(signals, Ce, x, bp, Cp, T_c, prior_cfg)
        VarPro cost over x = [gamma, beta, ...] only.
        Amplitudes n are solved analytically at each call via a MAP
        linear solve, reducing the nonlinear search from 3N to 2N dims.
        Used during the inner MAP optimisation loop.

Both are thin Python dispatchers around a jitted `_core` function: the
dispatcher selects `signals[data_level]` and slices the valid region
in plain Python, so that `data_level` (a string) and `valid` (an int
used as a slice bound) never enter the jit trace. A single
`static_argnums` cannot mark only *some* leaves of a dict-valued
argument as static, so `signals` cannot be passed directly into a
jitted function that slices by `valid`; splitting into a dispatcher +
core avoids that entirely.

Cost function structure (eq. 15, Yoko & Polifke 2026):

    J = J_prior + J_LFL + J_like

    J_prior = 0.5 * (b - bp)^T Cp^{-1} (b - bp)
              + 0.5 * log|2 pi Cp|

    J_LFL   = 0.5 * (sum(n) - LFL)^2 / LFL_sigma^2
              + 0.5 * log(2 pi LFL_sigma^2)      [only if LFL is set]

    J_like  = 0.5 * r^T r / Ce
              + 0.5 * Nd * log(2 pi Ce)

    where r = conv(u, h, 'valid') * dt - q   is the output residual.

The MacKay MML noise estimate (eq. 24-25) is also computed at each call:

    Ce_ME = r^T r / (Nd - gamma)
    gamma = P - trace(H^{-1} Cp^{-1})           effective parameter count

References:
    Yoko & Polifke (2026), Sections 3.1-3.3
    MacKay (1999), Neural Computation 11(5)
"""

import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
from typing import Optional

from core.impulse_response import calculate_impulse_response
from core.prior import PriorConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_convolution(u: jnp.ndarray,
                        h: jnp.ndarray,
                        dt: float) -> jnp.ndarray:
    """
    Compute the 'valid' discrete convolution p = conv(u, h) * dt.

    JAX's jnp.convolve returns the full convolution by default.
    The 'valid' region has length len(u) - len(h) + 1, corresponding
    to output samples for which all impulse response taps are supported
    by observed input data.

    Parameters
    ----------
    u  : jnp.ndarray, shape (M,)   Input signals.
    h  : jnp.ndarray, shape (L,)   Impulse response.
    dt : float                      Sampling interval [s].

    Returns
    -------
    p : jnp.ndarray, shape (M - L + 1,)
        Predicted output over the valid region.
    """
    full = jnp.convolve(u, h, mode='full')
    L    = h.shape[0]
    # 'valid' corresponds to indices L-1 : M in the full convolution
    p    = full[L - 1 : u.shape[0]] * dt
    return p


def _mackay_noise_estimate(r: jnp.ndarray,
                            d2J: jnp.ndarray,
                            inv_Cp_diag: jnp.ndarray,
                            Nd: int,
                            Nb: int) -> jnp.ndarray:
    """
    Compute the MacKay MML estimate of the noise variance Ce.

    Ce_ME = r^T r / (Nd - gamma)
    gamma = Nb - trace(H^{-1} * Cp^{-1})

    where H = d2J is the Gauss-Newton Hessian. Matches calculateCost.m /
    calculateCost_VarPro.m exactly: no clamping of gamma is applied.

    Parameters
    ----------
    r            : jnp.ndarray, shape (Nd,)   Residual vector.
    d2J          : jnp.ndarray, shape (Nb,Nb) Gauss-Newton Hessian.
    inv_Cp_diag  : jnp.ndarray, shape (Nb,)   Diagonal of Cp^{-1}.
    Nd           : int   Number of valid data points.
    Nb           : int   Number of parameters.

    Returns
    -------
    Ce_ME : jnp.ndarray, scalar
    """
    inv_Cp = jnp.diag(inv_Cp_diag)
    X      = jax.scipy.linalg.solve(d2J, inv_Cp, assume_a='pos')
    gamma  = Nb - jnp.trace(X)
    Ce_ME  = (r @ r) / (Nd - gamma)
    return Ce_ME


# ---------------------------------------------------------------------------
# Full cost:  J(b)
# ---------------------------------------------------------------------------

def calculate_cost(signals: dict,
                   Ce: float,
                   b: jnp.ndarray,
                   bp: jnp.ndarray,
                   Cp: jnp.ndarray,
                   T_c: float,
                   prior_cfg: PriorConfig,
                   data_level: str = 'coarse') -> tuple:
    """
    Evaluate the negative log-posterior and its derivatives w.r.t. b.

    Parameters
    ----------
    signals    : dict
        Output of prepare_signals. Uses signals[data_level] sub-dict.
    Ce         : float
        Data noise variance.
    b          : jnp.ndarray, shape (3N,)
        Full parameter vector.
    bp         : jnp.ndarray, shape (3N,)
        Prior mean.
    Cp         : jnp.ndarray, shape (3N, 3N)
        Prior covariance (diagonal).
    T_c        : float
        Convective timescale [s].
    prior_cfg  : PriorConfig
        Prior configuration (static - used for LFL and noise).
    data_level : str
        'coarse' or 'fine'. Default 'coarse'.

    Returns
    -------
    J      : jnp.ndarray, scalar   Total negative log-posterior.
    dJ     : jnp.ndarray, (3N,)   Gradient w.r.t. b.
    d2J    : jnp.ndarray, (3N,3N) Gauss-Newton Hessian.
    Ce_ME  : jnp.ndarray, scalar   MML noise estimate.
    J_like : jnp.ndarray, scalar   Likelihood contribution to J.
    p      : jnp.ndarray, (Nd,)   Predicted output over valid region.
    """
    sig    = signals[data_level]
    u      = sig['u']
    q_v    = sig['q'][sig['valid']:]        # truncate to valid region
    t_h_nd = sig['t_h'] / T_c               # non-dimensional impulse response time
    dt     = sig['dt']

    return _calculate_cost_core(u, q_v, t_h_nd, dt, Ce, b, bp, Cp, T_c, prior_cfg)


@partial(jit, static_argnums=(9,))
def _calculate_cost_core(u: jnp.ndarray,
                          q_v: jnp.ndarray,
                          t_h_nd: jnp.ndarray,
                          dt: float,
                          Ce: float,
                          b: jnp.ndarray,
                          bp: jnp.ndarray,
                          Cp: jnp.ndarray,
                          T_c: float,
                          prior_cfg: PriorConfig) -> tuple:
    """Jitted core of calculate_cost. See calculate_cost for parameter docs."""
    Nd = q_v.shape[0]
    Nb = b.shape[0]

    # Precompute diagonal of Cp^{-1}
    Cp_diag     = jnp.diag(Cp)
    inv_Cp_diag = 1.0 / Cp_diag

    # -----------------------------------------------------------------------
    # Prior cost
    # -----------------------------------------------------------------------
    diff    = b - bp
    J_prior = 0.5 * jnp.dot(diff, inv_Cp_diag * diff)
    J_prior = J_prior + 0.5 * jnp.sum(jnp.log(2.0 * jnp.pi * Cp_diag))

    dJ_prior = inv_Cp_diag * diff                          # (3N,)
    d2J_prior = jnp.diag(inv_Cp_diag)                     # (3N, 3N)

    # -----------------------------------------------------------------------
    # LFL soft constraint
    # -----------------------------------------------------------------------
    J_LFL  = jnp.zeros(())
    dJ_LFL = jnp.zeros(Nb)
    d2J_LFL = jnp.zeros((Nb, Nb))

    if prior_cfg.LFL is not None:
        n_idx   = jnp.arange(b.shape[0] // 3) * 3         # 0, 3, 6, ...
        n_vals  = b[n_idx]
        err     = jnp.sum(n_vals) - prior_cfg.LFL
        s2      = prior_cfg.LFL_sigma ** 2

        J_LFL   = 0.5 * err ** 2 / s2 + 0.5 * jnp.log(2.0 * jnp.pi * s2)

        ones_n  = jnp.ones(n_vals.shape[0])
        dJ_n    = (err / s2) * ones_n                      # (N,)
        dJ_LFL  = dJ_LFL.at[n_idx].set(dJ_n)

        d2J_block = (1.0 / s2) * jnp.outer(ones_n, ones_n)   # (N, N)
        d2J_LFL   = d2J_LFL.at[jnp.ix_(n_idx, n_idx)].set(d2J_block)

    # -----------------------------------------------------------------------
    # Data likelihood
    # -----------------------------------------------------------------------
    h, dhdb, _, _ = calculate_impulse_response(b, t_h_nd, T_c)

    # Predicted output over valid region: conv(u, h, 'valid') * dt
    p    = _valid_convolution(u, h, dt)                    # (Nd,)
    r    = p - q_v                                         # residual

    # Jacobian of p w.r.t. b: each column is conv(u, dhdb[:,k], 'valid')*dt
    # Vectorised over parameter columns using vmap
    def conv_col(dhdb_col):
        return _valid_convolution(u, dhdb_col, dt)

    dpdb = jax.vmap(conv_col, in_axes=1, out_axes=1)(dhdb)  # (Nd, 3N)

    J_like  = 0.5 * jnp.dot(r, r) / Ce
    J_like  = J_like + 0.5 * Nd * jnp.log(2.0 * jnp.pi * Ce)

    dJ_like  = dpdb.T @ (r / Ce)                           # (3N,)
    d2J_like = (dpdb.T @ dpdb) / Ce                        # (3N, 3N)

    # -----------------------------------------------------------------------
    # Totals
    # -----------------------------------------------------------------------
    J   = J_prior + J_LFL + J_like
    dJ  = dJ_prior + dJ_LFL + dJ_like
    d2J = d2J_prior + d2J_LFL + d2J_like

    # -----------------------------------------------------------------------
    # MacKay MML noise estimate
    # -----------------------------------------------------------------------
    Ce_ME = _mackay_noise_estimate(r, d2J, inv_Cp_diag, Nd, Nb)

    return J, dJ, d2J, Ce_ME, J_like, p


# ---------------------------------------------------------------------------
# VarPro cost:  J(x),  n solved analytically
# ---------------------------------------------------------------------------

def calculate_cost_varpro(signals: dict,
                           Ce: float,
                           x: jnp.ndarray,
                           bp: jnp.ndarray,
                           Cp: jnp.ndarray,
                           T_c: float,
                           prior_cfg: PriorConfig,
                           data_level: str = 'coarse') -> tuple:
    """
    Variable-projection cost over nonlinear parameters x = [gamma, beta, ...].

    For fixed x, amplitudes n are solved analytically via MAP linear regression,
    reducing the nonlinear search from 3N to 2N dimensions.

    Parameters
    ----------
    signals    : dict    - Prepared signals struct.
    Ce         : float   - Data noise variance.
    x          : jnp.ndarray, shape (2N,)
                 Nonlinear parameter vector [gamma_1, beta_1, gamma_2, ...].
    bp         : jnp.ndarray, shape (3N,)  - Prior mean (full).
    Cp         : jnp.ndarray, shape (3N, 3N) - Prior covariance (full, diagonal).
    T_c        : float   - Convective timescale [s].
    prior_cfg  : PriorConfig  (static)
    data_level : str     - 'coarse' or 'fine'.

    Returns
    -------
    J      : scalar   Total negative log-posterior at projected solution.
    dJ     : (2N,)    Gradient w.r.t. x.
    d2J    : (2N,2N)  Gauss-Newton Hessian w.r.t. x.
    Ce_ME  : scalar   MML noise estimate.
    J_like : scalar   Likelihood contribution.
    p      : (Nd,)    Predicted output.
    b_out  : (3N,)    Full parameter vector with projected n.
    """
    sig    = signals[data_level]
    u      = sig['u']
    q_v    = sig['q'][sig['valid']:]
    t_h_nd = sig['t_h'] / T_c
    dt     = sig['dt']

    return _calculate_cost_varpro_core(
        u, q_v, t_h_nd, dt, Ce, x, bp, Cp, T_c, prior_cfg)


@partial(jit, static_argnums=(9,))
def _calculate_cost_varpro_core(u: jnp.ndarray,
                                 q_v: jnp.ndarray,
                                 t_h_nd: jnp.ndarray,
                                 dt: float,
                                 Ce: float,
                                 x: jnp.ndarray,
                                 bp: jnp.ndarray,
                                 Cp: jnp.ndarray,
                                 T_c: float,
                                 prior_cfg: PriorConfig) -> tuple:
    """Jitted core of calculate_cost_varpro. See calculate_cost_varpro for parameter docs."""
    Nd = q_v.shape[0]
    Nx = x.shape[0]
    N  = Nx // 2
    Nb = 3 * N

    # Index helpers (static, computed from N at trace time)
    # Interleaved layout: b = [n1, g1, b1, n2, g2, b2, ...]
    # x layout:               [g1, b1, g2, b2, ...]
    idx_n_b     = jnp.arange(N) * 3        # 0, 3, 6  (n in b)
    idx_gamma_b = jnp.arange(N) * 3 + 1   # 1, 4, 7  (gamma in b)
    idx_beta_b  = jnp.arange(N) * 3 + 2   # 2, 5, 8  (beta in b)
    idx_gamma_x = jnp.arange(N) * 2        # 0, 2, 4  (gamma in x)
    idx_beta_x  = jnp.arange(N) * 2 + 1   # 1, 3, 5  (beta in x)

    # Extract sub-priors
    np_  = bp[idx_n_b]                    # prior mean for n,     (N,)
    xp   = jnp.zeros(Nx)
    xp   = xp.at[idx_gamma_x].set(bp[idx_gamma_b])
    xp   = xp.at[idx_beta_x ].set(bp[idx_beta_b ])

    Cp_diag     = jnp.diag(Cp)
    Cn_diag     = Cp_diag[idx_n_b]        # prior var for n
    Cx_diag     = jnp.zeros(Nx)
    Cx_diag     = Cx_diag.at[idx_gamma_x].set(Cp_diag[idx_gamma_b])
    Cx_diag     = Cx_diag.at[idx_beta_x ].set(Cp_diag[idx_beta_b ])

    inv_Cn = 1.0 / Cn_diag               # (N,)
    inv_Cx = 1.0 / Cx_diag               # (2N,)

    # -----------------------------------------------------------------------
    # Build b_tmp with n=0 to get Gaussian shape matrix G
    # -----------------------------------------------------------------------
    b_tmp = jnp.zeros(Nb)
    b_tmp = b_tmp.at[idx_gamma_b].set(x[idx_gamma_x])
    b_tmp = b_tmp.at[idx_beta_b ].set(x[idx_beta_x ])
    # n is zero in b_tmp; we only need G from calculateImpulseResponse
    _, _, G, _ = calculate_impulse_response(b_tmp, t_h_nd, T_c)   # (T_h, N)

    # -----------------------------------------------------------------------
    # Build design matrix A: A[:,i] = conv(u, G[:,i], 'valid') * dt
    # -----------------------------------------------------------------------
    def conv_basis(g_col):
        return _valid_convolution(u, g_col, dt)

    A = jax.vmap(conv_basis, in_axes=1, out_axes=1)(G)    # (Nd, N)

    # -----------------------------------------------------------------------
    # MAP linear solve for n given x
    #   H_n * n = g_n
    #   H_n = A^T A / Ce + diag(inv_Cn)
    #   g_n = A^T q / Ce + diag(inv_Cn) * np_
    # -----------------------------------------------------------------------
    Hn = (A.T @ A) / Ce + jnp.diag(inv_Cn)               # (N, N)
    gn = (A.T @ q_v) / Ce + inv_Cn * np_                 # (N,)

    if prior_cfg.LFL is not None:
        s2  = prior_cfg.LFL_sigma ** 2
        Hn  = Hn + (1.0 / s2) * jnp.ones((N, N))
        gn  = gn + (prior_cfg.LFL / s2) * jnp.ones(N)

    n_MAP = jnp.linalg.solve(Hn, gn)                     # (N,)

    # Assemble full b_out
    b_out = b_tmp.at[idx_n_b].set(n_MAP)

    # -----------------------------------------------------------------------
    # Prior cost (evaluated at b_out)
    # -----------------------------------------------------------------------
    diff_b  = b_out - bp
    J_prior = 0.5 * jnp.dot(diff_b, (1.0 / Cp_diag) * diff_b)
    J_prior = J_prior + 0.5 * jnp.sum(jnp.log(2.0 * jnp.pi * Cp_diag))

    # Gradient w.r.t. x only (n terms vanish at n_MAP by construction)
    dJ_prior = inv_Cx * (x - xp)                         # (2N,)

    # -----------------------------------------------------------------------
    # LFL cost (gradient w.r.t. x is zero - LFL only couples to n)
    # -----------------------------------------------------------------------
    J_LFL = jnp.zeros(())
    if prior_cfg.LFL is not None:
        err   = jnp.sum(n_MAP) - prior_cfg.LFL
        s2    = prior_cfg.LFL_sigma ** 2
        J_LFL = 0.5 * err ** 2 / s2 + 0.5 * jnp.log(2.0 * jnp.pi * s2)

    # -----------------------------------------------------------------------
    # Likelihood cost (w.r.t. x via dhdb columns for gamma and beta only)
    # -----------------------------------------------------------------------
    _, dhdb, _, _ = calculate_impulse_response(b_out, t_h_nd, T_c)

    p = A @ n_MAP                                         # (Nd,) via design matrix

    r = p - q_v                                           # (Nd,)

    # Jacobian of p w.r.t. x: only gamma and beta columns of dhdb
    gamma_cols = idx_gamma_b                              # columns in dhdb
    beta_cols  = idx_beta_b

    def conv_col(col):
        return _valid_convolution(u, col, dt)

    dpdgamma = jax.vmap(conv_col, in_axes=1, out_axes=1)(
        dhdb[:, gamma_cols])                              # (Nd, N)
    dpdbeta  = jax.vmap(conv_col, in_axes=1, out_axes=1)(
        dhdb[:, beta_cols])                               # (Nd, N)

    # Interleave into dpdx shape (Nd, 2N)
    dpdx = jnp.zeros((Nd, Nx))
    dpdx = dpdx.at[:, idx_gamma_x].set(dpdgamma)
    dpdx = dpdx.at[:, idx_beta_x ].set(dpdbeta)

    J_like  = 0.5 * jnp.dot(r, r) / Ce
    J_like  = J_like + 0.5 * Nd * jnp.log(2.0 * jnp.pi * Ce)

    dJ_like  = dpdx.T @ (r / Ce)                         # (2N,)
    d2J_like = (dpdx.T @ dpdx) / Ce                      # (2N, 2N)

    # -----------------------------------------------------------------------
    # Totals
    # -----------------------------------------------------------------------
    J   = J_prior + J_LFL + J_like
    dJ  = dJ_prior + dJ_like
    d2J = jnp.diag(inv_Cx) + d2J_like                    # (2N, 2N)

    # -----------------------------------------------------------------------
    # MacKay MML noise estimate (separate gamma_x and gamma_n contributions,
    # matching calculateCost_VarPro.m exactly: no clamping is applied)
    # -----------------------------------------------------------------------
    X_n     = jnp.linalg.solve(Hn, jnp.diag(inv_Cn))
    gamma_n = N - jnp.trace(X_n)

    X_x     = jax.scipy.linalg.solve(d2J, jnp.diag(inv_Cx), assume_a='pos')
    gamma_x = Nx - jnp.trace(X_x)

    gamma_tot = gamma_n + gamma_x
    Ce_ME     = (r @ r) / (Nd - gamma_tot)

    return J, dJ, d2J, Ce_ME, J_like, p, b_out


# ---------------------------------------------------------------------------
# Finite-difference Hessian  (fallback / validation)
# ---------------------------------------------------------------------------

def fd_hessian(signals: dict,
               Ce: float,
               b: jnp.ndarray,
               bp: jnp.ndarray,
               Cp: jnp.ndarray,
               T_c: float,
               prior_cfg: PriorConfig,
               eps: float = 1e-5,
               data_level: str = 'coarse') -> jnp.ndarray:
    """
    Compute the Hessian of J w.r.t. b using central finite differences.

    Used as a fallback when the Gauss-Newton approximation is inaccurate,
    and for validation of the analytic Hessian.

    Parameters
    ----------
    eps : float   Finite difference step size.

    Returns
    -------
    H : jnp.ndarray, shape (3N, 3N)
    """
    def neg_log_post(b_):
        J, _, _, _, _, _ = calculate_cost(
            signals, Ce, b_, bp, Cp, T_c, prior_cfg, data_level)
        return J

    # Use JAX's own Hessian for accuracy and simplicity
    H = jax.hessian(neg_log_post)(b)
    return H
