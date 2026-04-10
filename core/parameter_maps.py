"""
parameter_maps.py
=================
Maps between the unconstrained parameter space used during optimization
and the physical parameter space of the Gaussian pulse model.

Parameter space (b):
    b = [n_1, gamma_1, beta_1, n_2, gamma_2, beta_2, ...]
    - n_i     : pulse amplitude (unconstrained)
    - gamma_i : log of non-dimensional delay gap,  gamma_i = log(Delta_tau_i / T_c)
    - beta_i  : log of non-dimensional width,      beta_i  = log(sigma_i / T_c)

Physical space (a):
    a = [n_1, tau_1, sigma_1, n_2, tau_2, sigma_2, ...]
    - n_i     : pulse amplitude
    - tau_i   : cumulative time delay,  tau_i = T_c * sum_{k=1}^{i} exp(gamma_k)
    - sigma_i : pulse width,            sigma_i = T_c * exp(beta_i)

The cumulative-sum structure of tau enforces strict ordering tau_1 < tau_2 < ...
and guarantees all delays and widths are strictly positive without hard constraints.

References:
    Yoko & Polifke (2026), Section 4.1
"""

import jax
import jax.numpy as jnp
from jax import jit
from functools import partial


# ---------------------------------------------------------------------------
# Forward map: parameter space -> physical space
# ---------------------------------------------------------------------------

@jit
def map_to_physical(b: jnp.ndarray, T_c: float) -> tuple:
    """
    Map parameter vector b to physical parameters a.

    Parameters
    ----------
    b : jnp.ndarray, shape (3N,)
        Parameter vector [n_1, gamma_1, beta_1, n_2, gamma_2, beta_2, ...].
    T_c : float
        Convective timescale used for non-dimensionalisation.

    Returns
    -------
    a : jnp.ndarray, shape (3N,)
        Physical parameter vector [n_1, tau_1, sig_1, n_2, tau_2, sig_2, ...].
    params : dict
        Dictionary with keys 'n', 'tau', 'sigma', each shape (N,).
    jac : dict
        Dictionary of Jacobian blocks:
            'dn_dbn'       : (N, N)  d(n)   / d(n)     = I
            'dtau_dgamma'  : (N, N)  d(tau) / d(gamma) = lower triangular
            'dsig_dbeta'   : (N, N)  d(sig) / d(beta)  = diagonal
    """
    b = b.ravel()
    N = b.shape[0] // 3

    # Unpack interleaved parameter vector
    b_n     = b[0::3]   # shape (N,)
    b_gamma = b[1::3]   # shape (N,)
    b_beta  = b[2::3]   # shape (N,)

    # --- Forward maps ---
    a_n     = b_n
    a_gaps  = jnp.exp(b_gamma) * T_c          # Delta_tau_i in physical time
    a_tau   = jnp.cumsum(a_gaps)              # cumulative delays, strictly ordered
    a_sigma = jnp.exp(b_beta) * T_c           # widths, strictly positive

    # Pack into interleaved physical vector
    a = jnp.stack([a_n, a_tau, a_sigma], axis=1).ravel()  # (3N,)

    params = {'n': a_n, 'tau': a_tau, 'sigma': a_sigma}

    # --- Jacobian blocks ---
    # d(n_i) / d(n_j) = delta_ij
    dn_dbn = jnp.eye(N)

    # d(tau_i) / d(gamma_j):
    #   tau_i = T_c * sum_{k=1}^{i} exp(gamma_k)
    #   => d(tau_i)/d(gamma_j) = T_c * exp(gamma_j)  if j <= i, else 0
    # This is a lower-triangular matrix where each column j has
    # exp(gamma_j)*T_c in rows j, j+1, ..., N-1
    j_idx = jnp.arange(N)
    i_idx = jnp.arange(N)
    # entry (i,j) = exp(gamma_j)*T_c * (j <= i)
    lower_mask = (i_idx[:, None] >= j_idx[None, :])   # (N, N) bool
    dtau_dgamma = lower_mask * (jnp.exp(b_gamma) * T_c)[None, :]  # (N, N)

    # d(sigma_i) / d(beta_j) = delta_ij * exp(beta_i) * T_c
    dsig_dbeta = jnp.diag(jnp.exp(b_beta) * T_c)

    jac = {
        'dn_dbn'      : dn_dbn,
        'dtau_dgamma' : dtau_dgamma,
        'dsig_dbeta'  : dsig_dbeta,
    }

    return a, params, jac


@jit
def map_to_physical_covariance(b: jnp.ndarray, T_c: float,
                                Cb: jnp.ndarray) -> jnp.ndarray:
    """
    Propagate posterior covariance from parameter space to physical space.

    Uses first-order uncertainty propagation:
        Ca = J_full @ Cb @ J_full.T

    where J_full is the (3N x 3N) full Jacobian of the map b -> a,
    assembled from the three diagonal blocks (n, tau, sigma are decoupled
    in b-space).

    Parameters
    ----------
    b : jnp.ndarray, shape (3N,)
        MAP parameter vector.
    T_c : float
        Convective timescale.
    Cb : jnp.ndarray, shape (3N, 3N)
        Posterior covariance in parameter space.

    Returns
    -------
    Ca : jnp.ndarray, shape (3N, 3N)
        Posterior covariance in physical space.
    """
    b = b.ravel()
    N = b.shape[0] // 3
    Nb = 3 * N

    _, _, jac = map_to_physical(b, T_c)

    # Index arrays for interleaved layout
    idx_n     = jnp.arange(N) * 3       # 0, 3, 6, ...
    idx_gamma = jnp.arange(N) * 3 + 1   # 1, 4, 7, ...
    idx_beta  = jnp.arange(N) * 3 + 2   # 2, 5, 8, ...

    # Assemble full (3N x 3N) Jacobian
    J = jnp.zeros((Nb, Nb))
    J = J.at[jnp.ix_(idx_n,     idx_n    )].set(jac['dn_dbn'])
    J = J.at[jnp.ix_(idx_gamma, idx_gamma)].set(jac['dtau_dgamma'])
    J = J.at[jnp.ix_(idx_beta,  idx_beta )].set(jac['dsig_dbeta'])

    Ca = J @ Cb @ J.T
    return Ca


# ---------------------------------------------------------------------------
# Inverse map: physical space -> parameter space
# ---------------------------------------------------------------------------

@jit
def map_to_parameter(a: jnp.ndarray, T_c: float) -> jnp.ndarray:
    """
    Map physical parameter vector a back to parameter space b.

    Inverts the cumulative delay structure and applies log transforms
    to recover unconstrained parameters.

    Parameters
    ----------
    a : jnp.ndarray, shape (3N,)
        Physical parameter vector [n_1, tau_1, sig_1, n_2, tau_2, sig_2, ...].
    T_c : float
        Convective timescale.

    Returns
    -------
    b : jnp.ndarray, shape (3N,)
        Parameter vector [n_1, gamma_1, beta_1, n_2, gamma_2, beta_2, ...].
    """
    a = a.ravel()

    # Unpack
    a_n     = a[0::3]   # shape (N,)
    a_tau   = a[1::3]
    a_sigma = a[2::3]

    # Invert cumulative sum: recover delay gaps
    gaps    = jnp.diff(jnp.concatenate([jnp.array([0.0]), a_tau]))
    b_n     = a_n
    b_gamma = jnp.log(gaps   / T_c)
    b_beta  = jnp.log(a_sigma / T_c)

    # Pack
    b = jnp.stack([b_n, b_gamma, b_beta], axis=1).ravel()
    return b


# ---------------------------------------------------------------------------
# Convenience: unpack b or a into named arrays
# ---------------------------------------------------------------------------

def unpack_b(b: jnp.ndarray) -> tuple:
    """Return (n, gamma, beta) arrays from parameter vector b."""
    b = b.ravel()
    return b[0::3], b[1::3], b[2::3]


def unpack_a(a: jnp.ndarray) -> tuple:
    """Return (n, tau, sigma) arrays from physical vector a."""
    a = a.ravel()
    return a[0::3], a[1::3], a[2::3]