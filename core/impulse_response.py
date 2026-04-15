"""
impulse_response.py
===================
Evaluates the N-Gaussian distributed time delay (DTD) impulse response
and its Jacobian with respect to the parameter vector b.

The impulse response model is:

    h(t) = sum_{i=1}^{N} n_i * G_i(t)

where each basis function is a normalised Gaussian:

    G_i(t) = (2 * pi * sigma_i^2)^{-1/2} * exp(-(t - tau_i)^2 / (2*sigma_i^2))

All computations are carried out in non-dimensional time (t / T_c), with
T_c reintroduced only when returning physical-space outputs.

Two evaluation modes are provided:
    - calculate_impulse_response : full Jacobian, used during MAP optimisation
    - impulse_response_val       : value only, used for forward passes and sampling

References:
    Yoko & Polifke (2026), Sections 2 and 4.1
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, jacfwd
from functools import partial

from parameter_maps import map_to_physical, unpack_b


# ---------------------------------------------------------------------------
# Core: single Gaussian basis function
# ---------------------------------------------------------------------------

@jit
def gaussian_basis(t: jnp.ndarray, tau: float,
                   sigma: float) -> jnp.ndarray:
    """
    Evaluate a single normalised Gaussian pulse.

    Parameters
    ----------
    t     : jnp.ndarray, shape (T,)
        Time vector in physical units.
    tau   : float
        Centre (time delay) in physical units.
    sigma : float
        Width (standard deviation) in physical units.

    Returns
    -------
    g : jnp.ndarray, shape (T,)
        Gaussian evaluated at each time point.
    """
    return (1.0 / jnp.sqrt(2.0 * jnp.pi * sigma ** 2)) * \
           jnp.exp(-0.5 * ((t - tau) / sigma) ** 2)


# ---------------------------------------------------------------------------
# Gaussian basis matrix: all N pulses evaluated at all T time points
# ---------------------------------------------------------------------------

@partial(jit, static_argnums=())
def gaussian_basis_matrix(t: jnp.ndarray,
                           tau: jnp.ndarray,
                           sigma: jnp.ndarray) -> jnp.ndarray:
    """
    Build the (T x N) Gaussian basis matrix G.

    Entry G[t_idx, i] = G_i(t[t_idx]).

    Parameters
    ----------
    t     : jnp.ndarray, shape (T,)
        Time vector in physical units.
    tau   : jnp.ndarray, shape (N,)
        Delay vector in physical units.
    sigma : jnp.ndarray, shape (N,)
        Width vector in physical units.

    Returns
    -------
    G : jnp.ndarray, shape (T, N)
        Gaussian basis matrix.
    """
    # Outer broadcast: dt[i,j] = t[i] - tau[j],  shape (T, N)
    dt = t[:, None] - tau[None, :]
    S  = sigma[None, :]                           # broadcast across time
    G  = (1.0 / jnp.sqrt(2.0 * jnp.pi * S ** 2)) * jnp.exp(-0.5 * (dt / S) ** 2)
    return G                                       # (T, N)


# ---------------------------------------------------------------------------
# Value-only forward pass  (cheap, used in VI/sampling inner loop)
# ---------------------------------------------------------------------------

@jit
def impulse_response_val(b: jnp.ndarray,
                          t_nd: jnp.ndarray,
                          T_c: float) -> jnp.ndarray:
    """
    Evaluate the impulse response h(t) without computing the Jacobian.

    Suitable for use inside normalising-flow / ELBO training loops where
    only the forward pass is needed and jax.grad handles differentiation.

    Parameters
    ----------
    b    : jnp.ndarray, shape (3N,)
        Parameter vector in parameter space.
    t_nd : jnp.ndarray, shape (T,)
        Non-dimensional time vector (t / T_c).
    T_c  : float
        Convective timescale [s].

    Returns
    -------
    h : jnp.ndarray, shape (T,)
        Impulse response values.
    """
    _, params, _ = map_to_physical(b, T_c)
    t_phys = t_nd * T_c
    G = gaussian_basis_matrix(t_phys, params['tau'], params['sigma'])
    return G @ params['n']                         # (T,)


# ---------------------------------------------------------------------------
# Full forward pass with analytic Jacobian  (used during MAP optimisation)
# ---------------------------------------------------------------------------

@jit
def calculate_impulse_response(b: jnp.ndarray,
                                t_nd: jnp.ndarray,
                                T_c: float,
                                Cb: jnp.ndarray | None = None
                                ) -> tuple:
    """
    Evaluate the impulse response and its Jacobian dh/db.

    Analytic expressions are used for the physical-space gradients;
    the chain rule through the parameter map is applied via the Jacobian
    blocks returned by map_to_physical.

    Parameters
    ----------
    b    : jnp.ndarray, shape (3N,)
        Parameter vector in parameter space.
    t_nd : jnp.ndarray, shape (T,)
        Non-dimensional time vector (t / T_c).
    T_c  : float
        Convective timescale [s].
    Cb   : jnp.ndarray, shape (3N, 3N), optional
        Posterior covariance in parameter space.
        When provided, the pointwise variance of h is also returned.

    Returns
    -------
    h      : jnp.ndarray, shape (T,)
        Impulse response.
    dhdb   : jnp.ndarray, shape (T, 3N)
        Jacobian of h w.r.t. b, in interleaved column order
        [dh/dn_1, dh/dgamma_1, dh/dbeta_1, dh/dn_2, ...].
    G      : jnp.ndarray, shape (T, N)
        Gaussian basis matrix (unit-amplitude pulses).
    h_var  : jnp.ndarray, shape (T,), or None
        Pointwise variance of h when Cb is supplied.
    """
    b = b.ravel()
    N = b.shape[0] // 3

    # Map to physical space and get Jacobian blocks
    _, params, jac = map_to_physical(b, T_c)
    a_n    = params['n']      # (N,)
    a_tau  = params['tau']    # (N,)
    a_sig  = params['sigma']  # (N,)

    # Physical time vector
    t_phys = t_nd * T_c       # (T,)

    # Gaussian basis matrix, shape (T, N)
    G  = gaussian_basis_matrix(t_phys, a_tau, a_sig)

    # Impulse response
    h  = G @ a_n              # (T,)

    # -----------------------------------------------------------------------
    # Analytic gradients in physical space
    # -----------------------------------------------------------------------
    # Reuse broadcast quantities from basis matrix construction
    dt = t_phys[:, None] - a_tau[None, :]    # (T, N)
    S  = a_sig[None, :]                       # (1, N)

    # dh/d(n_i) = G[:, i]
    dh_dan   = G                              # (T, N)

    # dh/d(tau_i) = G[:, i] * (dt[:, i] / sigma_i^2) * n_i
    dh_datau = G * (dt / S ** 2) * a_n[None, :]       # (T, N)

    # dh/d(sigma_i) = G[:, i] * ((dt[:,i]^2 - sigma_i^2) / sigma_i^3) * n_i
    dh_dasig = G * ((dt ** 2 - S ** 2) / S ** 3) * a_n[None, :]  # (T, N)

    # -----------------------------------------------------------------------
    # Chain rule: physical gradients -> parameter-space gradients
    # -----------------------------------------------------------------------
    # dh/d(n)     shape (T, N)  -- pass through identity
    dh_dbn     = dh_dan @ jac['dn_dbn']           # (T, N)

    # dh/d(gamma) shape (T, N)
    # dtau_dgamma is (N, N) lower triangular
    dh_dbgamma = dh_datau @ jac['dtau_dgamma']    # (T, N)

    # dh/d(beta)  shape (T, N)
    dh_dbbeta  = dh_dasig @ jac['dsig_dbeta']     # (T, N)

    # -----------------------------------------------------------------------
    # Assemble Jacobian in interleaved column order [n1, g1, b1, n2, g2, b2, ...]
    # -----------------------------------------------------------------------
    # Stack along a new axis then reshape: (T, N, 3) -> (T, 3N)
    dhdb = jnp.stack([dh_dbn, dh_dbgamma, dh_dbbeta], axis=2)  # (T, N, 3)
    dhdb = dhdb.reshape(dhdb.shape[0], -1)                       # (T, 3N)

    # -----------------------------------------------------------------------
    # Optional: pointwise variance  diag(dhdb @ Cb @ dhdb.T)
    # Computed as row-wise dot product to avoid forming the full (T x T) matrix
    # -----------------------------------------------------------------------
    h_var = None
    if Cb is not None:
        # (T, 3N) @ (3N, 3N) -> (T, 3N), then row dot with dhdb -> (T,)
        h_var = jnp.sum((dhdb @ Cb) * dhdb, axis=1)

    return h, dhdb, G, h_var


# ---------------------------------------------------------------------------
# JAX-grad based Jacobian  (alternative, used in VI / flow training)
# ---------------------------------------------------------------------------

@partial(jit, static_argnums=(2,))
def impulse_response_jacobian_ad(b: jnp.ndarray,
                                  t_nd: jnp.ndarray,
                                  T_c: float) -> jnp.ndarray:
    """
    Compute dh/db using forward-mode automatic differentiation.

    This is an alternative to the analytic Jacobian in
    calculate_impulse_response and is provided for validation and for
    use in contexts where the analytic chain rule is inconvenient
    (e.g., when the parameter map changes).

    Parameters
    ----------
    b    : jnp.ndarray, shape (3N,)
    t_nd : jnp.ndarray, shape (T,)
    T_c  : float

    Returns
    -------
    dhdb : jnp.ndarray, shape (T, 3N)
    """
    # jacfwd is efficient when output dim (T) >> input dim (3N),
    # which is the typical case here
    return jacfwd(impulse_response_val, argnums=0)(b, t_nd, T_c)


# ---------------------------------------------------------------------------
# Batched evaluation: vmap over a collection of parameter vectors
# ---------------------------------------------------------------------------

# Evaluate h for a batch of parameter vectors, e.g. MCMC samples or
# flow samples.  Shape: (B, 3N) -> (B, T)
impulse_response_batch = jit(
    vmap(impulse_response_val, in_axes=(0, None, None))
)