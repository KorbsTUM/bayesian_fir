"""
inference/seeds.py
==================
Generates deterministic, space-filling restart seeds for the multi-start
MAP optimisation in estimatePosterior.

Rather than drawing random samples from the prior (which can cluster by
chance), we use a Sobol quasi-random low-discrepancy sequence to produce
seeds that cover the prior support uniformly. This makes the multi-start
strategy more reliable at finding the global MAP optimum, especially in
higher-dimensional parameter spaces (large N).

Seed generation pipeline:
    1. Generate N Sobol points uniformly on [0, 1]^d using scipy's
       Sobol engine (equivalent to MATLAB's sobolset).
    2. Rescale to [Phi(-3), Phi(3)] to concentrate seeds within the
       ±3-sigma region of the prior and avoid pathological initialisations
       in the tails.
    3. Apply the inverse normal CDF (probit transform) to map uniform
       samples to N(0, I).
    4. Apply the Cholesky factor of the prior covariance C to map
       N(0, I) -> N(m, C).

All seeds are generated in NumPy (runs once, not JIT-compiled) and
returned as a JAX array for use in the vmapped optimizer.

References:
    Yoko & Polifke (2026), Section 4.2
    Joe & Kuo (2008), 'Constructing Sobol sequences with better
        two-dimensional projections', SIAM J. Sci. Comput.
"""

import numpy as np
import jax.numpy as jnp
from scipy.stats import norm
from scipy.stats.qmc import Sobol


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_restart_seeds(m: jnp.ndarray,
                            C: jnp.ndarray,
                            n_restarts: int,
                            sigma_clip: float = 3.0,
                            scramble: bool = False) -> jnp.ndarray:
    """
    Generate deterministic Gaussian restart seeds via a Sobol sequence.

    Produces space-filling samples concentrated within a `sigma_clip`-sigma
    region of the Gaussian prior N(m, C).

    Parameters
    ----------
    m           : jnp.ndarray, shape (d,)
        Mean vector of the target Gaussian (prior mean over x or b).
    C           : jnp.ndarray, shape (d, d)
        Covariance matrix of the target Gaussian (prior covariance).
    n_restarts  : int
        Number of restart seeds to generate.
    sigma_clip  : float
        Number of standard deviations to clip the Sobol samples to.
        Default 3.0, matching the MATLAB implementation.
    scramble    : bool
        Whether to scramble the Sobol sequence for randomisation.
        Default False (deterministic, matching MATLAB's sobolset behaviour).

    Returns
    -------
    seeds : jnp.ndarray, shape (d, n_restarts)
        Matrix of restart seeds, one per column.
        Compatible with the vmapped optimizer which expects (d, n_restarts).
    """
    m = np.asarray(m, dtype=np.float64).ravel()
    C = np.asarray(C, dtype=np.float64)
    d = m.shape[0]

    if C.shape != (d, d):
        raise ValueError(f"C must be ({d}, {d}), got {C.shape}.")
    if n_restarts < 1:
        raise ValueError(f"n_restarts must be >= 1, got {n_restarts}.")

    # ------------------------------------------------------------------
    # Step 1: Sobol sequence on [0, 1]^d
    # ------------------------------------------------------------------
    # scipy's Sobol engine is equivalent to MATLAB's sobolset.
    # We skip the first n_restarts points (Skip=N in MATLAB) because the
    # very first points of a Sobol sequence are less well-distributed.
    sobol_engine = Sobol(d=d, scramble=scramble)

    # Fast-forward (skip) n_restarts points to match MATLAB behaviour
    sobol_engine.fast_forward(n_restarts)

    # Draw n_restarts points: shape (n_restarts, d)
    U = sobol_engine.random(n_restarts)           # (n_restarts, d) in [0,1]

    # ------------------------------------------------------------------
    # Step 2: Rescale to [Phi(-sigma_clip), Phi(+sigma_clip)]
    # ------------------------------------------------------------------
    p_lo = norm.cdf(-sigma_clip)
    p_hi = norm.cdf(+sigma_clip)
    U    = p_lo + (p_hi - p_lo) * U              # (n_restarts, d)

    # ------------------------------------------------------------------
    # Step 3: Probit transform -> N(0, I)
    # ------------------------------------------------------------------
    Z = norm.ppf(U).T                             # (d, n_restarts)

    # ------------------------------------------------------------------
    # Step 4: Cholesky transform -> N(m, C)
    # ------------------------------------------------------------------
    L = np.linalg.cholesky(C)                     # lower triangular (d, d)
    seeds = m[:, None] + L @ Z                    # (d, n_restarts)

    return jnp.array(seeds, dtype=jnp.float32)


# ---------------------------------------------------------------------------
# Nonlinear-only seeds  (for VarPro optimizer over x = [gamma, beta, ...])
# ---------------------------------------------------------------------------

def generate_nonlinear_seeds(bp: jnp.ndarray,
                              Cp: jnp.ndarray,
                              N: int,
                              n_restarts: int,
                              sigma_clip: float = 3.0,
                              scramble: bool = False) -> jnp.ndarray:
    """
    Generate restart seeds over the nonlinear parameter subspace only.

    Extracts the gamma and beta entries from the full prior (bp, Cp) and
    generates seeds in the 2N-dimensional space x = [gamma_1, beta_1, ...].

    This is the seed generator used by the VarPro optimizer, which only
    iterates over x while solving for n analytically.

    Parameters
    ----------
    bp         : jnp.ndarray, shape (3N,)   Full prior mean.
    Cp         : jnp.ndarray, shape (3N,3N) Full prior covariance (diagonal).
    N          : int                         Model order.
    n_restarts : int                         Number of seeds.
    sigma_clip : float                       Sigma clipping level.
    scramble   : bool                        Sobol scrambling flag.

    Returns
    -------
    seeds : jnp.ndarray, shape (2N, n_restarts)
        Restart seeds over x = [gamma_1, beta_1, gamma_2, beta_2, ...].
    """
    bp = np.asarray(bp, dtype=np.float64).ravel()
    Cp = np.asarray(Cp, dtype=np.float64)

    # Extract gamma and beta indices from full b layout
    # b = [n_1, gamma_1, beta_1, n_2, gamma_2, beta_2, ...]
    idx_gamma = np.arange(N) * 3 + 1    # 1, 4, 7, ...
    idx_beta  = np.arange(N) * 3 + 2    # 2, 5, 8, ...

    # Interleave into x layout: [gamma_1, beta_1, gamma_2, beta_2, ...]
    idx_x = np.empty(2 * N, dtype=int)
    idx_x[0::2] = idx_gamma
    idx_x[1::2] = idx_beta

    xp = bp[idx_x]                       # (2N,)
    Cx = np.diag(np.diag(Cp)[idx_x])    # (2N, 2N) diagonal

    return generate_restart_seeds(xp, Cx, n_restarts,
                                   sigma_clip=sigma_clip,
                                   scramble=scramble)


# ---------------------------------------------------------------------------
# Utility: recommended number of restarts
# ---------------------------------------------------------------------------

def recommended_restarts(N: int,
                          restart_scaling: int = 10) -> int:
    """
    Compute the recommended number of random restarts for a model of order N.

    Follows the MATLAB convention of restart_scaling restarts per nonlinear
    parameter (default 10), giving 10 * 2N restarts for a model of order N.

    Parameters
    ----------
    N               : int   Model order.
    restart_scaling : int   Restarts per nonlinear parameter (default 10).

    Returns
    -------
    n_restarts : int
    """
    Nx = 2 * N    # number of nonlinear parameters
    return restart_scaling * Nx


# ---------------------------------------------------------------------------
# Utility: recommended number of optimizer iterations
# ---------------------------------------------------------------------------

def recommended_iterations(N: int,
                             iteration_scaling: int = 100) -> int:
    """
    Compute the recommended maximum number of LM iterations for order N.

    Follows the MATLAB convention of iteration_scaling iterations per
    nonlinear parameter (default 100).

    Parameters
    ----------
    N                 : int   Model order.
    iteration_scaling : int   Iterations per nonlinear parameter.

    Returns
    -------
    max_iter : int
    """
    Nx = 2 * N
    return iteration_scaling * Nx