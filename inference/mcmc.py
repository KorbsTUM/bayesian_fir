"""
inference/mcmc.py
=================
Metropolis-Hastings (MH) random-walk MCMC sampler for posterior validation.

This module provides a JAX-native implementation of the MH algorithm used
in Section 5.3 of Yoko & Polifke (2026) to validate the accuracy of the
Laplace approximation. The sampler makes no assumptions about the shape of
the posterior and is used to obtain reference samples against which the
Laplace-approximate covariance is compared.

Two implementations are provided:

    run_mcmc_scan(...)
        JAX-native implementation using jax.lax.scan. The entire chain
        runs on-device with no Python-level iteration overhead. Suitable
        for GPU execution and for vmapping over multiple chains.

    run_mcmc_python(...)
        Plain Python loop implementation. Slower but easier to debug,
        and supports arbitrary Python-level callbacks (e.g. progress bars).
        Useful for initial validation on CPU.

Usage notes
-----------
* The sampler is initialised at the MAP estimate b_map with a proposal
  covariance scaled from the Laplace covariance: C_prop = (2.38^2/d)*Cb_map.
  This is the optimal scaling for a Gaussian target in d dimensions
  (Gelman, Roberts & Gilks 1996) and targets roughly 23% acceptance.

* The noise variance Ce is fixed at the MAP estimate during MCMC -- only
  the impulse response parameters b are sampled. This matches the MATLAB
  implementation.

* The log-posterior function is constructed from calculate_cost, which
  returns the negative log-posterior J. The log-posterior is therefore
  simply -J (up to a constant that cancels in the MH ratio).

* Window acceptance rates are tracked in blocks of 100 iterations for
  diagnostics, matching the MATLAB implementation.

References:
    Yoko & Polifke (2026), Section 5.3.
    Hastings (1970), 'Monte Carlo sampling methods using Markov chains
        and their applications', Biometrika 57(1).
    Gelman, Roberts & Gilks (1996), 'Efficient Metropolis jumping rules',
        Bayesian Statistics 5.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, vmap
from functools import partial
from typing import Callable, Optional, NamedTuple

from core.cost import calculate_cost
from core.prior import PriorConfig


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

class MCMCResult:
    """
    Container for MCMC chain output.

    Attributes
    ----------
    chain        : jnp.ndarray (d, n_iter)    Full parameter chain.
    log_post     : jnp.ndarray (n_iter,)      Log-posterior at each step.
    accepted     : jnp.ndarray (n_iter,) bool Acceptance indicators.
    post_samples : jnp.ndarray (d, n_post)    Post-burn-in thinned samples.
    post_log_post: jnp.ndarray (n_post,)      Log-posterior for post samples.
    accept_rate  : float                       Post-burn-in acceptance rate.
    window_ar    : np.ndarray                  Per-100-iter acceptance rates.
    burn_in      : int                         Burn-in length used.
    thin         : int                         Thinning interval used.
    prop_cov     : jnp.ndarray (d, d)         Proposal covariance used.
    """

    def __init__(self, chain, log_post, accepted, post_samples,
                 post_log_post, accept_rate, window_ar,
                 burn_in, thin, prop_cov):
        self.chain         = chain
        self.log_post      = log_post
        self.accepted      = accepted
        self.post_samples  = post_samples
        self.post_log_post = post_log_post
        self.accept_rate   = float(accept_rate)
        self.window_ar     = np.asarray(window_ar)
        self.burn_in       = int(burn_in)
        self.thin          = int(thin)
        self.prop_cov      = prop_cov

    def __repr__(self):
        n_post = self.post_samples.shape[1]
        return (
            f"MCMCResult("
            f"n_iter={self.chain.shape[1]}, "
            f"n_post={n_post}, "
            f"accept_rate={self.accept_rate:.1%}, "
            f"burn_in={self.burn_in}, "
            f"thin={self.thin})"
        )

    def effective_sample_size(self) -> jnp.ndarray:
        """
        Estimate the effective sample size (ESS) for each parameter
        using the autocorrelation method.

        Returns
        -------
        ess : jnp.ndarray, shape (d,)
        """
        samples = np.array(self.post_samples)   # (d, n_post)
        d, n    = samples.shape
        ess     = np.zeros(d)

        for i in range(d):
            s   = samples[i] - samples[i].mean()
            ac  = np.correlate(s, s, mode='full')[n-1:]
            ac  = ac / ac[0]
            # Geyer's initial positive sequence estimator
            T   = 1
            for k in range(1, n // 2):
                pair = ac[2*k] + ac[2*k+1]
                if pair < 0:
                    break
                T += 2 * pair
            ess[i] = n / T

        return jnp.array(ess)


# ---------------------------------------------------------------------------
# Proposal covariance construction
# ---------------------------------------------------------------------------

def make_proposal_covariance(Cb_map: jnp.ndarray) -> jnp.ndarray:
    """
    Construct the MH proposal covariance from the Laplace posterior
    covariance using the optimal Gelman-Roberts-Gilks scaling.

    C_prop = (2.38^2 / d) * Cb_map

    Parameters
    ----------
    Cb_map : jnp.ndarray, shape (d, d)
        Laplace-approximate posterior covariance at MAP.

    Returns
    -------
    C_prop : jnp.ndarray, shape (d, d)
    """
    d = Cb_map.shape[0]
    return (2.38 ** 2 / d) * Cb_map


# ---------------------------------------------------------------------------
# Log-posterior function factory
# ---------------------------------------------------------------------------

def make_log_posterior(signals    : dict,
                        Ce         : float,
                        bp         : jnp.ndarray,
                        Cp         : jnp.ndarray,
                        T_c        : float,
                        prior_cfg  : PriorConfig) -> Callable:
    """
    Construct a log-posterior function for use in the MH sampler.

    The log-posterior is -J(b), where J is the negative log-posterior
    returned by calculate_cost. The Ce is fixed at the MAP estimate.

    Parameters
    ----------
    signals   : dict         Prepared signal struct.
    Ce        : float        Fixed noise variance (MAP estimate).
    bp        : jnp.ndarray  Prior mean, shape (3N,).
    Cp        : jnp.ndarray  Prior covariance, shape (3N, 3N).
    T_c       : float        Convective timescale [s].
    prior_cfg : PriorConfig  Prior configuration (static).

    Returns
    -------
    log_post : Callable  b -> scalar log-posterior.
    """
    @jit
    def log_post(b: jnp.ndarray) -> jnp.ndarray:
        J, _, _, _, _, _ = calculate_cost(
            signals, Ce, b, bp, Cp, T_c, prior_cfg)
        return -J

    return log_post


# ---------------------------------------------------------------------------
# JAX-native scan-based MCMC  (recommended)
# ---------------------------------------------------------------------------

class _ScanState(NamedTuple):
    """Carry state for jax.lax.scan MH loop."""
    b        : jnp.ndarray   # current sample        (d,)
    lp       : jnp.ndarray   # current log-posterior (scalar)
    key      : jnp.ndarray   # JAX PRNG key


def run_mcmc_scan(log_post_fn : Callable,
                   b0          : jnp.ndarray,
                   prop_cov    : jnp.ndarray,
                   n_iter      : int,
                   burn_frac   : float = 0.25,
                   thin        : int   = 1,
                   seed        : int   = 0) -> MCMCResult:
    """
    Run Metropolis-Hastings using jax.lax.scan (fully on-device).

    The entire chain is generated in a single scan call with no Python
    iteration overhead, making this suitable for GPU execution and for
    vmapping over multiple independent chains.

    Parameters
    ----------
    log_post_fn : Callable     b -> scalar log-posterior (JIT-compiled).
    b0          : jnp.ndarray  Initial parameter vector, shape (d,).
    prop_cov    : jnp.ndarray  Proposal covariance, shape (d, d).
    n_iter      : int          Total number of MCMC iterations.
    burn_frac   : float        Fraction of chain discarded as burn-in.
    thin        : int          Thinning interval for post-burn-in samples.
    seed        : int          JAX PRNG seed for reproducibility.

    Returns
    -------
    result : MCMCResult
    """
    d   = b0.shape[0]
    key = jax.random.PRNGKey(seed)

    # Cholesky of proposal covariance for efficient sampling
    L_prop = jnp.linalg.cholesky(prop_cov)           # lower triangular (d,d)

    # Initial log-posterior
    lp0 = log_post_fn(b0)

    # ------------------------------------------------------------------
    # Single MH step as a scan body
    # ------------------------------------------------------------------
    def mh_step(carry: _ScanState, _) -> tuple:
        b, lp, key = carry

        # Split key
        key, key_prop, key_accept = jax.random.split(key, 3)

        # Propose: b_prop = b + L_prop @ z,  z ~ N(0, I)
        z      = jax.random.normal(key_prop, shape=(d,))
        b_prop = b + L_prop @ z

        # Evaluate log-posterior at proposal
        lp_prop = log_post_fn(b_prop)
        lp_prop = jnp.where(jnp.isfinite(lp_prop), lp_prop, -jnp.inf)

        # MH acceptance
        log_alpha = jnp.minimum(0.0, lp_prop - lp)
        u         = jax.random.uniform(key_accept)
        accept    = jnp.log(u) < log_alpha

        # Update state
        b_new  = jnp.where(accept, b_prop,  b)
        lp_new = jnp.where(accept, lp_prop, lp)

        new_carry = _ScanState(b=b_new, lp=lp_new, key=key)

        # Output: (sample, log_post, accepted)
        output = (b_new, lp_new, accept)
        return new_carry, output

    # ------------------------------------------------------------------
    # Run chain via scan
    # ------------------------------------------------------------------
    init_carry = _ScanState(b=b0, lp=lp0, key=key)
    _, (chain_T, log_post_arr, accepted_arr) = jax.lax.scan(
        mh_step, init_carry, None, length=n_iter)

    # chain_T      : (n_iter, d)  -- scan stacks outputs along axis 0
    # log_post_arr : (n_iter,)
    # accepted_arr : (n_iter,)

    chain    = chain_T.T                              # (d, n_iter)
    log_post_arr = log_post_arr                       # (n_iter,)

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------
    result = _postprocess_chain(
        chain, log_post_arr, accepted_arr,
        prop_cov, burn_frac, thin, n_iter)

    return result


# ---------------------------------------------------------------------------
# Python-loop MCMC  (debugging / CPU fallback)
# ---------------------------------------------------------------------------

def run_mcmc_python(log_post_fn : Callable,
                     b0          : jnp.ndarray,
                     prop_cov    : jnp.ndarray,
                     n_iter      : int,
                     burn_frac   : float = 0.25,
                     thin        : int   = 1,
                     seed        : int   = 0,
                     print_every : int   = 2000) -> MCMCResult:
    """
    Run Metropolis-Hastings with a plain Python loop.

    Slower than run_mcmc_scan but easier to debug and supports
    Python-level progress reporting. Intended for CPU use and initial
    validation.

    Parameters
    ----------
    log_post_fn : Callable     b -> scalar log-posterior.
    b0          : jnp.ndarray  Initial parameter vector, shape (d,).
    prop_cov    : jnp.ndarray  Proposal covariance, shape (d, d).
    n_iter      : int          Total number of MCMC iterations.
    burn_frac   : float        Fraction of chain discarded as burn-in.
    thin        : int          Thinning interval.
    seed        : int          Random seed.
    print_every : int          Print progress every this many iterations.

    Returns
    -------
    result : MCMCResult
    """
    d     = b0.shape[0]
    rng   = np.random.default_rng(seed)
    L     = np.linalg.cholesky(np.array(prop_cov))

    # Pre-allocate in NumPy for speed
    chain        = np.zeros((d, n_iter), dtype=np.float32)
    log_post_arr = np.full(n_iter, -np.inf, dtype=np.float64)
    accepted_arr = np.zeros(n_iter, dtype=bool)

    b_cur  = np.array(b0, dtype=np.float64)
    lp_cur = float(log_post_fn(jnp.array(b_cur)))

    chain[:, 0]    = b_cur
    log_post_arr[0]= lp_cur

    for i in range(1, n_iter):
        # Propose
        z       = rng.standard_normal(d)
        b_prop  = b_cur + L @ z
        lp_prop = float(log_post_fn(jnp.array(b_prop, dtype=jnp.float32)))

        if not np.isfinite(lp_prop):
            lp_prop = -np.inf

        # Accept / reject
        log_alpha = min(0.0, lp_prop - lp_cur)
        if np.log(rng.random()) < log_alpha:
            b_cur  = b_prop
            lp_cur = lp_prop
            accepted_arr[i] = True

        chain[:, i]     = b_cur
        log_post_arr[i] = lp_cur

        if print_every > 0 and i % print_every == 0:
            ar_recent = accepted_arr[max(0, i-print_every):i].mean()
            print(f"  MCMC iteration {i:>6}/{n_iter}  "
                  f"accept rate (recent): {ar_recent:.1%}")

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------
    result = _postprocess_chain(
        jnp.array(chain),
        jnp.array(log_post_arr),
        jnp.array(accepted_arr),
        prop_cov, burn_frac, thin, n_iter)

    return result


# ---------------------------------------------------------------------------
# Post-processing helper
# ---------------------------------------------------------------------------

def _postprocess_chain(chain        : jnp.ndarray,
                        log_post_arr : jnp.ndarray,
                        accepted_arr : jnp.ndarray,
                        prop_cov     : jnp.ndarray,
                        burn_frac    : float,
                        thin         : int,
                        n_iter       : int) -> MCMCResult:
    """
    Apply burn-in and thinning, compute acceptance rates and window ARs.

    Parameters
    ----------
    chain        : (d, n_iter)
    log_post_arr : (n_iter,)
    accepted_arr : (n_iter,) bool
    prop_cov     : (d, d)
    burn_frac    : float
    thin         : int
    n_iter       : int

    Returns
    -------
    MCMCResult
    """
    burn_in = int(np.floor(burn_frac * n_iter))

    # Post-burn-in thinned indices
    post_idx     = np.arange(burn_in, n_iter, thin)
    post_samples = chain[:, post_idx]
    post_lp      = log_post_arr[post_idx]

    # Overall post-burn-in acceptance rate
    accept_rate = float(jnp.mean(accepted_arr[burn_in:]))

    # Window acceptance rates in blocks of 100 iterations
    accepted_np  = np.array(accepted_arr)
    n_windows    = int(np.ceil(n_iter / 100))
    window_ar    = np.zeros(n_windows)
    for w in range(n_windows):
        lo = w * 100
        hi = min((w + 1) * 100, n_iter)
        window_ar[w] = accepted_np[lo:hi].mean()

    return MCMCResult(
        chain         = chain,
        log_post      = log_post_arr,
        accepted      = accepted_arr,
        post_samples  = post_samples,
        post_log_post = post_lp,
        accept_rate   = accept_rate,
        window_ar     = window_ar,
        burn_in       = burn_in,
        thin          = thin,
        prop_cov      = prop_cov,
    )


# ---------------------------------------------------------------------------
# Convenience: run MCMC from a PosteriorResult
# ---------------------------------------------------------------------------

def run_mcmc_from_posterior(posterior,
                             signals    : dict,
                             bp         : jnp.ndarray,
                             Cp         : jnp.ndarray,
                             T_c        : float,
                             prior_cfg  : PriorConfig,
                             n_iter     : int   = 200_000,
                             burn_frac  : float = 0.25,
                             thin       : int   = 1,
                             seed       : int   = 0,
                             use_scan   : bool  = True) -> MCMCResult:
    """
    Run MCMC validation from a PosteriorResult, using the MAP estimate
    as the starting point and the Laplace covariance for the proposal.

    Parameters
    ----------
    posterior  : PosteriorResult   Output of estimate_posterior.
    signals    : dict              Prepared signal struct.
    bp         : jnp.ndarray       Prior mean.
    Cp         : jnp.ndarray       Prior covariance.
    T_c        : float             Convective timescale [s].
    prior_cfg  : PriorConfig       Prior configuration.
    n_iter     : int               Number of MCMC iterations.
    burn_frac  : float             Burn-in fraction.
    thin       : int               Thinning interval.
    seed       : int               PRNG seed.
    use_scan   : bool              Use JAX scan (True) or Python loop (False).

    Returns
    -------
    mcmc : MCMCResult
    """
    b0       = posterior.b_map
    Ce       = posterior.Ce
    Cb_map   = posterior.Cb_map
    prop_cov = make_proposal_covariance(Cb_map)

    log_post_fn = make_log_posterior(
        signals, Ce, bp, Cp, T_c, prior_cfg)

    print(f"\nRunning MCMC for N={posterior.N} "
          f"({'scan' if use_scan else 'python'} mode):")
    print(f"  n_iter={n_iter}, burn_frac={burn_frac}, thin={thin}")

    if use_scan:
        mcmc = run_mcmc_scan(
            log_post_fn, b0, prop_cov,
            n_iter, burn_frac, thin, seed)
    else:
        mcmc = run_mcmc_python(
            log_post_fn, b0, prop_cov,
            n_iter, burn_frac, thin, seed)

    print(f"  Acceptance rate (post burn-in): {mcmc.accept_rate:.1%}")
    ess = mcmc.effective_sample_size()
    print(f"  Min ESS: {float(ess.min()):.0f}  |  "
          f"Mean ESS: {float(ess.mean()):.0f}")

    return mcmc