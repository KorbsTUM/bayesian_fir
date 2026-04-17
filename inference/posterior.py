"""
inference/posterior.py
======================
Estimates the MAP posterior for a given model order N, computes the
Laplace-approximate posterior covariance, and returns model scoring
metrics (log marginal likelihood, Occam factor, best-fit likelihood).

This module orchestrates the full inference pipeline for a single model
order:

    1. Generate restart seeds over the nonlinear parameter space x.
    2. Run the multi-start LM optimizer (vmapped over restarts).
    3. Compute the full (3N x 3N) Hessian at the MAP estimate.
    4. Invert the Hessian via Cholesky to get the posterior covariance.
    5. Compute the Laplace-approximate log marginal likelihood.
    6. Evaluate the impulse response and its pointwise uncertainty.
    7. Map MAP parameters back to physical space.

Model scoring (eq. 21, Yoko & Polifke 2026):

    log p(q|N) ≈ log p(q|a*, N)        [best-fit likelihood, logBFL]
               + log p(a*|N)            [prior at MAP]
               + (P/2) log(2*pi)
               + 0.5 * log|Sigma_a*|   [posterior volume]

Equivalently:

    logML  = -J_MAP + (P/2)*log(2*pi) - sum(log(diag(R)))
    logBFL = -J_like
    logOF  = logML - logBFL            [Occam factor]

where R is the upper Cholesky factor of the Hessian H = Sigma_a*^{-1}.

References:
    Yoko & Polifke (2026), Sections 3.2-3.4 and Algorithm 1.
    MacKay (2003), 'Information Theory, Inference, and Learning Algorithms'.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional

from core.cost import calculate_cost
from core.impulse_response import calculate_impulse_response
from core.parameter_maps import map_to_physical, map_to_physical_covariance
from core.prior import PriorConfig, generate_prior
from inference.seeds import (generate_nonlinear_seeds,
                              recommended_restarts,
                              recommended_iterations)
from inference.optimizer import (OptimizerConfig,
                                  run_all_restarts,
                                  compute_final_hessian)


# ---------------------------------------------------------------------------
# Output containers
# ---------------------------------------------------------------------------

class PosteriorResult:
    """
    Container for the posterior estimate of a single model order.

    Attributes
    ----------
    b_map   : jnp.ndarray (3N,)   MAP parameters in parameter space.
    Cb_map  : jnp.ndarray (3N,3N) Posterior covariance in parameter space.
    a_map   : jnp.ndarray (3N,)   MAP parameters in physical space.
    Ca_map  : jnp.ndarray (3N,3N) Posterior covariance in physical space.
    Ce      : float                MAP noise variance estimate.
    h       : ImpulseResponse      Impulse response struct.
    logML   : float                Log marginal likelihood.
    logBFL  : float                Log best-fit likelihood.
    logOF   : float                Log Occam factor.
    N       : int                  Model order.
    names   : list of str          Parameter names.
    """

    def __init__(self,
                 b_map, Cb_map, a_map, Ca_map,
                 Ce, h, logML, logBFL, logOF, N, names):
        self.b_map  = b_map
        self.Cb_map = Cb_map
        self.a_map  = a_map
        self.Ca_map = Ca_map
        self.Ce     = Ce
        self.h      = h
        self.logML  = float(logML)
        self.logBFL = float(logBFL)
        self.logOF  = float(logOF)
        self.N      = N
        self.names  = names

    def __repr__(self):
        return (
            f"PosteriorResult(N={self.N}, "
            f"logML={self.logML:.2f}, "
            f"logBFL={self.logBFL:.2f}, "
            f"logOF={self.logOF:.2f})"
        )


class ImpulseResponse:
    """
    Container for the impulse response evaluated at the MAP estimate.

    Attributes
    ----------
    time : jnp.ndarray (T,)   Time vector in physical units [s].
    val  : jnp.ndarray (T,)   Impulse response values.
    var  : jnp.ndarray (T,)   Pointwise variance (from Laplace approx).
    """

    def __init__(self, time, val, var):
        self.time = time
        self.val  = val
        self.var  = var


class ModelRanking:
    """
    Container for model ranking metrics across all candidate orders.

    Attributes
    ----------
    orders  : list of int          Candidate model orders.
    logML   : jnp.ndarray (K,)    Log marginal likelihoods.
    logBFL  : jnp.ndarray (K,)    Log best-fit likelihoods.
    logOF   : jnp.ndarray (K,)    Log Occam factors.
    best_N  : int                  Model order with highest logML.
    """

    def __init__(self, orders, logML, logBFL, logOF):
        self.orders  = list(orders)
        self.logML   = jnp.array(logML)
        self.logBFL  = jnp.array(logBFL)
        self.logOF   = jnp.array(logOF)
        best_idx     = int(jnp.argmax(self.logML))
        self.best_N  = self.orders[best_idx]

    def normalized(self):
        """Return logML and logBFL normalized to the best model."""
        ml_max  = jnp.max(self.logML)
        return (self.logML  - ml_max,
                self.logBFL - ml_max,
                self.logOF)

    def print_table(self):
        """Print a formatted model ranking table to stdout."""
        logML_n, logBFL_n, logOF = self.normalized()
        header = f"{'N':>4}  {'logML':>10}  {'logBFL':>10}  {'logOF':>10}"
        print("\nModel ranking:\n" + "-" * len(header))
        print(header)
        print("-" * len(header))
        for i, N in enumerate(self.orders):
            marker = " <-- best" if N == self.best_N else ""
            print(f"{N:>4}  {float(logML_n[i]):>10.2f}  "
                  f"{float(logBFL_n[i]):>10.2f}  "
                  f"{float(logOF[i]):>10.2f}{marker}")
        print("-" * len(header))


# ---------------------------------------------------------------------------
# Core: estimate posterior for a single model order
# ---------------------------------------------------------------------------

def estimate_posterior(signals    : dict,
                        bp         : jnp.ndarray,
                        Cp         : jnp.ndarray,
                        T_c        : float,
                        prior_cfg  : PriorConfig,
                        opt_cfg    : OptimizerConfig,
                        N          : int,
                        names      : list,
                        Ce0        : Optional[float] = None,
                        n_eval_pts : int = 500) -> PosteriorResult:
    """
    Estimate the MAP posterior for a model of order N.

    Parameters
    ----------
    signals    : dict           Prepared signal struct from prepare_signals.
    bp         : jnp.ndarray   Prior mean, shape (3N,).
    Cp         : jnp.ndarray   Prior covariance, shape (3N, 3N).
    T_c        : float          Convective timescale [s].
    prior_cfg  : PriorConfig    Prior configuration (static).
    opt_cfg    : OptimizerConfig Optimizer configuration (static).
    N          : int            Model order.
    names      : list of str    Parameter names (from generate_prior).
    Ce0        : float or None  Initial noise variance. If None, uses 1e-4.
    n_eval_pts : int            Number of points for impulse response eval.

    Returns
    -------
    result : PosteriorResult
    """
    print(f"\nEstimating posterior for N={N} delays:")

    # ------------------------------------------------------------------
    # Initial noise variance
    # ------------------------------------------------------------------
    if Ce0 is None:
        Ce0 = 1e-4 if opt_cfg.infer_noise else None
        if Ce0 is None:
            raise ValueError("Must provide Ce0 when infer_noise=False.")

    # ------------------------------------------------------------------
    # Generate restart seeds over nonlinear parameter space x
    # ------------------------------------------------------------------
    n_restarts = recommended_restarts(N, opt_cfg.restart_scaling)
    max_iter   = recommended_iterations(N, opt_cfg.iteration_scaling)

    print(f"  Restarts : {n_restarts}  |  Max iter : {max_iter}")

    seeds = generate_nonlinear_seeds(bp, Cp, N, n_restarts)  # (2N, n_restarts)

    # ------------------------------------------------------------------
    # Run multi-start LM optimization
    # ------------------------------------------------------------------
    best = run_all_restarts(
        seeds, Ce0, signals, bp, Cp, T_c, prior_cfg, opt_cfg, max_iter)

    b_map  = best['b_full']     # (3N,)
    Ce     = best['Ce']         # scalar
    J_like = best['J_like']     # scalar
    A_bfgs = best['A']          # (2N, 2N)

    print(f"  MAP cost : {float(best['f']):.4f}  |  "
          f"Ce : {float(Ce):.3e}  |  "
          f"Best restart : {int(best['best_idx'])}")

    # ------------------------------------------------------------------
    # Full Hessian at MAP (with optional BFGS correction)
    # ------------------------------------------------------------------
    H_full = compute_final_hessian(
        b_map, Ce, A_bfgs, signals, bp, Cp, T_c, prior_cfg, opt_cfg)

    # ------------------------------------------------------------------
    # Posterior covariance via Cholesky inversion
    # ------------------------------------------------------------------
    H_full = (H_full + H_full.T) / 2.0          # enforce symmetry

    Cb_map, R, success = _robust_cholesky_inverse(H_full)

    if not success:
        raise RuntimeError(
            f"Hessian at MAP is not positive definite for N={N}. "
            f"Cannot compute posterior covariance. "
            f"Consider increasing the signal length or adjusting the prior.")

    # ------------------------------------------------------------------
    # Model scoring  (eq. 21)
    # ------------------------------------------------------------------
    # Recompute full cost at MAP for accurate J_MAP
    J_MAP, _, _, _, J_like_full, _ = calculate_cost(
        signals, Ce, b_map, bp, Cp, T_c, prior_cfg)

    P      = b_map.shape[0]
    logML  = (-J_MAP
              + 0.5 * P * jnp.log(2.0 * jnp.pi)
              - jnp.sum(jnp.log(jnp.diag(R))))
    logBFL = -J_like_full
    logOF  = logML - logBFL

    # ------------------------------------------------------------------
    # Physical-space parameters and covariance
    # ------------------------------------------------------------------
    a_map, _, _ = map_to_physical(b_map, T_c)
    Ca_map      = map_to_physical_covariance(b_map, T_c, Cb_map)

    # ------------------------------------------------------------------
    # Impulse response at MAP with uncertainty
    # ------------------------------------------------------------------
    sig_coarse = signals['coarse']
    t_h_nd     = sig_coarse['t_h'] / T_c
    t_eval_nd  = jnp.linspace(0.0, float(t_h_nd[-1]), n_eval_pts)

    h_val, _, _, h_var = calculate_impulse_response(
        b_map, t_eval_nd, T_c, Cb_map)

    # Convert time back to physical units
    t_eval_phys = t_eval_nd * T_c

    h = ImpulseResponse(
        time = t_eval_phys,
        val  = h_val,
        var  = h_var,
    )

    return PosteriorResult(
        b_map  = b_map,
        Cb_map = Cb_map,
        a_map  = a_map,
        Ca_map = Ca_map,
        Ce     = float(Ce),
        h      = h,
        logML  = float(logML),
        logBFL = float(logBFL),
        logOF  = float(logOF),
        N      = N,
        names  = names,
    )


# ---------------------------------------------------------------------------
# Model ranking across multiple orders
# ---------------------------------------------------------------------------

def rank_models(signals    : dict,
                T_c        : float,
                prior_cfg  : PriorConfig,
                opt_cfg    : OptimizerConfig,
                model_orders: list,
                Ce0        : Optional[float] = None,
                n_eval_pts : int = 500) -> tuple:
    """
    Estimate posteriors for all candidate model orders and rank them
    by log marginal likelihood.

    Parameters
    ----------
    signals      : dict           Prepared signal struct.
    T_c          : float          Convective timescale [s].
    prior_cfg    : PriorConfig    Prior configuration.
    opt_cfg      : OptimizerConfig Optimizer configuration.
    model_orders : list of int    Candidate model orders to evaluate.
    Ce0          : float or None  Initial noise variance.
    n_eval_pts   : int            Points for impulse response evaluation.

    Returns
    -------
    best_result  : PosteriorResult   Result for the best model order.
    all_results  : list of PosteriorResult  Results for all orders.
    ranking      : ModelRanking      Model ranking metrics.
    """
    # Guard: cannot rank with fixed T_h
    if prior_cfg.T_h is not None and len(model_orders) > 1:
        raise ValueError(
            "Model ranking across candidate orders is not meaningful for "
            "fixed T_h (see Section 4.4 of Yoko & Polifke 2026). "
            "Either set prior_cfg.T_h = None, or specify a single model order."
        )

    all_results = []
    logML_list  = []
    logBFL_list = []
    logOF_list  = []

    for N in model_orders:
        # Generate order-specific prior (includes t_max and T_h for signals)
        bp, Cp, names, t_max = generate_prior(N, T_c, prior_cfg)
        T_h = t_max * T_c

        # Re-prepare signals with order-specific T_h
        # (signals must be re-prepared per order when T_h varies)
        from signal.prepare import prepare_signals
        sig = prepare_signals(
            np.array(signals['fine']['u']),
            np.array(signals['fine']['q']),
            signals['fine']['fs'],
            T_h,
        )

        result = estimate_posterior(
            sig, bp, Cp, T_c, prior_cfg, opt_cfg,
            N, names, Ce0, n_eval_pts)

        all_results.append(result)
        logML_list.append(result.logML)
        logBFL_list.append(result.logBFL)
        logOF_list.append(result.logOF)

    ranking = ModelRanking(model_orders, logML_list, logBFL_list, logOF_list)
    ranking.print_table()

    best_result = all_results[
        model_orders.index(ranking.best_N)]

    return best_result, all_results, ranking


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _robust_cholesky_inverse(H: jnp.ndarray,
                              max_jitter_steps: int = 10
                              ) -> tuple:
    """
    Invert a symmetric positive semi-definite matrix via Cholesky
    decomposition, with progressive jitter for numerical stability.

    Parameters
    ----------
    H                : jnp.ndarray (P, P)   Symmetric matrix to invert.
    max_jitter_steps : int                  Maximum jitter doubling steps.

    Returns
    -------
    H_inv   : jnp.ndarray (P, P)   Inverse of H (posterior covariance).
    R       : jnp.ndarray (P, P)   Upper Cholesky factor of H.
    success : bool                  Whether inversion succeeded.
    """
    P = H.shape[0]

    # Try plain Cholesky first
    try:
        R      = jnp.linalg.cholesky(H).T          # upper triangular
        H_inv  = jax.scipy.linalg.cho_solve(
            (R.T, True), jnp.eye(P))
        return H_inv, R, True
    except Exception:
        pass

    # Progressive jitter
    jitter = 1e-6 * jnp.eye(P)
    for _ in range(max_jitter_steps):
        try:
            R     = jnp.linalg.cholesky(H + jitter).T
            H_inv = jax.scipy.linalg.cho_solve(
                (R.T, True), jnp.eye(P))
            print(f"  Warning: added jitter {float(jitter[0,0]):.2e} "
                  f"for Hessian stability.")
            return H_inv, R, True
        except Exception:
            jitter = jitter * 10.0

    # Failed
    return jnp.zeros_like(H), jnp.zeros_like(H), False