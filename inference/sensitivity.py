"""
inference/sensitivity.py
=========================
Differentiable path from raw input/output signals (u, q) to the fitted
DTD model parameters (n_i, tau_i, sigma_i), via implicit differentiation
through the MAP optimum. Everything else in the inference pipeline
(MCMC, model-order ranking across N, the TFDSI baseline) is untouched
and remains non-differentiable, as intended.

Why implicit differentiation
-----------------------------
inference.optimizer's multi-start Levenberg-Marquardt loop runs inside
jax.lax.while_loop, which has no reverse-mode VJP rule - jax.grad cannot
trace through it directly. Rather than unrolling the loop (whose cost
would scale with the number of LM iterations, up to ~1000 per restart,
times the number of restarts), this module treats the converged
solution z* = (x*, Ce*) as an implicit function of the data, defined by
its stationarity conditions:

    dJ(x*, Ce*, u, q) = 0                     VarPro gradient (always)
    Ce* - Ce_ME(x*, Ce*, u, q) = 0             MacKay MML noise fixed
                                                point (only when
                                                opt_cfg.infer_noise=True,
                                                matching how the LM loop
                                                itself jointly updates Ce)

and differentiates through THAT condition via the implicit function
theorem, instead of through the iterations that found it:

    dz*/dtheta = -(dF/dz)^-1 (dF/dtheta)

The cost of one backward pass is one (2N- or (2N+1)-dimensional) linear
solve plus a handful of Jacobian/VJP evaluations of the (already
differentiable) VarPro cost function - independent of how many restarts
or LM iterations the forward solve took. This is the standard
"unrolling-free" implicit-differentiation technique (e.g. Blondel et al.
2022, 'Efficient and Modular Implicit Differentiation'), implemented
here directly via jax.custom_vjp rather than a general-purpose library,
since the residual functions (dJ, Ce_ME) are already sitting right there
as differentiable outputs of core.cost.calculate_cost_varpro.

Because reverse-mode is what's implemented (via custom_vjp, not
custom_jvp), jax.jacobian(..., argnums=(0,1)) on the functions in this
module computes the (3N, M) sensitivity matrix efficiently via O(3N)
backward passes - appropriate here since the number of fitted
parameters (3N) is typically much smaller than the signal length (M).

Scope
-----
Only the 'no downsampling' preprocessing path is supported
(signals.prepare.prepare_signals_diff): scipy.signal.resample_poly, used
by prepare_signals when downsampling is requested, has no differentiable
JAX equivalent in this codebase.

Validity - read before trusting the output
--------------------------------------------
The implicit function theorem gives the sensitivity of the TRUE root of
the stationarity conditions above. It says nothing useful if the forward
solve didn't actually reach that root. Two ways that can happen, both
found while validating this module against finite differences:

    1. Non-convergence. If the LM loop exits via the ill-conditioned-
       Hessian bailout (inference.optimizer's cond_max check) or simply
       hasn't reached a small gradient norm yet, x* is not a root and
       the "sensitivity" computed at it is meaningless - not merely
       imprecise. This is most likely with a single restart landing in
       a bad basin; using enough restarts (the make_map_estimator
       default, via recommended_restarts) makes it much less likely in
       practice. When it matters, verify convergence directly, e.g.:

           b_map, Ce = map_estimate(u, q)
           signals = prepare_signals_diff(u, q, fs, T_h)
           x_star = jnp.stack([b_map[1::3], b_map[2::3]]).T.ravel()  # gamma,beta interleaved
           _, dJ, *_ = calculate_cost_varpro(signals, Ce, x_star, bp, Cp, T_c, prior_cfg)
           assert jnp.linalg.norm(dJ) < 1e-6   # tune to your problem's scale

    2. Ce* -> 0. With opt_cfg.infer_noise=True and (near-)noise-free
       data, the MacKay MML noise estimate drives Ce* towards 0, and the
       1/Ce likelihood weighting in the cost function becomes singular
       there - both the implicit-diff Jacobian and any finite-difference
       check of it become unreliable in that limit. This is a property
       of the noise-free limit itself (the noise level becomes
       unidentifiable), not specific to this implementation. If your
       data really is near noise-free, prefer opt_cfg.infer_noise=False
       with an explicit small Ce0 over letting Ce be estimated.
"""

from typing import Optional

import jax
import jax.numpy as jnp

from core.cost import calculate_cost_varpro
from core.parameter_maps import map_to_physical
from core.prior import PriorConfig
from inference.optimizer import OptimizerConfig, run_all_restarts
from inference.seeds import (generate_nonlinear_seeds,
                              recommended_restarts,
                              recommended_iterations)
from signals.prepare import prepare_signals_diff


def make_map_estimator(fs: float,
                        T_h: float,
                        T_c: float,
                        bp: jnp.ndarray,
                        Cp: jnp.ndarray,
                        prior_cfg: PriorConfig,
                        opt_cfg: OptimizerConfig,
                        n_restarts: Optional[int] = None,
                        max_iter: Optional[int] = None,
                        Ce0: Optional[float] = None):
    """
    Build a differentiable MAP estimator b_map(u, q) for a fixed model
    order N (inferred from bp.shape[0] // 3), fixed prior, and fixed
    optimizer settings.

    The returned function is differentiable w.r.t. u and q via implicit
    differentiation (see module docstring):

        map_estimate = make_map_estimator(...)
        b_map, Ce = map_estimate(u, q)
        d_bmap_du, d_bmap_dq = jax.jacobian(
            lambda u, q: map_estimate(u, q)[0], argnums=(0, 1))(u, q)

    Parameters
    ----------
    fs         : float               Sampling frequency [Hz].
    T_h        : float               Impulse response duration [s].
    T_c        : float               Convective timescale [s].
    bp         : jnp.ndarray, (3N,)    Prior mean.
    Cp         : jnp.ndarray, (3N,3N)  Prior covariance.
    prior_cfg  : PriorConfig         Prior configuration (static).
    opt_cfg    : OptimizerConfig     Optimizer configuration (static).
    n_restarts : int, optional       Defaults to recommended_restarts(N, ...).
    max_iter   : int, optional       Defaults to recommended_iterations(N, ...).
    Ce0        : float, optional     Initial noise variance when
                                     opt_cfg.infer_noise=True (default 1e-4),
                                     or the fixed noise variance when
                                     opt_cfg.infer_noise=False (required).

    Returns
    -------
    map_estimate : Callable[[u, q], (b_map, Ce)]
        b_map : jnp.ndarray, (3N,)   MAP parameter vector, differentiable
                                     w.r.t. u and q.
        Ce    : jnp.ndarray, scalar  Converged noise variance (constant
                                     w.r.t. u, q when infer_noise=False).
    """
    N = bp.shape[0] // 3

    if n_restarts is None:
        n_restarts = recommended_restarts(N, opt_cfg.restart_scaling)
    if max_iter is None:
        max_iter = recommended_iterations(N, opt_cfg.iteration_scaling)
    if Ce0 is None:
        if not opt_cfg.infer_noise:
            raise ValueError("Ce0 must be provided when opt_cfg.infer_noise=False.")
        Ce0 = 1e-4

    # Seeds depend only on the prior (bp, Cp), never on u, q - safe to
    # precompute once here (NumPy/Sobol, matches estimate_posterior).
    seeds = generate_nonlinear_seeds(bp, Cp, N, n_restarts)  # (2N, n_restarts)

    def _signals(u, q):
        return prepare_signals_diff(u, q, fs, T_h)

    def _dJ(x, Ce, u, q):
        """VarPro gradient dJ/dx, as a function of (x, Ce, u, q)."""
        _, dJ, _, _, _, _, _ = calculate_cost_varpro(
            _signals(u, q), Ce, x, bp, Cp, T_c, prior_cfg)
        return dJ

    def _F2(x, Ce, u, q):
        """MacKay MML noise residual: zero at the joint (x, Ce) fixed point."""
        _, _, _, Ce_ME, _, _, _ = calculate_cost_varpro(
            _signals(u, q), Ce, x, bp, Cp, T_c, prior_cfg)
        return Ce - Ce_ME

    # -------------------------------------------------------------------
    # Forward solve: run the existing while_loop-based multi-restart LM
    # optimizer unchanged. Opaque to autodiff - custom_vjp below replaces
    # its (nonexistent) VJP rule entirely, so this can use lax.while_loop
    # freely.
    # -------------------------------------------------------------------
    def _solve(u, q):
        signals = _signals(u, q)
        best = run_all_restarts(seeds, Ce0, signals, bp, Cp, T_c, prior_cfg,
                                 opt_cfg, max_iter)
        return best['x'], best['Ce']

    @jax.custom_vjp
    def find_root(u, q):
        return _solve(u, q)

    def find_root_fwd(u, q):
        x_star, Ce_star = _solve(u, q)
        return (x_star, Ce_star), (x_star, Ce_star, u, q)

    def find_root_bwd(residuals, cotangent):
        x_star, Ce_star, u, q = residuals
        dL_dx, dL_dCe = cotangent
        Nx = x_star.shape[0]

        if opt_cfg.infer_noise:
            # Joint (x, Ce) fixed point: F = [dJ(x,Ce,u,q); Ce - Ce_ME(x,Ce,u,q)] = 0
            dF1_dx, dF1_dCe = jax.jacobian(_dJ, argnums=(0, 1))(x_star, Ce_star, u, q)
            dF2_dx, dF2_dCe = jax.jacobian(_F2, argnums=(0, 1))(x_star, Ce_star, u, q)

            A = jnp.zeros((Nx + 1, Nx + 1))
            A = A.at[:Nx, :Nx].set(dF1_dx)
            A = A.at[:Nx, Nx].set(dF1_dCe)
            A = A.at[Nx, :Nx].set(dF2_dx)
            A = A.at[Nx, Nx].set(dF2_dCe)

            v = jnp.concatenate([dL_dx, jnp.array([dL_dCe])])
            w = jnp.linalg.solve(A.T, v)
            w1, w2 = w[:Nx], w[Nx]

            def F_joint(u_, q_):
                return _dJ(x_star, Ce_star, u_, q_), _F2(x_star, Ce_star, u_, q_)

            _, vjp_fn = jax.vjp(F_joint, u, q)
            dF_du, dF_dq = vjp_fn((w1, w2))
        else:
            # Ce is held fixed (= Ce0), never updated by the LM loop, so
            # it carries no dependence on u, q - only x* is implicit.
            A = jax.jacobian(_dJ, argnums=0)(x_star, Ce_star, u, q)   # (Nx, Nx)
            w = jnp.linalg.solve(A.T, dL_dx)

            def F_x(u_, q_):
                return _dJ(x_star, Ce_star, u_, q_)

            _, vjp_fn = jax.vjp(F_x, u, q)
            dF_du, dF_dq = vjp_fn(w)

        return (-dF_du, -dF_dq)

    find_root.defvjp(find_root_fwd, find_root_bwd)

    # -------------------------------------------------------------------
    # Public entry point: compose the implicit-diff root with an ordinary
    # (autodiff-transparent) reconstruction of the full parameter vector,
    # so JAX's normal chain rule combines the explicit dependence of
    # b_map on (u, q) (through signals) with the implicit dependence
    # (through x*, Ce*) automatically.
    # -------------------------------------------------------------------
    def map_estimate(u, q):
        x_star, Ce_star = find_root(u, q)
        signals = _signals(u, q)
        _, _, _, _, _, _, b_map = calculate_cost_varpro(
            signals, Ce_star, x_star, bp, Cp, T_c, prior_cfg)
        return b_map, Ce_star

    return map_estimate


def sensitivity_to_physical(u: jnp.ndarray,
                             q: jnp.ndarray,
                             fs: float,
                             T_h: float,
                             T_c: float,
                             bp: jnp.ndarray,
                             Cp: jnp.ndarray,
                             prior_cfg: PriorConfig,
                             opt_cfg: OptimizerConfig,
                             **kwargs) -> jnp.ndarray:
    """
    Fully composed differentiable path: u, q -> fitted DTD parameters in
    physical space (n_i, tau_i, sigma_i), for a fixed model order N
    (from bp.shape[0] // 3).

    Example
    -------
        a_map = sensitivity_to_physical(u, q, fs, T_h, T_c, bp, Cp,
                                         prior_cfg, opt_cfg)
        da_du, da_dq = jax.jacobian(
            sensitivity_to_physical, argnums=(0, 1)
        )(u, q, fs, T_h, T_c, bp, Cp, prior_cfg, opt_cfg)
        # da_du, da_dq: shape (3N, M) - sensitivity of every (n_i, tau_i,
        # sigma_i) to every sample of u / q.

    Parameters
    ----------
    u, q       : jnp.ndarray, (M,)   Input / output signals.
    fs         : float                Sampling frequency [Hz].
    T_h        : float                Impulse response duration [s].
    T_c        : float                Convective timescale [s].
    bp, Cp     : jnp.ndarray          Prior mean / covariance, shape (3N,), (3N,3N).
    prior_cfg  : PriorConfig
    opt_cfg    : OptimizerConfig
    **kwargs   : forwarded to make_map_estimator (n_restarts, max_iter, Ce0).

    Returns
    -------
    a_map : jnp.ndarray, (3N,)
        Physical parameter vector [n_1, tau_1, sig_1, n_2, ...].
    """
    map_estimate = make_map_estimator(fs, T_h, T_c, bp, Cp, prior_cfg,
                                       opt_cfg, **kwargs)
    b_map, _ = map_estimate(u, q)
    a_map, _, _ = map_to_physical(b_map, T_c)
    return a_map
