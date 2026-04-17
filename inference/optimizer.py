"""
inference/optimizer.py
======================
Levenberg-Marquardt (LM) trust-region optimizer for MAP estimation of the
flame impulse response parameters, with variable projection (VarPro) over
the linear amplitude parameters.

Architecture
------------
The optimizer is structured in three layers:

    1. lm_step(x, f, g, H, Ce, lambda_, ...)
       A single LM step: solves the trust-region subproblem, evaluates the
       new cost, and accepts or rejects the step based on the improvement
       ratio rho. Returns the updated state. Written as a pure function so
       it can be used inside jax.lax.while_loop.

    2. run_one_restart(x0, Ce0, signals, bp, Cp, T_c, prior_cfg, config)
       Runs the full LM loop from a single starting point x0, updating
       the noise estimate Ce jointly with the parameters. Returns the
       converged MAP estimate, cost, and BFGS curvature correction.

    3. run_all_restarts(seeds, Ce0, signals, bp, Cp, T_c, prior_cfg, config)
       Runs all restarts and selects the best solution.
       Two execution modes:
           * JAX vmap  : all restarts in parallel on GPU (recommended)
           * Python for-loop : fallback for debugging / CPU

Design choices
--------------
* The LM loop is implemented with jax.lax.while_loop so the entire
  optimisation runs on-device without Python-level iteration overhead.
  This is essential for vmapping over restarts.

* VarPro: only the nonlinear parameters x = [gamma_1, beta_1, ...] are
  iterated. Amplitudes n are solved analytically at each step via the
  MAP linear solve in calculate_cost_varpro.

* BFGS curvature correction: a rank-2 BFGS update accumulates the
  difference between the true gradient change and the GN prediction,
  capturing second-order terms dropped by the GN approximation.
  This correction is optionally added to the final Hessian before the
  Laplace covariance is computed.

* Joint noise update: after iteration 5, Ce is updated on accepted steps
  as a weighted average 0.6*Ce + 0.4*Ce_new (Ce_new from MacKay MML).
  This avoids the expensive nested optimisation described in Section 3.3.

References:
    Nielsen (1999), 'Damping parameter in Marquardt's method', IMM.
    Yoko & Polifke (2026), Section 4.2.
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
from typing import NamedTuple

from core.cost import calculate_cost_varpro, calculate_cost
from core.prior import PriorConfig


# ---------------------------------------------------------------------------
# Optimizer configuration
# ---------------------------------------------------------------------------

class OptimizerConfig(NamedTuple):
    """
    Static configuration for the LM optimizer.

    All fields are Python scalars (not JAX arrays) so they are treated as
    compile-time constants by JAX and do not affect tracing.

    Attributes
    ----------
    restart_scaling   : int    Restarts per nonlinear parameter (default 10).
    iteration_scaling : int    Max iterations per nonlinear parameter (default 100).
    tol_optimality    : float  Gradient norm convergence tolerance.
    tol_step          : float  Relative step size convergence tolerance.
    tol_fn            : float  Function change convergence tolerance.
    lam_inc0          : float  Base lambda increment on rejected step.
    lam_dec           : float  Base lambda decrement on accepted step.
    lam_p             : float  Lambda scaling exponent.
    lam_min           : float  Minimum lambda value.
    noise_update_weight : float  Weight for joint Ce update (0.4 in MATLAB).
    noise_update_start  : int    Iteration after which Ce updates begin.
    use_bfgs_correction : bool   Whether to apply BFGS curvature correction.
    use_parallel        : bool   Whether to vmap over restarts (GPU).
    solution_selection  : str    'log_prob' or 'log_ml'.
    infer_noise         : bool   Whether to estimate Ce from data (True) or
                                 use a fixed value (False).
    """
    restart_scaling     : int   = 10
    iteration_scaling   : int   = 100
    tol_optimality      : float = 1e-6
    tol_step            : float = 1e-6
    tol_fn              : float = 1e-8
    lam_inc0            : float = 2.0
    lam_dec             : float = 3.0
    lam_p               : float = 3.0
    lam_min             : float = 1e-8
    noise_update_weight : float = 0.4
    noise_update_start  : int   = 5
    use_bfgs_correction : bool  = True
    use_parallel        : bool  = True
    solution_selection  : str   = 'log_prob'
    infer_noise         : bool  = True


# ---------------------------------------------------------------------------
# Loop carry state
# ---------------------------------------------------------------------------

class LMState(NamedTuple):
    """
    Carry state for the LM while_loop.

    All fields must be JAX arrays or Python scalars (no Python containers).
    """
    x        : jnp.ndarray   # current nonlinear parameters  (2N,)
    b_full   : jnp.ndarray   # current full parameters       (3N,)
    f        : jnp.ndarray   # current cost (scalar)
    g        : jnp.ndarray   # current gradient              (2N,)
    H        : jnp.ndarray   # current GN Hessian            (2N,2N)
    Ce       : jnp.ndarray   # current noise estimate        (scalar)
    J_like   : jnp.ndarray   # likelihood part of cost       (scalar)
    lambda_  : jnp.ndarray   # LM damping parameter          (scalar)
    lam_inc  : jnp.ndarray   # current lambda increment      (scalar)
    A        : jnp.ndarray   # BFGS curvature matrix         (2N,2N)
    db       : jnp.ndarray   # last step                     (2N,)
    df       : jnp.ndarray   # last function change          (scalar)
    k        : jnp.ndarray   # iteration counter             (scalar int)
    converged: jnp.ndarray   # convergence flag              (scalar bool)


# ---------------------------------------------------------------------------
# Single LM step  (pure function, no side effects)
# ---------------------------------------------------------------------------

def _make_lm_step(signals, bp, Cp, T_c, prior_cfg, cfg):
    """
    Factory that closes over static arguments and returns a JIT-compiled
    single-step function suitable for use inside jax.lax.while_loop.
    """

    @jit
    def lm_step(state: LMState) -> LMState:
        x       = state.x
        b_full  = state.b_full
        f       = state.f
        g       = state.g
        H       = state.H
        Ce      = state.Ce
        J_like  = state.J_like
        lambda_ = state.lambda_
        lam_inc = state.lam_inc
        A       = state.A
        k       = state.k

        Nx = x.shape[0]
        I  = jnp.eye(Nx)

        # ----------------------------------------------------------------
        # Solve trust-region subproblem: (H + lambda*I) db = -g
        # ----------------------------------------------------------------
        H_reg = H + lambda_ * I
        db    = jnp.linalg.solve(H_reg, -g)          # (2N,)

        predicted_improvement = 0.5 * db @ (lambda_ * db - g)

        # ----------------------------------------------------------------
        # Evaluate cost at proposed point
        # ----------------------------------------------------------------
        x_new = x + db
        f_new, g_new, H_new, Ce_new, J_like_new, _, b_full_new = \
            calculate_cost_varpro(signals, Ce, x_new, bp, Cp, T_c, prior_cfg)
        H_new = (H_new + H_new.T) / 2.0

        actual_improvement = f - f_new

        # ----------------------------------------------------------------
        # Compute improvement ratio rho
        # ----------------------------------------------------------------
        rho = jnp.where(
            predicted_improvement <= 0,
            -jnp.inf,
            actual_improvement / predicted_improvement
        )

        # ----------------------------------------------------------------
        # Accept / reject step
        # ----------------------------------------------------------------
        step_accepted = rho > 0.0

        # Trust region update
        scale = jnp.where(
            step_accepted,
            jnp.maximum(1.0 / cfg.lam_dec,
                        1.0 - (cfg.lam_inc0 - 1.0) *
                        (2.0 * rho - 1.0) ** cfg.lam_p),
            lam_inc
        )
        lambda_new = jnp.where(
            step_accepted,
            jnp.maximum(lambda_ * scale, cfg.lam_min),
            lambda_ * lam_inc
        )
        lam_inc_new = jnp.where(step_accepted, cfg.lam_inc0, 2.0 * lam_inc)

        # Accept: update state
        x_out      = jnp.where(step_accepted, x_new,      x)
        b_full_out = jnp.where(step_accepted, b_full_new,  b_full)
        f_out      = jnp.where(step_accepted, f_new,       f)
        g_out      = jnp.where(step_accepted, g_new,       g)
        H_out      = jnp.where(step_accepted, H_new,       H)
        J_like_out = jnp.where(step_accepted, J_like_new,  J_like)
        df_out     = jnp.where(step_accepted, f - f_new,
                               jnp.zeros(()))

        # ----------------------------------------------------------------
        # BFGS curvature correction (only on accepted steps, after burn-in)
        # ----------------------------------------------------------------
        do_bfgs = step_accepted & (jnp.linalg.norm(g_out) < 1.0)

        s   = x_out - x
        y   = g_out - g
        yGN = H_out @ s
        u   = y - yGN

        aa  = jnp.where(s @ u > 1e-12, 1.0 / (u @ s),      0.0)
        v   = A @ s
        bb  = jnp.where(s @ (A @ s) > 0, 1.0 / (s @ A @ s), 0.0)

        A_new = A + aa * jnp.outer(u, u) - bb * jnp.outer(v, v)
        A_out = jnp.where(do_bfgs, A_new, A)

        # ----------------------------------------------------------------
        # Joint noise update (after burn-in, accepted steps only)
        # ----------------------------------------------------------------
        do_noise = step_accepted & (k > cfg.noise_update_start) & \
                   cfg.infer_noise
        w        = cfg.noise_update_weight
        Ce_out   = jnp.where(do_noise,
                             (1.0 - w) * Ce + w * Ce_new,
                             Ce)

        # ----------------------------------------------------------------
        # Convergence check
        # ----------------------------------------------------------------
        converged = (
            (jnp.linalg.norm(g_out) < cfg.tol_optimality) |
            (jnp.linalg.norm(db) / jnp.maximum(jnp.linalg.norm(x_out),
                                                1e-12) < cfg.tol_step) |
            (jnp.abs(df_out) < cfg.tol_fn)
        ) & (k > 5)

        return LMState(
            x        = x_out,
            b_full   = b_full_out,
            f        = f_out,
            g        = g_out,
            H        = H_out,
            Ce       = Ce_out,
            J_like   = J_like_out,
            lambda_  = lambda_new,
            lam_inc  = lam_inc_new,
            A        = A_out,
            db       = db,
            df       = df_out,
            k        = k + 1,
            converged= converged,
        )

    return lm_step


# ---------------------------------------------------------------------------
# Single restart
# ---------------------------------------------------------------------------

def run_one_restart(x0         : jnp.ndarray,
                    Ce0        : float,
                    signals    : dict,
                    bp         : jnp.ndarray,
                    Cp         : jnp.ndarray,
                    T_c        : float,
                    prior_cfg  : PriorConfig,
                    cfg        : OptimizerConfig,
                    max_iter   : int) -> dict:
    """
    Run the LM optimizer from a single starting point x0.

    Parameters
    ----------
    x0       : jnp.ndarray, shape (2N,)   Initial nonlinear parameters.
    Ce0      : float                       Initial noise variance.
    signals  : dict                        Prepared signal struct.
    bp       : jnp.ndarray, shape (3N,)   Prior mean.
    Cp       : jnp.ndarray, shape (3N,3N) Prior covariance.
    T_c      : float                       Convective timescale.
    prior_cfg: PriorConfig                 Prior configuration (static).
    cfg      : OptimizerConfig             Optimizer configuration (static).
    max_iter : int                         Maximum number of iterations.

    Returns
    -------
    result : dict with keys:
        'x'       : (2N,)   converged nonlinear parameters
        'b_full'  : (3N,)   converged full parameters
        'f'       : scalar  final cost
        'J_like'  : scalar  final likelihood cost
        'Ce'      : scalar  final noise estimate
        'A'       : (2N,2N) BFGS curvature correction
        'n_iter'  : scalar  number of iterations taken
    """
    Nx = x0.shape[0]
    Ce0_arr = jnp.array(Ce0)

    # Initial cost evaluation
    f0, g0, H0, Ce_new0, J_like0, _, b_full0 = \
        calculate_cost_varpro(signals, Ce0_arr, x0, bp, Cp, T_c, prior_cfg)
    H0 = (H0 + H0.T) / 2.0

    # Initial damping: max diagonal of H
    lambda0 = jnp.maximum(jnp.max(jnp.diag(H0)), 1e-6)

    # Initial BFGS matrix: near-zero (will be built up during iteration)
    A0 = 1e-10 * jnp.eye(Nx)

    # Initial state
    state = LMState(
        x        = x0,
        b_full   = b_full0,
        f        = f0,
        g        = g0,
        H        = H0,
        Ce       = Ce0_arr,
        J_like   = J_like0,
        lambda_  = lambda0,
        lam_inc  = jnp.array(cfg.lam_inc0),
        A        = A0,
        db       = jnp.ones(Nx),
        df       = jnp.array(1.0),
        k        = jnp.array(0),
        converged= jnp.array(False),
    )

    # Build step function
    lm_step = _make_lm_step(signals, bp, Cp, T_c, prior_cfg, cfg)

    # Run loop
    def cond_fn(state):
        return (~state.converged) & (state.k < max_iter)

    def body_fn(state):
        return lm_step(state)

    final = jax.lax.while_loop(cond_fn, body_fn, state)

    return {
        'x'      : final.x,
        'b_full' : final.b_full,
        'f'      : final.f,
        'J_like' : final.J_like,
        'Ce'     : final.Ce,
        'A'      : final.A,
        'n_iter' : final.k,
    }


# ---------------------------------------------------------------------------
# All restarts  (vmapped or serial)
# ---------------------------------------------------------------------------

def run_all_restarts(seeds      : jnp.ndarray,
                     Ce0        : float,
                     signals    : dict,
                     bp         : jnp.ndarray,
                     Cp         : jnp.ndarray,
                     T_c        : float,
                     prior_cfg  : PriorConfig,
                     cfg        : OptimizerConfig,
                     max_iter   : int) -> dict:
    """
    Run the LM optimizer from all restart seeds and select the best solution.

    Parameters
    ----------
    seeds    : jnp.ndarray, shape (2N, n_restarts)
        Restart seeds from generate_nonlinear_seeds, one per column.
    Ce0      : float   Initial noise variance.
    ...      : (remaining args as in run_one_restart)

    Returns
    -------
    best : dict
        Result dict for the best restart (lowest cost or highest log-ML),
        with the same keys as run_one_restart, plus:
            'all_f'  : (n_restarts,)  costs from all restarts
            'best_idx': int           index of the selected restart
    """
    seeds_T = seeds.T                      # (n_restarts, 2N)
    n_restarts = seeds_T.shape[0]

    if cfg.use_parallel:
        # -----------------------------------------------------------------
        # Parallel execution via vmap over restarts
        # All restarts run simultaneously on GPU
        # -----------------------------------------------------------------
        def one_restart(x0):
            return run_one_restart(
                x0, Ce0, signals, bp, Cp, T_c, prior_cfg, cfg, max_iter)

        results = vmap(one_restart)(seeds_T)

    else:
        # -----------------------------------------------------------------
        # Serial execution (debugging / CPU fallback)
        # -----------------------------------------------------------------
        result_list = []
        for i in range(n_restarts):
            r = run_one_restart(
                seeds_T[i], Ce0, signals, bp, Cp, T_c, prior_cfg,
                cfg, max_iter)
            result_list.append(r)
            print(f"  Restart {i+1}/{n_restarts}  "
                  f"J={float(r['f']):.4f}  "
                  f"Ce={float(r['Ce']):.2e}  "
                  f"iters={int(r['n_iter'])}")

        # Stack into arrays matching vmap output format
        results = {
            k: jnp.stack([r[k] for r in result_list], axis=0)
            for k in result_list[0]
        }

    # ---------------------------------------------------------------------
    # Select best restart
    # ---------------------------------------------------------------------
    all_f   = results['f']                 # (n_restarts,)

    if cfg.solution_selection == 'log_prob':
        best_idx = jnp.argmin(all_f)
    else:
        raise NotImplementedError(
            "solution_selection='log_ml' requires per-restart Hessian "
            "computation and is not yet implemented in the vmapped path. "
            "Use solution_selection='log_prob' for now."
        )

    best = {k: results[k][best_idx] for k in results}
    best['all_f']    = all_f
    best['best_idx'] = best_idx

    return best


# ---------------------------------------------------------------------------
# Final Hessian computation at MAP
# ---------------------------------------------------------------------------

def compute_final_hessian(b_map      : jnp.ndarray,
                           Ce         : jnp.ndarray,
                           A_bfgs     : jnp.ndarray,
                           signals    : dict,
                           bp         : jnp.ndarray,
                           Cp         : jnp.ndarray,
                           T_c        : float,
                           prior_cfg  : PriorConfig,
                           cfg        : OptimizerConfig) -> jnp.ndarray:
    """
    Compute the full (3N x 3N) Hessian at the MAP estimate b_map,
    optionally with the BFGS curvature correction applied to the
    nonlinear (gamma, beta) block.

    This is the Hessian used for the Laplace covariance and model scoring.

    Parameters
    ----------
    b_map   : jnp.ndarray, shape (3N,)   MAP parameter vector (full).
    Ce      : jnp.ndarray, scalar        MAP noise estimate.
    A_bfgs  : jnp.ndarray, shape (2N,2N) BFGS curvature correction.
    ...

    Returns
    -------
    H_full : jnp.ndarray, shape (3N, 3N)
    """
    # Full GN Hessian at b_map
    _, _, H_full, _, _, _ = calculate_cost(
        signals, Ce, b_map, bp, Cp, T_c, prior_cfg)

    if not cfg.use_bfgs_correction:
        return H_full

    # Apply BFGS correction to the nonlinear (gamma, beta) block
    Nb = b_map.shape[0]
    N  = Nb // 3

    # Indices of gamma and beta in b
    idx_gamma = jnp.arange(N) * 3 + 1    # 1, 4, 7, ...
    idx_beta  = jnp.arange(N) * 3 + 2    # 2, 5, 8, ...

    # Interleaved nonlinear indices in b: [gamma_1, beta_1, gamma_2, ...]
    idx_x = jnp.stack(
        [idx_gamma, idx_beta], axis=1).ravel()              # (2N,)

    H_full = H_full.at[jnp.ix_(idx_x, idx_x)].add(A_bfgs)
    return H_full