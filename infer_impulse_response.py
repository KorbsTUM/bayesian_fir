"""
infer_impulse_response.py
=========================
Top-level entry point for Bayesian impulse response inference.

This module mirrors the MATLAB function inferImpulseResponse and provides
a clean, single-function interface to the full inference pipeline:

    1. Validate inputs and configuration.
    2. For each candidate model order N:
        a. Generate the order-specific prior (bp, Cp, t_max).
        b. Prepare signals with the order-specific impulse response
           support T_h = t_max * T_c.
        c. Estimate the MAP posterior via multi-start LM optimisation.
        d. Compute the Laplace-approximate covariance and model scores.
    3. Rank candidate models by log marginal likelihood.
    4. Return the best model's posterior, impulse response, and ranking.
    5. Optionally run MCMC validation for the best model.

Minimal usage example
---------------------
    from infer_impulse_response import infer_impulse_response
    from core.prior import PriorConfig
    from inference.optimizer import OptimizerConfig
    import numpy as np
    import scipy.io as sio

    # Load LES data
    data = sio.loadmat('data/BRS_EderSilva23/data_raw_incomp.mat')
    u = data['u'].ravel()
    q = data['q'].ravel()
    t = data['t'].ravel()

    # Physical parameters
    L_ref = 50e-3     # flame length [m]
    U_ref = 11.3      # bulk velocity [m/s]
    T_c   = L_ref / U_ref

    # Run inference
    result = infer_impulse_response(
        u, q, t, T_c,
        model_orders = [1, 2, 3, 4, 5],
    )

    # Access results
    print(result.ranking)
    print(result.best.h.time)
    print(result.best.h.val)

References:
    Yoko & Polifke (2026), Algorithm 1.
"""

import numpy as np
import jax.numpy as jnp
from dataclasses import dataclass, field
from typing import Optional

from core.prior import PriorConfig, generate_prior
from inference.optimizer import OptimizerConfig
from inference.posterior import (
    PosteriorResult,
    ModelRanking,
    estimate_posterior,
)
from inference.mcmc import (
    MCMCResult,
    run_mcmc_from_posterior,
)
from signals.prepare import prepare_signals


# ---------------------------------------------------------------------------
# Top-level configuration
# ---------------------------------------------------------------------------

@dataclass
class InferenceConfig:
    """
    Top-level configuration combining prior, optimizer, and MCMC settings.

    Attributes
    ----------
    prior    : PriorConfig       Prior hyperparameters.
    optimizer: OptimizerConfig   LM optimizer settings.
    run_mcmc : bool              Whether to run MCMC for the best model.
    mcmc_iter: int               Number of MCMC iterations.
    mcmc_burn: float             MCMC burn-in fraction.
    mcmc_thin: int               MCMC thinning interval.
    mcmc_seed: int               MCMC random seed.
    mcmc_scan: bool              Use JAX scan (True) or Python loop (False).
    Ce0      : float or None     Initial noise variance. None -> 1e-4.
    ds_limit : int or None       Max downsampling factor. None -> no limit.
    n_eval_pts: int              Points for impulse response evaluation.
    verbose  : bool              Whether to print progress to stdout.
    """
    prior     : PriorConfig      = field(default_factory=PriorConfig)
    optimizer : OptimizerConfig  = field(default_factory=OptimizerConfig)
    run_mcmc  : bool             = False
    mcmc_iter : int              = 200_000
    mcmc_burn : float            = 0.25
    mcmc_thin : int              = 1
    mcmc_seed : int              = 0
    mcmc_scan : bool             = True
    Ce0       : Optional[float]  = None
    ds_limit  : Optional[int]    = None
    n_eval_pts: int              = 500
    verbose   : bool             = True


def default_config() -> InferenceConfig:
    """Return the default InferenceConfig matching paper settings."""
    return InferenceConfig()


# ---------------------------------------------------------------------------
# Top-level result container
# ---------------------------------------------------------------------------

@dataclass
class InferenceResult:
    """
    Container for the full inference result.

    Attributes
    ----------
    best     : PosteriorResult   Posterior for the best model order.
    all      : list              PosteriorResult for each candidate order.
    ranking  : ModelRanking      Model ranking metrics.
    mcmc     : MCMCResult or None  MCMC result if run_mcmc=True.
    config   : InferenceConfig   Configuration used.
    T_c      : float             Convective timescale used.
    """
    best    : PosteriorResult
    all     : list
    ranking : ModelRanking
    mcmc    : Optional[MCMCResult]
    config  : InferenceConfig
    T_c     : float

    def summary(self) -> str:
        """Return a human-readable summary of the inference result."""
        lines = [
            "\n" + "="*60,
            "Bayesian Impulse Response Inference - Summary",
            "="*60,
            f"Best model order : N = {self.best.N}",
            f"logML            : {self.best.logML:.2f}",
            f"logBFL           : {self.best.logBFL:.2f}",
            f"logOF            : {self.best.logOF:.2f}",
            f"Noise std (Ce^0.5): {self.best.Ce**0.5:.4e}",
            "",
            "MAP parameters (physical space):",
            f"  {'i':>3}  {'n_i':>10}  {'tau_i/T_c':>12}  {'sig_i/T_c':>12}",
            "-"*42,
        ]
        N = self.best.N
        a = np.array(self.best.a_map)
        Ca = np.array(self.best.Ca_map)
        for i in range(N):
            n_i   = a[3*i]
            tau_i = a[3*i+1] / self.T_c
            sig_i = a[3*i+2] / self.T_c
            # 1-sigma uncertainties from diagonal of Ca
            dn    = Ca[3*i,   3*i  ] ** 0.5
            dtau  = Ca[3*i+1, 3*i+1] ** 0.5 / self.T_c
            dsig  = Ca[3*i+2, 3*i+2] ** 0.5 / self.T_c
            lines.append(
                f"  {i+1:>3}  "
                f"{n_i:>+8.3f}±{dn:.3f}  "
                f"{tau_i:>10.3f}±{dtau:.3f}  "
                f"{sig_i:>10.3f}±{dsig:.3f}"
            )
        lines.append("="*60)
        if self.mcmc is not None:
            lines.append(
                f"MCMC: {self.mcmc.post_samples.shape[1]} post-burn-in samples, "
                f"accept rate {self.mcmc.accept_rate:.1%}"
            )
            lines.append("="*60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def infer_impulse_response(u             : np.ndarray,
                            q             : np.ndarray,
                            t             : np.ndarray,
                            T_c           : float,
                            model_orders  : list = None,
                            config        : InferenceConfig = None
                            ) -> InferenceResult:
    """
    Infer the flame impulse response from input-output time-series data.

    Parameters
    ----------
    u            : np.ndarray, shape (M,)
        Input velocity fluctuation signals (normalised, zero-mean).
    q            : np.ndarray, shape (M,)
        Output heat release rate fluctuation (normalised, zero-mean).
    t            : np.ndarray, shape (M,)
        Time vector [s].
    T_c          : float
        Convective timescale [s] (e.g. flame_length / bulk_velocity).
    model_orders : list of int, optional
        Candidate model orders to evaluate. Default [1, 2, 3, 4, 5].
    config       : InferenceConfig, optional
        Full configuration. Default: default_config().

    Returns
    -------
    result : InferenceResult
        Container with best model posterior, all posteriors, model
        ranking, and optional MCMC result.

    Raises
    ------
    ValueError
        If inputs are inconsistent (shape mismatch, fixed T_h with
        multiple model orders, etc.).
    """
    # ------------------------------------------------------------------
    # Defaults
    # ------------------------------------------------------------------
    if model_orders is None:
        model_orders = [1, 2, 3, 4, 5]
    if config is None:
        config = default_config()

    prior_cfg = config.prior
    opt_cfg   = config.optimizer

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    u = np.asarray(u, dtype=np.float64).ravel()
    q = np.asarray(q, dtype=np.float64).ravel()
    t = np.asarray(t, dtype=np.float64).ravel()

    if not (len(u) == len(q) == len(t)):
        raise ValueError(
            f"u, q, t must all have the same length. "
            f"Got {len(u)}, {len(q)}, {len(t)}."
        )
    if T_c <= 0:
        raise ValueError(f"T_c must be positive, got {T_c}.")
    if len(model_orders) == 0:
        raise ValueError("model_orders must contain at least one entry.")

    # Guard: fixed T_h + multiple model orders biases model ranking
    if prior_cfg.T_h is not None and len(model_orders) > 1:
        raise ValueError(
            "Model ranking across multiple orders is not meaningful when "
            "prior_cfg.T_h is fixed (see Section 4.4 of Yoko & Polifke "
            "2026). Either set prior_cfg.T_h = None for automatic support "
            "selection, or pass a single model order."
        )

    if config.verbose:
        print("\n" + "="*60)
        print("Bayesian Impulse Response Inference")
        print("="*60)
        print(f"  Signal length  : {len(u)} samples")
        print(f"  Duration       : {(t[-1]-t[0])*1e3:.1f} ms")
        print(f"  fs             : {1/np.mean(np.diff(t)):.1f} Hz")
        print(f"  T_c            : {T_c*1e3:.2f} ms")
        print(f"  Model orders   : {model_orders}")
        print(f"  Parallel       : {opt_cfg.use_parallel}")
        print("="*60)

    # ------------------------------------------------------------------
    # Sampling frequency
    # ------------------------------------------------------------------
    fs = 1.0 / float(np.mean(np.diff(t)))

    # ------------------------------------------------------------------
    # Prepare a base fine-resolution signals struct (no downsampling).
    # This is re-used across model orders: for each N we re-prepare
    # with the order-specific T_h by calling prepare_signals again.
    # We store u and q as plain numpy for re-preparation.
    # ------------------------------------------------------------------
    all_results : list[PosteriorResult] = []
    logML_list  = []
    logBFL_list = []
    logOF_list  = []

    for N in model_orders:
        if config.verbose:
            print(f"\n[Model order N={N}]")

        # --------------------------------------------------------------
        # Generate order-specific prior
        # --------------------------------------------------------------
        bp, Cp, names, t_max = generate_prior(N, T_c, prior_cfg)
        T_h = t_max * T_c

        if config.verbose:
            print(f"  T_h = {T_h*1e3:.2f} ms  "
                  f"(t_max = {t_max:.2f} T_c)")

        # --------------------------------------------------------------
        # Prepare signals with order-specific T_h
        # --------------------------------------------------------------
        signals = prepare_signals(u, q, fs, T_h,
                                   ds_limit=config.ds_limit)

        if config.verbose:
            sig = signals['coarse']
            Nd  = sig['n'] - sig['valid']
            print(f"  Coarse fs = {sig['fs']:.1f} Hz  |  "
                  f"DS = {sig['ds_factor']}x  |  "
                  f"Valid samples = {Nd}")

        # --------------------------------------------------------------
        # Estimate posterior
        # --------------------------------------------------------------
        result = estimate_posterior(
            signals   = signals,
            bp        = bp,
            Cp        = Cp,
            T_c       = T_c,
            prior_cfg = prior_cfg,
            opt_cfg   = opt_cfg,
            N         = N,
            names     = names,
            Ce0       = config.Ce0,
            n_eval_pts= config.n_eval_pts,
        )

        all_results.append(result)
        logML_list .append(result.logML)
        logBFL_list.append(result.logBFL)
        logOF_list .append(result.logOF)

    # ------------------------------------------------------------------
    # Model ranking
    # ------------------------------------------------------------------
    ranking = ModelRanking(model_orders, logML_list, logBFL_list, logOF_list)

    if config.verbose:
        ranking.print_table()

    # ------------------------------------------------------------------
    # Select best model
    # ------------------------------------------------------------------
    best_idx    = model_orders.index(ranking.best_N)
    best_result = all_results[best_idx]

    if config.verbose:
        print(f"\nSelected model: N = {ranking.best_N}")

    # ------------------------------------------------------------------
    # Optional MCMC for the best model
    # ------------------------------------------------------------------
    mcmc_result = None
    if config.run_mcmc:
        # Re-prepare signals with best model's T_h for MCMC
        _, Cp_best, _, t_max_best = generate_prior(
            ranking.best_N, T_c, prior_cfg)
        bp_best, _, _, _ = generate_prior(
            ranking.best_N, T_c, prior_cfg)
        T_h_best = t_max_best * T_c
        signals_best = prepare_signals(
            u, q, fs, T_h_best, ds_limit=config.ds_limit)

        mcmc_result = run_mcmc_from_posterior(
            posterior  = best_result,
            signals    = signals_best,
            bp         = bp_best,
            Cp         = Cp_best,
            T_c        = T_c,
            prior_cfg  = prior_cfg,
            n_iter     = config.mcmc_iter,
            burn_frac  = config.mcmc_burn,
            thin       = config.mcmc_thin,
            seed       = config.mcmc_seed,
            use_scan   = config.mcmc_scan,
        )

    # ------------------------------------------------------------------
    # Package and return
    # ------------------------------------------------------------------
    inference_result = InferenceResult(
        best    = best_result,
        all     = all_results,
        ranking = ranking,
        mcmc    = mcmc_result,
        config  = config,
        T_c     = T_c,
    )

    if config.verbose:
        print(inference_result.summary())

    return inference_result