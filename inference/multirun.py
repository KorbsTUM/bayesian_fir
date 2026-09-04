"""
inference/multirun.py
======================
Joint inference across multiple runs of the same nominal condition (e.g.
several repeated measurements of one physical operating point): one
shared parameter vector b - hence one shared model order N and one shared
noise variance Ce - fit against the combined likelihood of all runs at
once, rather than independent per-run fits. This is a different mechanism
from pooling independent cases that genuinely differ (each getting its
own parameters, only the model order shared) - here every run is assumed
to be a noisy realisation of the *same* underlying impulse response, so
combining them tightens one shared estimate instead of producing several
separately weaker ones. A single short run risks an unreliable fit; 3
runs combined roughly triples the informative (post-downsampling, post-T_h)
sample count feeding that one estimate.

The mechanism itself lives in core.cost: calculate_cost, calculate_cost_varpro,
and calculate_cost_val all accept either their usual single `signals` dict
or a list of per-run `signals` dicts (each independently prepared via
prepare_signals, so each run keeps its own correct convolution/valid-region
- no cross-run contamination). Every other module in the inference
pipeline (inference/optimizer.py's LM restart loop, inference/posterior.py's
Hessian/logML step, inference/variational.py's ELBO training and Laplace
warm start, inference/mcmc.py's log-posterior) already treats `signals` as
opaque and passes it straight through to those three functions, so none
of them needed changes for the multi-run case - this module is purely
orchestration: prepare a list of per-run signals instead of one, and call
the exact same estimate_posterior/estimate_posterior_vi entry points.

Two-stage method selection (mirrors the WET Kornilov pooled-inference
design from the sibling multi-dataset-model-selection branch, applied to
one joint fit instead of many independent per-case ones, since that
design isn't present on this branch): a ranking_method/ranking_optimizer/
ranking_vi choice used for the per-order joint-evidence sweep, and an
independently-selectable param_method/param_optimizer/param_vi choice
(defaulting to the ranking choice) used for the final joint parameter fit
at the winning order - useful because VI is substantially more expensive
than Laplace, so a cheap Laplace sweep across candidate orders followed
by a single VI fit only at the winning order avoids paying VI's cost once
per candidate order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

import numpy as np
import jax.numpy as jnp

from core.prior import PriorConfig, generate_prior
from inference.optimizer import OptimizerConfig
from inference.posterior import PosteriorResult, ModelRanking, estimate_posterior
from inference.variational import VIConfig, estimate_posterior_vi
from inference.mcmc import MCMCResult, run_mcmc_from_posterior

if TYPE_CHECKING:
    # config.defaults imports inference.optimizer, and inference/multirun.py
    # is the first thing inside the inference package to reach outside it
    # via a top-level config.defaults import - importing it back here at
    # module load time creates a config <-> inference cycle that manifests
    # depending on which module a caller happens to import first (same
    # issue, same fix, as inference/pooled.py on the sibling
    # multi-dataset-model-selection branch). Imported lazily at call time
    # instead - see MultiRunConfig.preproc / infer_impulse_response_multirun.
    from config.defaults import PreprocConfig
from signals.prepare import prepare_signals


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MultiRunConfig:
    """
    Configuration for infer_impulse_response_multirun.

    Attributes
    ----------
    prior             : PriorConfig       Shared across all runs (there is
                                          one joint parameter vector).
    preproc           : PreprocConfig or None   Shared downsampling settings.
                                          None -> PreprocConfig() (no
                                          downsampling), resolved lazily by
                                          infer_impulse_response_multirun.
    ranking_method    : str               'laplace' (default) or 'vi', used
                                          for the per-order joint-evidence
                                          sweep.
    ranking_optimizer : OptimizerConfig   Used for the ranking sweep.
    ranking_vi        : VIConfig          Used for the ranking sweep when
                                          ranking_method='vi'.
    param_method      : str or None       Backend for the final parameter
                                          fit at the winning order. None ->
                                          ranking_method.
    param_optimizer   : OptimizerConfig or None   None -> ranking_optimizer.
    param_vi          : VIConfig or None          None -> ranking_vi.
    Ce0               : float or None     Initial/fixed noise variance
                                          (shared across all runs).
                                          None -> 1e-4 (estimated) or
                                          required if infer_noise=False.
    n_eval_pts        : int               Impulse-response evaluation points.
    run_mcmc          : bool              Whether to run MCMC validation at
                                          the final joint fit.
    mcmc_iter         : int               "
    mcmc_burn         : float             "
    mcmc_thin         : int               "
    mcmc_seed         : int               "
    mcmc_scan         : bool              "
    verbose           : bool              Whether to print progress.
    """
    prior             : PriorConfig             = field(default_factory=PriorConfig)
    preproc           : Optional[PreprocConfig] = None   # None -> PreprocConfig(), resolved lazily
    ranking_method    : str                     = 'laplace'
    ranking_optimizer : OptimizerConfig         = field(default_factory=OptimizerConfig)
    ranking_vi        : VIConfig                = field(default_factory=VIConfig)
    param_method      : Optional[str]           = None
    param_optimizer   : Optional[OptimizerConfig] = None
    param_vi          : Optional[VIConfig]        = None
    Ce0               : Optional[float]         = None
    n_eval_pts        : int                     = 500
    run_mcmc          : bool                    = False
    mcmc_iter         : int                     = 200_000
    mcmc_burn         : float                   = 0.25
    mcmc_thin         : int                     = 1
    mcmc_seed         : int                     = 0
    mcmc_scan         : bool                    = True
    verbose           : bool                    = True


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class MultiRunResult:
    """
    Attributes
    ----------
    N_star    : int              Model order selected by the joint-evidence sweep.
    ranking   : ModelRanking     Per-order joint logML/logBFL/logOF across the sweep.
    best      : PosteriorResult  Final joint DTD parameter fit at N_star.
    mcmc      : MCMCResult or None
    run_names : list of str      Labels for the runs that were combined.
    T_c       : float
    """
    N_star    : int
    ranking   : ModelRanking
    best      : PosteriorResult
    mcmc      : Optional[MCMCResult]
    run_names : List[str]
    T_c       : float

    def summary(self) -> str:
        """Return a human-readable summary of the joint fit."""
        lines = [
            "\n" + "=" * 60,
            "Joint Multi-Run Inference - Summary",
            "=" * 60,
            f"Runs combined    : {', '.join(self.run_names)}",
            f"Method           : {self.best.method}",
            f"Best model order : N = {self.best.N}",
            f"logML            : {self.best.logML:.2f}",
            f"logBFL           : {self.best.logBFL:.2f}",
            f"logOF            : {self.best.logOF:.2f}",
            f"Noise std (Ce^0.5): {self.best.Ce**0.5:.4e}",
            "",
            "MAP parameters (physical space, shared across all runs):",
            f"  {'i':>3}  {'n_i':>10}  {'tau_i/T_c':>12}  {'sig_i/T_c':>12}",
            "-" * 42,
        ]
        N = self.best.N
        a  = np.array(self.best.a_map)
        Ca = np.array(self.best.Ca_map)
        for i in range(N):
            n_i, tau_i, sig_i = a[3*i], a[3*i+1] / self.T_c, a[3*i+2] / self.T_c
            dn   = Ca[3*i,   3*i  ] ** 0.5
            dtau = Ca[3*i+1, 3*i+1] ** 0.5 / self.T_c
            dsig = Ca[3*i+2, 3*i+2] ** 0.5 / self.T_c
            lines.append(
                f"  {i+1:>3}  {n_i:>+8.3f}±{dn:.3f}  "
                f"{tau_i:>10.3f}±{dtau:.3f}  {sig_i:>10.3f}±{dsig:.3f}"
            )
        lines.append("=" * 60)
        if self.mcmc is not None:
            lines.append(
                f"MCMC: {self.mcmc.post_samples.shape[1]} post-burn-in samples, "
                f"accept rate {self.mcmc.accept_rate:.1%}"
            )
            lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prepare_run_signals(u_list, q_list, t_list, T_h, preproc_cfg):
    """Prepare each run's signals independently (own convolution/valid region)."""
    signals_list = []
    for u, q, t in zip(u_list, q_list, t_list):
        fs = 1.0 / float(np.mean(np.diff(t)))
        signals_list.append(prepare_signals(
            u, q, fs, T_h, ds_mode=preproc_cfg.DSmode, ds_value=preproc_cfg.DSvalue))
    return signals_list


def _estimate(signals_list, bp, Cp, T_c, prior_cfg, opt_cfg, method, vi_cfg,
              N, names, Ce0, n_eval_pts) -> PosteriorResult:
    if method == 'vi':
        return estimate_posterior_vi(
            signals=signals_list, bp=bp, Cp=Cp, T_c=T_c, prior_cfg=prior_cfg,
            opt_cfg=opt_cfg, vi_cfg=vi_cfg, N=N, names=names, Ce0=Ce0,
            n_eval_pts=n_eval_pts)
    return estimate_posterior(
        signals=signals_list, bp=bp, Cp=Cp, T_c=T_c, prior_cfg=prior_cfg,
        opt_cfg=opt_cfg, N=N, names=names, Ce0=Ce0, n_eval_pts=n_eval_pts)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def infer_impulse_response_multirun(u_list       : List[np.ndarray],
                                     q_list       : List[np.ndarray],
                                     t_list       : List[np.ndarray],
                                     T_c          : float,
                                     model_orders : Optional[List[int]] = None,
                                     config       : Optional[MultiRunConfig] = None,
                                     run_names    : Optional[List[str]] = None
                                     ) -> MultiRunResult:
    """
    Jointly infer one shared flame impulse response from several runs of
    the same nominal condition. See the module docstring for the design.

    Parameters
    ----------
    u_list, q_list, t_list : list of np.ndarray
        Input/output/time arrays, one triple per run. Runs may have
        different lengths - each is prepared independently.
    T_c          : float   Convective timescale [s], shared across all runs.
    model_orders : list of int, optional   Default [1, 2, 3, 4, 5].
    config       : MultiRunConfig, optional   Default MultiRunConfig().
    run_names    : list of str, optional   Labels for diagnostics/summary.
                   Default ["run1", "run2", ...].

    Returns
    -------
    result : MultiRunResult
    """
    if model_orders is None:
        model_orders = [1, 2, 3, 4, 5]
    if config is None:
        config = MultiRunConfig()
    if run_names is None:
        run_names = [f"run{i+1}" for i in range(len(u_list))]

    prior_cfg = config.prior

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if not (len(u_list) == len(q_list) == len(t_list) == len(run_names)):
        raise ValueError(
            f"u_list, q_list, t_list, run_names must all have the same "
            f"length. Got {len(u_list)}, {len(q_list)}, {len(t_list)}, "
            f"{len(run_names)}.")
    if len(u_list) == 0:
        raise ValueError("At least one run is required.")
    for name, u, q, t in zip(run_names, u_list, q_list, t_list):
        u, q, t = np.asarray(u), np.asarray(q), np.asarray(t)
        if not (len(u) == len(q) == len(t)):
            raise ValueError(
                f"[{name}] u, q, t must have the same length. "
                f"Got {len(u)}, {len(q)}, {len(t)}.")
    if T_c <= 0:
        raise ValueError(f"T_c must be positive, got {T_c}.")
    if len(model_orders) == 0:
        raise ValueError("model_orders must contain at least one entry.")
    if config.ranking_method not in ('laplace', 'vi'):
        raise ValueError(
            f"config.ranking_method must be 'laplace' or 'vi', got "
            f"{config.ranking_method!r}.")
    param_method = config.param_method or config.ranking_method
    if param_method not in ('laplace', 'vi'):
        raise ValueError(
            f"config.param_method must be 'laplace' or 'vi', got {param_method!r}.")
    if prior_cfg.T_h is not None and len(model_orders) > 1:
        raise ValueError(
            "Model ranking across multiple orders is not meaningful when "
            "prior_cfg.T_h is fixed. Either set prior_cfg.T_h = None for "
            "automatic support selection, or pass a single model order.")

    u_list = [np.asarray(u, dtype=np.float64).ravel() for u in u_list]
    q_list = [np.asarray(q, dtype=np.float64).ravel() for q in q_list]
    t_list = [np.asarray(t, dtype=np.float64).ravel() for t in t_list]

    preproc_cfg = config.preproc
    if preproc_cfg is None:
        from config.defaults import PreprocConfig
        preproc_cfg = PreprocConfig()

    if config.verbose:
        print("\n" + "=" * 60)
        print("Joint Multi-Run Bayesian Impulse Response Inference")
        print("=" * 60)
        print(f"  Runs           : {run_names}")
        for name, u, t in zip(run_names, u_list, t_list):
            fs = 1.0 / float(np.mean(np.diff(t)))
            print(f"    [{name}] {len(u)} samples at {fs:.1f} Hz "
                  f"({(t[-1]-t[0])*1e3:.1f} ms)")
        print(f"  T_c            : {T_c*1e3:.2f} ms")
        print(f"  Model orders   : {model_orders}")
        print("=" * 60)

    # ------------------------------------------------------------------
    # Stage 1: joint-evidence sweep across candidate orders
    # ------------------------------------------------------------------
    ranking_results = []
    logML_list, logBFL_list, logOF_list = [], [], []

    for N in model_orders:
        if config.verbose:
            print(f"\n[Model order N={N}]")

        bp, Cp, names, t_max = generate_prior(N, T_c, prior_cfg)
        T_h = t_max * T_c

        if config.verbose:
            print(f"  T_h = {T_h*1e3:.2f} ms  (t_max = {t_max:.2f} T_c)")

        signals_list = _prepare_run_signals(u_list, q_list, t_list, T_h, preproc_cfg)

        if config.verbose:
            for name, sig in zip(run_names, signals_list):
                coarse = sig['coarse']
                print(f"  [{name}] coarse fs = {coarse['fs']:.1f} Hz  |  "
                      f"DS = {coarse['ds_factor']}x  |  "
                      f"valid samples = {coarse['n'] - coarse['valid']}")

        result = _estimate(signals_list, bp, Cp, T_c, prior_cfg,
                            config.ranking_optimizer, config.ranking_method,
                            config.ranking_vi, N, names, config.Ce0, config.n_eval_pts)

        ranking_results.append(result)
        logML_list.append(result.logML)
        logBFL_list.append(result.logBFL)
        logOF_list.append(result.logOF)

    ranking = ModelRanking(model_orders, logML_list, logBFL_list, logOF_list)
    if config.verbose:
        ranking.print_table()

    N_star = ranking.best_N
    if config.verbose:
        print(f"\nSelected shared model order: N = {N_star}")

    # ------------------------------------------------------------------
    # Stage 2: final joint parameter fit at N_star. Reuse stage 1's
    # result for N_star directly if param_method/param_optimizer/param_vi
    # weren't overridden (there's only one fit per order here, not one
    # per dataset x order like the Kornilov pooling case, so avoiding a
    # redundant refit is worth doing rather than always re-running).
    # ------------------------------------------------------------------
    param_opt = config.param_optimizer or config.ranking_optimizer
    param_vi  = config.param_vi or config.ranking_vi
    needs_refit = (
        param_method != config.ranking_method or
        param_opt    != config.ranking_optimizer or
        (param_method == 'vi' and param_vi != config.ranking_vi)
    )

    best_idx = model_orders.index(N_star)
    if not needs_refit:
        best_result = ranking_results[best_idx]
    else:
        if config.verbose:
            print(f"\n=== Final parameter fit at N={N_star} ({param_method}) ===")
        bp, Cp, names, t_max = generate_prior(N_star, T_c, prior_cfg)
        T_h = t_max * T_c
        signals_list = _prepare_run_signals(u_list, q_list, t_list, T_h, preproc_cfg)
        best_result = _estimate(signals_list, bp, Cp, T_c, prior_cfg,
                                 param_opt, param_method, param_vi,
                                 N_star, names, config.Ce0, config.n_eval_pts)

    # ------------------------------------------------------------------
    # Optional MCMC validation at the final joint fit
    # ------------------------------------------------------------------
    mcmc_result = None
    if config.run_mcmc:
        _, Cp_best, names_best, t_max_best = generate_prior(N_star, T_c, prior_cfg)
        bp_best, _, _, _ = generate_prior(N_star, T_c, prior_cfg)
        T_h_best = t_max_best * T_c
        signals_list_best = _prepare_run_signals(u_list, q_list, t_list, T_h_best, preproc_cfg)

        mcmc_result = run_mcmc_from_posterior(
            posterior  = best_result,
            signals    = signals_list_best,
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

    result = MultiRunResult(
        N_star    = N_star,
        ranking   = ranking,
        best      = best_result,
        mcmc      = mcmc_result,
        run_names = run_names,
        T_c       = T_c,
    )

    if config.verbose:
        print(result.summary())

    return result
