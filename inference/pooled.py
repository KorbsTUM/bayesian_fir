"""
inference/pooled.py
====================
Pooled model-order selection across multiple datasets: find the single
model order N best supported by the *combined* evidence of several
independent time series, then fit each dataset's own DTD parameters at
that shared N.

This is an orchestration layer over the existing, unmodified
infer_impulse_response - not new inference machinery. For independent
datasets sharing a hypothesis N, Bayes' rule gives

    logML_total(N) = sum_i logML_i(N)

and every logML_i(N) is exactly what infer_impulse_response already
computes for one dataset. So "the most likely N for the whole set" is
just: sweep every (dataset, order) pair, sum the per-dataset logML across
datasets for each order, and take the argmax.

Two-stage design
-------------------
    Stage 1 (ranking):   every dataset x every candidate order, using
                          config.ranking_method - this is the expensive,
                          repeated-many-times part, so a cheap backend
                          (typically Laplace) is the sensible default.
    Stage 2 (parameters): every dataset, once, at the single N selected
                          by stage 1, using config.param_method (defaults
                          to ranking_method if not set) - this can afford
                          a richer backend (e.g. 'vi') since it only runs
                          once per dataset instead of once per (dataset,
                          order) pair.

The two method choices are each uniform across the whole dataset pool
(not per-dataset): mixing methods *within* a single summed logML would
combine two different approximation biases into one number, which stage 1
would do if datasets used different methods there. Stage 2 has no such
constraint (nothing is summed), but is still kept pool-uniform for
simplicity and comparability across datasets.

The prior (prior_cfg) is shared across all datasets - it lives entirely
in non-dimensional space (gamma = log(delta_tau/T_c), beta = log(sigma/T_c)),
so each dataset's own physical scale enters only through its own T_c, not
through the prior. Per-dataset noise handling (Ce0/infer_noise) can still
differ per dataset via DatasetSpec, since that's orthogonal to the shared
prior.

Output volume: with potentially many datasets, per-dataset
InferenceConfig.verbose is always forced to False here regardless of
PooledInferenceConfig.verbose (letting every dataset print its full
restart-by-restart progress would flood stdout); the orchestrator prints
one summary line per dataset per stage plus the final pooled ranking
table when config.verbose=True.

Explicitly out of scope: this loops over datasets sequentially in Python,
it does not vmap/batch the per-dataset loop on-device. That would require
padding/masking every dataset to a uniform shape (threaded through
core.cost's residual/likelihood/MML Nd computation), dropping
prepare_signals's anti-aliased downsampling (not JAX-traceable), and
removing per-dataset progress printing - real rework of already-validated
core files, deliberately deferred rather than built here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, NamedTuple, Optional, TYPE_CHECKING

import numpy as np
import jax.numpy as jnp

from core.prior import PriorConfig
from inference.optimizer import OptimizerConfig
from inference.variational import VIConfig

if TYPE_CHECKING:
    # config.defaults imports inference.optimizer, and infer_impulse_response
    # imports config.defaults, so importing either back here at module load
    # time would create a config/infer_impulse_response <-> inference import
    # cycle (it only manifests depending on which module a caller happens to
    # import first, since inference/pooled.py is the first thing inside the
    # inference package to reach outside it). Both are instead imported
    # lazily at call time, once every package involved is already fully
    # loaded - see _dataset_inference_config and infer_shared_model_order.
    from config.defaults import PreprocConfig
    from infer_impulse_response import InferenceConfig, InferenceResult


# ---------------------------------------------------------------------------
# Per-dataset input
# ---------------------------------------------------------------------------

class DatasetSpec(NamedTuple):
    """
    One dataset's input to pooled inference - same shapes
    infer_impulse_response takes directly, plus optional per-dataset noise
    overrides.

    Attributes
    ----------
    name        : str          Label, used in diagnostics/print_table.
    u, q, t     : np.ndarray, shape (M_i,)   Input/output/time - M_i may
                                differ freely between datasets (each is
                                processed independently; see module
                                docstring for why this isn't vmapped).
    T_c         : float        Convective timescale [s] for this dataset.
    Ce0         : float or None   Per-dataset noise override. None -> use
                                PooledInferenceConfig.Ce0.
    infer_noise : bool or None    Per-dataset override of
                                OptimizerConfig.infer_noise (e.g. for a
                                laminar/noise-free dataset in an otherwise
                                noisy pool). None -> use the pool's
                                optimizer config unchanged.
    """
    name        : str
    u           : np.ndarray
    q           : np.ndarray
    t           : np.ndarray
    T_c         : float
    Ce0         : Optional[float] = None
    infer_noise : Optional[bool]  = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PooledInferenceConfig:
    """
    Configuration for infer_shared_model_order.

    Attributes
    ----------
    prior             : PriorConfig       Shared across all datasets (see
                                          module docstring for why this is
                                          the physically correct default).
    preproc           : PreprocConfig or None   Shared downsampling settings.
                                          None -> PreprocConfig() (no
                                          downsampling), resolved lazily by
                                          infer_shared_model_order.
    ranking_method    : str               'laplace' (default) or 'vi', used
                                          for the stage-1 sweep.
    ranking_optimizer : OptimizerConfig   Used for stage 1.
    ranking_vi        : VIConfig          Used for stage 1 when
                                          ranking_method='vi'.
    param_method      : str or None       Backend for stage 2. None ->
                                          ranking_method.
    param_optimizer   : OptimizerConfig or None   None -> ranking_optimizer.
    param_vi          : VIConfig or None          None -> ranking_vi.
    Ce0               : float or None     Pool-wide default noise variance;
                                          overridden per dataset by
                                          DatasetSpec.Ce0 when set.
    n_eval_pts        : int               Impulse-response evaluation points,
                                          forwarded to infer_impulse_response.
    run_mcmc          : bool              Whether to run MCMC validation in
                                          stage 2 (once per dataset, at the
                                          shared N_star). Never runs during
                                          the stage-1 ranking sweep. Default
                                          False.
    mcmc_iter         : int               Forwarded to InferenceConfig when
                                          run_mcmc=True (same defaults).
    mcmc_burn         : float             "
    mcmc_thin         : int               "
    mcmc_seed         : int               "
    mcmc_scan         : bool              "
    verbose           : bool              Pooled-level output only - see
                                          module docstring's "Output volume".
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
# Pooled ranking result
# ---------------------------------------------------------------------------

class PooledModelRanking:
    """
    Model ranking pooled across datasets, mirroring
    inference.posterior.ModelRanking's style.

    Attributes
    ----------
    orders        : list of int              Candidate model orders.
    dataset_names : list of str               Dataset labels, row order
                                              matches logML_matrix's rows.
    logML_matrix  : jnp.ndarray (n_datasets, n_orders)
                                              Per-dataset, per-order logML
                                              (or ELBO, if ranking_method='vi').
    logML_total   : jnp.ndarray (n_orders,)   Summed across datasets.
    best_N        : int                       Order maximizing logML_total.
    """

    def __init__(self, orders, dataset_names, logML_matrix):
        self.orders        = list(orders)
        self.dataset_names = list(dataset_names)
        self.logML_matrix  = jnp.asarray(logML_matrix)
        self.logML_total   = jnp.sum(self.logML_matrix, axis=0)
        best_idx           = int(jnp.argmax(self.logML_total))
        self.best_N        = self.orders[best_idx]

    def print_table(self):
        """
        Print a per-dataset breakdown (each row normalized to that
        dataset's own best order, so individual disagreement is visible
        at a glance) plus the pooled TOTAL row.
        """
        col_w  = 10
        header = f"{'dataset':>20}  " + "  ".join(
            f"{'N=' + str(N):>{col_w}}" for N in self.orders)
        rule = "-" * len(header)

        print("\nPooled model ranking (logML, normalized per row to its own max):")
        print(rule)
        print(header)
        print(rule)
        for i, name in enumerate(self.dataset_names):
            row   = self.logML_matrix[i]
            row_n = row - jnp.max(row)
            vals  = "  ".join(f"{float(v):>{col_w}.2f}" for v in row_n)
            print(f"{name:>20}  {vals}")
        print(rule)
        total_n = self.logML_total - jnp.max(self.logML_total)
        vals    = "  ".join(f"{float(v):>{col_w}.2f}" for v in total_n)
        print(f"{'TOTAL (pooled)':>20}  {vals}")
        print(rule)
        print(f"Best shared N = {self.best_N}")


# ---------------------------------------------------------------------------
# Pooled result container
# ---------------------------------------------------------------------------

@dataclass
class PooledInferenceResult:
    """
    Attributes
    ----------
    N_star          : int                  Shared model order selected by
                                           the pooled ranking.
    pooled_ranking  : PooledModelRanking   Stage-1 ranking details.
    ranking_results : list of InferenceResult
                                           Stage-1 result per dataset (all
                                           candidate orders) - kept for
                                           diagnostics.
    param_results   : list of InferenceResult
                                           Stage-2 result per dataset (a
                                           single order, N_star). `.best`
                                           on each is the final per-dataset
                                           PosteriorResult.
    dataset_names   : list of str
    """
    N_star          : int
    pooled_ranking  : PooledModelRanking
    ranking_results : List[InferenceResult]
    param_results   : List[InferenceResult]
    dataset_names   : List[str]

    def per_dataset(self, name: str):
        """Return the final PosteriorResult (stage 2, at N_star) for a dataset by name."""
        idx = self.dataset_names.index(name)
        return self.param_results[idx].best


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dataset_inference_config(spec       : DatasetSpec,
                               prior_cfg  : PriorConfig,
                               preproc_cfg: PreprocConfig,
                               method     : str,
                               opt_cfg    : OptimizerConfig,
                               vi_cfg     : VIConfig,
                               pool_ce0   : Optional[float],
                               n_eval_pts : int,
                               run_mcmc   : bool = False,
                               mcmc_iter  : int = 200_000,
                               mcmc_burn  : float = 0.25,
                               mcmc_thin  : int = 1,
                               mcmc_seed  : int = 0,
                               mcmc_scan  : bool = True) -> InferenceConfig:
    """Build one dataset's InferenceConfig, applying its noise overrides.

    MCMC args default to off/InferenceConfig's own defaults; only the
    stage-2 call site in infer_shared_model_order passes non-default
    values (see module docstring: MCMC never runs during the stage-1
    ranking sweep).
    """
    from infer_impulse_response import InferenceConfig   # see import note at top of file
    if spec.infer_noise is not None:
        opt_cfg = opt_cfg._replace(infer_noise=spec.infer_noise)
    ce0 = spec.Ce0 if spec.Ce0 is not None else pool_ce0
    return InferenceConfig(
        prior      = prior_cfg,
        preproc    = preproc_cfg,
        optimizer  = opt_cfg,
        method     = method,
        vi         = vi_cfg,
        Ce0        = ce0,
        n_eval_pts = n_eval_pts,
        run_mcmc   = run_mcmc,
        mcmc_iter  = mcmc_iter,
        mcmc_burn  = mcmc_burn,
        mcmc_thin  = mcmc_thin,
        mcmc_seed  = mcmc_seed,
        mcmc_scan  = mcmc_scan,
        verbose    = False,   # see module docstring's "Output volume"
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def infer_shared_model_order(datasets     : List[DatasetSpec],
                              model_orders : List[int],
                              config       : Optional[PooledInferenceConfig] = None
                              ) -> PooledInferenceResult:
    """
    Find the model order best supported by the combined evidence of
    several datasets, then fit each dataset's own DTD parameters at that
    shared order. See the module docstring for the two-stage design.

    Parameters
    ----------
    datasets     : list of DatasetSpec
    model_orders : list of int          Candidate orders swept in stage 1.
    config       : PooledInferenceConfig, optional   Default: PooledInferenceConfig().

    Returns
    -------
    result : PooledInferenceResult
    """
    from infer_impulse_response import infer_impulse_response   # see import note at top of file

    if len(datasets) == 0:
        raise ValueError("datasets must contain at least one entry.")
    if len(model_orders) == 0:
        raise ValueError("model_orders must contain at least one entry.")
    if config is None:
        config = PooledInferenceConfig()

    preproc_cfg = config.preproc
    if preproc_cfg is None:
        from config.defaults import PreprocConfig
        preproc_cfg = PreprocConfig()

    param_method = config.param_method    or config.ranking_method
    param_opt    = config.param_optimizer or config.ranking_optimizer
    param_vi     = config.param_vi        or config.ranking_vi

    # ------------------------------------------------------------------
    # Stage 1: ranking sweep - every dataset x every candidate order
    # ------------------------------------------------------------------
    if config.verbose:
        print(f"\n=== Stage 1: ranking sweep ({config.ranking_method}) - "
              f"{len(datasets)} datasets x {len(model_orders)} orders ===")

    ranking_results = []
    for spec in datasets:
        cfg = _dataset_inference_config(
            spec, config.prior, preproc_cfg,
            config.ranking_method, config.ranking_optimizer, config.ranking_vi,
            config.Ce0, config.n_eval_pts)
        result = infer_impulse_response(
            spec.u, spec.q, spec.t, spec.T_c,
            model_orders=model_orders, config=cfg)
        ranking_results.append(result)
        if config.verbose:
            print(f"  [{spec.name}] best N={result.ranking.best_N}")

    logML_matrix   = jnp.stack([r.ranking.logML for r in ranking_results])
    pooled_ranking = PooledModelRanking(model_orders, [s.name for s in datasets], logML_matrix)
    N_star         = pooled_ranking.best_N

    if config.verbose:
        pooled_ranking.print_table()

    # ------------------------------------------------------------------
    # Stage 2: per-dataset parameter fit at the shared N_star
    # ------------------------------------------------------------------
    if config.verbose:
        print(f"\n=== Stage 2: parameter fit at N={N_star} ({param_method}) - "
              f"{len(datasets)} datasets ===")

    param_results = []
    for spec in datasets:
        cfg = _dataset_inference_config(
            spec, config.prior, preproc_cfg,
            param_method, param_opt, param_vi,
            config.Ce0, config.n_eval_pts,
            run_mcmc  = config.run_mcmc,
            mcmc_iter = config.mcmc_iter,
            mcmc_burn = config.mcmc_burn,
            mcmc_thin = config.mcmc_thin,
            mcmc_seed = config.mcmc_seed,
            mcmc_scan = config.mcmc_scan)
        result = infer_impulse_response(
            spec.u, spec.q, spec.t, spec.T_c,
            model_orders=[N_star], config=cfg)
        param_results.append(result)
        if config.verbose:
            a = np.asarray(result.best.a_map)
            mcmc_note = ""
            if result.mcmc is not None:
                mcmc_note = f"  MCMC accept={result.mcmc.accept_rate:.1%}"
            print(f"  [{spec.name}] Ce={result.best.Ce:.3e}  n_1={a[0]:+.3f}{mcmc_note}")

    return PooledInferenceResult(
        N_star          = N_star,
        pooled_ranking  = pooled_ranking,
        ranking_results = ranking_results,
        param_results   = param_results,
        dataset_names   = [s.name for s in datasets],
    )
