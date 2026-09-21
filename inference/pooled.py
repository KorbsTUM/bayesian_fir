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

config.param_prior lets stage 2 use a *different* PriorConfig than stage 1
(default: same as config.prior).

config.param_T_h_floor addresses a specific problem: some pools have
datasets whose T_c varies enough that the automatic, order-scaled T_h =
t_max(N_star) * T_c undershoots a small-T_c dataset's real impulse
response extent (cutting its stage-2 fit off early) - the fix isn't to
fix T_h to one literal value (which infer_impulse_response refuses
together with more than one candidate model order anyway, since a fixed
window would bias stage 1's order comparison - Section 4.4 of Yoko &
Polifke 2026), because a single flat number then *undershoots in the
other direction* for a large-T_c dataset at a higher order: its own
automatic T_h can be several times the flat value, and forcing a much
shorter absolute window onto more parameters (higher N) is exactly the
kind of overparameterized-for-its-support situation that leaves the
Laplace Hessian ill-conditioned (see DegenerateFitError below - this is
exactly what was observed with the WET_Kornilov-style Kornilov pool:
fine in stage 1 at N=4 for a large-T_c case, NaN in stage 2 once a flat
25ms window replaced the automatic ~81ms one). So instead,
param_T_h_floor only raises the floor: for each dataset, stage 2 uses
max(t_max(N_star, param_prior) * spec.T_c, param_T_h_floor) - large-T_c
datasets keep their own (already generous) automatic window untouched,
and only small-T_c datasets whose automatic window falls under the floor
get lifted up to it. Requires param_prior.T_h (or prior.T_h, if
param_prior is unset) to be None - it's ill-defined to take a max against
an already-fixed T_h.

Stage 2 also tolerates a single dataset's fit going numerically
degenerate at N_star (DegenerateFitError from inference.posterior,
raised when infer_impulse_response's single candidate order comes back
NaN): that dataset is skipped (warned about, excluded from
param_results/dataset_names, and listed in PooledInferenceResult.
stage2_excluded) rather than aborting the whole pool's stage 2 - mirrors
how PooledModelRanking already tolerates a NaN cell in stage 1's sweep,
which stage 2 had no equivalent for previously.

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

from dataclasses import dataclass, field, replace
from typing import List, NamedTuple, Optional, TYPE_CHECKING

import numpy as np
import jax
import jax.numpy as jnp

from core.prior import PriorConfig, estimate_t_max
from inference.optimizer import OptimizerConfig
from inference.posterior import DegenerateFitError
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
                                          Used for stage 1 always, and for
                                          stage 2 too unless param_prior is
                                          set.
    param_prior       : PriorConfig or None   Stage-2-only prior override.
                                          None -> prior.
    param_T_h_floor   : float or None     Stage-2-only minimum T_h [s].
                                          None -> stage 2 uses param_prior's
                                          (or prior's) T_h/automatic
                                          Fenton-Wilkinson estimate
                                          unmodified, same as stage 1.
                                          When set, stage 2 instead uses,
                                          per dataset, max(automatic T_h at
                                          N_star, param_T_h_floor) - see
                                          module docstring for why this
                                          (not a single fixed T_h) is the
                                          safe way to raise a too-short
                                          automatic window without
                                          under-cutting datasets whose own
                                          automatic window is already
                                          larger than the floor.
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
    param_prior       : Optional[PriorConfig]   = None   # None -> prior
    param_T_h_floor   : Optional[float]         = None
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
    best_N        : int                       Order maximizing logML_total,
                                              among orders with a finite
                                              total (see below).

    A NaN logML for some (dataset, order) pair means that dataset's Laplace
    covariance was numerically degenerate at that order (e.g. an
    ill-conditioned Hessian - most often an overparameterized order
    chasing structure that dataset's data doesn't support), not that the
    order is literally the best fit. Since summing propagates NaN, one
    bad (dataset, order) pair would otherwise poison that order's entire
    pooled total, and - because plain jnp.argmax/jnp.max are not NaN-safe -
    could even get that order silently crowned "best", or make a single
    NaN cell corrupt an entire displayed row. Orders with a NaN pooled
    total are excluded from best_N selection (with a printed warning) and
    still shown as NaN in print_table() (that order's contribution really
    is unavailable) rather than silently hidden or spread to unrelated
    orders.
    """

    def __init__(self, orders, dataset_names, logML_matrix):
        self.orders        = list(orders)
        self.dataset_names = list(dataset_names)
        self.logML_matrix  = jnp.asarray(logML_matrix)
        self.logML_total   = jnp.sum(self.logML_matrix, axis=0)

        total_nan_mask = jnp.isnan(self.logML_total)
        if bool(jnp.all(total_nan_mask)):
            raise ValueError(
                "Pooled logML is NaN for every candidate model order - "
                "cannot select a shared N. Check the optimizer/prior "
                "settings (a fixed, very small Ce0 combined with an "
                "overparameterized order is a common cause).")

        cell_nan_mask = jnp.isnan(self.logML_matrix)
        if bool(jnp.any(cell_nan_mask)):
            rows, cols = np.nonzero(np.asarray(cell_nan_mask))
            bad = sorted({(self.dataset_names[r], self.orders[c]) for r, c in zip(rows, cols)})
            print(f"  WARNING: logML is NaN for {len(bad)} (dataset, order) "
                  f"pair(s) {bad} - Laplace covariance was numerically "
                  f"degenerate there (likely overparameterized for that "
                  f"dataset). Any order with at least one NaN dataset is "
                  f"excluded from shared-N selection.")

        safe_total   = jnp.where(total_nan_mask, -jnp.inf, self.logML_total)
        best_idx     = int(jnp.argmax(safe_total))
        self.best_N  = self.orders[best_idx]

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
            # nanmax is safe here: __init__ already rejects any dataset
            # whose row is entirely NaN (that would force logML_total to
            # be all-NaN too, via summation, and raise there).
            row_n = row - jnp.nanmax(row)
            vals  = "  ".join(f"{float(v):>{col_w}.2f}" for v in row_n)
            print(f"{name:>20}  {vals}")
        print(rule)
        total_n = self.logML_total - jnp.nanmax(self.logML_total)
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
                                           PosteriorResult. Only datasets
                                           that fit successfully at N_star
                                           are included - see
                                           stage2_excluded.
    dataset_names   : list of str         Names matching param_results,
                                           row for row (NOT necessarily
                                           all input datasets - see
                                           stage2_excluded).
    stage2_excluded : list of str         Names of datasets whose stage-2
                                           fit at N_star was numerically
                                           degenerate (DegenerateFitError)
                                           and were skipped rather than
                                           aborting the whole pool. Empty
                                           in the common case.
    """
    N_star          : int
    pooled_ranking  : PooledModelRanking
    ranking_results : List[InferenceResult]
    param_results   : List[InferenceResult]
    dataset_names   : List[str]
    stage2_excluded : List[str] = field(default_factory=list)

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
    param_prior  = config.param_prior     if config.param_prior is not None else config.prior

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
        # inference.optimizer's LM step is rebuilt as a fresh jit closure on
        # every call (see its module docstring / _make_lm_step), so JAX's
        # compilation cache never gets reused across datasets anyway -
        # confirmed empirically (identical-shape calls each independently
        # pay the full ~250ms+ while_loop compile, zero cache hits). Given
        # that reuse is already unavailable, clearing here costs little and
        # keeps compiled-artifact memory from growing unbounded across a
        # long sweep (root cause of the Ubuntu "cannot allocate memory"
        # LLVM failures after a few cases) - a stopgap until the optimizer
        # itself is restructured to make that reuse possible.
        jax.clear_caches()

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

    # param_T_h_floor: see module docstring. Computed once (t_max is pure
    # prior-space, independent of T_c) and combined with each dataset's
    # own T_c below.
    t_max_star = None
    if config.param_T_h_floor is not None:
        if param_prior.T_h is not None:
            raise ValueError(
                "config.param_T_h_floor requires param_prior.T_h (or "
                "prior.T_h, if param_prior is unset) to be None - it "
                "computes each dataset's T_h from the automatic "
                "Fenton-Wilkinson estimate at N_star and takes the max "
                "against the floor, which is ill-defined when T_h is "
                "already fixed to a single literal value.")
        t_max_star = estimate_t_max(N_star, param_prior)

    param_results   = []
    param_names     = []
    stage2_excluded = []
    for spec in datasets:
        dataset_prior = param_prior
        if t_max_star is not None:
            T_h_i = max(t_max_star * spec.T_c, config.param_T_h_floor)
            dataset_prior = replace(param_prior, T_h=T_h_i)

        cfg = _dataset_inference_config(
            spec, dataset_prior, preproc_cfg,
            param_method, param_opt, param_vi,
            config.Ce0, config.n_eval_pts,
            run_mcmc  = config.run_mcmc,
            mcmc_iter = config.mcmc_iter,
            mcmc_burn = config.mcmc_burn,
            mcmc_thin = config.mcmc_thin,
            mcmc_seed = config.mcmc_seed,
            mcmc_scan = config.mcmc_scan)
        try:
            result = infer_impulse_response(
                spec.u, spec.q, spec.t, spec.T_c,
                model_orders=[N_star], config=cfg)
        except DegenerateFitError as e:
            print(f"  WARNING: [{spec.name}] stage-2 fit at N={N_star} was "
                  f"numerically degenerate - skipped, excluded from "
                  f"results ({e}).")
            stage2_excluded.append(spec.name)
            jax.clear_caches()   # see stage 1's comment above
            continue
        param_results.append(result)
        param_names.append(spec.name)
        if config.verbose:
            a = np.asarray(result.best.a_map)
            mcmc_note = ""
            if result.mcmc is not None:
                mcmc_note = f"  MCMC accept={result.mcmc.accept_rate:.1%}"
            t_h_note = f"  T_h={T_h_i*1e3:.1f}ms" if t_max_star is not None else ""
            print(f"  [{spec.name}] Ce={result.best.Ce:.3e}  n_1={a[0]:+.3f}{t_h_note}{mcmc_note}")
        jax.clear_caches()   # see stage 1's comment above

    if stage2_excluded and config.verbose:
        print(f"\n  {len(stage2_excluded)} dataset(s) excluded from stage-2 "
              f"results due to a degenerate fit at N={N_star}: {stage2_excluded}")

    return PooledInferenceResult(
        N_star          = N_star,
        pooled_ranking  = pooled_ranking,
        ranking_results = ranking_results,
        param_results   = param_results,
        dataset_names   = param_names,
        stage2_excluded = stage2_excluded,
    )
