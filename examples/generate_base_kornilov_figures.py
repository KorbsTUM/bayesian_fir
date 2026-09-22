#!/usr/bin/env python3
"""
examples/generate_base_kornilov_figures.py
==========================================
Bayesian DTD analysis across the 20 (new, plain) Kornilov cases
(data/Kornilov/, C01-C20 - see data/Kornilov/metadata.py for provenance
of the per-case L_ref/U_ref used to build T_c). Modelled on
examples/generate_WET_kornilov_figures.py (the WET_Kornilov analogue), with
one difference forced by what this dataset actually has: no
series/water-gas-ratio structure - the 20 cases are independent flame
conditions, not organised into subsets x WGR, so the plot grid is just
laid out case-by-case rather than 5x4.

Each case ships a SysID reference impulse response (fir_sysid.csv -
curated from .../Kornilov_timeseries/SysID/System Identification/FIRs/
FIR_{case}.txt) via data.Kornilov.loader.load_kornilov_fir_sysid, overlaid
on the Bayesian fit and scored (correlation, RMSE) same as
generate_WET_kornilov_figures.py does for WET_Kornilov. That loader still
returns None (not raising) for any case missing a fir_sysid.csv, so this
script tolerates a partially-covered dataset - a future case can be added
without its own SysID reference and will just plot Bayesian-only.

As in generate_WET_kornilov_figures.py: Laplace everywhere (both the shared-N
ranking sweep and the per-dataset parameter fit - see inference/pooled.py's
module docstring for why mixing methods across datasets in a summed logML
would be unsound), all 20 cases treated as noise-free by default (Ce held
fixed, not estimated - see --ce0), MCMC validation available but off by
default.

Steps:
    1. Load all 20 Kornilov cases (data.Kornilov.loader).
    2. Stage 1: rank candidate model orders by evidence summed across all
       20 cases (inference.pooled.infer_shared_model_order) to pick one
       shared N.
    3. Stage 2: fit each case's own DTD parameters at that N.
    4. For each case, plot the Bayesian impulse response (plus the SysID
       reference, if data.Kornilov.loader.load_kornilov_fir_sysid finds
       one) in a grid, one panel per case.
    5. Save a per-dataset summary CSV, including SysID agreement metrics
       (correlation, RMSE) where a reference is available.

Usage:
    python examples/generate_base_kornilov_figures.py
    python examples/generate_base_kornilov_figures.py --model-orders 1 2 3 4
    python examples/generate_base_kornilov_figures.py --mcmc --mcmc-iter 200000
    python examples/generate_base_kornilov_figures.py --cases C01 C02 C03

Output:
    examples/outputs/kornilov_new_bayesian.png
    examples/outputs/kornilov_new_summary.csv
"""

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.prior import PriorConfig
from core.impulse_response import calculate_impulse_response
from config.defaults import PreprocConfig
from inference.optimizer import OptimizerConfig
from inference.pooled import PooledInferenceConfig, infer_shared_model_order
from utils.plotting import get_colours, errorpatch
from data.Kornilov.loader import load_all_kornilov_datasets, load_kornilov_fir_sysid
from data.Kornilov.metadata import CASES

OUTPUT_DIR = REPO_ROOT / "examples" / "outputs"

N_COLS = 5  # plot-grid layout: 20 cases -> 4 rows x 5 cols


def bayesian_h_on_sysid_grid(best, T_c, t_sysid):
    """
    Evaluate the Bayesian impulse response at the SysID reference's own
    time grid, so the two curves sit on identical x-axes with no
    resampling. See examples/generate_WET_kornilov_figures.py's identically
    named helper for the full rationale (unchanged here).

    Returns (h_val, h_std) - h_std is the pointwise Gaussian marginal std
    from Cb_map (a linearised/Laplace uncertainty, same convention used
    elsewhere in this codebase, e.g. PosteriorResult.h.var).
    """
    t_nd = jnp.asarray(t_sysid) / T_c
    h_val, _, _, h_var = calculate_impulse_response(
        best.b_map, t_nd, T_c, best.Cb_map)
    return np.asarray(h_val), np.sqrt(np.asarray(h_var))


def sysid_h_continuous(fir):
    """
    Convert a curated SysID reference (discrete FIR taps) to the same
    continuous-time impulse-response *density* convention this codebase's
    Bayesian h(t) uses. See examples/generate_WET_kornilov_figures.py's
    identically named helper for the full derivation/validation (unchanged
    here) - raw SysID taps need dividing by dt_sysid to become comparable
    to h(t).
    """
    dt_sysid = float(fir['time'][1] - fir['time'][0])
    return fir['val'] / dt_sysid


def plot_grid(result, out_path):
    print("\n=== Plotting Bayesian impulse-response grid ===")
    names = result.dataset_names
    n_rows = -(-len(names) // N_COLS)  # ceil division
    fig, axes = plt.subplots(n_rows, N_COLS,
                              figsize=(3.2 * N_COLS, 2.4 * n_rows),
                              sharex=False, squeeze=False)

    # get_colours' palette has only 4 real colours - index 5 is white (a
    # gradient blend-target, not a usable series colour; see
    # utils/plotting.py's _PALETTE comment), so cycle through 1..4 rather
    # than indexing one colour per case.
    by_name = dict(zip(names, result.param_results))

    any_sysid = False
    for idx, name in enumerate(names):
        ax = axes[idx // N_COLS, idx % N_COLS]
        colour = get_colours((idx % 4) + 1)

        ir = by_name[name]
        best = ir.best
        h_val = np.asarray(best.h.val)
        h_std = np.sqrt(np.asarray(best.h.var))
        t_ms = np.asarray(best.h.time) * 1e3

        errorpatch(ax, t_ms, h_val, 1.96 * h_std, 1.96 * h_std,
                   color=colour, line_kwargs={'label': 'Bayesian DTD'})

        fir = load_kornilov_fir_sysid(name)
        if fir is not None:
            any_sysid = True
            t_sysid_ms = fir['time'] * 1e3
            h_ref = sysid_h_continuous(fir)
            ax.plot(t_sysid_ms, h_ref, color='k', lw=1.0, ls='--', label='SysID')

        ax.set_title(name)
        ax.grid(True, alpha=0.3)
        if idx // N_COLS == n_rows - 1:
            ax.set_xlabel('t [ms]')
        if idx % N_COLS == 0:
            ax.set_ylabel('h(t)')

    # blank out unused panels
    for idx in range(len(names), n_rows * N_COLS):
        axes[idx // N_COLS, idx % N_COLS].axis('off')

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    subtitle = "no SysID reference yet" if not any_sysid else "SysID reference overlaid where available"
    fig.suptitle(f"Kornilov (new) - Bayesian DTD impulse response "
                 f"(shared N = {result.N_star}, {subtitle})", y=1.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {out_path}")


def save_summary_csv(result, out_path):
    print("\n=== Saving per-dataset summary (incl. SysID agreement where available) ===")
    N = result.N_star
    header = ['dataset', 'N', 'Ce', 'logML', 'logBFL', 'logOF']
    for i in range(1, N + 1):
        header += [f'n_{i}', f'tau_{i}_s', f'sigma_{i}_s']
    header += ['sysid_corr', 'sysid_rmse']

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for name, ir in zip(result.dataset_names, result.param_results):
            best = ir.best

            corr, rmse = '', ''
            fir = load_kornilov_fir_sysid(name)
            if fir is not None:
                h_val, _ = bayesian_h_on_sysid_grid(best, ir.T_c, fir['time'])
                sysid_val = sysid_h_continuous(fir)
                corr = float(np.corrcoef(h_val, sysid_val)[0, 1])
                rmse = float(np.sqrt(np.mean((h_val - sysid_val) ** 2)))

            row = [name, best.N, float(best.Ce), best.logML, best.logBFL, best.logOF]
            row += [float(v) for v in np.asarray(best.a_map)]
            row += [corr, rmse]
            writer.writerow(row)
    print(f"  saved {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--cases", nargs="+", choices=CASES, default=None,
        help="Restrict to these cases (default: all 20).")
    parser.add_argument(
        "--model-orders", type=int, nargs="+", default=[1, 2, 3, 4, 5],
        help="Candidate model orders for the stage-1 ranking sweep (default: 1 2 3 4 5).")
    parser.add_argument(
        "--method", choices=["laplace", "vi"], default="laplace",
        help="Backend used for BOTH stages (default: laplace). Unlike "
             "run_pooled_inference.py's independent ranking/param method "
             "choice, this script applies one uniform choice everywhere.")
    parser.add_argument(
        "--lfl", type=float, default=None,
        help="Low-frequency (DC/steady-state) gain constraint: a soft "
             "Gaussian prior pulling sum(n_i) toward this value (see "
             "core/prior.py's PriorConfig.LFL/core/cost.py's J_LFL term). "
             "None (default) leaves it unconstrained, i.e. off - matching "
             "this script's behaviour before this flag existed. Applies "
             "to BOTH stages automatically (param_prior isn't set "
             "separately in this script, so stage 2 inherits --lfl/"
             "--lfl-sigma from stage 1's prior, same as every other "
             "hyperparameter). LFL=1.0 (unit steady-state gain) is the "
             "physically appropriate value for this dataset.")
    parser.add_argument(
        "--lfl-sigma", type=float, default=0.01,
        help="Std of the soft LFL constraint (default: 0.01, matching "
             "PriorConfig.LFL_sigma's own default). Only used when --lfl "
             "is set.")
    parser.add_argument(
        "--ce0", type=float, default=4e-7,
        help="Pool-wide fallback noise variance (default: 4e-7), used only "
             "when --ce0-alpha 0 disables the per-case scaling below. All "
             "cases are treated as noise-free here - infer_noise=False is "
             "hardcoded in this script, not a flag; only the fixed value "
             "is configurable.")
    parser.add_argument(
        "--ce0-alpha", type=float, default=0.005,
        help="Per-case Ce0 = alpha * var(q_i), overriding --ce0 for every "
             "case (default alpha: 0.0197). q is normalised per case "
             "((q-mean)/mean) before fitting, so a single flat --ce0 "
             "implicitly assumes a different *relative* noise fraction "
             "for every case depending on how strong that case's own "
             "output fluctuation is - e.g. at --ce0=4e-7, C12 (weak "
             "signal, var(q)~7e-6) was getting ~5.7%% of its variance "
             "treated as noise vs. C18's ~0.2%% (var(q)~1.9e-4), making "
             "C12's evidence-based fit far more conservative about small "
             "features than C18's. Scaling by each case's own var(q) "
             "puts every case on the same relative noise footing instead. "
             "The default 0.0197 is this dataset's own median Ce0/var(q) "
             "ratio at the old flat --ce0=4e-7 (i.e. matches a typical "
             "already-well-behaved case, e.g. C01, almost exactly - "
             "chosen so this default reproduces the old fit for most "
             "cases and only really changes the outliers). Pass 0 to "
             "disable and use the flat --ce0 for every case instead.")
    parser.add_argument(
        "--param-t-h-floor", type=float, default=0.025,
        help="Minimum T_h [s] for stage 2 only (default: 0.025 = 25 ms) - "
             "NOT a flat override: stage 2 uses, per case, max(automatic "
             "T_h at N_star, this floor). Several cases in this dataset "
             "have a small enough T_c that the automatic, order-scaled "
             "T_h (t_max(N) * T_c, same one stage 1 always uses "
             "unmodified) undershoots the impulse response's real "
             "physical extent, cutting the stage-2 fit off early - the "
             "floor lifts only those cases. A large-T_c case's own "
             "automatic T_h is left untouched whenever it's already above "
             "the floor (which is usually true) - a single flat T_h "
             "instead would undershoot such a case at a high enough N "
             "(observed in practice: fine in stage 1, numerically "
             "degenerate in stage 2 once a flat window replaced a much "
             "larger automatic one - see "
             "inference/pooled.py's PooledInferenceConfig.param_T_h_floor "
             "docstring). Pass 0 to disable and use the automatic "
             "per-order, per-case T_h for stage 2 too.")
    parser.add_argument(
        "--mcmc", action="store_true",
        help="Run MCMC validation in stage 2 (once per dataset, at the "
             "shared N_star). Off by default.")
    parser.add_argument(
        "--mcmc-iter", type=int, default=200_000,
        help="MCMC iterations when --mcmc is set (default: 200000).")
    parser.add_argument(
        "--ds-mode", choices=["factor", "frequency", "rate"], default="rate",
        help="Downsampling mode passed to prepare_signals (default: rate).")
    parser.add_argument(
        "--ds-value", type=float, default=3e-4,
        help="Downsampling value, interpreted per --ds-mode (default: 3e-4).")
    parser.add_argument(
        "--no-parallel", action="store_true",
        help="Disable vmap-parallel restarts (Python for-loop instead); "
             "useful for debugging on CPU.")
    args = parser.parse_args()

    print("Loading Kornilov (new) datasets...")
    datasets = load_all_kornilov_datasets(cases=args.cases)
    print(f"  {len(datasets)} cases: {[d.name for d in datasets]}")

    if args.ce0_alpha > 0:
        datasets = [d._replace(Ce0=args.ce0_alpha * float(np.var(d.q))) for d in datasets]
        ce0_vals = [d.Ce0 for d in datasets]
        print(f"  per-case Ce0 = {args.ce0_alpha:g} * var(q_i)  "
              f"(range: {min(ce0_vals):.3e} - {max(ce0_vals):.3e})")

    if args.lfl is not None:
        print(f"  LFL constraint: sum(n_i) -> {args.lfl:g}  (sigma={args.lfl_sigma:g})")

    preproc_cfg = PreprocConfig(DSmode=args.ds_mode, DSvalue=args.ds_value)
    opt_cfg     = OptimizerConfig(use_parallel=not args.no_parallel, infer_noise=False)

    config = PooledInferenceConfig(
        prior             = PriorConfig(LFL=args.lfl, LFL_sigma=args.lfl_sigma),
        param_T_h_floor   = args.param_t_h_floor if args.param_t_h_floor else None,
        preproc           = preproc_cfg,
        ranking_method    = args.method,
        ranking_optimizer = opt_cfg,
        Ce0               = args.ce0,
        run_mcmc          = args.mcmc,
        mcmc_iter         = args.mcmc_iter,
        verbose           = True,
    )

    result = infer_shared_model_order(datasets, args.model_orders, config)

    print(f"\nShared model order across {len(datasets)} datasets: N = {result.N_star}")
    if result.stage2_excluded:
        print(f"WARNING: {len(result.stage2_excluded)} case(s) excluded from "
              f"the figure/summary below (degenerate stage-2 fit at "
              f"N={result.N_star}): {result.stage2_excluded}")

    plot_grid(result, OUTPUT_DIR / "kornilov_new_bayesian.png")
    save_summary_csv(result, OUTPUT_DIR / "kornilov_new_summary.csv")

    print("\nDone.")


if __name__ == "__main__":
    main()
