#!/usr/bin/env python3
"""
examples/generate_kornilov_figures.py
=======================================
In the spirit of the original MATLAB generateFigures.m: run the full
Bayesian DTD analysis across all 20 WET Kornilov cases and compare the
result against an independent SysID (TFDSI.m) impulse-response estimate,
as a sanity check that the Bayesian fit agrees with a completely
different estimation method.

Unlike examples/run_pooled_inference.py (a general-purpose CLI for the
Kornilov dataset), this is a fixed-recipe script for this specific
comparison: Laplace everywhere (both the shared-N ranking sweep and the
per-dataset parameter fit - see inference/pooled.py's module docstring
for why mixing methods across dataset in a summed logML would be
unsound), all 20 cases treated as noise-free (Ce held fixed, not
estimated - see --ce0), and MCMC validation available but off by default.

Steps:
    1. Load all 20 WET Kornilov cases (data.WET_Kornilov.loader).
    2. Stage 1: rank candidate model orders by evidence summed across all
       20 cases (inference.pooled.infer_shared_model_order) to pick one
       shared N.
    3. Stage 2: fit each case's own DTD parameters at that N.
    4. For each case, evaluate the Bayesian impulse response on the
       SysID reference's own time grid (data.WET_Kornilov.loader.
       load_kornilov_fir_sysid) so the two curves are directly
       comparable with no resampling, and plot both together in a 5x4
       grid (series x water-gas-ratio).
    5. Save a per-dataset summary CSV, including agreement metrics
       (correlation, RMSE) against the SysID reference.

Usage:
    python examples/generate_kornilov_figures.py
    python examples/generate_kornilov_figures.py --model-orders 1 2 3 4
    python examples/generate_kornilov_figures.py --mcmc --mcmc-iter 200000
    python examples/generate_kornilov_figures.py --subsets U_const --wgr WGR0 WGR231

Output:
    examples/outputs/kornilov_bayesian_vs_sysid.png
    examples/outputs/kornilov_bayesian_vs_sysid_summary.csv
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
from data.WET_Kornilov.loader import load_all_kornilov_datasets, load_kornilov_fir_sysid
from data.WET_Kornilov.metadata import SUBSETS, WGR_LABELS

OUTPUT_DIR = REPO_ROOT / "examples" / "outputs"


def bayesian_h_on_sysid_grid(best, T_c, t_sysid):
    """
    Evaluate the Bayesian impulse response at the SysID reference's own
    time grid, so the two curves sit on identical x-axes with no
    resampling. The DTD model is a smooth sum of Gaussians with no hard
    cutoff at the fitted order's own T_h, so this is valid regardless of
    how that T_h compares to the SysID grid's ~20 ms span.

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
    Convert the curated SysID reference (discrete FIR taps: q[n] = sum_k
    b[k] u[n-k], no dt factor) to the same continuous-time impulse-response
    *density* convention this codebase's Bayesian h(t) uses, where
    core.cost's own convolution is p = conv(u, h, 'valid') * dt - i.e.
    h(t) needs multiplying by dt to become an equivalent discrete tap, so
    converting the other way (discrete tap -> density) divides by dt.

    Confirmed empirically before relying on it here: cross-checking
    against an independent DTD reconstruction (same approach used to
    verify fir_sysid.csv's row order), raw SysID taps sit ~1/dt_sysid
    below the Bayesian curve (peak ratios 9900-10600 against
    1/dt_sysid=10000); after this scaling, relative RMSE drops to 0.6-2.1%.
    """
    dt_sysid = float(fir['time'][1] - fir['time'][0])
    return fir['val'] / dt_sysid


def plot_comparison_grid(result, out_path):
    print("\n=== Plotting Bayesian vs. SysID comparison grid ===")
    fig, axes = plt.subplots(len(SUBSETS), len(WGR_LABELS),
                              figsize=(4 * len(WGR_LABELS), 3 * len(SUBSETS)),
                              sharex=True)

    by_name = dict(zip(result.dataset_names, result.param_results))

    # get_colours' palette has only 4 real colours - index 5 is white
    # (a gradient blend-target, not a usable series colour; see
    # utils/plotting.py's _PALETTE comment). With 5 subsets, index 1..4
    # covers 4 of them and the 5th gets an explicit non-white colour
    # rather than silently plotting white-on-white.
    subset_colours = list(get_colours([1, 2, 3, 4])) + [np.array([0.2, 0.2, 0.2])]

    for i, subset in enumerate(SUBSETS):
        colour = subset_colours[i]
        for j, wgr in enumerate(WGR_LABELS):
            ax = axes[i, j]
            name = f"{subset}_{wgr}"
            if name not in by_name:
                ax.axis('off')
                continue

            ir = by_name[name]
            best = ir.best
            fir = load_kornilov_fir_sysid(subset, wgr)
            t_ms = fir['time'] * 1e3

            h_val, h_std = bayesian_h_on_sysid_grid(best, ir.T_c, fir['time'])

            errorpatch(ax, t_ms, h_val, 1.96 * h_std, 1.96 * h_std,
                       color=colour, line_kwargs={'label': 'Bayesian DTD'})
            ax.plot(t_ms, sysid_h_continuous(fir), color='k', lw=1.0, ls='--', label='SysID')

            if i == 0:
                ax.set_title(wgr)
            if j == 0:
                ax.set_ylabel(subset)
            if i == len(SUBSETS) - 1:
                ax.set_xlabel('t [ms]')
            ax.grid(True, alpha=0.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"WET Kornilov - Bayesian DTD vs. SysID (shared N = {result.N_star})",
                 y=1.05)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {out_path}")


def save_summary_csv(result, out_path):
    print("\n=== Saving per-dataset summary (incl. SysID agreement) ===")
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
            subset, wgr = name.rsplit("_", 1)
            fir = load_kornilov_fir_sysid(subset, wgr)
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
        "--subsets", nargs="+", choices=SUBSETS, default=None,
        help="Restrict to these series (default: all 5).")
    parser.add_argument(
        "--wgr", nargs="+", choices=WGR_LABELS, default=None,
        help="Restrict to these water-gas-ratio cases (default: all 4).")
    parser.add_argument(
        "--model-orders", type=int, nargs="+", default=[1, 2, 3, 4, 5],
        help="Candidate model orders for the stage-1 ranking sweep (default: 1 2 3 4 5).")
    parser.add_argument(
        "--method", choices=["laplace", "vi"], default="laplace",
        help="Backend used for BOTH stages (default: laplace). Unlike "
             "run_pooled_inference.py's independent ranking/param method "
             "choice, this script applies one uniform choice everywhere.")
    parser.add_argument(
        "--ce0", type=float, default=1e-6,
        help="Fixed noise variance (default: 1e-6). All 20 cases are "
             "treated as noise-free here - infer_noise=False is hardcoded "
             "in this script, not a flag; only the fixed value is "
             "configurable.")
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

    print("Loading WET Kornilov datasets...")
    datasets = load_all_kornilov_datasets(subsets=args.subsets, wgr_labels=args.wgr)
    print(f"  {len(datasets)} cases: {[d.name for d in datasets]}")

    preproc_cfg = PreprocConfig(DSmode=args.ds_mode, DSvalue=args.ds_value)
    opt_cfg     = OptimizerConfig(use_parallel=not args.no_parallel, infer_noise=False)

    config = PooledInferenceConfig(
        prior             = PriorConfig(),
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

    plot_comparison_grid(result, OUTPUT_DIR / "kornilov_bayesian_vs_sysid.png")
    save_summary_csv(result, OUTPUT_DIR / "kornilov_bayesian_vs_sysid_summary.csv")

    print("\nDone.")


if __name__ == "__main__":
    main()
