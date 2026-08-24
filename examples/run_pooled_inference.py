#!/usr/bin/env python3
"""
examples/run_pooled_inference.py
==================================
End-to-end recipe for pooled model-order selection across the WET
Kornilov dataset: find the DTD model order N best supported by the
combined evidence of several/all of its 20 cases, then fit each case's
own DTD parameters at that shared N. See inference/pooled.py's module
docstring for the two-stage design (ranking sweep summed across datasets,
then a per-dataset parameter fit at the winning order) and why it isn't
new inference machinery - it's an orchestration layer over
infer_impulse_response.

Steps:
    1. Load some or all of the 20 WET Kornilov cases
       (data.WET_Kornilov.loader.load_all_kornilov_datasets), each with
       its own T_c = L_ref / U_ref.
    2. Stage 1: for every candidate order, run Bayesian inference on every
       selected dataset (--ranking-method, default 'laplace' - cheap,
       since this runs len(datasets) x len(model_orders) times) and sum
       the log evidence across datasets to pick the shared N.
    3. Stage 2: fit each dataset's own DTD parameters at that N
       (--param-method, defaults to --ranking-method; pass 'vi' here for
       a richer posterior since this only runs once per dataset).
    4. Plot every dataset's inferred FTF (colored by series, shaded by
       water-gas-ratio) and save a per-dataset parameter summary CSV.

Using a subset of cases
-------------------------
All 20 cases run by default. For a quicker/smaller run, restrict with
--subsets and/or --wgr, e.g.:

    python examples/run_pooled_inference.py --subsets U_const P_const
    python examples/run_pooled_inference.py --wgr WGR0 WGR231

Usage:
    python examples/run_pooled_inference.py
    python examples/run_pooled_inference.py --model-orders 1 2 3 4
    python examples/run_pooled_inference.py --ranking-method laplace --param-method vi
    python examples/run_pooled_inference.py --subsets U_const --wgr WGR0 WGR231

Output:
    examples/outputs/kornilov_ftf_pooled.png
    examples/outputs/kornilov_pooled_summary.csv
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
from core.ftf import calculate_ftf
from config.defaults import PreprocConfig
from inference.optimizer import OptimizerConfig
from inference.pooled import PooledInferenceConfig, infer_shared_model_order
from utils.plotting import get_colours
from data.WET_Kornilov.loader import load_all_kornilov_datasets
from data.WET_Kornilov.metadata import SUBSETS, WGR_LABELS

OUTPUT_DIR = REPO_ROOT / "examples" / "outputs"


def plot_pooled_ftf(result, out_path):
    print("\n=== Plotting pooled FTF comparison ===")
    omega = jnp.linspace(0.0, 2 * jnp.pi * 500, 200)
    freq = omega / (2 * jnp.pi)

    # get_colours' palette has only 4 real colours - index 5 is white
    # (a gradient blend-target, not a usable series colour; see
    # utils/plotting.py's _PALETTE comment). With 5 subsets, index 1..4
    # covers 4 of them and the 5th gets an explicit non-white colour
    # rather than silently plotting white-on-white.
    subset_colours_list = list(get_colours([1, 2, 3, 4])) + [np.array([0.2, 0.2, 0.2])]
    subset_colour = dict(zip(SUBSETS, subset_colours_list))
    wgr_alpha     = {w: a for w, a in zip(WGR_LABELS, [0.35, 0.55, 0.75, 1.0])}

    fig, (ax_gain, ax_phase) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

    for name, ir in zip(result.dataset_names, result.param_results):
        subset, wgr = name.rsplit("_", 1)
        ftf = calculate_ftf(ir.best.a_map, omega, ir.best.Ca_map)
        colour = subset_colour.get(subset, "k")
        alpha  = wgr_alpha.get(wgr, 1.0)
        label  = subset if wgr == WGR_LABELS[0] else None
        ax_gain.plot(freq, ftf['gain'], color=colour, alpha=alpha, label=label)
        ax_phase.plot(freq, ftf['phase'], color=colour, alpha=alpha)

    ax_gain.set_ylabel('|F|')
    ax_gain.legend(frameon=False, title='series (lighter = lower WGR)')
    ax_gain.grid(True, alpha=0.3)

    ax_phase.set_ylabel('phase [rad]')
    ax_phase.set_xlabel('frequency [Hz]')
    ax_phase.grid(True, alpha=0.3)

    fig.suptitle(f"WET Kornilov - pooled FTF comparison (shared N = {result.N_star})")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def save_summary_csv(result, out_path):
    print("\n=== Saving per-dataset parameter summary ===")
    N = result.N_star
    header = ['dataset', 'N', 'Ce', 'logML', 'logBFL', 'logOF']
    for i in range(1, N + 1):
        header += [f'n_{i}', f'tau_{i}_s', f'sigma_{i}_s']

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for name, ir in zip(result.dataset_names, result.param_results):
            best = ir.best
            row = [name, best.N, float(best.Ce), best.logML, best.logBFL, best.logOF]
            row += [float(v) for v in np.asarray(best.a_map)]
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
        "--model-orders", type=int, nargs="+", default=[1, 2, 3],
        help="Candidate model orders for the stage-1 ranking sweep (default: 1 2 3).")
    parser.add_argument(
        "--ranking-method", choices=["laplace", "vi"], default="laplace",
        help="Backend for stage 1 (default: laplace - runs "
             "len(datasets) x len(model_orders) times, so cheap is sensible).")
    parser.add_argument(
        "--param-method", choices=["laplace", "vi"], default=None,
        help="Backend for stage 2 (default: same as --ranking-method). "
             "Only runs once per dataset, so 'vi' is affordable here even "
             "when --ranking-method is 'laplace'.")
    parser.add_argument(
        "--ds-mode", choices=["factor", "frequency", "rate"], default="rate",
        help="Downsampling mode passed to prepare_signals (default: rate). "
             "The raw Kornilov signals are 1 MHz / 115k-500k samples per "
             "case, same order as the shipped BRS_EderSilva23 dataset.")
    parser.add_argument(
        "--ds-value", type=float, default=3e-4,
        help="Downsampling value, interpreted per --ds-mode (default: 3e-4).")
    parser.add_argument(
        "--ce0", type=float, default=None,
        help="Pool-wide noise-variance override (MML initial guess unless "
             "a dataset has its own DatasetSpec.Ce0). Default: 1e-4.")
    parser.add_argument(
        "--no-parallel", action="store_true",
        help="Disable vmap-parallel restarts (Python for-loop instead); "
             "useful for debugging on CPU.")
    args = parser.parse_args()

    print("Loading WET Kornilov datasets...")
    datasets = load_all_kornilov_datasets(subsets=args.subsets, wgr_labels=args.wgr)
    print(f"  {len(datasets)} cases: {[d.name for d in datasets]}")

    preproc_cfg = PreprocConfig(DSmode=args.ds_mode, DSvalue=args.ds_value)
    opt_cfg     = OptimizerConfig(use_parallel=not args.no_parallel)

    config = PooledInferenceConfig(
        prior             = PriorConfig(),
        preproc           = preproc_cfg,
        ranking_method    = args.ranking_method,
        ranking_optimizer = opt_cfg,
        param_method      = args.param_method,
        Ce0               = args.ce0,
        verbose           = True,
    )

    result = infer_shared_model_order(datasets, args.model_orders, config)

    print(f"\nShared model order across {len(datasets)} datasets: N = {result.N_star}")

    plot_pooled_ftf(result, OUTPUT_DIR / "kornilov_ftf_pooled.png")
    save_summary_csv(result, OUTPUT_DIR / "kornilov_pooled_summary.csv")

    print("\nDone.")


if __name__ == "__main__":
    main()
