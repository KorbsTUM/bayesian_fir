#!/usr/bin/env python3
"""
examples/run_spray_multirun.py
================================
Joint multi-run inference recipe for the spray burner dataset
(data/SprayV3/, data/SprayV4/): combine all 3 runs of one version into a
single joint DTD parameter fit
(inference.multirun.infer_impulse_response_multirun) rather than fitting
each run independently. Each run alone is short enough (77k-95k raw
samples, and far fewer valid samples after downsampling) that a
single-run fit risks being unreliable; combining all 3 runs of the same
nominal condition - one shared parameter vector, one shared noise
variance - roughly triples the informative sample count feeding one
estimate instead of producing 3 separately-shaky ones. See
inference/multirun.py's module docstring for the full design/why.

Unlike the WET Kornilov dataset (noise-free, fixed Ce0), this data is
treated as noisy - OptimizerConfig's default infer_noise=True estimates
Ce from the data itself.

Laplace/VI are independently selectable for the model-order sweep
(--ranking-method) and the final parameter fit (--param-method, defaults
to --ranking-method) - useful since VI is much more expensive than
Laplace, so a cheap Laplace sweep across candidate orders followed by one
VI fit only at the winning order avoids paying VI's cost once per
candidate order.

By default both versions are fit and an IR + FTF comparison plot is saved
under examples/outputs/ (spray_ir_comparison.png, spray_ftf_comparison.png).

Usage:
    python examples/run_spray_multirun.py
    python examples/run_spray_multirun.py --version V3
    python examples/run_spray_multirun.py --version V4 --ranking-method laplace --param-method vi
    python examples/run_spray_multirun.py --model-orders 1 2 3 --mcmc
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from utils.io import load_raw_npy
from utils.plotting import get_colours, errorpatch
from core.prior import PriorConfig
from core.ftf import calculate_ftf
from config.defaults import PreprocConfig
from inference.optimizer import OptimizerConfig
from inference.variational import VIConfig
from inference.multirun import MultiRunConfig, MultiRunResult, infer_impulse_response_multirun

DATA_ROOT   = REPO_ROOT / "data"
OUTPUT_DIR  = REPO_ROOT / "examples" / "outputs"
RUN_NAMES   = ["run1", "run2", "run3"]
VERSION_COLOUR = {"V3": get_colours(4), "V4": get_colours(3)}

# Confirmed shared across all runs of both SprayV3 and SprayV4.
L_REF = 7.14e-2   # reference length [m]
U_REF = 25.0      # reference velocity [m/s]
DT    = 1e-6      # sampling interval [s] - the raw arrays carry no timebase


def load_version(version: str, dt: float):
    data_dir = DATA_ROOT / f"Spray{version}"
    u_list, q_list, t_list = [], [], []
    for run in RUN_NAMES:
        run_dir = data_dir / run
        data = load_raw_npy(run_dir, dt=dt)
        print(f"  [{run}] {data['u'].shape[0]} samples at {data['fs']:.1f} Hz "
              f"({data['t'][-1] * 1e3:.2f} ms)")
        u_list.append(data['u'])
        q_list.append(data['q'])
        t_list.append(data['t'])
    return u_list, q_list, t_list


def run_version(version: str, args) -> MultiRunResult:
    T_c = args.l_ref / args.u_ref

    print(f"\nLoading Spray{version} (3 runs)...")
    u_list, q_list, t_list = load_version(version, args.dt)

    preproc_cfg = PreprocConfig(DSmode=args.ds_mode, DSvalue=args.ds_value)
    opt_cfg     = OptimizerConfig(use_parallel=not args.no_parallel)

    config = MultiRunConfig(
        prior             = PriorConfig(),
        preproc           = preproc_cfg,
        ranking_method    = args.ranking_method,
        ranking_optimizer = opt_cfg,
        param_method      = args.param_method,
        run_mcmc          = args.mcmc,
        mcmc_iter         = args.mcmc_iter,
        verbose           = True,
    )

    result = infer_impulse_response_multirun(
        u_list, q_list, t_list, T_c,
        model_orders=args.model_orders, config=config, run_names=RUN_NAMES)

    print(f"\nDone. Spray{version}: shared N = {result.N_star}, "
          f"method = {result.best.method}")
    return result


def plot_ir_comparison(results: dict, out_path: Path):
    """Impulse response h(t), MAP mean +/- 95% band, for each version."""
    print("\n=== Impulse response comparison ===")
    fig, ax = plt.subplots(figsize=(6, 4))
    for version, result in results.items():
        h = result.best.h
        t_ms = np.asarray(h.time) * 1e3
        band = 1.96 * np.sqrt(np.asarray(h.var))
        errorpatch(ax, t_ms, h.val, band, band,
                   color=VERSION_COLOUR[version],
                   line_kwargs={'label': f"Spray{version} (N={result.N_star}, "
                                          f"{result.best.method})"})
    ax.set_xlabel('time [ms]')
    ax.set_ylabel('h(t)')
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_ftf_comparison(results: dict, out_path: Path):
    """FTF gain/phase, MAP +/- 95% credible band, for each version."""
    print("\n=== FTF comparison ===")
    omega = jnp.linspace(0.0, 2 * jnp.pi * 500, 200)
    freq  = omega / (2 * jnp.pi)

    fig, (ax_gain, ax_phase) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    for version, result in results.items():
        ftf   = calculate_ftf(result.best.a_map, omega, result.best.Ca_map)
        color = VERSION_COLOUR[version]
        label = f"Spray{version} (N={result.N_star}, {result.best.method})"
        errorpatch(ax_gain, freq, ftf['gain'], ftf['gain95lo'], ftf['gain95hi'],
                   color=color, line_kwargs={'label': label})
        errorpatch(ax_phase, freq, ftf['phase'], ftf['phase95lo'], ftf['phase95hi'],
                   color=color)
    ax_gain.set_ylabel('|F|')
    ax_gain.legend(frameon=False)
    ax_gain.grid(True, alpha=0.3)
    ax_phase.set_ylabel('phase [rad]')
    ax_phase.set_xlabel('frequency [Hz]')
    ax_phase.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--version", choices=["V3", "V4", "both"], default="both",
        help="Spray burner version(s) to fit (default: both, needed for the "
             "IR/FTF comparison plots).")
    parser.add_argument(
        "--model-orders", type=int, nargs="+", default=[1, 2, 3],
        help="Candidate model orders for the joint-evidence sweep (default: 1 2 3).")
    parser.add_argument(
        "--ranking-method", choices=["laplace", "vi"], default="laplace",
        help="Backend for the model-order sweep (default: laplace - runs "
             "once per candidate order, so cheap is sensible).")
    parser.add_argument(
        "--param-method", choices=["laplace", "vi"], default=None,
        help="Backend for the final joint parameter fit (default: same as "
             "--ranking-method). Only runs once, so 'vi' is affordable here "
             "even when --ranking-method is 'laplace'.")
    parser.add_argument(
        "--l-ref", type=float, default=L_REF,
        help=f"Reference length [m] for T_c = l_ref / u_ref (default: {L_REF}).")
    parser.add_argument(
        "--u-ref", type=float, default=U_REF,
        help=f"Reference velocity [m/s] for T_c = l_ref / u_ref (default: {U_REF}).")
    parser.add_argument(
        "--dt", type=float, default=DT,
        help=f"Sampling interval [s] (default: {DT}).")
    parser.add_argument(
        "--ds-mode", choices=["factor", "frequency", "rate"], default="rate",
        help="Downsampling mode passed to prepare_signals (default: rate). "
             "The raw spray signals are ~1 MHz, same order as the shipped "
             "BRS_EderSilva23/WET Kornilov datasets.")
    parser.add_argument(
        "--ds-value", type=float, default=3e-4,
        help="Downsampling value, interpreted per --ds-mode (default: 3e-4, "
             "tuned to ~1 MHz data - use --ds-mode factor --ds-value 1 to "
             "disable downsampling).")
    parser.add_argument(
        "--mcmc", action="store_true",
        help="Run MCMC validation on the final joint fit.")
    parser.add_argument(
        "--mcmc-iter", type=int, default=200_000,
        help="MCMC iterations when --mcmc is set (default: 200000).")
    parser.add_argument(
        "--no-parallel", action="store_true",
        help="Disable vmap-parallel restarts (Python for-loop instead); "
             "useful for debugging on CPU.")
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip the IR/FTF comparison plots (just print the fit summaries).")
    args = parser.parse_args()

    versions = ["V3", "V4"] if args.version == "both" else [args.version]
    results = {version: run_version(version, args) for version in versions}

    if not args.no_plots:
        plot_ir_comparison(results, OUTPUT_DIR / "spray_ir_comparison.png")
        plot_ftf_comparison(results, OUTPUT_DIR / "spray_ftf_comparison.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
