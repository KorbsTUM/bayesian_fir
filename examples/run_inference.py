#!/usr/bin/env python3
"""
examples/run_inference.py
==========================
End-to-end recipe for running the Bayesian impulse-response inference
pipeline on the BRS_EderSilva23 dataset, mirroring the analysis in
generateFigures.m (Figures 5-7, and optionally 10-11).

Steps:
    1. Load the raw input/output data (utils.io.load_raw_incomp), which
       unpacks the MATLAB iddata object in data_raw_incomp.mat and
       applies the same (x - mean(x))/mean(x) normalisation as MATLAB.
    2. Configure preprocessing to match the paper's downsampling
       (DSmode='rate', DSvalue=3e-4 -> ~1,570 samples instead of 470,000).
    3. Run multi-order Bayesian inference with the prior free (LFL unset).
    4. Repeat with the low-frequency-limit constraint LFL=1.
    5. Evaluate the flame transfer function from each posterior and plot
       gain/phase against the experimental reference.
    6. Optionally (--mcmc) run MCMC to validate the Laplace approximation
       for a single fixed model order, and plot the corner heatmap.

Usage:
    python examples/run_inference.py
    python examples/run_inference.py --model-orders 1 2 3
    python examples/run_inference.py --mcmc

Output:
    examples/outputs/ftf_comparison.png
    examples/outputs/corner_mcmc.png   (only with --mcmc)
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.io import load_raw_incomp, load_ftf_experiment
from utils.plotting import errorpatch, get_colours, corner_heatmap
from core.prior import PriorConfig
from core.ftf import calculate_ftf
from config.defaults import PreprocConfig
from infer_impulse_response import infer_impulse_response, InferenceConfig


DATA_DIR   = REPO_ROOT / "data" / "BRS_EderSilva23"
OUTPUT_DIR = REPO_ROOT / "examples" / "outputs"

# Physical parameters (generateFigures.m, "System properties")
L_REF = 50e-3    # flame length [m]
U_REF = 11.3     # bulk flow velocity [m/s]
T_C   = L_REF / U_REF

# Downsampling matching the paper's analysis (generateFigures.m, Figures 5-7):
# DSmode='rate' with DSvalue=3e-4 gives ds_factor = floor(fs * DSvalue) = 300
# at this dataset's 1 MHz sampling rate.
PREPROC_CFG = PreprocConfig(DSmode='rate', DSvalue=3e-4)


def load_data():
    path = DATA_DIR / "data_raw_incomp.mat"
    print(f"Loading data from {path}")
    data = load_raw_incomp(path)
    print(f"  {data['u'].shape[0]} samples at {data['fs']:.1f} Hz "
          f"({data['t'][-1] * 1e3:.1f} ms)")
    return data


def run_baseline(u, q, t, model_orders):
    # infer_impulse_response prints its own progress/summary (InferenceConfig
    # defaults to verbose=True), so nothing further is printed here.
    print("\n=== Step 1: baseline inference (LFL free) ===")
    config = InferenceConfig(preproc=PREPROC_CFG)
    return infer_impulse_response(u, q, t, T_C, model_orders=model_orders, config=config)


def run_lfl(u, q, t, model_orders):
    print("\n=== Step 2: inference with LFL = 1 ===")
    config = InferenceConfig(preproc=PREPROC_CFG, prior=PriorConfig(LFL=1.0))
    return infer_impulse_response(u, q, t, T_C, model_orders=model_orders, config=config)


def plot_ftf_comparison(result_baseline, result_lfl, exp_ftf, out_path):
    print("\n=== Step 3: flame transfer function comparison ===")
    omega = jnp.linspace(0.0, 2 * jnp.pi * 500, 200)
    ftf_baseline = calculate_ftf(result_baseline.best.a_map, omega, result_baseline.best.Ca_map)
    ftf_lfl      = calculate_ftf(result_lfl.best.a_map,      omega, result_lfl.best.Ca_map)
    freq = omega / (2 * jnp.pi)

    fig, (ax_gain, ax_phase) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)

    ax_gain.plot(exp_ftf['freq'], exp_ftf['gain'], 'o', color=get_colours(2), label='Exp')
    errorpatch(ax_gain, freq, ftf_baseline['gain'],
               ftf_baseline['gain95lo'], ftf_baseline['gain95hi'],
               color=get_colours(4), line_kwargs={'label': 'BI (LFL free)'})
    errorpatch(ax_gain, freq, ftf_lfl['gain'],
               ftf_lfl['gain95lo'], ftf_lfl['gain95hi'],
               color=get_colours(1), line_kwargs={'label': 'BI (LFL = 1)'})
    ax_gain.set_ylabel('|F|')
    ax_gain.legend(frameon=False)
    ax_gain.grid(True, alpha=0.3)

    ax_phase.plot(exp_ftf['freq'], exp_ftf['phase'], 'o', color=get_colours(2))
    errorpatch(ax_phase, freq, ftf_baseline['phase'],
               ftf_baseline['phase95lo'], ftf_baseline['phase95hi'], color=get_colours(4))
    errorpatch(ax_phase, freq, ftf_lfl['phase'],
               ftf_lfl['phase95lo'], ftf_lfl['phase95hi'], color=get_colours(1))
    ax_phase.set_ylabel('phase [rad]')
    ax_phase.set_xlabel('frequency [Hz]')
    ax_phase.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def run_mcmc_validation(u, q, t, model_order, out_path):
    print(f"\n=== Optional: MCMC validation (N={model_order}, matches Figures 10-11) ===")
    config = InferenceConfig(
        preproc  = PREPROC_CFG,
        prior    = PriorConfig(LFL=1.0, T_h=0.015),
        run_mcmc = True,
        mcmc_iter= 500_000,
    )
    result = infer_impulse_response(u, q, t, T_C, model_orders=[model_order], config=config)
    print(f"  MCMC acceptance rate: {result.mcmc.accept_rate:.1%}")

    fig = plt.figure(figsize=(8, 8))
    corner_heatmap(fig, result.mcmc.post_samples.T,
                   result.best.b_map, result.best.Cb_map, result.best.names)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model-orders", type=int, nargs="+", default=[1, 2, 3, 4, 5],
        help="Candidate model orders to rank (default: 1 2 3 4 5).")
    parser.add_argument(
        "--mcmc", action="store_true",
        help="Also run the MCMC validation pass (Figures 10-11): fixes "
             "N=--mcmc-order, T_h=15ms, LFL=1, 500k MH iterations.")
    parser.add_argument(
        "--mcmc-order", type=int, default=3,
        help="Model order for the MCMC validation pass (default: 3).")
    args = parser.parse_args()

    data = load_data()
    u, q, t = data['u'], data['q'], data['t']
    exp_ftf = load_ftf_experiment(DATA_DIR / "FTF_exp_30kW_Front_lambda1.3.mat")

    result_baseline = run_baseline(u, q, t, args.model_orders)
    result_lfl      = run_lfl(u, q, t, args.model_orders)

    plot_ftf_comparison(result_baseline, result_lfl, exp_ftf,
                         OUTPUT_DIR / "ftf_comparison.png")

    if args.mcmc:
        run_mcmc_validation(u, q, t, args.mcmc_order, OUTPUT_DIR / "corner_mcmc.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
