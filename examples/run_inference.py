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
    2. Configure preprocessing (downsampling), by default matching the
       paper's turbulent-case settings (DSmode='rate', DSvalue=3e-4 ->
       ~1,570 samples instead of 470,000).
    3. Run multi-order Bayesian inference with the prior free (LFL unset).
    4. Repeat with the low-frequency-limit constraint LFL=1.
    5. Evaluate the flame transfer function from each posterior and plot
       gain/phase against the experimental reference (if available).
    6. Optionally (--mcmc) run MCMC to validate the Laplace approximation
       for a single fixed model order, and plot the corner heatmap.

Using your own (e.g. laminar) dataset
--------------------------------------
By default this loads data/BRS_EderSilva23/, the turbulent dataset shipped
with the repo, and treats it as noisy (the optimizer estimates the noise
level from the data itself). To try a different dataset - e.g. laminar
data that is effectively noise-free - point --data-dir at a directory
containing a data_raw_incomp.mat file in the same MATLAB iddata format
(InputData/OutputData/Ts fields; see utils.io.load_raw_incomp), and pass
--noise-free. An experimental FTF reference file is optional: if
--ftf-path isn't given or doesn't exist, the comparison plot just shows
the inferred FTF without an experimental overlay. Downsampling and the
convective timescale T_c = L_ref / U_ref are also very sampling-rate- and
geometry-dependent, so both are exposed as flags rather than hardcoded.

    python examples/run_inference.py \
        --data-dir data/my_laminar_case \
        --noise-free --ce0 1e-8 \
        --l-ref 0.02 --u-ref 3.5 \
        --ds-mode factor --ds-value 1

Noise-free note: with --noise-free, Ce is held fixed at --ce0 rather than
estimated (MacKay MML estimation of Ce on near-noise-free data drives
Ce -> 0 and becomes numerically singular - see inference/sensitivity.py's
docstring for the same issue in the differentiable path). Pick --ce0
small relative to the signal's variance, not exactly zero.

Usage:
    python examples/run_inference.py
    python examples/run_inference.py --model-orders 1 2 3
    python examples/run_inference.py --mcmc
    python examples/run_inference.py --data-dir data/my_laminar_case --noise-free --ce0 1e-8

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
from inference.optimizer import OptimizerConfig
from infer_impulse_response import infer_impulse_response, InferenceConfig


DEFAULT_DATA_DIR = REPO_ROOT / "data" / "BRS_EderSilva23"
DEFAULT_FTF_PATH = DEFAULT_DATA_DIR / "FTF_exp_30kW_Front_lambda1.3.mat"
OUTPUT_DIR = REPO_ROOT / "examples" / "outputs"

# Physical parameters (generateFigures.m, "System properties") - defaults
# for the shipped turbulent dataset; override with --l-ref/--u-ref for a
# different flame geometry (e.g. a laminar case).
L_REF = 50e-3    # flame length [m]
U_REF = 11.3     # bulk flow velocity [m/s]


def load_data(data_dir: Path):
    path = data_dir / "data_raw_incomp.mat"
    print(f"Loading data from {path}")
    data = load_raw_incomp(path)
    print(f"  {data['u'].shape[0]} samples at {data['fs']:.1f} Hz "
          f"({data['t'][-1] * 1e3:.1f} ms)")
    return data


def build_optimizer_config(args) -> OptimizerConfig:
    if args.noise_free:
        return OptimizerConfig(infer_noise=False, use_parallel=not args.no_parallel)
    return OptimizerConfig(use_parallel=not args.no_parallel)


def run_baseline(u, q, t, model_orders, preproc_cfg, opt_cfg, T_c, ce0, method):
    # infer_impulse_response prints its own progress/summary (InferenceConfig
    # defaults to verbose=True), so nothing further is printed here.
    print("\n=== Step 1: baseline inference (LFL free) ===")
    config = InferenceConfig(preproc=preproc_cfg, optimizer=opt_cfg, Ce0=ce0, method=method)
    return infer_impulse_response(u, q, t, T_c, model_orders=model_orders, config=config)


def run_lfl(u, q, t, model_orders, preproc_cfg, opt_cfg, T_c, ce0, method):
    print("\n=== Step 2: inference with LFL = 1 ===")
    config = InferenceConfig(preproc=preproc_cfg, optimizer=opt_cfg,
                              prior=PriorConfig(LFL=1.0), Ce0=ce0, method=method)
    return infer_impulse_response(u, q, t, T_c, model_orders=model_orders, config=config)


def plot_ftf_comparison(result_baseline, result_lfl, exp_ftf, out_path):
    print("\n=== Step 3: flame transfer function comparison ===")
    omega = jnp.linspace(0.0, 2 * jnp.pi * 500, 200)
    ftf_baseline = calculate_ftf(result_baseline.best.a_map, omega, result_baseline.best.Ca_map)
    ftf_lfl      = calculate_ftf(result_lfl.best.a_map,      omega, result_lfl.best.Ca_map)
    freq = omega / (2 * jnp.pi)

    fig, (ax_gain, ax_phase) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)

    if exp_ftf is not None:
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

    if exp_ftf is not None:
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


def run_mcmc_validation(u, q, t, model_order, preproc_cfg, opt_cfg, T_c, ce0, method, out_path):
    print(f"\n=== Optional: MCMC validation (N={model_order}, matches Figures 10-11) ===")
    config = InferenceConfig(
        preproc  = preproc_cfg,
        optimizer= opt_cfg,
        prior    = PriorConfig(LFL=1.0, T_h=0.015),
        run_mcmc = True,
        mcmc_iter= 500_000,
        Ce0      = ce0,
        method   = method,
    )
    result = infer_impulse_response(u, q, t, T_c, model_orders=[model_order], config=config)
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
    parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR,
        help=f"Directory containing data_raw_incomp.mat (default: {DEFAULT_DATA_DIR}).")
    parser.add_argument(
        "--ftf-path", type=Path, default=None,
        help="Optional experimental FTF .mat file for the comparison plot "
             "(freq_exp/gain_exp/phase_exp). Skipped if not given or missing "
             "(default: FTF_exp_30kW_Front_lambda1.3.mat under --data-dir "
             "when using the default dataset).")
    parser.add_argument(
        "--noise-free", action="store_true",
        help="Hold the noise variance fixed at --ce0 instead of estimating "
             "it from the data (MacKay MML). Use for near-noise-free data "
             "(e.g. laminar cases) - MML estimation drives Ce -> 0 there, "
             "which is numerically singular.")
    parser.add_argument(
        "--ce0", type=float, default=None,
        help="Noise variance: fixed value when --noise-free is set "
             "(required in that case; try something small relative to the "
             "signal's variance, e.g. 1e-8), otherwise the MML estimator's "
             "initial guess (default 1e-4).")
    parser.add_argument(
        "--l-ref", type=float, default=L_REF,
        help=f"Reference length [m] for T_c = l_ref / u_ref (default: {L_REF}).")
    parser.add_argument(
        "--u-ref", type=float, default=U_REF,
        help=f"Reference velocity [m/s] for T_c = l_ref / u_ref (default: {U_REF}).")
    parser.add_argument(
        "--ds-mode", choices=["factor", "frequency", "rate"], default="rate",
        help="Downsampling mode passed to prepare_signals (default: rate). "
             "Use 'factor' with --ds-value 1 to disable downsampling.")
    parser.add_argument(
        "--ds-value", type=float, default=3e-4,
        help="Downsampling value, interpreted per --ds-mode (default: 3e-4, "
             "tuned to the shipped dataset's 1 MHz sampling rate - pick a "
             "value appropriate to your own data's fs, or use --ds-mode "
             "factor --ds-value 1 for no downsampling).")
    parser.add_argument(
        "--no-parallel", action="store_true",
        help="Disable vmap-parallel restarts (Python for-loop instead); "
             "useful for debugging on CPU.")
    parser.add_argument(
        "--method", choices=["laplace", "vi"], default="laplace",
        help="Inference backend (default: laplace). 'laplace' is the "
             "MAP + Hessian approximation used throughout the paper. 'vi' "
             "fits a normalizing flow via the ELBO, warm-started from the "
             "same Laplace MAP - see inference/variational.py.")
    args = parser.parse_args()

    if args.noise_free and args.ce0 is None:
        parser.error("--noise-free requires --ce0 (a fixed noise variance).")

    preproc_cfg = PreprocConfig(DSmode=args.ds_mode, DSvalue=args.ds_value)
    opt_cfg     = build_optimizer_config(args)
    T_c         = args.l_ref / args.u_ref

    data = load_data(args.data_dir)
    u, q, t = data['u'], data['q'], data['t']

    ftf_path = args.ftf_path
    if ftf_path is None and args.data_dir == DEFAULT_DATA_DIR:
        ftf_path = DEFAULT_FTF_PATH
    exp_ftf = None
    if ftf_path is not None and Path(ftf_path).exists():
        exp_ftf = load_ftf_experiment(ftf_path)
    else:
        print("No experimental FTF reference found - plotting inferred FTF only.")

    result_baseline = run_baseline(u, q, t, args.model_orders, preproc_cfg, opt_cfg, T_c,
                                    args.ce0, args.method)
    result_lfl      = run_lfl(u, q, t, args.model_orders, preproc_cfg, opt_cfg, T_c,
                               args.ce0, args.method)

    plot_ftf_comparison(result_baseline, result_lfl, exp_ftf,
                         OUTPUT_DIR / "ftf_comparison.png")

    if args.mcmc:
        run_mcmc_validation(u, q, t, args.mcmc_order, preproc_cfg, opt_cfg, T_c, args.ce0,
                             args.method, OUTPUT_DIR / "corner_mcmc.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
