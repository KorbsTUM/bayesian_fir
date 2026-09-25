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

Zero-DC-gain hypothesis (--lfl 0): the fitted N=1 impulse response has an
implausible nonzero FTF(0) (a single Gaussian lobe can only ever integrate
to a positive area). PriorConfig.LFL is the existing soft constraint on
sum(n_i) already used for the BRS_EderSilva23 case (there pinned to 1);
setting it to 0 here tests whether the data actually support a zero-DC-gain
(signed, multi-lobe) impulse response instead. When --lfl is given, EACH
version is fit twice - once unconstrained, once with the LFL prior - and
both are overlaid on the IR/FTF plots plus compared via their logML, which
is a direct Bayes-factor test of the zero-DC-gain hypothesis against the
unconstrained fit (same data, same noise model, only the prior differs).
Note N=1 is degenerate under LFL=0 (forces n_1 -> 0); include N>=2 in
--model-orders for this to be meaningful - the evidence sweep will reject
N=1 on its own once N=2 is available, no special-casing needed.

Usage:
    python examples/run_spray_multirun.py
    python examples/run_spray_multirun.py --version V3
    python examples/run_spray_multirun.py --version V4 --ranking-method laplace --param-method vi
    python examples/run_spray_multirun.py --model-orders 1 2 3 --mcmc
    python examples/run_spray_multirun.py --model-orders 2 3 --lfl 0
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


def fit_version(version: str, u_list, q_list, t_list, args, lfl=None) -> MultiRunResult:
    """Run the joint multi-run fit for one version, optionally under the
    LFL prior (sum(n_i) ~ N(lfl, lfl_sigma^2)). lfl=None reproduces the
    unconstrained fit."""
    T_c   = args.l_ref / args.u_ref
    label = f"Spray{version}" + (f" (LFL={lfl})" if lfl is not None else " (unconstrained)")

    preproc_cfg = PreprocConfig(DSmode=args.ds_mode, DSvalue=args.ds_value)
    opt_cfg     = OptimizerConfig(use_parallel=not args.no_parallel)
    prior_cfg   = PriorConfig(LFL=lfl, LFL_sigma=args.lfl_sigma)

    if lfl == 0.0 and args.model_orders == [1]:
        print(f"\nWARNING: --lfl 0 with --model-orders 1 only is degenerate "
              f"(forces n_1 -> 0, collapsing the impulse response) - "
              f"include N>=2 for a meaningful fit.")

    config = MultiRunConfig(
        prior             = prior_cfg,
        preproc           = preproc_cfg,
        ranking_method    = args.ranking_method,
        ranking_optimizer = opt_cfg,
        param_method      = args.param_method,
        run_mcmc          = args.mcmc,
        mcmc_iter         = args.mcmc_iter,
        verbose           = True,
    )

    print(f"\n=== {label} ===")
    result = infer_impulse_response_multirun(
        u_list, q_list, t_list, T_c,
        model_orders=args.model_orders, config=config, run_names=RUN_NAMES)

    print(f"\nDone. {label}: shared N = {result.N_star}, "
          f"method = {result.best.method}, logML = {result.best.logML:.2f}")
    return result


def print_evidence_comparison(fits: dict, versions: list):
    """fits: dict[(version, 'free'|'lfl')] -> MultiRunResult. Prints the
    logML gap between the unconstrained and LFL-constrained fit for each
    version - a direct Bayes-factor test of the LFL hypothesis, since both
    fits share the same data and noise model and differ only in the prior."""
    print("\n" + "=" * 60)
    print("Zero-DC-gain (LFL) hypothesis test")
    print("=" * 60)
    print(f"{'version':<10}{'N (free)':<10}{'N (LFL)':<10}"
          f"{'logML free':<14}{'logML LFL':<14}{'delta':<10}")
    print("-" * 68)
    for version in versions:
        r_free = fits[(version, 'free')]
        r_lfl  = fits.get((version, 'lfl'))
        if r_lfl is None:
            continue
        delta = r_lfl.best.logML - r_free.best.logML
        print(f"{version:<10}{r_free.N_star:<10}{r_lfl.N_star:<10}"
              f"{r_free.best.logML:<14.2f}{r_lfl.best.logML:<14.2f}{delta:<+10.2f}")
    print("-" * 68)
    print("delta = logML(LFL) - logML(free): near 0 means the data support "
          "zero DC gain about as well as an unconstrained gain; strongly "
          "negative means the data pay a real evidence cost to enforce it.")
    print("=" * 60)


def plot_ir_comparison(entries: list, out_path: Path):
    """Impulse response h(t), MAP mean +/- 95% band, for each entry.

    entries: list of (label, result, color, linestyle).
    """
    print("\n=== Impulse response comparison ===")
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, result, color, linestyle in entries:
        h = result.best.h
        t_ms = np.asarray(h.time) * 1e3
        band = 1.96 * np.sqrt(np.asarray(h.var))
        errorpatch(ax, t_ms, h.val, band, band, color=color,
                   line_kwargs={'label': label, 'linestyle': linestyle})
    ax.set_xlabel('time [ms]')
    ax.set_ylabel('h(t)')
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_ftf_comparison(entries: list, out_path: Path):
    """FTF gain/phase, MAP +/- 95% credible band, for each entry.

    entries: list of (label, result, color, linestyle).
    """
    print("\n=== FTF comparison ===")
    omega = jnp.linspace(0.0, 2 * jnp.pi * 500, 200)
    freq  = omega / (2 * jnp.pi)

    fig, (ax_gain, ax_phase) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    for label, result, color, linestyle in entries:
        ftf = calculate_ftf(result.best.a_map, omega, result.best.Ca_map)
        errorpatch(ax_gain, freq, ftf['gain'], ftf['gain95lo'], ftf['gain95hi'],
                   color=color, line_kwargs={'label': label, 'linestyle': linestyle})
        errorpatch(ax_phase, freq, ftf['phase'], ftf['phase95lo'], ftf['phase95hi'],
                   color=color, line_kwargs={'linestyle': linestyle})
    ax_gain.axhline(0.0, color='grey', linewidth=0.8, linestyle=':')
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
    parser.add_argument(
        "--lfl", type=float, default=None,
        help="Low-frequency (DC gain) constraint: soft-constrains sum(n_i) "
             "to this value (e.g. 0 to test the zero-DC-gain hypothesis, "
             "mirroring the LFL=1 pass on the BRS_EderSilva23 case). "
             "Default: unconstrained. When set, each version is fit BOTH "
             "unconstrained and with this constraint, and both are overlaid "
             "on the plots plus compared via logML (see module docstring). "
             "N=1 is degenerate under --lfl 0 - include N>=2 in "
             "--model-orders.")
    parser.add_argument(
        "--lfl-sigma", type=float, default=0.01,
        help="Std of the soft prior on sum(n_i) - lfl (default: 0.01, same "
             "default PriorConfig uses for the BRS LFL=1 case). Loosen this "
             "if the optimizer fights the constraint too hard on Spray's "
             "noisier data.")
    args = parser.parse_args()

    versions = ["V3", "V4"] if args.version == "both" else [args.version]

    fits = {}
    for version in versions:
        print(f"\nLoading Spray{version} (3 runs)...")
        u_list, q_list, t_list = load_version(version, args.dt)
        fits[(version, 'free')] = fit_version(version, u_list, q_list, t_list, args, lfl=None)
        if args.lfl is not None:
            fits[(version, 'lfl')] = fit_version(version, u_list, q_list, t_list, args, lfl=args.lfl)

    if args.lfl is not None:
        print_evidence_comparison(fits, versions)

    if not args.no_plots:
        entries = []
        for version in versions:
            base_color = VERSION_COLOUR[version]
            r_free = fits[(version, 'free')]
            entries.append((f"Spray{version} (N={r_free.N_star}, {r_free.best.method})",
                             r_free, base_color, '-'))
            r_lfl = fits.get((version, 'lfl'))
            if r_lfl is not None:
                entries.append((f"Spray{version} LFL={args.lfl} (N={r_lfl.N_star}, "
                                 f"{r_lfl.best.method})", r_lfl, base_color, '--'))
        # Tag output filenames with the param-fit method so re-runs under a
        # different backend (e.g. --param-method vi) don't clobber a
        # previous comparison's plots.
        tag = args.param_method or args.ranking_method
        plot_ir_comparison(entries, OUTPUT_DIR / f"spray_ir_comparison_{tag}.png")
        plot_ftf_comparison(entries, OUTPUT_DIR / f"spray_ftf_comparison_{tag}.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
