#!/usr/bin/env python3
"""
examples/run_sensitivity.py
============================
Compute the sensitivity of the fitted DTD parameters (n_i, tau_i, sigma_i)
to every sample of your input/output signals (u, q), via implicit
differentiation through the MAP optimum (inference.sensitivity).

WHAT TO EDIT
------------
Everything you need to change is in the "USER SETTINGS" block below:
    1. Point DATA_PATH at your own .npz file containing arrays 'u', 'q'
       (equal length) and a scalar 'fs' (sampling frequency in Hz). If
       your signals live in MATLAB .mat / .csv / etc., convert once with
       e.g.:
           import numpy as np, scipy.io as sio
           d = sio.loadmat("my_signals.mat")
           np.savez("my_signals.npz", u=d['u'].ravel(), q=d['q'].ravel(), fs=1000.0)
       If DATA_PATH is left as None, a small synthetic demo signal is
       generated instead so the script runs out of the box.
    2. Set MODEL_ORDER (N, number of Gaussian pulses) and T_C (convective
       timescale [s] = flame length / bulk velocity, or whatever is
       physically appropriate for your case) to match your problem.
    3. Set the prior hyperparameters (PRIOR_CFG) to whatever you'd
       normally use for a standard inference run on this data.
    4. Set NOISY = True if your signal has real measurement noise
       (recommended default), or False only if it is essentially
       noise-free (see note below - mixing this up silently breaks the
       result).

Everything after that runs unattended.

Noisy vs. noise-free
---------------------
* NOISY = True  -> optimizer estimates the noise level Ce from the data
                    itself (MacKay MML). This is what you want for real,
                    measured signals.
* NOISY = False -> Ce is held fixed at CE0 (see inference/sensitivity.py
                    docstring). Only use this for signals with no
                    meaningful measurement noise (e.g. clean simulation
                    output): with NOISY=True on noise-free data, the
                    estimated noise Ce* collapses to ~0 and the
                    sensitivity becomes numerically singular.

Usage
-----
    python examples/run_sensitivity.py

Output
------
Prints the fitted physical parameters, a convergence check, and the
sensitivity matrices' shapes/summary stats. Saves the full Jacobians to
examples/outputs/sensitivity.npz (da_du, da_dq, a_map, param_names).
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from core.prior import PriorConfig, generate_prior
from core.parameter_maps import map_to_physical
from core.cost import calculate_cost_varpro
from inference.optimizer import OptimizerConfig
from inference.sensitivity import make_map_estimator
from signals.prepare import prepare_signals_diff


# =============================================================================
# USER SETTINGS - edit this block for your own problem
# =============================================================================

# Path to a .npz file with arrays 'u', 'q' (same length) and scalar 'fs' [Hz].
# Leave as None to run on a generated synthetic demo signal instead.
DATA_PATH = None  # e.g. REPO_ROOT / "data" / "my_signals.npz"

MODEL_ORDER = 1        # N: number of Gaussian pulses in the DTD model
T_C = 50e-3 / 11.3     # convective timescale [s] (flame length / bulk velocity)
T_H = 0.05              # impulse response duration [s] to fit over

PRIOR_CFG = PriorConfig(T_h=T_H)   # use your usual prior here, e.g. PriorConfig(LFL=1.0, T_h=T_H)

NOISY = True            # True: estimate noise from data. False: fixed, near-zero noise.
CE0 = None               # None -> sensible default (1e-3 if NOISY else 1e-6). Override if needed.

OUTPUT_PATH = REPO_ROOT / "examples" / "outputs" / "sensitivity.npz"

# =============================================================================


def load_data():
    if DATA_PATH is not None:
        print(f"Loading signals from {DATA_PATH}")
        d = np.load(DATA_PATH)
        u, q, fs = d['u'], d['q'], float(d['fs'])
    else:
        print("DATA_PATH not set - generating a synthetic demo signal.")
        print("  (edit DATA_PATH in examples/run_sensitivity.py to use your own data)")
        rng = np.random.default_rng(0)
        # Pulse placed at the prior's median delay/width so it both fits
        # inside T_H and sits near where the optimizer starts searching -
        # this is purely to make the *demo* converge cleanly; your real
        # data has no such constraint.
        n_true = 1.0
        tau_true = T_C * np.exp(PRIOR_CFG.mu_gamma)
        sig_true = T_C * np.exp(PRIOR_CFG.mu_beta)
        fs = max(200.0, 20.0 / sig_true)     # >= ~20 samples across the pulse width
        M = int(np.ceil(2 * T_H * fs))       # duration = 2 * T_H, comfortably covers tau_true
        u = rng.standard_normal(M) * 0.3
        n_h = int(np.ceil(T_H * fs)) + 1
        t_h = np.arange(n_h) / fs
        h_true = (n_true / np.sqrt(2 * np.pi * sig_true ** 2)
                  * np.exp(-0.5 * ((t_h - tau_true) / sig_true) ** 2))
        q = np.convolve(u, h_true, mode='full')[:M] * (1.0 / fs)
        if NOISY:
            q = q + 0.02 * rng.standard_normal(M)
    print(f"  {u.shape[0]} samples at {fs:.1f} Hz")
    return jnp.array(u), jnp.array(q), fs


def check_convergence(map_estimate, u, q, fs, T_h, T_c, bp, Cp, prior_cfg):
    b_map, Ce = map_estimate(u, q)
    signals = prepare_signals_diff(u, q, fs, T_h)
    x_star = jnp.stack([b_map[1::3], b_map[2::3]]).T.ravel()
    _, dJ, *_ = calculate_cost_varpro(signals, Ce, x_star, bp, Cp, T_c, prior_cfg)
    grad_norm = float(jnp.linalg.norm(dJ))
    # 1e-4 is a reasonable rule of thumb, not a hard guarantee - tune to
    # your problem's scale if in doubt (see inference/sensitivity.py docstring).
    print(f"  ||dJ|| at reported optimum: {grad_norm:.3e} "
          f"({'OK' if grad_norm < 1e-4 else 'NOT CONVERGED - see docstring caveats'})")
    return b_map, Ce, grad_norm


def main():
    u, q, fs = load_data()

    bp, Cp, names, _ = generate_prior(MODEL_ORDER, T_C, PRIOR_CFG)

    ce0 = CE0
    if ce0 is None:
        ce0 = 1e-3 if NOISY else 1e-6

    opt_cfg = OptimizerConfig(infer_noise=NOISY)

    print(f"\n=== Fitting N={MODEL_ORDER} model (infer_noise={NOISY}, Ce0={ce0:.1e}) ===")
    map_estimate = make_map_estimator(
        fs, T_H, T_C, bp, Cp, PRIOR_CFG, opt_cfg, Ce0=ce0)

    b_map, Ce, grad_norm = check_convergence(
        map_estimate, u, q, fs, T_H, T_C, bp, Cp, PRIOR_CFG)

    a_map, params, _ = map_to_physical(b_map, T_C)
    print("\nFitted physical parameters:")
    for i in range(MODEL_ORDER):
        print(f"  pulse {i + 1}: n={float(params['n'][i]): .4f}  "
              f"tau={float(params['tau'][i]) * 1e3: .3f} ms  "
              f"sigma={float(params['sigma'][i]) * 1e3: .3f} ms")
    print(f"  Ce (noise variance): {float(Ce):.3e}")

    print("\n=== Computing sensitivities d(n_i,tau_i,sigma_i) / d(u,q) ===")
    def a_of_signals(u_, q_):
        b, _ = map_estimate(u_, q_)
        return map_to_physical(b, T_C)[0]

    da_du, da_dq = jax.jacobian(a_of_signals, argnums=(0, 1))(u, q)
    print(f"  da_du shape: {da_du.shape}  (rows = {names})")
    print(f"  da_dq shape: {da_dq.shape}")
    print(f"  max |da_du|: {float(jnp.abs(da_du).max()):.3e}")
    print(f"  max |da_dq|: {float(jnp.abs(da_dq).max()):.3e}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUTPUT_PATH,
              da_du=np.array(da_du), da_dq=np.array(da_dq),
              a_map=np.array(a_map), param_names=np.array(names),
              grad_norm=grad_norm, Ce=float(Ce))
    print(f"\nSaved sensitivities to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
