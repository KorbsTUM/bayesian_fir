"""
core/ftf.py
============
Evaluates the flame transfer function (FTF) from pulse-model parameters
in physical space.

The FTF is the Fourier transform of the impulse response:

    FTF(omega) = sum_i n_i * exp(-1j*omega*tau_i - 0.5*omega^2*sigma_i^2)

When a physical-space covariance Ca is supplied, 95% credible intervals
for gain and (unwrapped) phase are estimated by Monte Carlo sampling
from N(a, Ca), matching calculateFTF.m.

References:
    Yoko & Polifke (2026), Section 2.
"""

import jax
import jax.numpy as jnp
import numpy as np


def _ftf_complex(a: jnp.ndarray, omega: jnp.ndarray) -> jnp.ndarray:
    """
    Evaluate the complex FTF for a single physical parameter vector.

    Parameters
    ----------
    a     : jnp.ndarray, shape (3P,)   [n_1, tau_1, sig_1, n_2, ...].
    omega : jnp.ndarray, shape (T,)    Angular frequencies [rad/s].

    Returns
    -------
    ftf : jnp.ndarray, shape (T,) complex
    """
    n   = a[0::3]
    tau = a[1::3]
    sig = a[2::3]

    # (T, P) broadcast
    g = jnp.exp(-1j * omega[:, None] * tau[None, :]
                - 0.5 * (omega[:, None] ** 2) * (sig[None, :] ** 2))
    return jnp.sum(n[None, :] * g, axis=1)          # (T,)


def calculate_ftf(a: jnp.ndarray,
                   omega: jnp.ndarray,
                   Ca: jnp.ndarray | None = None,
                   n_samples: int = 5000,
                   seed: int = 0) -> dict:
    """
    Evaluate the flame transfer function gain and phase, with optional
    95% credible intervals from Monte Carlo sampling of the physical-
    space posterior N(a, Ca).

    Parameters
    ----------
    a         : jnp.ndarray, shape (3P,)
        MAP physical parameter vector [n_1, tau_1, sig_1, n_2, ...].
    omega     : jnp.ndarray, shape (T,)
        Angular frequency vector [rad/s].
    Ca        : jnp.ndarray, shape (3P, 3P), optional
        Physical-space posterior covariance. When supplied, 95%
        credible intervals are estimated by sampling n_samples draws
        from N(a, Ca) (matches calculateFTF.m's mvnrnd-based intervals).
    n_samples : int
        Number of Monte Carlo samples used for the credible intervals.
    seed      : int
        Seed for the NumPy RNG used to draw samples (MATLAB's mvnrnd
        call is unseeded; an explicit seed is used here for
        reproducibility, consistent with the rest of this JAX port).

    Returns
    -------
    ftf : dict with keys
        'gain'  : jnp.ndarray, (T,)   MAP gain |FTF(omega)|.
        'phase' : jnp.ndarray, (T,)   MAP unwrapped phase.
        (only when Ca is supplied)
        'gain95lo', 'gain95hi'   : jnp.ndarray, (T,)  Gain credible band,
                                    given as offsets from 'gain' (lo <= 0 <= hi).
        'phase95lo', 'phase95hi' : jnp.ndarray, (T,)  Phase credible band,
                                    given as offsets from 'phase'.
    """
    omega = jnp.asarray(omega).ravel()
    a     = jnp.asarray(a)

    ftf_map    = _ftf_complex(a, omega)
    gain_map   = jnp.abs(ftf_map)
    phase0_map = jnp.angle(ftf_map)
    phase_map  = jnp.unwrap(phase0_map)

    result = {'gain': gain_map, 'phase': phase_map}

    if Ca is not None:
        rng     = np.random.default_rng(seed)
        samples = rng.multivariate_normal(np.asarray(a), np.asarray(Ca),
                                           size=n_samples)          # (N, 3P)

        ftf_batch = jax.vmap(_ftf_complex, in_axes=(0, None))(
            jnp.asarray(samples), omega)                            # (N, T)
        gain  = jnp.abs(ftf_batch)                                  # (N, T)
        phase = jnp.angle(ftf_batch)                                # (N, T)

        # Gain credible band, centred on the MAP gain
        dgain             = gain - gain_map[None, :]
        result['gain95lo'] = -jnp.quantile(dgain, 0.025, axis=0)
        result['gain95hi'] =  jnp.quantile(dgain, 0.975, axis=0)

        # Phase credible band: wrap the deviation from the MAP's
        # (non-unwrapped) phase before taking quantiles, to avoid
        # 2*pi discontinuities biasing the interval.
        dphase    = jnp.angle(jnp.exp(1j * (phase - phase0_map[None, :])))
        dphase_lo = jnp.quantile(dphase, 0.025, axis=0)
        dphase_hi = jnp.quantile(dphase, 0.975, axis=0)

        phase95lo_raw = -dphase_lo
        phase95hi_raw =  dphase_hi

        # Unwrap the bounds relative to the (already unwrapped) MAP phase
        result['phase95lo'] = phase_map - jnp.unwrap(phase_map - phase95lo_raw)
        result['phase95hi'] = jnp.unwrap(phase_map + phase95hi_raw) - phase_map

    return result
