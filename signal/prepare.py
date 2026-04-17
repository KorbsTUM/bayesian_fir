"""
signal/prepare.py
=================
Prepares input/output time-series signals for Bayesian impulse response
inference.

Responsibilities:
    1. Estimate the signal bandwidth from the input PSD and compute an
       appropriate integer downsampling factor.
    2. Apply anti-aliased downsampling via scipy.signal.resample_poly with
       symmetric edge padding to suppress boundary artefacts.
    3. Package both the downsampled ('coarse') and original-resolution
       ('fine') signals into a nested dict consumed by calculate_cost and
       calculate_cost_varpro.
    4. Enforce the valid convolution region (Section 4.3 of Yoko & Polifke
       2026) by recording the slice of output samples for which the discrete
       convolution is fully supported by observed input data.

The 'coarse' signals are used during MAP optimisation (fast).
The 'fine' signals are retained for diagnostic evaluation at full resolution.

Signal dict layout (one entry per level: 'coarse' / 'fine'):

    signals['coarse'] = {
        'u'     : jnp.ndarray (M,)     input signal
        'q'     : jnp.ndarray (M,)     output signal
        'fs'    : float                 sampling frequency [Hz]
        'dt'    : float                 sampling interval  [s]
        't'     : jnp.ndarray (M,)     time vector        [s]
        't_h'   : jnp.ndarray (L,)     impulse response time vector [s]
        'n'     : int                   number of samples
        'valid' : jnp.ndarray (int)     index of first valid output sample
                                        (i.e. valid region = valid : n)
        'ds_factor' : int               downsampling factor applied
    }

Notes:
    - All heavy signal processing (FFT, resampling) is done in NumPy/SciPy
      since it runs once and does not need to be JIT-compiled.
    - Only the final packaged arrays are converted to jnp for use in JAX.
    - The 'valid' entry is a Python int (static), not a JAX array, so it
      can be used directly as a slice index inside JIT-compiled functions.

References:
    Yoko & Polifke (2026), Sections 4.3-4.4
"""

import numpy as np
import jax.numpy as jnp
from scipy.signal import resample_poly
from typing import Optional


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def prepare_signals(u: np.ndarray,
                    q: np.ndarray,
                    fs: float,
                    T_h: float,
                    ds_limit: Optional[int] = None) -> dict:
    """
    Prepare coarse and fine signal representations for inference.

    Parameters
    ----------
    u        : np.ndarray, shape (M,)
        Input fluctuation signal (zero-mean, normalised).
    q        : np.ndarray, shape (M,)
        Output fluctuation signal (zero-mean, normalised).
    fs       : float
        Original sampling frequency [Hz].
    T_h      : float
        Desired impulse response duration [s].
    ds_limit : int or None
        Maximum allowed downsampling factor (config.preproc.DSlimit).
        None means no limit.

    Returns
    -------
    signals : dict
        Nested dict with keys 'coarse' and 'fine', each containing
        the packaged signal struct described in the module docstring.
    """
    u = np.asarray(u, dtype=np.float64).ravel()
    q = np.asarray(q, dtype=np.float64).ravel()

    if len(u) != len(q):
        raise ValueError(f"u and q must have the same length, "
                         f"got {len(u)} and {len(q)}.")

    # ------------------------------------------------------------------
    # Step 1: estimate bandwidth and downsampling factor
    # ------------------------------------------------------------------
    f_cut     = _estimate_bandwidth(u, fs, T_h, energy_threshold=0.999)
    f_cut     *= 1.1                          # 10% margin
    ds_factor = max(1, int(np.floor(fs / (2.0 * f_cut))))

    if ds_limit is not None:
        ds_factor = min(ds_factor, int(ds_limit))

    # ------------------------------------------------------------------
    # Step 2: package fine (original resolution) signals
    # ------------------------------------------------------------------
    fine = _package_signals(u, q, fs, ds_factor=1, T_h=T_h)

    # ------------------------------------------------------------------
    # Step 3: downsample and package coarse signals
    # ------------------------------------------------------------------
    if ds_factor == 1:
        coarse = fine
    else:
        u_ds = _safe_resample(u, ds_factor)
        q_ds = _safe_resample(q, ds_factor)
        fs_ds = fs / ds_factor
        coarse = _package_signals(u_ds, q_ds, fs_ds,
                                   ds_factor=ds_factor, T_h=T_h)

    return {'coarse': coarse, 'fine': fine}


# ---------------------------------------------------------------------------
# Bandwidth estimation
# ---------------------------------------------------------------------------

def _estimate_bandwidth(u: np.ndarray,
                         fs: float,
                         T_h: float,
                         energy_threshold: float = 0.999) -> float:
    """
    Estimate the one-sided bandwidth of u that retains `energy_threshold`
    fraction of the total signal energy.

    The FFT is zero-padded to the next power of 2 above len(u) + n_h - 1,
    where n_h = ceil(T_h * fs) + 1 is the impulse response length in samples.
    This padding length matches the one used in the convolution so that the
    spectral estimate is consistent with the inference problem.

    Parameters
    ----------
    u                : np.ndarray  Input signal.
    fs               : float       Sampling frequency [Hz].
    T_h              : float       Impulse response duration [s].
    energy_threshold : float       Fraction of energy to retain (default 0.999).

    Returns
    -------
    f_cut : float   Cutoff frequency [Hz].
    """
    n     = len(u)
    n_h   = int(np.ceil(T_h * fs)) + 1
    Nfft  = int(2 ** np.ceil(np.log2(n + n_h - 1)))

    # Single-sided spectrum
    U     = np.fft.rfft(u, n=Nfft)
    freqs = np.fft.rfftfreq(Nfft, d=1.0 / fs)

    # Cumulative energy
    psd      = np.abs(U) ** 2
    psd_cum  = np.cumsum(psd)
    psd_cum /= psd_cum[-1]                    # normalise to [0, 1]

    idx = np.searchsorted(psd_cum, energy_threshold)
    idx = min(idx, len(freqs) - 1)

    return float(freqs[idx])


# ---------------------------------------------------------------------------
# Anti-aliased downsampling
# ---------------------------------------------------------------------------

def _safe_resample(x: np.ndarray, ds_factor: int) -> np.ndarray:
    """
    Downsample x by integer factor ds_factor with anti-alias filtering.

    Symmetric edge-padding (N samples at each end) is applied before
    resampling to suppress the boundary artefacts that scipy's
    resample_poly introduces at signal edges.

    Parameters
    ----------
    x         : np.ndarray, shape (N,)   Signal to downsample.
    ds_factor : int                       Downsampling factor (> 1).

    Returns
    -------
    x_ds : np.ndarray, shape (ceil(N / ds_factor),)
        Downsampled signal.
    """
    x  = x.ravel()
    N  = len(x)
    P  = 1          # resample ratio numerator
    Q  = ds_factor  # resample ratio denominator (= up/down = 1/ds_factor)

    # Pad symmetrically
    pad   = np.concatenate([np.full(N, x[0]), x, np.full(N, x[-1])])
    pad_ds = resample_poly(pad, P, Q)

    # Expected output length for the original signal
    n_out = int(np.floor(N / ds_factor)) + (1 if N % ds_factor else 0)

    # The padded prefix contributes floor(N / ds_factor) samples;
    # extract the central portion corresponding to the original signal
    n_pad_out = int(np.floor(N / ds_factor))
    start     = n_pad_out
    end       = start + n_out
    x_ds      = pad_ds[start:end]

    return x_ds.astype(np.float64)


# ---------------------------------------------------------------------------
# Signal packaging
# ---------------------------------------------------------------------------

def _package_signals(u: np.ndarray,
                      q: np.ndarray,
                      fs: float,
                      ds_factor: int,
                      T_h: float) -> dict:
    """
    Package time-domain signals and derived quantities into a signal dict.

    The valid convolution region is defined as the set of output samples
    for which the full impulse response support [0, T_h] is covered by
    observed input data.  The first n_h - 1 output samples require input
    prehistory (unobserved) and are excluded.

    Parameters
    ----------
    u         : np.ndarray, shape (M,)   Input signal (possibly downsampled).
    q         : np.ndarray, shape (M,)   Output signal.
    fs        : float                     Sampling frequency [Hz].
    ds_factor : int                       Downsampling factor applied.
    T_h       : float                     Impulse response duration [s].

    Returns
    -------
    sig : dict
        Signal struct with fields described in the module docstring.
    """
    u = u.ravel()
    q = q.ravel()

    n   = len(u)
    dt  = 1.0 / fs
    n_h = int(np.ceil(T_h * fs)) + 1          # impulse response length [samples]

    # Valid region: output samples n_h-1 ... n-1  (0-indexed)
    # This corresponds to MATLAB's  valid = n_h : n  (1-indexed)
    valid_start = n_h - 1                       # first valid output index (0-based)

    # Time vectors
    t   = np.arange(n)   * dt                  # signal time vector [s]
    t_h = np.arange(n_h) * dt                  # impulse response time vector [s]

    sig = {
        'u'         : jnp.array(u,   dtype=jnp.float32),
        'q'         : jnp.array(q,   dtype=jnp.float32),
        'fs'        : float(fs),
        'dt'        : float(dt),
        't'         : jnp.array(t,   dtype=jnp.float32),
        't_h'       : jnp.array(t_h, dtype=jnp.float32),
        'n'         : int(n),
        'valid'     : int(valid_start),         # Python int -> static in JAX
        'ds_factor' : int(ds_factor),
    }

    return sig


# ---------------------------------------------------------------------------
# Convenience: truncate q to valid region (used outside JIT)
# ---------------------------------------------------------------------------

def get_valid_output(sig: dict) -> jnp.ndarray:
    """
    Return the valid portion of the output signal q[valid:].

    Parameters
    ----------
    sig : dict   Signal struct from _package_signals.

    Returns
    -------
    q_valid : jnp.ndarray, shape (Nd,)
    """
    return sig['q'][sig['valid']:]


# ---------------------------------------------------------------------------
# Convenience: reconstruct fs and signal length from a signal dict
# ---------------------------------------------------------------------------

def signal_info(signals: dict, level: str = 'coarse') -> str:
    """
    Return a human-readable summary of a prepared signal struct.

    Parameters
    ----------
    signals : dict   Output of prepare_signals.
    level   : str    'coarse' or 'fine'.

    Returns
    -------
    info : str
    """
    sig  = signals[level]
    Nd   = sig['n'] - sig['valid']
    dur  = sig['n'] * sig['dt']
    return (
        f"Level       : {level}\n"
        f"Samples     : {sig['n']}  (valid: {Nd})\n"
        f"fs          : {sig['fs']:.1f} Hz\n"
        f"dt          : {sig['dt']*1e3:.4f} ms\n"
        f"Duration    : {dur*1e3:.1f} ms\n"
        f"DS factor   : {sig['ds_factor']}\n"
        f"n_h (L)     : {int(sig['t_h'].shape[0])}\n"
        f"Valid start : sample {sig['valid']}\n"
    )