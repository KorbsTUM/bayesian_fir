"""
signals/prepare.py
=================
Prepares input/output time-series signals for Bayesian impulse response
inference.

Responsibilities:
    1. Compute an integer downsampling factor from config.preproc.DSmode /
       DSvalue (matches the active switch in prepareSignals.m: 'factor'
       uses DSvalue directly, 'frequency' divides fs by DSvalue, 'rate'
       multiplies fs by DSvalue). The default ('factor', 1) means no
       downsampling, matching loadDefaultConfig.m.

       Note: prepareSignals.m also contains a commented-out "legacy v1.0"
       block that estimates a bandwidth from the input PSD and derives
       ds_factor from it. That block is inactive in the current MATLAB
       source (kept only as a future-reference comment) and is
       intentionally not ported here, to match current MATLAB behaviour.
    2. Apply anti-aliased downsampling via scipy.signals.resample_poly with
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
        'u'     : jnp.ndarray (M,)     input signals
        'q'     : jnp.ndarray (M,)     output signals
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
    - All heavy signals processing (FFT, resampling) is done in NumPy/SciPy
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


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def prepare_signals(u: np.ndarray,
                    q: np.ndarray,
                    fs: float,
                    T_h: float,
                    ds_mode: str = 'factor',
                    ds_value: float = 1) -> dict:
    """
    Prepare coarse and fine signals representations for inference.

    Parameters
    ----------
    u        : np.ndarray, shape (M,)
        Input fluctuation signals (zero-mean, normalised).
    q        : np.ndarray, shape (M,)
        Output fluctuation signals (zero-mean, normalised).
    fs       : float
        Original sampling frequency [Hz].
    T_h      : float
        Desired impulse response duration [s].
    ds_mode  : str
        How ds_value is interpreted (config.preproc.DSmode):
            'factor'    - ds_value is used directly as the integer
                          downsampling factor.
            'frequency' - ds_value is a target sample rate [Hz];
                          ds_factor = floor(fs / ds_value).
            'rate'      - ds_value is a downsampling rate in [0, 1];
                          ds_factor = floor(fs * ds_value).
    ds_value : float
        Value interpreted according to ds_mode. Default 1 with
        ds_mode='factor' means no downsampling, matching
        loadDefaultConfig.m.

    Returns
    -------
    signals : dict
        Nested dict with keys 'coarse' and 'fine', each containing
        the packaged signals struct described in the module docstring.
    """
    u = np.asarray(u, dtype=np.float64).ravel()
    q = np.asarray(q, dtype=np.float64).ravel()

    if len(u) != len(q):
        raise ValueError(f"u and q must have the same length, "
                         f"got {len(u)} and {len(q)}.")

    # ------------------------------------------------------------------
    # Step 1: compute the integer downsampling factor from DSmode/DSvalue
    # ------------------------------------------------------------------
    if ds_mode == 'factor':
        ds_factor = int(ds_value)
    elif ds_mode == 'frequency':
        ds_factor = int(fs / ds_value)
    elif ds_mode == 'rate':
        ds_factor = int(fs * ds_value)
    else:
        raise ValueError(
            f"ds_mode must be 'factor', 'frequency', or 'rate', got {ds_mode!r}.")

    # ------------------------------------------------------------------
    # Step 2: package fine (original resolution) signals
    # ------------------------------------------------------------------
    fine = _package_signals(u, q, fs, ds_factor=1, T_h=T_h)

    # ------------------------------------------------------------------
    # Step 3: downsample and package coarse signals
    # ------------------------------------------------------------------
    if ds_factor <= 1:
        coarse = fine
    else:
        u_ds = _safe_resample(u, ds_factor)
        q_ds = _safe_resample(q, ds_factor)
        fs_ds = fs / ds_factor
        coarse = _package_signals(u_ds, q_ds, fs_ds,
                                   ds_factor=ds_factor, T_h=T_h)

    return {'coarse': coarse, 'fine': fine}


# ---------------------------------------------------------------------------
# Differentiable variant (no downsampling)
# ---------------------------------------------------------------------------

def prepare_signals_diff(u: jnp.ndarray,
                          q: jnp.ndarray,
                          fs: float,
                          T_h: float) -> dict:
    """
    JAX-differentiable signal preparation, restricted to ds_factor <= 1
    (no downsampling).

    prepare_signals routes u, q through np.asarray and, when downsampling
    is requested, scipy.signal.resample_poly - neither of which accepts
    JAX tracers, so gradients w.r.t. u, q cannot flow through it. This
    variant keeps u, q as jnp arrays throughout and skips the
    (SciPy-only) resampling path entirely, so it is only valid when the
    caller does not downsample - i.e. it reproduces exactly the
    ds_factor <= 1 branch of prepare_signals ('factor' mode with
    ds_value <= 1, matching loadDefaultConfig.m's default). It exists
    for use inside gradient-tracked code, e.g. inference.sensitivity's
    implicit-differentiation MAP estimator, where the pipeline needs to
    stay differentiable from raw u, q through to the fitted DTD
    parameters.

    Parameters
    ----------
    u   : jnp.ndarray, shape (M,)   Input fluctuation signal.
    q   : jnp.ndarray, shape (M,)   Output fluctuation signal.
    fs  : float                      Sampling frequency [Hz].
    T_h : float                      Desired impulse response duration [s].

    Returns
    -------
    signals : dict
        Nested dict with keys 'coarse' and 'fine' (identical, since
        ds_factor=1), matching the layout produced by prepare_signals.
    """
    u = jnp.asarray(u, dtype=jnp.float64).ravel()
    q = jnp.asarray(q, dtype=jnp.float64).ravel()

    if u.shape[0] != q.shape[0]:
        raise ValueError(f"u and q must have the same length, "
                         f"got {u.shape[0]} and {q.shape[0]}.")

    n   = u.shape[0]
    dt  = 1.0 / fs
    n_h = int(np.ceil(T_h * fs)) + 1
    valid_start = n_h - 1

    sig = {
        'u'         : u,
        'q'         : q,
        'fs'        : float(fs),
        'dt'        : float(dt),
        't'         : jnp.arange(n)   * dt,
        't_h'       : jnp.arange(n_h) * dt,
        'n'         : int(n),
        'valid'     : int(valid_start),
        'ds_factor' : 1,
    }
    return {'coarse': sig, 'fine': sig}


# ---------------------------------------------------------------------------
# Anti-aliased downsampling
# ---------------------------------------------------------------------------

def _safe_resample(x: np.ndarray, ds_factor: int) -> np.ndarray:
    """
    Downsample x by integer factor ds_factor with anti-alias filtering.

    Symmetric edge-padding (N samples at each end) is applied before
    resampling to suppress the boundary artefacts that scipy's
    resample_poly introduces at signals edges.

    Parameters
    ----------
    x         : np.ndarray, shape (N,)   Signal to downsample.
    ds_factor : int                       Downsampling factor (> 1).

    Returns
    -------
    x_ds : np.ndarray, shape (ceil(N / ds_factor),)
        Downsampled signals.
    """
    x  = x.ravel()
    N  = len(x)
    P  = 1          # resample ratio numerator
    Q  = ds_factor  # resample ratio denominator (= up/down = 1/ds_factor)

    # Pad symmetrically
    pad   = np.concatenate([np.full(N, x[0]), x, np.full(N, x[-1])])
    pad_ds = resample_poly(pad, P, Q)

    # Expected output length for the original signals
    n_out = int(np.floor(N / ds_factor)) + (1 if N % ds_factor else 0)

    # The padded prefix contributes floor(N / ds_factor) samples;
    # extract the central portion corresponding to the original signals
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
    Package time-domain signals and derived quantities into a signals dict.

    The valid convolution region is defined as the set of output samples
    for which the full impulse response support [0, T_h] is covered by
    observed input data.  The first n_h - 1 output samples require input
    prehistory (unobserved) and are excluded.

    Parameters
    ----------
    u         : np.ndarray, shape (M,)   Input signals (possibly downsampled).
    q         : np.ndarray, shape (M,)   Output signals.
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
    t   = np.arange(n)   * dt                  # signals time vector [s]
    t_h = np.arange(n_h) * dt                  # impulse response time vector [s]

    sig = {
        'u'         : jnp.array(u,   dtype=jnp.float64),
        'q'         : jnp.array(q,   dtype=jnp.float64),
        'fs'        : float(fs),
        'dt'        : float(dt),
        't'         : jnp.array(t,   dtype=jnp.float64),
        't_h'       : jnp.array(t_h, dtype=jnp.float64),
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
    Return the valid portion of the output signals q[valid:].

    Parameters
    ----------
    sig : dict   Signal struct from _package_signals.

    Returns
    -------
    q_valid : jnp.ndarray, shape (Nd,)
    """
    return sig['q'][sig['valid']:]


# ---------------------------------------------------------------------------
# Convenience: reconstruct fs and signals length from a signals dict
# ---------------------------------------------------------------------------

def signal_info(signals: dict, level: str = 'coarse') -> str:
    """
    Return a human-readable summary of a prepared signals struct.

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