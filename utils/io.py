"""
utils/io.py
============
Loaders for the experimental datasets shipped in data/BRS_EderSilva23/,
ported from the loading logic at the top of generateFigures.m:

    load(sprintf('%sdata_raw_incomp.mat',ptd));
    exp_FTF = load(sprintf('%sFTF_exp_30kW_Front_lambda1.3.mat',ptd));

    u_full = data_raw_incomp.u;
    q_full = data_raw_incomp.y;
    dt = data_raw_incomp.Ts;
    t_full = (0:length(u_full)-1)*dt;

    u_full = (u_full - mean(u_full))/mean(u_full);
    q_full = (q_full - mean(q_full))/mean(q_full);

data_raw_incomp.mat does not store plain arrays: its single top-level
variable is a serialized MATLAB System Identification Toolbox `iddata`
object. scipy.io.loadmat exposes that object's raw stored fields
(InputData, OutputData, Ts, ...) rather than its `.u` / `.y` convenience
properties, so load_raw_incomp below unpacks those fields directly and
reproduces the exact preprocessing above - including that MATLAB's own
script builds t starting at 0 and does not apply the iddata object's
Tstart field, even though that field is present in the file.
"""

from pathlib import Path

import numpy as np
import scipy.io as sio


def load_raw_incomp(path, normalise: bool = True) -> dict:
    """
    Load the BRS_EderSilva23 raw input/output dataset.

    Parameters
    ----------
    path      : str or Path   Path to data_raw_incomp.mat.
    normalise : bool          Apply the (x - mean(x)) / mean(x) relative-
                              fluctuation normalisation generateFigures.m
                              applies before inference. Default True.

    Returns
    -------
    data : dict with keys
        'u'  : np.ndarray, (M,)   Input signal (uref) [m/s].
        'q'  : np.ndarray, (M,)   Output signal (dQ) [W].
        't'  : np.ndarray, (M,)   Time vector [s], starting at 0.
        'dt' : float               Sampling interval [s].
        'fs' : float               Sampling frequency [Hz].
    """
    mat = sio.loadmat(str(path))
    var_name = next(k for k in mat.keys() if not k.startswith('__'))
    obj = mat[var_name][0, 0]

    u  = obj['InputData'][0, 0].ravel().astype(np.float64)
    q  = obj['OutputData'][0, 0].ravel().astype(np.float64)
    dt = float(np.asarray(obj['Ts'][0, 0]).item())

    if normalise:
        u = (u - u.mean()) / u.mean()
        q = (q - q.mean()) / q.mean()

    t = np.arange(u.shape[0]) * dt

    return {'u': u, 'q': q, 't': t, 'dt': dt, 'fs': 1.0 / dt}


def load_ftf_experiment(path) -> dict:
    """
    Load the experimental flame transfer function reference data.

    Parameters
    ----------
    path : str or Path   Path to FTF_exp_30kW_Front_lambda1.3.mat.

    Returns
    -------
    ftf : dict with keys
        'freq'  : np.ndarray, (K,)   Frequency [Hz].
        'gain'  : np.ndarray, (K,)   Linear gain |F(f)|.
        'phase' : np.ndarray, (K,)   Phase [rad].
    """
    mat = sio.loadmat(str(path))
    return {
        'freq' : mat['freq_exp'].ravel().astype(np.float64),
        'gain' : mat['gain_exp'].ravel().astype(np.float64),
        'phase': mat['phase_exp'].ravel().astype(np.float64),
    }
