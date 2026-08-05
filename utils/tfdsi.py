"""
utils/tfdsi.py
===============
Approximate substitute for TFDSI.m, an independent-method impulse
response / transfer function estimate used in comparison plots against
the paper's Bayesian model.

TFDSI.m calls MATLAB's System Identification Toolbox (impulseest with a
'CS' cubic-spline regularization kernel, plus bode). That toolbox's
kernel-based empirical-Bayes hyperparameter optimisation is proprietary
and has no open equivalent to port exactly.

What's implemented here instead is a standard ridge-regularized FIR
estimate with an exponentially-decaying prior on the impulse response
(regularization strength and decay rate chosen by generalized
cross-validation), which is the same underlying idea - a smooth,
decaying impulse response, hyperparameters selected from the data - but
will NOT numerically match MATLAB's impulseest output. Treat this as an
approximate reference baseline, not a validated port of TFDSI.m.

Frequency-response uncertainty is propagated from the impulse-response
posterior covariance via the delta method, consistent with how
uncertainty is propagated elsewhere in this codebase (e.g.
core.parameter_maps.map_to_physical_covariance).
"""

import numpy as np


def _design_matrix(u: np.ndarray, n_taps: int) -> np.ndarray:
    """Build the (T, n_taps) FIR regressor Phi[t, k] = u[t - k] (0 if t < k)."""
    u = np.asarray(u, dtype=np.float64).ravel()
    T = u.shape[0]
    Phi = np.zeros((T, n_taps))
    for k in range(n_taps):
        Phi[k:, k] = u[:T - k]
    return Phi


def estimate_impulse_siid(u: np.ndarray,
                           q: np.ndarray,
                           dt: float,
                           n_taps: int,
                           rho_grid: np.ndarray | None = None,
                           n_alpha: int = 25) -> dict:
    """
    Estimate a regularized FIR impulse response via ridge regression
    with an exponentially-decaying prior, matching hyperparameters
    (decay rate rho, regularization strength alpha) to the data by
    generalized cross-validation (GCV).

    Parameters
    ----------
    u        : np.ndarray, shape (T,)   Input signal.
    q        : np.ndarray, shape (T,)   Output signal.
    dt       : float                     Sampling interval [s].
    n_taps   : int                       Number of FIR taps (impulse response length).
    rho_grid : np.ndarray, optional      Candidate decay rates in (0, 1).
    n_alpha  : int                        Number of log-spaced regularization
                                          strengths to search.

    Returns
    -------
    result : dict with keys
        'time'  : np.ndarray, (n_taps,)  Lag time vector [s].
        'val'   : np.ndarray, (n_taps,)  Estimated impulse response.
        'std'   : np.ndarray, (n_taps,)  Pointwise posterior std.
        'cov'   : np.ndarray, (n_taps, n_taps)  Posterior covariance.
        'rho'   : float   Selected decay rate.
        'alpha' : float   Selected regularization strength.
    """
    u = np.asarray(u, dtype=np.float64).ravel()
    q = np.asarray(q, dtype=np.float64).ravel()
    T = u.shape[0]

    Phi     = _design_matrix(u, n_taps)      # (T, n_taps)
    PhiTPhi = Phi.T @ Phi                    # (n_taps, n_taps) - hoisted out of the search loop
    PhiTq   = Phi.T @ q                      # (n_taps,)

    lags = np.arange(n_taps)
    if rho_grid is None:
        rho_grid = np.array([0.90, 0.93, 0.95, 0.97, 0.99, 0.995])
    alpha_grid = np.logspace(-6.0, 2.0, n_alpha)

    best = None
    for rho in rho_grid:
        prior_var = np.maximum(rho ** (2.0 * lags), 1e-300)
        for alpha in alpha_grid:
            Rinv  = np.diag(1.0 / (alpha * prior_var))
            A     = PhiTPhi + Rinv
            h_hat = np.linalg.solve(A, PhiTq)
            resid = q - Phi @ h_hat

            dof_eff = max(T - np.trace(np.linalg.solve(A, PhiTPhi)), 1.0)
            gcv     = T * (resid @ resid) / dof_eff ** 2

            if best is None or gcv < best[0]:
                best = (gcv, rho, alpha, h_hat, A, resid, dof_eff)

    _, rho, alpha, h_hat, A, resid, dof_eff = best
    sigma2 = (resid @ resid) / max(T - dof_eff, 1.0)
    Cov_h  = sigma2 * np.linalg.inv(A)
    std_h  = np.sqrt(np.clip(np.diag(Cov_h), 0.0, None))

    return {
        'time' : lags * dt,
        'val'  : h_hat,
        'std'  : std_h,
        'cov'  : Cov_h,
        'rho'  : float(rho),
        'alpha': float(alpha),
    }


def transfer_function_siid(h_result: dict, omega: np.ndarray) -> dict:
    """
    Evaluate the transfer function (gain/phase) of an estimated impulse
    response at the given angular frequencies, with uncertainty
    propagated from the impulse-response posterior covariance via the
    delta method.

    Parameters
    ----------
    h_result : dict   Output of estimate_impulse_siid.
    omega    : np.ndarray, shape (W,)   Angular frequencies [rad/s].

    Returns
    -------
    result : dict with keys
        'w'         : np.ndarray, (W,)   Angular frequency vector.
        'gain'      : np.ndarray, (W,)   Magnitude |H(omega)|.
        'phase'     : np.ndarray, (W,)   Unwrapped phase [rad], normalised
                                         so its maximum value is 0 (matches
                                         TFDSI.m's `phase - max(phase)`).
        'std_gain'  : np.ndarray, (W,)   Delta-method gain std.
        'std_phase' : np.ndarray, (W,)   Delta-method phase std.
    """
    h      = h_result['val']
    Cov_h  = h_result['cov']
    time   = h_result['time']
    dt     = time[1] - time[0] if time.shape[0] > 1 else 1.0
    k      = np.arange(h.shape[0])
    omega  = np.asarray(omega, dtype=np.float64).ravel()

    phase_arg = -omega[:, None] * k[None, :] * dt      # (W, n_taps)
    cos_wt = np.cos(phase_arg)
    sin_wt = np.sin(phase_arg)

    Re_H = cos_wt @ h
    Im_H = sin_wt @ h

    var_re   = np.einsum('wk,kl,wl->w', cos_wt, Cov_h, cos_wt)
    var_im   = np.einsum('wk,kl,wl->w', sin_wt, Cov_h, sin_wt)
    cov_reim = np.einsum('wk,kl,wl->w', cos_wt, Cov_h, sin_wt)

    H     = Re_H + 1j * Im_H
    gain  = np.abs(H)
    phase = np.unwrap(np.angle(H))
    phase = phase - phase.max()

    gain2     = np.maximum(gain ** 2, 1e-300)
    var_gain  = (Re_H ** 2 * var_re + Im_H ** 2 * var_im
                 + 2.0 * Re_H * Im_H * cov_reim) / gain2
    var_phase = (Re_H ** 2 * var_im + Im_H ** 2 * var_re
                 - 2.0 * Re_H * Im_H * cov_reim) / gain2 ** 2

    return {
        'w'         : omega,
        'gain'      : gain,
        'phase'     : phase,
        'std_gain'  : np.sqrt(np.clip(var_gain, 0.0, None)),
        'std_phase' : np.sqrt(np.clip(var_phase, 0.0, None)),
    }


def tfdsi(u: np.ndarray, q: np.ndarray, t: np.ndarray,
          n_taps: int, omega: np.ndarray) -> tuple:
    """
    Convenience wrapper mirroring TFDSI.m's [hSI, FTFSI] = TFDSI(u,q,t,N,w).

    Parameters
    ----------
    u, q   : np.ndarray, shape (T,)   Input/output signals.
    t      : np.ndarray, shape (T,)   Time vector [s] (used only for dt).
    n_taps : int                       Number of FIR taps (MATLAB's N).
    omega  : np.ndarray, shape (W,)   Angular frequencies [rad/s] (MATLAB's w).

    Returns
    -------
    h_si   : dict   See estimate_impulse_siid.
    ftf_si : dict   See transfer_function_siid.
    """
    t  = np.asarray(t, dtype=np.float64).ravel()
    dt = float(np.mean(np.diff(t)))
    h_si   = estimate_impulse_siid(u, q, dt, n_taps)
    ftf_si = transfer_function_siid(h_si, omega)
    return h_si, ftf_si
