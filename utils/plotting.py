"""
utils/plotting.py
==================
matplotlib ports of the MATLAB plotting helpers in utils/*.m:

    get_colours          <- getColours.m
    errorpatch            <- errorpatch.m
    corner_heatmap         <- cornerHeatmap.m
    plot_model_comparison <- plotModelComparisonH.m

These reproduce the *data semantics* of the MATLAB originals (same
colour ramp, same uncertainty-band construction, same corner-plot
histograms/ellipses, same normalised model-comparison bars). They do
not attempt to reproduce MATLAB's pixel-level layout logic (manual
text-collision avoidance, Figure/Axes unit juggling), which doesn't
translate meaningfully to matplotlib and isn't needed for the plots to
convey the same information.

Note on plotModelComparisonH.m: its MATLAB defaults reference a
`colourLibrary(...)` helper that is not present anywhere in this
repository (only getColours.m is) - the MATLAB file is not fully
self-contained as committed. plot_model_comparison below uses
get_colours throughout instead, so this module has no such gap.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.stats import norm


# ---------------------------------------------------------------------------
# get_colours  <- getColours.m
# ---------------------------------------------------------------------------

# Paper colour ramp (Yoko & Polifke 2026), 1-indexed to match getColours.m:
#   1. Blue  2. Orange  3. Red  4. Blue2  5. White
_PALETTE = np.array([
    [41,  52, 122],
    [253, 129,  83],
    [221,  48,  37],
    [142, 152, 211],
    [255, 255, 255],
]) / 255.0


def get_colours(col_idx, grad_cnt: int = 1) -> np.ndarray:
    """
    Look up or interpolate colours from the paper's fixed colour ramp.

    Indexing is 1-based (colour 1 = blue, ...), matching getColours.m
    and every call site ported from cornerHeatmap.m / plotModelComparisonH.m.

    Parameters
    ----------
    col_idx  : int or sequence of int
        1-based colour index/indices into the ramp.
    grad_cnt : int
        1            - return the colour(s) at col_idx directly.
        >1, len(col_idx) > 1  - interpolate a gradient of length grad_cnt
                                 through the given colour(s).
        >1, scalar col_idx    - blend from that colour towards white,
                                 grad_cnt steps, covering positions [0, 0.6].

    Returns
    -------
    rgb : np.ndarray, shape (3,) or (grad_cnt, 3)
    """
    idx = np.atleast_1d(np.asarray(col_idx, dtype=int))

    if grad_cnt == 1:
        rgb = _PALETTE[idx - 1]
        return rgb[0] if rgb.shape[0] == 1 else rgb

    if grad_cnt > 1 and idx.shape[0] > 1:
        M = idx.shape[0]
        grad_points = _PALETTE[idx - 1]                    # (M, 3)
        positions = np.linspace(1, grad_cnt, M)
        targets = np.arange(1, grad_cnt + 1)
        rgb = np.zeros((grad_cnt, 3))
        for ch in range(3):
            rgb[:, ch] = np.interp(targets, positions, grad_points[:, ch])
        return rgb

    start = _PALETTE[idx[0] - 1]
    finish = np.array([1.0, 1.0, 1.0])
    position = np.linspace(0.0, 0.6, grad_cnt)[:, None]
    return start + position * (finish - start)


# ---------------------------------------------------------------------------
# errorpatch  <- errorpatch.m
# ---------------------------------------------------------------------------

def errorpatch(ax, x, mean, lo, hi,
                color='tab:blue',
                line_kwargs: dict | None = None,
                patch_kwargs: dict | None = None):
    """
    Plot a mean line with an asymmetric shaded uncertainty band.

    Parameters
    ----------
    ax    : matplotlib.axes.Axes
    x     : array-like, shape (T,)   Horizontal coordinates.
    mean  : array-like, shape (T,)   Mean line values.
    lo    : array-like, shape (T,)   Lower deviation from mean (>= 0).
    hi    : array-like, shape (T,)   Upper deviation from mean (>= 0).
    color : matplotlib colour spec
    line_kwargs  : dict, optional   Overrides for the mean line.
    patch_kwargs : dict, optional   Overrides for the shaded band.

    Returns
    -------
    line : matplotlib.lines.Line2D   The plotted mean line.
    """
    x    = np.ravel(x)
    mean = np.ravel(mean)
    lo   = np.ravel(lo)
    hi   = np.ravel(hi)

    verts_x = np.concatenate([x, x[::-1]])
    verts_y = np.concatenate([mean + hi, (mean - lo)[::-1]])

    patch_opts = dict(facecolor=color, edgecolor='none', alpha=0.2)
    patch_opts.update(patch_kwargs or {})
    ax.fill(verts_x, verts_y, **patch_opts)

    line_opts = dict(color=color, linewidth=2)
    line_opts.update(line_kwargs or {})
    line, = ax.plot(x, mean, **line_opts)
    return line


# ---------------------------------------------------------------------------
# corner_heatmap  <- cornerHeatmap.m
# ---------------------------------------------------------------------------

def _draw_cov_ellipse(ax, mu, C, k, color):
    """Draw a k-sigma covariance ellipse for a 2D Gaussian N(mu, C)."""
    C = 0.5 * (C + C.T)
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(C))):
        return
    eigvals, V = np.linalg.eigh(C)          # ascending order, as MATLAB's eig
    eigvals = np.clip(eigvals, 0.0, None)
    t = np.linspace(0.0, 2.0 * np.pi, 200)
    a, b = k * np.sqrt(eigvals[0]), k * np.sqrt(eigvals[1])
    ellipse = V @ np.stack([a * np.cos(t), b * np.sin(t)])   # (2, 200)
    ax.plot(mu[0] + ellipse[0], mu[1] + ellipse[1], color=color, linewidth=1)


def corner_heatmap(fig, samples: np.ndarray,
                    b_map: np.ndarray,
                    Cb_map: np.ndarray,
                    param_names: list | None = None,
                    n_bins: int = 40):
    """
    Plot pairwise posterior marginals as a corner heatmap.

    Diagonal panels show the marginal sample histogram against the
    Laplace-approximate Gaussian N(b_map[i], Cb_map[i,i]). Off-diagonal
    panels show the joint 2D histogram (alpha-scaled by density) with
    1/2/3-sigma Laplace covariance ellipses overlaid.

    Parameters
    ----------
    fig         : matplotlib.figure.Figure   Figure to draw the grid into.
    samples     : np.ndarray, shape (N, D)   Posterior samples (e.g. MCMC).
    b_map       : np.ndarray, shape (D,)     MAP estimate.
    Cb_map      : np.ndarray, shape (D, D)   MAP covariance (Laplace approx).
    param_names : list of str, optional      Axis labels, length D.
    n_bins      : int                         Histogram bin count.

    Returns
    -------
    axes : np.ndarray, shape (D, D)   Grid of Axes.
    """
    samples = np.asarray(samples)
    b_map   = np.asarray(b_map)
    Cb_map  = np.asarray(Cb_map)
    N, D    = samples.shape

    if param_names is None:
        param_names = [f"$p_{{{i+1}}}$" for i in range(D)]

    cmap = ListedColormap(get_colours([3, 5], 50)[::-1])

    axes = fig.subplots(D, D, squeeze=False,
                         gridspec_kw={'wspace': 0.0, 'hspace': 0.0})

    for i in range(D):
        for j in range(D):
            ax = axes[i, j]
            if i == j:
                mu, sig = b_map[j], np.sqrt(Cb_map[j, j])
                edges = np.linspace(mu - 4 * sig, mu + 4 * sig, n_bins)

                ax.hist(samples[:, j], bins=edges, density=True,
                        color=get_colours(3), alpha=0.5, edgecolor='none')
                ax.hist(samples[:, j], bins=edges, density=True,
                        histtype='step', color=get_colours(3), linewidth=1.0)

                b = np.linspace(mu - 4 * sig, mu + 4 * sig, 200)
                ax.plot(b, norm.pdf(b, mu, sig),
                        color=get_colours(1), linewidth=1.5)

                ax.set_xlim(mu - 4 * sig, mu + 4 * sig)
                ax.set_yticks([])
            else:
                mux, muy   = b_map[j], b_map[i]
                sigx, sigy = np.sqrt(Cb_map[j, j]), np.sqrt(Cb_map[i, i])
                xedges = np.linspace(mux - 4 * sigx, mux + 4 * sigx, n_bins + 1)
                yedges = np.linspace(muy - 4 * sigy, muy + 4 * sigy, n_bins + 1)

                counts, _, _ = np.histogram2d(
                    samples[:, j], samples[:, i], bins=[xedges, yedges],
                    density=True)
                counts = counts.T
                counts[~np.isfinite(counts)] = 0.0

                cmax = np.percentile(counts, 99)
                if not np.isfinite(cmax) or cmax <= 0:
                    cmax = counts.max()
                cmax = max(cmax, np.finfo(float).eps)

                t = np.clip(counts / cmax, 0.0, 1.0)
                alpha_power, alpha_min, alpha_max = 0.6, 0.0, 0.95
                alpha = alpha_min + (alpha_max - alpha_min) * t ** alpha_power
                alpha[t < 0.02] = 0.0

                rgba = cmap(t)
                rgba[..., -1] = alpha
                ax.imshow(rgba, origin='lower', aspect='auto',
                          extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

                mu_vec = np.array([mux, muy])
                C = Cb_map[np.ix_([j, i], [j, i])]
                for k in (1, 2, 3):
                    _draw_cov_ellipse(ax, mu_vec, C, k, get_colours(1))

                ax.set_xlim(xedges[0], xedges[-1])
                ax.set_ylim(yedges[0], yedges[-1])
                ax.set_facecolor(cmap(0.0))

            if i < D - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(param_names[j])
                ax.tick_params(axis='x', rotation=45)

            if j > 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(param_names[i])

    return axes


# ---------------------------------------------------------------------------
# plot_model_comparison  <- plotModelComparisonH.m
# ---------------------------------------------------------------------------

def plot_model_comparison(ax, log_ml, log_bfl, log_of, model_labels=None):
    """
    Plot model-ranking metrics (ML, OF, BFL) as a grouped bar comparison.

    Scores are normalised to the best (highest logML) model, matching
    plotModelComparisonH.m. For each model, the log(BFL) and log(ML)
    bars are drawn side by side, and a third bar spans the gap between
    the BFL and ML tips to represent log(OF) = log(ML) - log(BFL).

    Parameters
    ----------
    ax           : matplotlib.axes.Axes
    log_ml       : array-like, shape (K,)   Log marginal likelihood per model.
    log_bfl      : array-like, shape (K,)   Log best-fit likelihood per model.
    log_of       : array-like, shape (K,)   Log Occam factor per model.
    model_labels : list of str, optional    Length-K labels. Default "Model i".

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    log_ml  = np.asarray(log_ml,  dtype=float).ravel()
    log_bfl = np.asarray(log_bfl, dtype=float).ravel()
    log_of  = np.asarray(log_of,  dtype=float).ravel()
    n_mod   = log_ml.shape[0]

    if model_labels is None:
        model_labels = [f"Model {i + 1}" for i in range(n_mod)]

    ind_ml  = int(np.argmax(log_ml))
    log_bfl = log_bfl - log_ml[ind_ml]
    log_ml  = log_ml - log_ml[ind_ml]

    x = np.arange(1, n_mod + 1)
    w, buffer = 0.28, 0.08
    col_ml, col_of, col_bfl = get_colours(1), get_colours(2), get_colours(3)

    ax.bar(x + w / 2 + buffer / 2, log_ml, w, color=col_ml, label='log(ML)')
    ax.bar(x - w / 2 - buffer / 2, log_bfl, w, color=col_bfl, label='log(BFL)')
    ax.bar(x[ind_ml] + w / 2 + buffer / 2, log_ml[ind_ml], w,
           color=col_ml, edgecolor='k', linewidth=2)

    for xi, bfl_i, ml_i in zip(x, log_bfl, log_ml):
        y0, y1 = sorted((bfl_i, ml_i))
        ax.bar(xi, y1 - y0, w, bottom=y0, color=col_of,
               edgecolor='k', linewidth=0.5,
               label='log(OF)' if xi == x[0] else None)

    ax.axhline(0.0, color='k', linewidth=1)
    ax.axhline(log_ml.max(), color=get_colours(4), linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels)
    ax.set_ylabel('log(ML) / log(BFL) / log(OF)')
    ax.legend(frameon=False)
    return ax
