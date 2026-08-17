"""
data/WET_Kornilov/loader.py
============================
Loaders for the WET Kornilov dataset (see this package's __init__.py and
metadata.py for the dataset structure and L_ref/U_ref provenance).

    load_kornilov_case(...)         one case's raw signals, as a plain
                                     dict (mirrors utils.io.load_raw_incomp's
                                     shape/normalisation convention).
    load_all_kornilov_datasets(...) all 20 cases, ready to hand to
                                     inference.pooled.infer_shared_model_order
                                     as a list of DatasetSpec.
"""

from pathlib import Path
from typing import List, Optional

import numpy as np

from data.WET_Kornilov.metadata import KORNILOV_CASES, SUBSETS, WGR_LABELS
from inference.pooled import DatasetSpec

DEFAULT_DATA_DIR = Path(__file__).resolve().parent


def load_kornilov_case(subset     : str,
                        wgr        : str,
                        data_dir   : Optional[Path] = None,
                        normalise  : bool = True) -> dict:
    """
    Load one WET Kornilov case's raw input/output signals.

    Parameters
    ----------
    subset    : str    One of data.WET_Kornilov.metadata.SUBSETS
                        (e.g. 'U_const').
    wgr       : str    One of data.WET_Kornilov.metadata.WGR_LABELS
                        (e.g. 'WGR0').
    data_dir  : Path, optional   Root containing '{subset}/{wgr}/'.
                        Defaults to this package's own directory.
    normalise : bool    Apply the same (x - mean(x)) / mean(x)
                        relative-fluctuation normalisation used by
                        utils.io.load_raw_incomp. Default True.

    Returns
    -------
    data : dict with keys 'u', 'q', 't', 'dt', 'fs' (see utils.io.load_raw_incomp
        for the exact meaning of each).
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    case_dir = Path(data_dir) / subset / wgr

    u_raw = np.loadtxt(case_dir / "velocityRef1.csv", delimiter=",")
    q_raw = np.loadtxt(case_dir / "heatRelease.csv", delimiter=",")

    t = u_raw[:, 0]
    u = u_raw[:, 1].astype(np.float64)
    q = q_raw[:, 1].astype(np.float64)

    if u.shape[0] != q.shape[0]:
        raise ValueError(
            f"{subset}/{wgr}: velocityRef1.csv has {u.shape[0]} samples, "
            f"heatRelease.csv has {q.shape[0]}.")

    if normalise:
        u = (u - u.mean()) / u.mean()
        q = (q - q.mean()) / q.mean()

    dt = float(np.mean(np.diff(t)))
    return {'u': u, 'q': q, 't': t, 'dt': dt, 'fs': 1.0 / dt}


def load_all_kornilov_datasets(data_dir   : Optional[Path] = None,
                                normalise  : bool = True,
                                subsets    : Optional[List[str]] = None,
                                wgr_labels : Optional[List[str]] = None
                                ) -> List[DatasetSpec]:
    """
    Load WET Kornilov cases as inference.pooled.DatasetSpec objects, with
    T_c = L_ref / U_ref from metadata.KORNILOV_CASES.

    Parameters
    ----------
    data_dir   : Path, optional        See load_kornilov_case.
    normalise  : bool                  See load_kornilov_case.
    subsets    : list of str, optional Restrict to these series (subset of
                                       metadata.SUBSETS). Default: all 5.
    wgr_labels : list of str, optional Restrict to these WGR cases (subset
                                       of metadata.WGR_LABELS). Default: all 4.

    Returns
    -------
    datasets : list of DatasetSpec, one per (subset, WGR) case (20 by default).
    """
    subsets    = SUBSETS     if subsets    is None else subsets
    wgr_labels = WGR_LABELS  if wgr_labels is None else wgr_labels

    datasets = []
    for subset in subsets:
        for wgr in wgr_labels:
            sig = load_kornilov_case(subset, wgr, data_dir=data_dir, normalise=normalise)
            meta = KORNILOV_CASES[subset][wgr]
            T_c = meta['L_ref'] / meta['U_ref']
            datasets.append(DatasetSpec(
                name = f"{subset}_{wgr}",
                u    = sig['u'],
                q    = sig['q'],
                t    = sig['t'],
                T_c  = T_c,
            ))
    return datasets
