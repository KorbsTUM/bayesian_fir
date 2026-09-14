"""
data/Kornilov/loader.py
=========================
Loaders for the (new, plain) Kornilov dataset: 20 independent cases
(C01-C20), each with its own input (velocityRef1.csv) / output
(heatRelease.csv) time series - same raw-CSV format and (x - mean(x)) /
mean(x) normalisation convention as data/WET_Kornilov/loader.py, but no
series/water-gas-ratio structure (see metadata.py).

    load_kornilov_case(...)         one case's raw signals, as a plain
                                     dict (mirrors data.WET_Kornilov.loader
                                     .load_kornilov_case).
    load_all_kornilov_datasets(...) all cases, ready to hand to
                                     inference.pooled.infer_shared_model_order
                                     as a list of DatasetSpec.
    load_kornilov_fir_sysid(...)    per-case reference SysID impulse
                                     response (fir_sysid.csv - curated from
                                     .../Kornilov_timeseries/SysID/System
                                     Identification/FIRs/FIR_{case}.txt,
                                     dt=1e-4 s, 100-200 taps depending on
                                     case), for comparison against the
                                     Bayesian DTD fit. Returns None if a
                                     case's fir_sysid.csv doesn't exist
                                     (rather than raising), so any future
                                     case can be added without one and
                                     still run through this loader
                                     unchanged.
"""

from pathlib import Path
from typing import List, Optional

import numpy as np

from data.Kornilov.metadata import KORNILOV_CASES, CASES
from inference.pooled import DatasetSpec

DEFAULT_DATA_DIR = Path(__file__).resolve().parent


def load_kornilov_case(case      : str,
                        data_dir  : Optional[Path] = None,
                        normalise : bool = True) -> dict:
    """
    Load one Kornilov case's raw input/output signals.

    Parameters
    ----------
    case      : str    One of data.Kornilov.metadata.CASES (e.g. 'C01').
    data_dir  : Path, optional   Root containing '{case}/'. Defaults to
                        this package's own directory.
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
    case_dir = Path(data_dir) / case

    u_raw = np.loadtxt(case_dir / "velocityRef1.csv", delimiter=",")
    q_raw = np.loadtxt(case_dir / "heatRelease.csv", delimiter=",")

    t = u_raw[:, 0]
    u = u_raw[:, 1].astype(np.float64)
    q = q_raw[:, 1].astype(np.float64)

    if u.shape[0] != q.shape[0]:
        raise ValueError(
            f"{case}: velocityRef1.csv has {u.shape[0]} samples, "
            f"heatRelease.csv has {q.shape[0]}.")

    if normalise:
        u = (u - u.mean()) / u.mean()
        q = (q - q.mean()) / q.mean()

    dt = float(np.mean(np.diff(t)))
    return {'u': u, 'q': q, 't': t, 'dt': dt, 'fs': 1.0 / dt}


def load_all_kornilov_datasets(data_dir  : Optional[Path] = None,
                                normalise : bool = True,
                                cases     : Optional[List[str]] = None
                                ) -> List[DatasetSpec]:
    """
    Load Kornilov cases as inference.pooled.DatasetSpec objects, with
    T_c = L_ref / U_ref from metadata.KORNILOV_CASES.

    Parameters
    ----------
    data_dir   : Path, optional   See load_kornilov_case.
    normalise  : bool             See load_kornilov_case.
    cases      : list of str, optional   Restrict to these cases (subset
                                of metadata.CASES). Default: all 20.

    Returns
    -------
    datasets : list of DatasetSpec, one per case (20 by default).
    """
    cases = CASES if cases is None else cases

    datasets = []
    for case in cases:
        sig = load_kornilov_case(case, data_dir=data_dir, normalise=normalise)
        meta = KORNILOV_CASES[case]
        T_c = meta['L_ref'] / meta['U_ref']
        datasets.append(DatasetSpec(
            name = case,
            u    = sig['u'],
            q    = sig['q'],
            t    = sig['t'],
            T_c  = T_c,
        ))
    return datasets


def load_kornilov_fir_sysid(case     : str,
                             data_dir : Optional[Path] = None
                             ) -> Optional[dict]:
    """
    Load one case's reference SysID impulse response (fir_sysid.csv), for
    comparison against the Bayesian DTD fit, if one has been dropped in.

    Parameters
    ----------
    case     : str    One of data.Kornilov.metadata.CASES.
    data_dir : Path, optional   Root containing '{case}/'. Defaults to
                       this package's own directory.

    Returns
    -------
    fir : dict with keys 'time' (lag time [s]) and 'val' (impulse
        response), or None if '{case}/fir_sysid.csv' doesn't exist.
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    fir_path = Path(data_dir) / case / "fir_sysid.csv"
    if not fir_path.exists():
        return None

    fir = np.loadtxt(fir_path, delimiter=",")
    return {'time': fir[:, 0], 'val': fir[:, 1]}