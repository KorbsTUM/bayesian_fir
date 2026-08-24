"""
data/WET_Kornilov
==================
20 WET Kornilov flame cases (5 series - U_const, P_const, Lf_const,
Tad_const, df_const - x 4 water-gas-ratio values WGR0/WGR091/WGR166/WGR231),
raw input (velocityRef1.csv) / output (heatRelease.csv) time series, a
reference SysID impulse response for comparison against the Bayesian DTD
fit (fir_sysid.csv), plus per-case L_ref/U_ref metadata (metadata.py).
See loader.py for how to load these as inference.pooled.DatasetSpec
objects and how the SysID reference is kept separate.
"""

from data.WET_Kornilov.loader import (
    load_kornilov_case,
    load_all_kornilov_datasets,
    load_kornilov_fir_sysid,
)
