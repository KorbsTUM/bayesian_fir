"""
data/Kornilov
==============
20 (new, plain) Kornilov flame cases (C01-C20), raw input
(velocityRef1.csv) / output (heatRelease.csv) time series, plus per-case
L_ref/U_ref metadata (metadata.py). Unlike data/WET_Kornilov/, these cases
have no series/water-gas-ratio structure. Each case also ships a SysID
reference impulse response (fir_sysid.csv, dt=1e-4 s - curated from
.../Kornilov_timeseries/SysID/System Identification/FIRs/FIR_{case}.txt,
one row of comma-separated discrete FIR taps per case, no time column in
the source) for comparison against the Bayesian DTD fit; see loader.py's
load_kornilov_fir_sysid, which returns None for any case missing that
file rather than raising.
See loader.py for how to load these as inference.pooled.DatasetSpec
objects.
"""

from data.Kornilov.loader import (
    load_kornilov_case,
    load_all_kornilov_datasets,
    load_kornilov_fir_sysid,
)