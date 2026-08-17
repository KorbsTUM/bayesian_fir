"""
data/WET_Kornilov/metadata.py
==============================
Per-case L_ref / U_ref for the 20 WET Kornilov cases.

Provenance: extracted from .../NieblYoko26/WET_Kornilov_Data.py's
`{prefix}_Lf_max` (L_ref [m]) and `{prefix}_ubulk` (U_ref [m/s]) arrays,
prefix in {U, P, Lf, Tad, df} - the only two arrays per prefix that are
relevant here (that file also defines `_Lf_maxslit`/`_Lf_max2`/
`_Lf_slitbase`/`_Lf_mp`/`_Lf_cg` variants and various Pi/DTD-fit values
used for an unrelated scaling-law analysis; none of those are used here).
Deliberately copied as static values into the repo rather than importing
that file directly, since it loads other inputs via relative paths that
only resolve from a specific external working directory.

Each array is ordered [WGR0, WGR091, WGR166, WGR231] (increasing
water-gas-ratio), matching the four WGR.../*.csv timeseries folders
under each subset directory.
"""

KORNILOV_CASES = {
    "U_const": {
        "WGR0":   {"L_ref": 0.0011000000000000003, "U_ref": 2.0},
        "WGR091": {"L_ref": 0.0026,                 "U_ref": 2.0},
        "WGR166": {"L_ref": 0.0046,                 "U_ref": 2.0},
        "WGR231": {"L_ref": 0.007299999999999999,   "U_ref": 2.0},
    },
    "P_const": {
        "WGR0":   {"L_ref": 0.0007000000000000001, "U_ref": 1.547230804},
        "WGR091": {"L_ref": 0.0023999999999999994, "U_ref": 1.77495666},
        "WGR166": {"L_ref": 0.0046,                 "U_ref": 1.850389284},
        "WGR231": {"L_ref": 0.008,                  "U_ref": 2.229656741},
    },
    "Lf_const": {
        "WGR0":   {"L_ref": 0.0060999999999999995, "U_ref": 2.0},
        "WGR091": {"L_ref": 0.0063,                 "U_ref": 2.0},
        "WGR166": {"L_ref": 0.0066,                 "U_ref": 2.0},
        "WGR231": {"L_ref": 0.0068000000000000005, "U_ref": 2.0},
    },
    "Tad_const": {
        "WGR0":   {"L_ref": 0.0052,                 "U_ref": 2.0},
        "WGR091": {"L_ref": 0.0049,                 "U_ref": 2.0},
        "WGR166": {"L_ref": 0.0054,                 "U_ref": 2.0},
        "WGR231": {"L_ref": 0.0068000000000000005, "U_ref": 2.0},
    },
    "df_const": {
        "WGR0":   {"L_ref": 0.0093,                 "U_ref": 2.0},
        "WGR091": {"L_ref": 0.0091,                 "U_ref": 2.0},
        "WGR166": {"L_ref": 0.0083,                 "U_ref": 2.0},
        "WGR231": {"L_ref": 0.0063999999999999994, "U_ref": 2.0},
    },
}

SUBSETS = list(KORNILOV_CASES.keys())
WGR_LABELS = ["WGR0", "WGR091", "WGR166", "WGR231"]
