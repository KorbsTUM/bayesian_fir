"""
data/Kornilov/metadata.py
==========================
Per-case L_ref / U_ref for the 20 (new, plain) Kornilov cases (C01-C20).
Unlike data/WET_Kornilov/, these 20 cases have no series/water-gas-ratio
structure - each is its own independent flame condition.

L_ref [m]: main flame length per case, the flame_length_main column of
.../NieblYoko26/flame_length_summary.csv (one directory above this repo's
own root - a CFD post-processing output, not tracked in this repo), minus
a 0.006 m offset (per the user: flame_length_main is measured from a
virtual origin 0.006 m upstream of the actual flame-anchoring point, so
this correction is needed before flame_length_main is usable as L_ref).

U_ref [m/s]: bulk reference velocity. 1.0 m/s for every case except the
following per-case operating points (provided directly by the user, not
derivable from the raw CSVs):
    C02: 0.5, C03: 1.5, C04: 2.0, C05: 2.5, C18: 2.5, C19: 1.5
"""

KORNILOV_CASES = {
    "C01": {"L_ref": 0.009125     - 0.006, "U_ref": 1.0},
    "C02": {"L_ref": 0.00765      - 0.006, "U_ref": 0.5},
    "C03": {"L_ref": 0.0103       - 0.006, "U_ref": 1.5},
    "C04": {"L_ref": 0.01137976   - 0.006, "U_ref": 2.0},
    "C05": {"L_ref": 0.01242216   - 0.006, "U_ref": 2.5},
    "C06": {"L_ref": 0.01470285   - 0.006, "U_ref": 1.0},
    "C07": {"L_ref": 0.011        - 0.006, "U_ref": 1.0},
    "C08": {"L_ref": 0.008025     - 0.006, "U_ref": 1.0},
    "C09": {"L_ref": 0.007675     - 0.006, "U_ref": 1.0},
    "C10": {"L_ref": 0.008425     - 0.006, "U_ref": 1.0},
    "C11": {"L_ref": 0.007875     - 0.006, "U_ref": 1.0},
    "C12": {"L_ref": 0.0074       - 0.006, "U_ref": 1.0},
    "C13": {"L_ref": 0.00685      - 0.006, "U_ref": 1.0},
    "C14": {"L_ref": 0.009375     - 0.006, "U_ref": 1.0},
    "C15": {"L_ref": 0.00895      - 0.006, "U_ref": 1.0},
    "C16": {"L_ref": 0.0087749995 - 0.006, "U_ref": 1.0},
    "C17": {"L_ref": 0.008575     - 0.006, "U_ref": 1.0},
    "C18": {"L_ref": 0.01980168   - 0.006, "U_ref": 2.5},
    "C19": {"L_ref": 0.009025     - 0.006, "U_ref": 1.5},
    "C20": {"L_ref": 0.008625     - 0.006, "U_ref": 1.0},
}

CASES = list(KORNILOV_CASES.keys())