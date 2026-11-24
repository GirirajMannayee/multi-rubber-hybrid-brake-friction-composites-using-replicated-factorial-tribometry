"""Shared constants and helpers.

All numeric constants that drive the synthetic generator are taken from the
summary statistics published in the manuscript (Tables 1, 3, 10-14, 17-19, 31)
so that the generated data reproduce the *reported* structure, NOT the original
physical measurements.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"
for _p in (RAW, PROC, TAB, FIG):
    _p.mkdir(parents=True, exist_ok=True)

SEED = 20260723
FORMS = [f"MR{i}" for i in range(7)]
BATCHES = ["B1", "B2", "B3"]
PRESS = [5, 10, 15]          # bar
SPEED = [300, 500, 660]      # rpm
TEMP = [150, 250, 350]       # degC setpoint

# --- Table 1: nominal composition (wt%) -------------------------------------
COMP = pd.DataFrame(
    {
        "NR":       [8, 10, 7, 7, 7, 7, 7.5],
        "NBR":      [6, 6, 9, 6, 6, 6, 7.5],
        "SBR":      [6, 5, 5, 8, 5, 5, 6],
        "EPDM":     [4, 3, 3, 3, 7, 4, 5],
        "MSR":      [2, 2, 2, 2, 1, 4, 3],
        "BaSO4":    [20, 20, 20, 20, 20, 20, 17],
        "Phenolic": [18] * 7,
        "Aramid":   [18] * 7,
        "Alumina":  [11] * 7,
        "Graphite": [7] * 7,
    },
    index=FORMS,
)
RUBBERS = ["NR", "NBR", "SBR", "EPDM", "MSR"]
ML_COMP = ["NR", "NBR", "SBR", "EPDM", "MSR", "BaSO4"]   # six varied constituents

# --- Responses ---------------------------------------------------------------
RESP = ["COF", "wear", "CV", "fade", "recovery"]
RESP_LABEL = {
    "COF": "Mean COF (-)",
    "wear": "Specific wear rate (1e-4 mm3/(N m))",
    "CV": "Friction CV (%)",
    "fade": "Fade (%)",
    "recovery": "Recovery (%)",
    "peakT": "Peak interface T (degC)",
}
RUN_COL = {   # run-level column names -> short names
    "mean_COF": "COF", "specific_wear_rate_e-4": "wear", "friction_CV_pct": "CV",
    "fade_pct": "fade", "recovery_pct": "recovery", "peak_interface_T_C": "peakT",
}


def code(P, V, T):
    """Coded factors of the manuscript (Sec. 2.3)."""
    return (np.asarray(P) - 10) / 5, (np.asarray(V) - 480) / 180, (np.asarray(T) - 250) / 100


def load_runs() -> pd.DataFrame:
    df = pd.read_csv(RAW / "run_level_567.csv")
    df = df.rename(columns=RUN_COL)
    df["xP"], df["xV"], df["xT"] = code(df.pressure_bar, df.speed_rpm, df.temp_setpoint_C)
    df["form_idx"] = df.formulation.str[2].astype(int)
    df["batch_idx"] = df.batch_id.str[-1].astype(int)
    for c in COMP.columns:
        df[c] = df.formulation.map(COMP[c])
    df["batch_uid"] = df.formulation + "-" + df.batch_id
    df["cond"] = (df.pressure_bar.astype(str) + "-" + df.speed_rpm.astype(str)
                  + "-" + df.temp_setpoint_C.astype(str))
    return df


def holm(p):
    """Holm step-down adjusted p-values."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, (m - rank) * p[idx])
        adj[idx] = min(1.0, run)
    return adj
