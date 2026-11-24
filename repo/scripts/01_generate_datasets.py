"""Generate the SYNTHETIC reconstruction of the replicated 7 x 3^3 tribometry
dataset (567 runs), the 31-point time series (17,577 rows), the 189 condition
summaries, the 21 batch records and the 81-run MR6 companion dataset.

IMPORTANT: every value produced here is simulated. The generator is calibrated
to the *published summary statistics* (additive quadratic RSM coefficients of
Table 18, marginal means of Table 14, formulation offsets of Table 12, noise
levels of Sec. 3.5, ICC values, ...). It is intended for code testing, teaching
and pipeline demonstration. It MUST NOT be reported as measured evidence.
Replace data/raw/run_level_567.csv with the real workbook export to analyse the
physical data - all downstream scripts read that single file.
"""
import itertools
import numpy as np
import pandas as pd
from common import *

rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------- parameters
OFF = {   # formulation offsets relative to MR0 (Table 12 / Table 18)
    "COF":      [0, .0131, .0265, .0211, .0358, .0433, .0535],
    "wear":     [0, -.321, -.574, -.463, -.833, -1.009, -1.275],
    "CV":       [0, -.12, -.32, -.33, -.49, -.61, -.75],
    "fade":     [0, -.17, -.59, -1.05, -1.35, -1.55, -1.99],
    "recovery": [0, -.09, .37, .18, .79, 1.30, 1.41],
}
# Table 18 additive quadratic coefficients (coded units)
B_COF = dict(b0=.4381, P=-.0021, V=-.0024, T=-.0026, PP=-.0094, VV=-.0063, TT=-.0053,
             PV=.0003, PT=.0004, VT=.0002)
B_WEAR = dict(b0=8.614, P=.407, V=.452, T=.360, PP=.046, VV=.007, TT=.024,
              PV=.008, PT=-.009, VT=-.015)
# Table 14 marginal means (levels 5/10/15 bar, 300/500/660 rpm, 150/250/350 C)
MARG = {
    "CV":       dict(P=[5.82, 6.43, 7.04], V=[5.67, 6.50, 7.11], T=[5.86, 6.45, 6.98]),
    "fade":     dict(P=[6.87, 8.04, 9.11], V=[6.73, 8.04, 9.25], T=[7.12, 7.91, 8.99]),
    "recovery": dict(P=[93.1, 92.3, 91.3], V=[93.1, 92.2, 91.4], T=[93.0, 92.3, 91.4]),
    "peakT":    dict(P=[273.5, 278.0, 282.8], V=[273.7, 278.6, 282.2], T=[177.3, 278.2, 378.9]),
}
GM = {"CV": 6.43, "fade": 8.01, "recovery": 92.24, "peakT": 278.1}      # Table 10 means
NOISE = {"COF": .0092, "wear": .219, "CV": .45, "fade": .95, "recovery": 1.55, "peakT": 2.9}
BATCH_SD = {"COF": .0009, "wear": .004, "CV": .09, "fade": .08, "recovery": .05, "peakT": .3}
DENS_MEAN = [1.910, 1.906, 1.904, 1.901, 1.899, 1.896, 1.893]     # Table 11
SHORE_MEAN = [76.0, 75.9, 75.8, 75.6, 75.5, 75.2, 75.1]


def lvl_effect(resp, fac, level, grid):
    m = np.array(MARG[resp][fac])
    return float(m[grid.index(level)] - m.mean())


def quad_model(b, xP, xV, xT):
    return (b["b0"] + b["P"] * xP + b["V"] * xV + b["T"] * xT + b["PP"] * xP**2
            + b["VV"] * xV**2 + b["TT"] * xT**2 + b["PV"] * xP * xV
            + b["PT"] * xP * xT + b["VT"] * xV * xT)


# ------------------------------------------------------------- batch records
batch_rows = []
for f in FORMS:
    for b in BATCHES:
        row = {"batch_uid": f"{f}-{b}", "formulation": f, "batch_id": b}
        total = 0.0
        for ing in COMP.columns:
            nominal = COMP.loc[f, ing] * 5.0                    # g in a 500 g batch (Table 3)
            actual = round(nominal + rng.normal(0, 0.15), 2)    # weighing scatter (simulated)
            row[f"{ing}_nominal_g"] = nominal
            row[f"{ing}_weighed_g"] = actual
            total += actual
        row["total_weighed_g"] = round(total, 2)
        row["mixer_make_model"] = "TO_BE_FILLED"
        row["mixing_time_min"] = "TO_BE_FILLED"
        row["mould_T_C"] = "TO_BE_FILLED"
        row["mould_pressure_MPa"] = "TO_BE_FILLED"
        row["post_cure"] = "TO_BE_FILLED"
        row["raw_material_lots"] = "TO_BE_FILLED"
        row["data_status"] = "SYNTHETIC"
        batch_rows.append(row)
batch_df = pd.DataFrame(batch_rows)
batch_df.to_csv(RAW / "batch_records_21.csv", index=False)

batch_eff = {r: {(f, b): rng.normal(0, BATCH_SD[r]) for f in FORMS for b in BATCHES}
             for r in BATCH_SD}

# ------------------------------------------------------------------ run level
rows = []
for fi, f in enumerate(FORMS):
    for b in BATCHES:
        conds = list(itertools.product(PRESS, SPEED, TEMP))
        spec_order = rng.permutation(len(conds))
        for k, (P, V, T) in enumerate(conds):
            xP, xV, xT = code(P, V, T)
            r = {
                "run_id": f"{f}-{b}-P{P:02d}-V{V}-T{T}",
                "formulation": f, "batch_id": b,
                "specimen_id": f"{f}-{b}-S{spec_order[k] + 1:02d}",
                "run_order_in_batch": int(spec_order[k] + 1),
                "pressure_bar": P, "speed_rpm": V, "temp_setpoint_C": T,
                "duration_min": 30, "ambient_C": 23,
                "counterface": "AISI 1045/EN8 steel disc, ground+polished",
                "counterface_Ra_um": round(float(rng.uniform(0.8, 1.6)), 2),
            }
            r["mean_COF"] = (quad_model(B_COF, xP, xV, xT) + OFF["COF"][fi]
                             + batch_eff["COF"][(f, b)] + rng.normal(0, NOISE["COF"]))
            r["specific_wear_rate_e-4"] = (quad_model(B_WEAR, xP, xV, xT) + OFF["wear"][fi]
                                           + batch_eff["wear"][(f, b)] + rng.normal(0, NOISE["wear"]))
            for resp, col in (("CV", "friction_CV_pct"), ("fade", "fade_pct"),
                              ("recovery", "recovery_pct"), ("peakT", "peak_interface_T_C")):
                off = 0.0 if resp == "peakT" else OFF[resp][fi] - np.mean(OFF[resp])
                eff = (lvl_effect(resp, "P", P, PRESS) + lvl_effect(resp, "V", V, SPEED)
                       + lvl_effect(resp, "T", T, TEMP))
                r[col] = (GM[resp] + off + eff + batch_eff[resp][(f, b)]
                          + rng.normal(0, NOISE[resp]))
            # mass loss: Archard-like power law in P and V, mild T term, NO formulation term
            r["_mass_raw"] = (P**1.09) * (V**1.14) * np.exp(0.45 * np.log((T) / 250.0)) \
                * np.exp(rng.normal(0, 0.05))
            r["density_g_cm3"] = float(np.clip(rng.normal(DENS_MEAN[fi], 0.0079), 1.8745, 1.9347))
            r["hardness_shoreD"] = float(np.clip(rng.normal(SHORE_MEAN[fi], 0.71), 73.22, 78.62))
            rows.append(r)

df = pd.DataFrame(rows)
df["mass_loss_g"] = df._mass_raw * (0.1451 / df._mass_raw.mean())
df = df.drop(columns="_mass_raw")
# physical bounds only (no forcing of the published min/max)
df["fade_pct"] = df.fade_pct.clip(lower=0.5)
df["recovery_pct"] = df.recovery_pct.clip(upper=100)
for c, d in (("mean_COF", 4), ("specific_wear_rate_e-4", 3), ("friction_CV_pct", 2), ("fade_pct", 2),
             ("recovery_pct", 2), ("peak_interface_T_C", 1), ("mass_loss_g", 4),
             ("density_g_cm3", 4), ("hardness_shoreD", 2)):
    df[c] = df[c].round(d)
df["dataset_status"] = "SYNTHETIC"
df["calibration_status"] = "NOT_APPLICABLE_SYNTHETIC"
df["test_basis"] = "SIMULATED_GENERATOR_V1"
df = df.sample(frac=1, random_state=1).sort_values(["formulation", "batch_id", "run_order_in_batch"]) \
       .reset_index(drop=True)
df.to_csv(RAW / "run_level_567.csv", index=False)

# ---------------------------------------------------------------- time series
t = np.arange(0, 31)
s = 1 - np.exp(-t / 6.0)
A = 0.0307
trend = A * (s - s.mean())
bump = (t / 12.0) * np.exp(1 - t / 12.0)
ts_rows = []
for _, r in df.iterrows():
    mu, cv = r.mean_COF, r.friction_CV_pct
    # AR(1) noise, projected orthogonal to constant & trend, then scaled so that the
    # series reproduces the run-level mean COF and CV exactly (ddof = 1).
    e = np.zeros(len(t))
    for i in range(1, len(t)):
        e[i] = 0.5 * e[i - 1] + rng.normal()
    X = np.column_stack([np.ones_like(t, float), trend])
    e = e - X @ np.linalg.lstsq(X, e, rcond=None)[0]
    var_tot = (cv / 100.0 * mu) ** 2
    var_tr = trend.var(ddof=1)
    var_n = max(var_tot - var_tr, 1e-8)
    e = e / e.std(ddof=1) * np.sqrt(var_n)
    cof_t = mu + trend + e
    # interface temperature: first-order rise + over-shoot bump, rescaled to the recorded peak
    Tset, Tpk = r.temp_setpoint_C, r.peak_interface_T_C
    Tt = 23 + (Tset - 23) * (1 - np.exp(-t / 7.0)) + (Tpk - Tset) * bump
    Tt = 23 + (Tt - 23) * (Tpk - 23) / (Tt.max() - 23)
    Tt = Tt + rng.normal(0, 0.8, len(t)); Tt[0] = 23 + abs(rng.normal(0, 0.5))
    Tt[np.argmax(Tt)] = Tpk if Tt.max() > Tpk else Tt[np.argmax(Tt)]
    for i in range(len(t)):
        ts_rows.append((r.run_id, int(t[i]), round(float(cof_t[i]), 5), round(float(Tt[i]), 1)))
ts = pd.DataFrame(ts_rows, columns=["run_id", "time_min", "COF_t", "interface_T_C"])
ts["signal_status"] = "SYNTHETIC_EXAMPLE_SIGNAL"
ts.to_csv(RAW / "time_series_17577.csv", index=False)

# --------------------------------------------------------- condition summary
d2 = df.rename(columns=RUN_COL)
g = d2.groupby(["formulation", "pressure_bar", "speed_rpm", "temp_setpoint_C"])
cs = g[["COF", "wear", "CV", "fade", "recovery", "peakT", "mass_loss_g"]].agg(["mean", "std"])
cs.columns = [f"{a}_{b}" for a, b in cs.columns]
cs = cs.reset_index()
cs.insert(4, "n_batches", 3)
cs["dataset_status"] = "SYNTHETIC"
cs.round(5).to_csv(RAW / "condition_summary_189.csv", index=False)

# ----------------------------------------------------- companion MR6 dataset
comp_rows = []
T_cov = {175: 70.2, 250: 65.5, 350: 50.0}
for P, V, T in itertools.product(PRESS, SPEED, [175, 250, 350]):
    xP, xV, xT = code(P, V, T)
    cov_mean = (T_cov[T] + {5: -2.0, 10: 4.0, 15: -2.0}[P] - 1.2 * xV
                - 1.6 * xP * (T == 350) - 1.2 * xV * (T == 350))
    for rep in (1, 2, 3):
        cov = float(np.clip(cov_mean + rng.normal(0, 1.8), 43.9, 77.9))
        comp_rows.append({
            "run_id": f"C-MR6-P{P:02d}-V{V}-T{T}-R{rep}", "formulation": "MR6",
            "pressure_bar": P, "speed_rpm": V, "temp_setpoint_C": T, "replicate": rep,
            "mean_COF": 0.472 + 0.0016 * (cov - 61.9) + rng.normal(0, 0.0045),
            "specific_wear_rate_e-6": 2.68 - 0.0207 * cov + rng.normal(0, 0.081),
            "friction_CV_pct": 5.0 - 0.14 * (cov - 61.9) + rng.normal(0, 0.5),
            "fade_pct": 8.0 - 0.22 * (cov - 61.9) + rng.normal(0, 0.45),
            "recovery_pct": min(100, 94.0 + 0.32 * (cov - 61.9) + rng.normal(0, 0.75)),
            "peak_interface_T_C": T + 28 + rng.normal(0, 3),
            "tribofilm_coverage_pct": cov,
            "dataset_status": "SYNTHETIC_COMPANION",
        })
comp_df = pd.DataFrame(comp_rows).round(4)
comp_df.to_csv(RAW / "companion_MR6_81.csv", index=False)

print("run_level", df.shape, "| time_series", ts.shape, "| condition_summary", cs.shape,
      "| batches", batch_df.shape, "| companion", comp_df.shape)
d = df.rename(columns=RUN_COL)
print(d[["COF", "wear", "CV", "fade", "recovery", "peakT", "mass_loss_g", "density_g_cm3",
         "hardness_shoreD"]].describe().loc[["mean", "std", "min", "max"]].round(4).T)
