"""Gaussian-process surrogates, Derringer-Suich desirability, Pareto screening and
sensitivity analysis (manuscript Sec. 2.9 / 3.9, Tables 27-28)."""
import itertools, json, warnings
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern, WhiteKernel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from common import *

warnings.filterwarnings("ignore")
df = load_runs()
FEATS = ["pressure_bar", "speed_rpm", "temp_setpoint_C"] + ML_COMP
X = df[FEATS].values.astype(float)


def make_gp():
    k = C(1.0) * Matern(np.ones(len(FEATS)), (1e-2, 1e3), nu=2.5) + WhiteKernel(0.1, (1e-4, 1.0))
    return Pipeline([("sc", StandardScaler()), ("m", TransformedTargetRegressor(
        GaussianProcessRegressor(k, random_state=0), transformer=StandardScaler()))])


# ---- candidate grid: 21 P x 19 V x 21 T x 7 formulations = 58,653
Pg = np.arange(5, 15.01, 0.5); Vg = np.arange(300, 660.01, 20); Tg = np.arange(150, 350.01, 10)
rows = [(f, P, V, T) for f in FORMS for P in Pg for V in Vg for T in Tg]
grid = pd.DataFrame(rows, columns=["formulation", "pressure_bar", "speed_rpm", "temp_setpoint_C"])
for c in COMP.columns:
    grid[c] = grid.formulation.map(COMP[c])
Xg = grid[FEATS].values.astype(float)
tested = df.groupby(["formulation", "pressure_bar", "speed_rpm", "temp_setpoint_C"])[RESP].mean().reset_index()

pred, pstd = {}, {}
for r in RESP:
    m = make_gp().fit(X, df[r].values)
    gpr = m.named_steps["m"].regressor_
    Xs = m.named_steps["sc"].transform(Xg)
    mu, sd = gpr.predict(Xs, return_std=True)
    sc = m.named_steps["m"].transformer_
    pred[r] = sc.inverse_transform(mu.reshape(-1, 1)).ravel()
    pstd[r] = sd * float(sc.scale_[0])
    # predictions at tested points
    Xt = pd.merge(tested, grid.drop_duplicates("formulation")[["formulation"] + ML_COMP], on="formulation")[FEATS].values
    print(r, "GP fitted", flush=True)
    grid[f"{r}_pred"] = pred[r]
    grid[f"{r}_sd"] = pstd[r]


def trap(x, z0, p0, p1, z1):
    return np.clip(np.minimum((x - z0) / (p0 - z0), (z1 - x) / (z1 - p1)), 0, 1)


def desir(g, plateau=(0.44, 0.48), zeros=(0.40, 0.50), w=None):
    d = {
        "COF": trap(g.COF_pred.values, zeros[0], plateau[0], plateau[1], zeros[1]),
        "wear": np.clip((10 - g.wear_pred.values) / 4, 0, 1),
        "CV": np.clip((9.5 - g.CV_pred.values) / 6, 0, 1),
        "fade": np.clip((13 - g.fade_pred.values) / 11, 0, 1),
        "recovery": np.clip((g.recovery_pred.values - 85) / 13, 0, 1),
    }
    w = w or {k: 1 for k in d}
    tot = sum(w.values())
    D = np.ones(len(g))
    for k, v in d.items():
        D = D * np.power(v, w[k] / tot)
    return D, d


D, dd = desir(grid)
grid["D"] = D
for k, v in dd.items():
    grid[f"d_{k}"] = v

grid.to_csv(PROC / "screening_grid_predictions.csv.gz", index=False, compression="gzip")

# ---- Table 27
key = ["formulation", "pressure_bar", "speed_rpm", "temp_setpoint_C"]
t27 = []
tst = grid.merge(tested[key], on=key)            # grid points that coincide with tested conditions
for f in FORMS:
    a = tst[tst.formulation == f].D; b = grid[grid.formulation == f].D
    t27.append(dict(formulation=f, mean_D_27_tested=a.mean(), min_D_27_tested=a.min(),
                    mean_D_full_grid=b.mean(), max_D_full_grid=b.max()))
pd.DataFrame(t27).to_csv(TAB / "T27_desirability_ranking.csv", index=False)

# ---- optimum + 5 % region for MR6
best = grid.loc[grid[grid.formulation == "MR6"].D.idxmax()]
reg = grid[(grid.formulation == "MR6") & (grid.D >= 0.95 * best.D)]
summary = dict(best_MR6=best[key + ["COF_pred", "wear_pred", "CV_pred", "fade_pred", "recovery_pred", "D",
                                    "COF_sd", "wear_sd"]].to_dict(),
               region5pct=dict(P=[reg.pressure_bar.min(), reg.pressure_bar.max()],
                               V=[reg.speed_rpm.min(), reg.speed_rpm.max()],
                               T=[reg.temp_setpoint_C.min(), reg.temp_setpoint_C.max()]))

# ---- Pareto (5 objectives) among feasible candidates, COF 0.40-0.50
feas = grid[(grid.COF_pred >= 0.40) & (grid.COF_pred <= 0.50)].copy()
feas["dev"] = np.maximum(0, np.maximum(0.44 - feas.COF_pred, feas.COF_pred - 0.48))
O = feas[["dev", "wear_pred", "CV_pred", "fade_pred"]].copy(); O["negrec"] = -feas.recovery_pred
O = O.values
order = np.argsort(O.sum(1))
front = []
for i in order:
    dominated = False
    for j in front:
        if np.all(O[j] <= O[i]) and np.any(O[j] < O[i]):
            dominated = True; break
    if not dominated:
        front = [j for j in front if not (np.all(O[i] <= O[j]) and np.any(O[i] < O[j]))] + [i]
nd = feas.iloc[front]
summary["pareto"] = dict(n_feasible=len(feas), n_nondominated=len(nd),
                         formulations=nd.formulation.value_counts().to_dict())
nd.to_csv(TAB / "T_pareto_nondominated.csv", index=False)

# ---- Table 28 sensitivity
scen = {
    "Base (0.44-0.48 plateau, equal weights)": dict(),
    "Narrow plateau 0.45-0.47": dict(plateau=(0.45, 0.47)),
    "Wide plateau 0.42-0.50": dict(plateau=(0.42, 0.50), zeros=(0.40, 0.52)),
    "Wear weighted x2": dict(w=dict(COF=1, wear=2, CV=1, fade=1, recovery=1)),
    "COF weighted x2": dict(w=dict(COF=2, wear=1, CV=1, fade=1, recovery=1)),
    "Stability (CV, fade) weighted x2": dict(w=dict(COF=1, wear=1, CV=2, fade=2, recovery=1)),
}
t28 = []
for n, kw in scen.items():
    Ds, _ = desir(grid, **kw)
    g2 = grid.assign(Dscn=Ds)
    mx = g2.groupby("formulation").Dscn.max()
    t28.append(dict(scenario=n, top_formulation=mx.idxmax(), D_MR6=mx["MR6"], D_MR5=mx["MR5"], D_MR0=mx["MR0"]))
pd.DataFrame(t28).to_csv(TAB / "T28_sensitivity.csv", index=False)
json.dump(summary, open(TAB / "screening_summary.json", "w"), indent=2, default=float)
print(json.dumps(summary, indent=2, default=float))
print(pd.DataFrame(t27).round(3).to_string()); print(pd.DataFrame(t28).round(3).to_string())
