"""Machine-learning benchmark under four validation schemes (S1-S4), grouped permutation
importance, SHAP, ungrouped importance, condition-grouped / LOFO benchmark and the companion
MR6 benchmark (manuscript Tables 22-26, 31). Hyper-parameters are fixed (no nested tuning) to keep
run time to a few minutes on one CPU core; edit MODELS to change them."""
import warnings, json, sys
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern, WhiteKernel
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from common import *

warnings.filterwarnings("ignore")
df = load_runs()
FEATS = ["pressure_bar", "speed_rpm", "temp_setpoint_C"] + ML_COMP
X = df[FEATS].values.astype(float)
FORMV = df.form_idx.values


def gp():
    k = C(1.0) * Matern(length_scale=np.ones(len(FEATS)), length_scale_bounds=(1e-2, 1e3), nu=2.5) \
        + WhiteKernel(0.1, (1e-4, 1.0))
    return Pipeline([("sc", StandardScaler()),
                     ("m", TransformedTargetRegressor(
                         GaussianProcessRegressor(k, normalize_y=False, n_restarts_optimizer=0, random_state=0),
                         transformer=StandardScaler()))])


def ridge():
    ct = ColumnTransformer([("poly", PolynomialFeatures(2, include_bias=False), [0, 1, 2])],
                           remainder="passthrough")
    return Pipeline([("ct", ct), ("sc", StandardScaler()), ("m", Ridge(alpha=1.0))])


MODELS = {
    "Mean baseline": lambda: DummyRegressor(),
    "Quadratic ridge": ridge,
    "Random forest": lambda: RandomForestRegressor(200, min_samples_leaf=2, random_state=0),
    "Extra trees": lambda: ExtraTreesRegressor(200, min_samples_leaf=2, random_state=0),
    "Gradient boosting": lambda: GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                                           max_depth=3, subsample=0.8, random_state=0),
    "RBF-SVR": lambda: Pipeline([("sc", StandardScaler()),
                                 ("m", TransformedTargetRegressor(SVR(C=10, epsilon=0.1, gamma="scale"),
                                                                  transformer=StandardScaler()))]),
    "Gaussian process": gp,
}


def rsm_design(Xo, f, levels):
    """Factorial RSM: formulation dummies (levels 1..6) + full quadratic in coded P,V,T."""
    xP, xV, xT = code(Xo[:, 0], Xo[:, 1], Xo[:, 2])
    Z = np.column_stack([xP, xV, xT, xP**2, xV**2, xT**2, xP*xV, xP*xT, xV*xT])
    if levels:
        Z = np.column_stack([Z] + [(f == l).astype(float) for l in levels])
    return Z


def fit_predict(name, Xtr, ytr, Xte, ftr, fte):
    if name == "RSM (formulation as factor)":
        seen = sorted(set(ftr) - {0})
        use = set(fte).issubset(set(ftr))
        lv = seen if use else []
        m = LinearRegression().fit(rsm_design(Xtr, ftr, lv), ytr)
        return m.predict(rsm_design(Xte, fte, lv))
    m = MODELS[name]().fit(Xtr, ytr)
    return m.predict(Xte)


ALL = list(MODELS) + ["RSM (formulation as factor)"]
cell = df.speed_rpm.astype(str) + "_" + df.temp_setpoint_C.astype(str)
SCHEMES = {
    "S1": list(KFold(5, shuffle=True, random_state=0).split(X)),
    "S2": list(LeaveOneGroupOut().split(X, groups=df.batch_idx)),
    "S3": list(LeaveOneGroupOut().split(X, groups=cell)),
    "S4": list(LeaveOneGroupOut().split(X, groups=df.form_idx)),
}


def score(y, p):
    return dict(R2=r2_score(y, p), MAE=mean_absolute_error(y, p), RMSE=mean_squared_error(y, p) ** 0.5)


def benchmark(schemes, targets, tag):
    rows, oof_rows = [], []
    for tg in targets:
        y = df[tg].values
        for sn, folds in schemes.items():
            for name in ALL:
                oof = np.zeros(len(y))
                for tr, te in folds:
                    oof[te] = fit_predict(name, X[tr], y[tr], X[te], FORMV[tr], FORMV[te])
                rows.append(dict(target=tg, scheme=sn, model=name, **score(y, oof)))
                if tg in ("COF", "wear") and sn in ("S3", "S4"):
                    oof_rows.append(pd.DataFrame({"run_id": df.run_id, "target": tg, "scheme": sn,
                                                  "model": name, "observed": y, "predicted": oof}))
            print(tag, tg, sn, "done", flush=True)
    return pd.DataFrame(rows), (pd.concat(oof_rows) if oof_rows else None)


def perm_importance():
    # Table 26: grouped permutation importance (extra trees, 9 random condition-grouped folds, 10 perms)
    rng = np.random.default_rng(0)
    groups = {"Pressure": [0], "Speed": [1], "Temperature": [2], "Composition block": list(range(3, 9))}
    rows = []
    for tg in RESP:
        y = df[tg].values
        per = {g: [] for g in groups}
        for tr, te in list(GroupKFold(9, shuffle=True, random_state=0).split(X, groups=df.formulation + "_" + df.cond)):
            m = ExtraTreesRegressor(200, min_samples_leaf=2, random_state=0).fit(X[tr], y[tr])
            base = mean_squared_error(y[te], m.predict(X[te]))
            for g, cols in groups.items():
                rat = []
                for _ in range(10):
                    Xp = X[te].copy()
                    idx = rng.permutation(len(te))
                    Xp[:, cols] = X[te][idx][:, cols]        # block permuted jointly
                    rat.append(mean_squared_error(y[te], m.predict(Xp)) / base)
                per[g].append(np.mean(rat))
        rows.append(dict(response=tg, **{f"{g}_mean": np.mean(v) for g, v in per.items()},
                         **{f"{g}_sd": np.std(v, ddof=1) for g, v in per.items()}))
    pd.DataFrame(rows).to_csv(TAB / "T26_grouped_permutation_importance.csv", index=False)
    print("T26 done", flush=True)


if __name__ == "__main__":
    res, oof = benchmark(SCHEMES, RESP, "main")
    res.to_csv(TAB / "T23_ml_benchmark_all.csv", index=False)
    oof.to_csv(PROC / "oof_predictions_COF_wear_S3_S4.csv", index=False)
    res.pivot_table(index=["target", "model"], columns="scheme", values="R2").to_csv(TAB / "T22_ml_R2_pivot.csv")

    # ---- Table 24 analogue: condition-grouped 5-fold (formulation-P-V-T groups) and LOFO
    grp = df.formulation + "_" + df.cond
    gk = list(GroupKFold(5).split(X, groups=grp))
    t24 = []
    for tg in ("wear", "COF"):
        y = df[tg].values
        for name in [m for m in ALL if m != "Mean baseline"]:
            oof = np.zeros(len(y))
            for tr, te in gk:
                oof[te] = fit_predict(name, X[tr], y[tr], X[te], FORMV[tr], FORMV[te])
            t24.append(dict(target=tg, model=name, scheme="condition_5fold", **score(y, oof)))
            oof2 = np.zeros(len(y))
            for tr, te in SCHEMES["S4"]:
                oof2[te] = fit_predict(name, X[tr], y[tr], X[te], FORMV[tr], FORMV[te])
            t24.append(dict(target=tg, model=name, scheme="LOFO_7fold", **score(y, oof2)))
    pd.DataFrame(t24).to_csv(TAB / "T24_condition_grouped_and_LOFO.csv", index=False)
    print("T24 done", flush=True)

    perm_importance()

    # ---- SHAP (extra trees, wear and COF)
    try:
        import shap
        sh = []
        for tg in ("wear", "COF"):
            m = ExtraTreesRegressor(300, min_samples_leaf=2, random_state=0).fit(X, df[tg].values)
            v = shap.TreeExplainer(m).shap_values(X)
            sh.append(pd.DataFrame({"target": tg, "feature": FEATS, "mean_abs_shap": np.abs(v).mean(0)}))
        pd.concat(sh).to_csv(TAB / "T_shap_mean_abs.csv", index=False)
    except Exception as e:
        print("SHAP skipped:", e)

    # ---- ungrouped permutation importance (GP -> wear, quadratic ridge -> COF)
    imp = []
    for tg, nm in (("wear", "Gaussian process"), ("COF", "Quadratic ridge")):
        y = df[tg].values
        gkf = list(LeaveOneGroupOut().split(X, groups=cell))[:]
        folds5 = list(GroupKFold(5).split(X, groups=cell))
        per = []
        for tr, te in folds5:
            m = MODELS[nm]().fit(X[tr], y[tr])
            pi = permutation_importance(m, X[te], y[te], n_repeats=20, random_state=0, scoring="r2")
            per.append(pi.importances_mean)
        per = np.array(per)
        imp.append(pd.DataFrame({"target": tg, "model": nm, "feature": FEATS,
                                 "mean": per.mean(0), "sd": per.std(0, ddof=1)}))
    pd.concat(imp).to_csv(TAB / "T_ungrouped_permutation_importance.csv", index=False)
    print("ungrouped importance done", flush=True)

    # ---- Companion MR6 benchmark (Table 31 analogue)
    comp = pd.read_csv(RAW / "companion_MR6_81.csv")
    Xc = comp[["pressure_bar", "speed_rpm", "temp_setpoint_C"]].values.astype(float)
    yc = comp["specific_wear_rate_e-6"].values
    condc = comp.pressure_bar.astype(str) + comp.speed_rpm.astype(str) + comp.temp_setpoint_C.astype(str)
    split = np.random.default_rng(1).permutation(len(yc))
    te_i, tr_i = split[:16], split[16:]
    cm = {"Random forest": lambda: RandomForestRegressor(200, random_state=0),
          "Extra trees": lambda: ExtraTreesRegressor(200, random_state=0),
          "Gradient boosting": lambda: GradientBoostingRegressor(random_state=0),
          "RBF-SVR": lambda: Pipeline([("sc", StandardScaler()), ("m", TransformedTargetRegressor(
              SVR(C=10), transformer=StandardScaler()))]),
          "Gaussian process": lambda: Pipeline([("sc", StandardScaler()), ("m", TransformedTargetRegressor(
              GaussianProcessRegressor(C(1.0) * Matern(np.ones(3), nu=2.5) + WhiteKernel(0.1),
                                       random_state=0), transformer=StandardScaler()))])}
    ct = []
    for n, f in cm.items():
        p = f().fit(Xc[tr_i], yc[tr_i]).predict(Xc[te_i])
        oof = np.zeros(len(yc))
        for tr, te in LeaveOneGroupOut().split(Xc, groups=condc):
            oof[te] = f().fit(Xc[tr], yc[tr]).predict(Xc[te])
        ct.append(dict(model=n, holdout_R2=r2_score(yc[te_i], p),
                       holdout_RMSE=mean_squared_error(yc[te_i], p) ** 0.5,
                       groupedCV_R2=r2_score(yc, oof), groupedCV_RMSE=mean_squared_error(yc, oof) ** 0.5))
    pd.DataFrame(ct).to_csv(TAB / "T31_companion_benchmark.csv", index=False)
    print("ALL DONE")
