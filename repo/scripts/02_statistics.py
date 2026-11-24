"""Descriptives, ANOVA, RSM, batch ICC, contrasts, marginal means (manuscript Tables 10-21)."""
import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
from scipy import stats
from common import *

df = load_runs()
R5 = RESP
R6 = RESP + ["peakT"]
out = {}

# ---- Table 10 / 11 descriptives
t10 = df[R6 + ["mass_loss_g", "density_g_cm3", "hardness_shoreD"]].agg(["count", "mean", "std", "min", "max"]).T
t10.to_csv(TAB / "T10_descriptives.csv")
agg = df.groupby("formulation")[R5 + ["density_g_cm3", "hardness_shoreD"]].agg(["mean", "std"])
agg.columns = [f"{a}_{b}" for a, b in agg.columns]
agg.to_csv(TAB / "T11_formulation_means.csv")

# ---- Table 12/13 paired contrasts vs MR0 over 27 conditions, Holm per response
cm = df.groupby(["formulation", "pressure_bar", "speed_rpm", "temp_setpoint_C"])[R6].mean()
sig = {"COF": .0093, "wear": .218}
rows = []
for r in R6:
    res = []
    for f in FORMS[1:]:
        d = (cm.loc[f][r].values - cm.loc["MR0"][r].values)
        t, p = stats.ttest_1samp(d, 0)
        se = d.std(ddof=1) / np.sqrt(len(d))
        ci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=se)
        res.append(dict(response=r, contrast=f"{f} vs MR0", delta=d.mean(),
                        pct=100 * d.mean() / cm.loc["MR0"][r].mean(), ci_lo=ci[0], ci_hi=ci[1],
                        d_z=d.mean() / d.std(ddof=1), p=p))
    adj = holm([x["p"] for x in res])
    for x, a in zip(res, adj):
        x["p_holm"] = a
    rows += res
t12 = pd.DataFrame(rows)
t12.to_csv(TAB / "T12_contrasts_vs_MR0.csv", index=False)
# neighbouring contrasts in sigma units
nb = []
for r, s_ in sig.items():
    for a, b in (("MR5", "MR4"), ("MR6", "MR5")):
        d = (cm.loc[b][r].values - cm.loc[a][r].values) if False else (cm.loc[a][r].mean() - cm.loc[b][r].mean())
        nb.append(dict(response=r, contrast=f"{a} vs {b}", delta=d, in_sigma=d / s_))
for f in FORMS[1:]:
    for r, s_ in sig.items():
        d = cm.loc[f][r].mean() - cm.loc["MR0"][r].mean()
        nb.append(dict(response=r, contrast=f"{f} vs MR0", delta=d, in_sigma=d / s_))
pd.DataFrame(nb).to_csv(TAB / "T13_contrasts_in_sigma_units.csv", index=False)

# ---- Table 14/15 marginal means
mm = []
for fac, col in (("Pressure (bar)", "pressure_bar"), ("Speed (rpm)", "speed_rpm"), ("Setpoint (C)", "temp_setpoint_C")):
    g = df.groupby(col)[R6].mean()
    for lv, row in g.iterrows():
        mm.append({"factor": fac, "level": lv, **row.to_dict()})
t14 = pd.DataFrame(mm)
t14.to_csv(TAB / "T14_marginal_means.csv", index=False)
rel = []
for fac in t14.factor.unique():
    s = t14[t14.factor == fac].reset_index(drop=True)
    row = {"factor": fac}
    for r in R6:
        row[r + "_low_to_high_pct"] = 100 * (s.loc[2, r] / s.loc[0, r] - 1)
    row["COF_low_to_mid_pct"] = 100 * (s.loc[1, "COF"] / s.loc[0, "COF"] - 1)
    rel.append(row)
pd.DataFrame(rel).to_csv(TAB / "T15_relative_change.csv", index=False)

# ---- Table 16 type-II ANOVA of full model (partial eta2)
full = ("{y} ~ C(formulation) + xP + xV + xT + I(xP**2) + I(xV**2) + I(xT**2) + xP:xV + xP:xT + xV:xT"
        " + C(formulation):xP + C(formulation):xV + C(formulation):xT")
nice = {"C(formulation)": "Formulation", "xP": "Pressure (P)", "xV": "Speed (V)", "xT": "Temperature (T)",
        "I(xP ** 2)": "P^2", "I(xV ** 2)": "V^2", "I(xT ** 2)": "T^2", "xP:xV": "P x V", "xP:xT": "P x T",
        "xV:xT": "V x T", "C(formulation):xP": "Formulation x P", "C(formulation):xV": "Formulation x V",
        "C(formulation):xT": "Formulation x T"}
t16 = {}
for r in R5:
    m = smf.ols(full.format(y=r), df).fit()
    a = anova_lm(m, typ=2)
    ss_res = a.loc["Residual", "sum_sq"]
    for term, row in a.drop("Residual").iterrows():
        t16.setdefault(nice.get(term, term), {})[f"{r}_eta2"] = row.sum_sq / (row.sum_sq + ss_res)
        t16[nice.get(term, term)][f"{r}_p"] = row["PR(>F)"]
pd.DataFrame(t16).T.to_csv(TAB / "T16_anova_partial_eta2.csv")

# ---- Table 17 categorical variance partition
cat = ("{y} ~ C(formulation) + C(pressure_bar) + C(speed_rpm) + C(temp_setpoint_C)"
       " + C(pressure_bar):C(speed_rpm) + C(pressure_bar):C(temp_setpoint_C)"
       " + C(speed_rpm):C(temp_setpoint_C) + C(formulation):C(temp_setpoint_C)")
nice2 = {"C(formulation)": "Formulation", "C(pressure_bar)": "Pressure", "C(speed_rpm)": "Speed",
         "C(temp_setpoint_C)": "Temperature", "C(pressure_bar):C(speed_rpm)": "P x V",
         "C(pressure_bar):C(temp_setpoint_C)": "P x T", "C(speed_rpm):C(temp_setpoint_C)": "V x T",
         "C(formulation):C(temp_setpoint_C)": "Formulation x T", "Residual": "Residual"}
t17, r2cat, inter_p = {}, {}, []
for r in R6:
    m = smf.ols(cat.format(y=r), df).fit()
    a = anova_lm(m, typ=2)
    tot = a.sum_sq.sum()
    r2cat[r] = m.rsquared
    for term, row in a.iterrows():
        t17.setdefault(nice2[term], {})[r] = 100 * row.sum_sq / tot
        if ":" in term and r in ("COF", "wear"):
            inter_p.append((r, nice2[term], row["PR(>F)"]))
t17 = pd.DataFrame(t17).T
t17.loc["Model R2"] = pd.Series(r2cat)
t17.to_csv(TAB / "T17_variance_share_pct.csv")

# ---- Table 18 additive quadratic model + LOF + pure error
add = ("{y} ~ C(formulation) + xP + xV + xT + I(xP**2) + I(xV**2) + I(xT**2) + xP:xV + xP:xT + xV:xT")
coef_rows, fit_stats, pred_tab, models = [], [], {}, {}
cellmeans = df.groupby("cond")
for r in R5:
    m = smf.ols(add.format(y=r), df).fit()
    models[r] = m
    ci = m.conf_int()
    for k in m.params.index:
        coef_rows.append(dict(response=r, term=k, coef=m.params[k], se=m.bse[k], p=m.pvalues[k]))
    # pure error from replicate batches (formulation x condition cells)
    cell = df.groupby(["formulation", "cond"])[r]
    sspe = ((df[r] - cell.transform("mean")) ** 2).sum()
    dfpe = len(df) - cell.ngroups
    dflof = m.df_resid - dfpe
    lof_F = ((m.ssr - sspe) / dflof) / (sspe / dfpe)
    fit_stats.append(dict(response=r, R2=m.rsquared, adjR2=m.rsquared_adj, RMSE=np.sqrt(m.mse_resid),
                          pure_error_SD=np.sqrt(sspe / dfpe), LOF_F=lof_F, LOF_df1=dflof, LOF_df2=dfpe,
                          LOF_p=1 - stats.f.cdf(lof_F, dflof, dfpe)))
pd.DataFrame(coef_rows).to_csv(TAB / "T18_additive_RSM_coefficients.csv", index=False)
pd.DataFrame(fit_stats).to_csv(TAB / "T18b_additive_RSM_fit.csv", index=False)

# COF stationary point (pooled model)
c = models["COF"].params
H = np.array([[2 * c["I(xP ** 2)"], c["xP:xV"], c["xP:xT"]], [c["xP:xV"], 2 * c["I(xV ** 2)"], c["xV:xT"]],
              [c["xP:xT"], c["xV:xT"], 2 * c["I(xT ** 2)"]]])
xs = np.linalg.solve(-H, np.array([c["xP"], c["xV"], c["xT"]]))
out["COF_optimum_pooled"] = dict(P_bar=10 + 5 * xs[0], V_rpm=480 + 180 * xs[1], T_C=250 + 100 * xs[2])

# ---- Table 19 predictions at mild / centre / severe
duty = {"mild": (5, 300, 150), "centre": (10, 500, 250), "severe": (15, 660, 350)}
rows = []
for f in FORMS:
    row = {"formulation": f}
    for r in ("COF", "wear"):
        for dn, (P, V, T) in duty.items():
            xP, xV, xT = code(P, V, T)
            nd = pd.DataFrame({"formulation": [f], "xP": xP, "xV": xV, "xT": xT})
            row[f"{r}_{dn}"] = float(models[r].predict(nd)[0])
    rows.append(row)
pd.DataFrame(rows).to_csv(TAB / "T19_predictions_duty_levels.csv", index=False)

# ---- Table 20: MR6-only full quadratic
quad = "{y} ~ xP + xV + xT + I(xP**2) + I(xV**2) + I(xT**2) + xP:xV + xP:xT + xV:xT"
d6 = df[df.formulation == "MR6"]
rows = []
for r in R6:
    m = smf.ols(quad.format(y=r), d6).fit()
    h = m.get_influence().hat_matrix_diag
    press = ((m.resid / (1 - h)) ** 2).sum()
    sst = ((d6[r] - d6[r].mean()) ** 2).sum()
    cell = d6.groupby("cond")[r]
    sspe = ((d6[r] - cell.transform("mean")) ** 2).sum()
    dfpe = len(d6) - cell.ngroups
    dflof = m.df_resid - dfpe
    F = ((m.ssr - sspe) / dflof) / (sspe / dfpe)
    rows.append(dict(response=r, R2=m.rsquared, adjR2=m.rsquared_adj, predR2=1 - press / sst,
                     RMSE=np.sqrt(m.mse_resid), LOF_p=1 - stats.f.cdf(F, dflof, dfpe),
                     LOF_df=f"{dflof},{dfpe}"))
    if r == "wear":
        sw = stats.shapiro(m.resid)
        out["MR6_wear_shapiro"] = dict(W=float(sw.statistic), p=float(sw.pvalue))
        d6 = d6.assign(wear_fit=m.fittedvalues, wear_res=m.resid)
        d6[["run_id", "wear_fit", "wear_res"]].to_csv(PROC / "MR6_wear_residuals.csv", index=False)
    if r == "COF":
        c6 = m.params
        H = np.array([[2 * c6["I(xP ** 2)"], c6["xP:xV"], c6["xP:xT"]],
                      [c6["xP:xV"], 2 * c6["I(xV ** 2)"], c6["xV:xT"]],
                      [c6["xP:xT"], c6["xV:xT"], 2 * c6["I(xT ** 2)"]]])
        x6 = np.linalg.solve(-H, np.array([c6["xP"], c6["xV"], c6["xT"]]))
        out["COF_optimum_MR6"] = dict(P_bar=10 + 5 * x6[0], V_rpm=480 + 180 * x6[1], T_C=250 + 100 * x6[2])
pd.DataFrame(rows).to_csv(TAB / "T20_MR6_quadratic_fit.csv", index=False)

# ---- Batch variability: mixed model ICC + Table 21 batch means
icc = {}
for r in R5:
    try:
        mm_ = smf.mixedlm(add.format(y=r), df, groups=df["batch_idx"].astype(str) + "_" + df.formulation).fit(reml=True)
        vg = float(mm_.cov_re.iloc[0, 0]); ve = float(mm_.scale)
        icc[r] = vg / (vg + ve)
    except Exception as e:  # singular fits are expected when the batch variance is ~0
        icc[r] = float("nan")
out["ICC_batch"] = icc
bm = df.groupby(["formulation", "batch_id"])[["COF", "wear"]].mean().unstack()
bm.columns = [f"{a}_{b}" for a, b in bm.columns]
bm.to_csv(TAB / "T21_batch_means.csv")

# ---- Spearman correlations (Sec. 3.6)
sp = df[R6].corr(method="spearman")
sp.to_csv(TAB / "T_spearman_responses.csv")

# ---- mass loss log-log exponents
ml = smf.ols("np.log(mass_loss_g) ~ np.log(pressure_bar) + np.log(speed_rpm) + np.log(temp_setpoint_C) + C(formulation)", df).fit()
out["massloss_loglog"] = dict(exp_P=ml.params["np.log(pressure_bar)"], exp_V=ml.params["np.log(speed_rpm)"],
                              R2=ml.rsquared)
out["interaction_tests_COF_wear_min_p"] = min(x[2] for x in inter_p)
json.dump(out, open(TAB / "stats_summary.json", "w"), indent=2, default=float)
print(json.dumps(out, indent=2, default=float))
print(t12[t12.response.isin(["COF", "wear"])].round(4).to_string())
print(pd.DataFrame(fit_stats).round(4).to_string())
print(pd.read_csv(TAB / "T20_MR6_quadratic_fit.csv").round(3).to_string())
