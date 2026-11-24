"""Generate all figures (PNG, 200 dpi) into results/figures/."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.formula.api as smf
from common import *

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.grid": True, "grid.alpha": .25,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100})
COL = plt.cm.viridis(np.linspace(0.05, 0.9, 7))
df = load_runs()
ts = pd.read_csv(RAW / "time_series_17577.csv")
WM = "SYNTHETIC DATA - not measured evidence"


def save(fig, name):
    fig.text(0.995, 0.005, WM, ha="right", va="bottom", fontsize=6, color="gray")
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


UN = {"COF": "COF (-)", "wear": "k (1e-4 mm$^3$/(N m))", "CV": "Friction CV (%)", "fade": "Fade (%)",
      "recovery": "Recovery (%)", "peakT": "Peak T (C)"}

# Fig 3 time series
r = df.merge(ts, on="run_id")
fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
sel = r[(r.pressure_bar == 10) & (r.speed_rpm == 500)]
for f, ls in (("MR0", "--"), ("MR6", "-")):
    for T, c in zip(TEMP, ["tab:blue", "tab:orange", "tab:red"]):
        s = sel[(sel.formulation == f) & (sel.temp_setpoint_C == T)].groupby("time_min")[["COF_t", "interface_T_C"]].mean()
        ax[0].plot(s.index, s.COF_t, ls, c=c, label=f"{f} {T}C")
        ax[1].plot(s.index, s.interface_T_C, ls, c=c)
ax[0].set(title="(a) COF, 10 bar, 500 rpm", xlabel="Time (min)", ylabel="COF"); ax[0].legend(fontsize=6, ncol=2)
ax[1].set(title="(b) Interface temperature", xlabel="Time (min)", ylabel="T (C)")
r["norm"] = r.COF_t / r.groupby("run_id").COF_t.transform(lambda s: s.iloc[:3].mean())
for T, c in zip(TEMP, ["tab:blue", "tab:orange", "tab:red"]):
    s = r[r.temp_setpoint_C == T].groupby("time_min").norm.agg(["mean", "std"])
    ax[2].plot(s.index, s["mean"], c=c, label=f"{T}C"); ax[2].fill_between(s.index, s["mean"] - s["std"], s["mean"] + s["std"], color=c, alpha=.15)
ax[2].set(title="(c) Normalised COF (all runs)", xlabel="Time (min)"); ax[2].legend(fontsize=7)
save(fig, "fig03_time_series")

# Fig 4 formulation means + 95% CI
fig, ax = plt.subplots(1, 5, figsize=(14, 2.8))
for a, rr in zip(ax, RESP):
    g = df.groupby("formulation")[rr].agg(["mean", "sem", "count"])
    a.errorbar(range(7), g["mean"], yerr=1.96 * g["sem"], fmt="o-", c="k", capsize=3)
    a.set(title=UN[rr], xticks=range(7), xticklabels=range(7), xlabel="MR index")
save(fig, "fig04_formulation_means")

# Fig 5 improvement vs MR0
t12 = pd.read_csv(TAB / "T12_contrasts_vs_MR0.csv")
fig, ax = plt.subplots(figsize=(8, 3.4))
w = .13
sign = {"COF": 1, "wear": -1, "CV": -1, "fade": -1, "recovery": 1}
for i, f in enumerate(FORMS[1:]):
    v = [sign[rr] * t12[(t12.response == rr) & (t12.contrast == f"{f} vs MR0")].pct.values[0] for rr in RESP]
    sg = [t12[(t12.response == rr) & (t12.contrast == f"{f} vs MR0")].p_holm.values[0] < .05 for rr in RESP]
    for j, (vv, s_) in enumerate(zip(v, sg)):
        ax.bar(j + (i - 2.5) * w, vv, w, color=COL[i + 1], hatch=None if s_ else "///", label=f if j == 0 else None)
ax.set(xticks=range(5), xticklabels=["COF", "Wear", "CV", "Fade", "Recovery"], ylabel="Improvement vs MR0 (%)")
ax.legend(ncol=6, fontsize=7); ax.axhline(0, c="k", lw=.5)
save(fig, "fig05_improvement_vs_MR0")

# Fig 6 main effects
fig, ax = plt.subplots(2, 3, figsize=(11, 6))
for i, rr in enumerate(["COF", "wear"]):
    for j, (col, lab) in enumerate((("pressure_bar", "Pressure (bar)"), ("speed_rpm", "Speed (rpm)"), ("temp_setpoint_C", "Setpoint (C)"))):
        for k, f in enumerate(FORMS):
            g = df[df.formulation == f].groupby(col)[rr].agg(["mean", "sem"])
            ax[i, j].errorbar(g.index, g["mean"], yerr=1.96 * g["sem"], marker="o", ms=3, c=COL[k], label=f, capsize=2)
        ax[i, j].set(xlabel=lab, ylabel=UN[rr])
ax[0, 0].legend(fontsize=6, ncol=2)
save(fig, "fig06_main_effects")

# Fig 7 relative change; Fig 8 marginal means
t14 = pd.read_csv(TAB / "T14_marginal_means.csv")
fig, ax = plt.subplots(figsize=(8, 3.4))
facs = t14.factor.unique(); resp7 = ["wear", "CV", "fade", "recovery", "COF"]
for i, fct in enumerate(facs):
    s = t14[t14.factor == fct].reset_index(drop=True)
    v = [100 * (s.loc[2, rr] / s.loc[0, rr] - 1) for rr in resp7]
    ax.bar(np.arange(5) + (i - 1) * .27, v, .27, label=fct)
ax.set(xticks=range(5), xticklabels=["Wear", "CV", "Fade", "Recovery", "COF"], ylabel="Change low -> high (%)")
ax.legend(); ax.axhline(0, c="k", lw=.5)
save(fig, "fig07_relative_change")

fig, ax = plt.subplots(1, 3, figsize=(11, 3))
for a, rr in zip(ax, ["CV", "fade", "recovery"]):
    for fct, mk in zip(facs, "osd"):
        s = t14[t14.factor == fct]
        a.plot(["low", "mid", "high"], s[rr], marker=mk, label=fct)
    a.set(title=UN[rr])
ax[0].legend(fontsize=6)
save(fig, "fig08_marginal_CV_fade_recovery")

# Fig 9 partial eta2 heatmap, Fig 10 variance share
t16 = pd.read_csv(TAB / "T16_anova_partial_eta2.csv", index_col=0)
H = t16[[f"{r}_eta2" for r in RESP]]; H.columns = RESP
fig, ax = plt.subplots(figsize=(6.5, 5))
im = ax.imshow(H.values, cmap="viridis", aspect="auto", vmin=0, vmax=.8)
ax.set_xticks(range(5)); ax.set_xticklabels(RESP); ax.set_yticks(range(len(H))); ax.set_yticklabels(H.index)
for i in range(H.shape[0]):
    for j in range(5):
        p = t16.iloc[i][f"{RESP[j]}_p"]; star = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
        ax.text(j, i, f"{H.values[i, j]:.2f}{star}", ha="center", va="center", color="w" if H.values[i, j] < .45 else "k", fontsize=7)
ax.grid(False); plt.colorbar(im, label="partial $\\eta^2$")
save(fig, "fig09_partial_eta2")

t17 = pd.read_csv(TAB / "T17_variance_share_pct.csv", index_col=0)
t17 = t17.drop("Model R2")
grp = pd.DataFrame({"Formulation": t17.loc["Formulation"], "Pressure": t17.loc["Pressure"], "Speed": t17.loc["Speed"],
                    "Temperature": t17.loc["Temperature"],
                    "Interactions": t17.loc[["P x V", "P x T", "V x T", "Formulation x T"]].sum(),
                    "Residual": t17.loc["Residual"]})
fig, ax = plt.subplots(figsize=(8, 3.2)); left = np.zeros(6)
for c, k in zip(plt.cm.tab10.colors, grp.index):
    ax.barh(grp.columns, grp.loc[k].values, left=left, color=c, label=k) if False else None
vals = grp.T
bottom = np.zeros(vals.shape[1])
fig, ax = plt.subplots(figsize=(8, 3.2))
for c, k in zip(plt.cm.tab10.colors, vals.index):
    ax.barh(vals.columns, vals.loc[k], left=bottom, color=c, label=k); bottom += vals.loc[k].values
ax.set(xlabel="Share of total sum of squares (%)"); ax.legend(ncol=3, fontsize=7, loc="lower center", bbox_to_anchor=(.5, 1))
save(fig, "fig10_variance_partition")

# Fig 11 response surfaces (additive model, 500 rpm)
add = "{y} ~ C(formulation) + xP + xV + xT + I(xP**2) + I(xV**2) + I(xT**2) + xP:xV + xP:xT + xV:xT"
mods = {r_: smf.ols(add.format(y=r_), df).fit() for r_ in RESP}
Pg, Tg = np.meshgrid(np.linspace(5, 15, 40), np.linspace(150, 350, 40))
fig, ax = plt.subplots(2, 2, figsize=(8.5, 6.5))
for i, rr in enumerate(["COF", "wear"]):
    vmin = vmax = None
    zs = []
    for f in ("MR0", "MR6"):
        nd = pd.DataFrame({"formulation": f, "xP": ((Pg - 10) / 5).ravel(), "xV": (500 - 480) / 180, "xT": ((Tg - 250) / 100).ravel()})
        zs.append(mods[rr].predict(nd).values.reshape(Pg.shape))
    lo, hi = min(z.min() for z in zs), max(z.max() for z in zs)
    for j, (f, z) in enumerate(zip(("MR0", "MR6"), zs)):
        cs = ax[i, j].contourf(Pg, Tg, z, 14, cmap="viridis", vmin=lo, vmax=hi)
        ax[i, j].set_title(f"{f} - {UN[rr]}", fontsize=9); ax[i, j].set_xlabel("Pressure (bar)", fontsize=8); ax[i, j].set_ylabel("Temperature (C)", fontsize=8); ax[i, j].tick_params(labelsize=7); ax[i, j].grid(False)
        plt.colorbar(cs, ax=ax[i, j])
fig.tight_layout(); save(fig, "fig11_response_surfaces")

# Fig 12 coefficients
co = pd.read_csv(TAB / "T18_additive_RSM_coefficients.csv")
fig, ax = plt.subplots(1, 2, figsize=(11, 5))
for a, rr in zip(ax, ["COF", "wear"]):
    s = co[(co.response == rr) & (co.term != "Intercept")].reset_index(drop=True)
    lab = s.term.str.replace("C(formulation)[T.", "", regex=False).str.replace("]", "", regex=False) \
        .str.replace("I(x", "x", regex=False).str.replace(" ** 2)", "^2", regex=False)
    a.errorbar(s.coef, range(len(s)), xerr=1.96 * s.se, fmt="o", c="k", capsize=2)
    a.set_yticks(range(len(s))); a.set_yticklabels(lab); a.axvline(0, c="r", lw=.6); a.invert_yaxis(); a.set_title(UN[rr])
save(fig, "fig12_RSM_coefficients")

# Fig 13 predictions at duty levels
t19 = pd.read_csv(TAB / "T19_predictions_duty_levels.csv")
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
for a, rr in zip(ax, ["COF", "wear"]):
    for i, f in enumerate(FORMS):
        a.plot(["mild", "centre", "severe"], t19.loc[i, [f"{rr}_mild", f"{rr}_centre", f"{rr}_severe"]], "o-", c=COL[i], label=f)
    a.set_ylabel(UN[rr])
ax[0].legend(fontsize=6, ncol=2)
save(fig, "fig13_predicted_duty_levels")

# Fig 14 MR6 wear residuals
res = pd.read_csv(PROC / "MR6_wear_residuals.csv")
fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
ax[0].scatter(res.wear_fit, res.wear_res, s=12); ax[0].axhline(0, c="r"); ax[0].set(xlabel="Fitted", ylabel="Residual")
stats.probplot(res.wear_res, plot=ax[1]); ax[2].hist(res.wear_res, bins=14); ax[2].set(xlabel="Residual")
save(fig, "fig14_MR6_wear_residuals")

# Fig 15 batch boxes
fig, ax = plt.subplots(1, 2, figsize=(12, 3.4))
for a, rr in zip(ax, ["COF", "wear"]):
    data, pos, cols = [], [], []
    for i, f in enumerate(FORMS):
        for j, b in enumerate(BATCHES):
            data.append(df[(df.formulation == f) & (df.batch_id == b)][rr]); pos.append(i * 4 + j)
    bp = a.boxplot(data, positions=pos, widths=.8, patch_artist=True, showfliers=False)
    for k, p in enumerate(bp["boxes"]):
        p.set_facecolor(COL[k // 3])
    a.set_xticks([i * 4 + 1 for i in range(7)]); a.set_xticklabels(FORMS); a.set_ylabel(UN[rr])
save(fig, "fig15_batch_boxplots")

# Fig 16 COF vs wear
cm = df.groupby(["formulation", "cond"])[["COF", "wear"]].mean().reset_index()
fig, ax = plt.subplots(figsize=(5.5, 4))
for i, f in enumerate(FORMS):
    s = cm[cm.formulation == f]; ax.scatter(s.COF, s.wear, s=14, color=COL[i], label=f)
ax.set(xlabel="COF", ylabel=UN["wear"]); ax.legend(fontsize=6, ncol=2)
save(fig, "fig16_COF_vs_wear")

# Fig 17/18 ML benchmark (needs 03)
try:
    res = pd.read_csv(TAB / "T23_ml_benchmark_all.csv")
    schemes = ["S1", "S2", "S3", "S4"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
    for a, tg in zip(ax, ["COF", "wear"]):
        s = res[res.target == tg].pivot(index="model", columns="scheme", values="R2")[schemes]
        s.clip(lower=-1).plot.bar(ax=a, width=.8); a.set_ylabel("Held-out R$^2$ (clipped at -1)"); a.set_title(UN[tg])
        a.axhline(0, c="k", lw=.5); a.tick_params(axis="x", rotation=40)
    save(fig, "fig17_ML_R2_by_scheme")
    fig, ax = plt.subplots(1, 5, figsize=(16, 3.8), sharey=True)
    for a, tg in zip(ax, RESP):
        s = res[res.target == tg].pivot(index="model", columns="scheme", values="R2")[schemes].clip(lower=-1)
        im = a.imshow(s.values, cmap="RdYlGn", vmin=-1, vmax=1); a.set_title(tg)
        a.set_xticks(range(4)); a.set_xticklabels(schemes); a.set_yticks(range(len(s))); a.set_yticklabels(s.index); a.grid(False)
        for i in range(s.shape[0]):
            for j in range(4):
                a.text(j, i, f"{s.values[i, j]:.2f}", ha="center", va="center", fontsize=6)
    save(fig, "fig18_ML_R2_heatmap")
    oof = pd.read_csv(PROC / "oof_predictions_COF_wear_S3_S4.csv")
    fig, ax = plt.subplots(2, 2, figsize=(8, 7))
    for i, tg in enumerate(["COF", "wear"]):
        for j, sc in enumerate(["S3", "S4"]):
            sub = res[(res.target == tg) & (res.scheme == sc)]
            bm = sub.loc[sub.R2.idxmax(), "model"]
            o = oof[(oof.target == tg) & (oof.scheme == sc) & (oof.model == bm)].merge(df[["run_id", "form_idx"]], on="run_id")
            ax[i, j].scatter(o.observed, o.predicted, c=o.form_idx, cmap="viridis", s=8)
            lim = [o.observed.min(), o.observed.max()]; ax[i, j].plot(lim, lim, "r--", lw=.8)
            ax[i, j].set(title=f"{tg} {sc}: {bm} (R2={sub.R2.max():.2f})", xlabel="Observed", ylabel="Predicted")
    save(fig, "fig19_pred_vs_obs")
except FileNotFoundError:
    print("ML results not found; skipping figs 17-19")

try:
    t26 = pd.read_csv(TAB / "T26_grouped_permutation_importance.csv")
    cols = ["Pressure", "Speed", "Temperature", "Composition block"]
    M = t26[[c + "_mean" for c in cols]].values
    fig, ax = plt.subplots(figsize=(6, 3.4)); im = ax.imshow(M, cmap="magma_r", aspect="auto"); ax.grid(False)
    ax.set_xticks(range(4)); ax.set_xticklabels(cols, rotation=20); ax.set_yticks(range(5)); ax.set_yticklabels(t26.response)
    for i in range(5):
        for j in range(4):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color="w" if M[i, j] > 3 else "k", fontsize=8)
    plt.colorbar(im, label="MSE ratio (1 = no effect)")
    save(fig, "fig20_grouped_permutation_importance")
    ug = pd.read_csv(TAB / "T_ungrouped_permutation_importance.csv")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for a, tg in zip(ax, ["wear", "COF"]):
        s = ug[ug.target == tg].sort_values("mean")
        a.barh(s.feature, s["mean"], xerr=s.sd, color=["tab:blue" if f in FEATS_OP else "gray" for f in s.feature]) \
            if (FEATS_OP := ["pressure_bar", "speed_rpm", "temp_setpoint_C"]) else None
        a.set_title(f"{tg} ({s.model.iloc[0]})")
    save(fig, "fig21_ungrouped_importance")
except FileNotFoundError:
    print("importance results not found")

# Fig 22/23 screening
try:
    g = pd.read_csv(PROC / "screening_grid_predictions.csv.gz")
    t27 = pd.read_csv(TAB / "T27_desirability_ranking.csv")
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    ax[0].axvspan(.44, .48, color="g", alpha=.12)
    sub = g.sample(6000, random_state=0)
    for i, f in enumerate(FORMS):
        s = sub[sub.formulation == f]; ax[0].scatter(s.COF_pred, s.wear_pred, s=3, color=COL[i], alpha=.4)
        b = g[g.formulation == f].sort_values("D").iloc[-1]; ax[0].scatter(b.COF_pred, b.wear_pred, marker="*", s=90, color=COL[i], edgecolor="k")
    ax[0].set(xlabel="Predicted COF", ylabel="Predicted wear", title="(a) Candidates; stars = optima")
    x = np.arange(7)
    ax[1].bar(x - .2, t27.mean_D_27_tested, .4, label="mean"); ax[1].bar(x + .2, t27.min_D_27_tested, .4, label="min", hatch="//")
    ax[1].set(xticks=x, xticklabels=FORMS, ylabel="D", title="(b) D over 27 tested conditions"); ax[1].legend()
    m6 = g[(g.formulation == "MR6") & (g.speed_rpm == 300)].pivot_table(index="temp_setpoint_C", columns="pressure_bar", values="D")
    im = ax[2].imshow(m6.values, origin="lower", aspect="auto", cmap="viridis", extent=[5, 15, 150, 350]); ax[2].grid(False)
    ax[2].set(title="(c) MR6 desirability, 300 rpm", xlabel="Pressure (bar)", ylabel="Temperature (C)"); plt.colorbar(im, ax=ax[2])
    save(fig, "fig22_screening")
    t28 = pd.read_csv(TAB / "T28_sensitivity.csv")
    fig, ax = plt.subplots(figsize=(9, 3.6)); x = np.arange(len(t28))
    ax.bar(x - .2, t28.D_MR6, .4, label="MR6"); ax.bar(x + .2, t28.D_MR0, .4, label="MR0")
    for i, tpf in enumerate(t28.top_formulation):
        ax.text(i, max(t28.D_MR6[i], t28.D_MR5[i]) + .02, f"top: {tpf}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels([s.replace(" (", "\n(") for s in t28.scenario], fontsize=6); ax.set_ylabel("D at own optimum"); ax.legend()
    save(fig, "fig23_sensitivity")
except FileNotFoundError:
    print("screening results not found")

# Fig 24 composition
fig, ax = plt.subplots(1, 2, figsize=(11, 3.4)); bottom = np.zeros(7)
for c, k in zip(plt.cm.Set2.colors, ML_COMP):
    ax[0].bar(FORMS, COMP[k], bottom=bottom, color=c, label=k); bottom += COMP[k].values
ax[0].legend(fontsize=7, ncol=3); ax[0].set_ylabel("wt%")
t12c = t12[(t12.response.isin(["COF", "wear"]))]
cp = t12c[t12c.response == "COF"].pct.values; wp = t12c[t12c.response == "wear"].pct.values
ax[1].bar(np.arange(6) - .2, cp, .4, label="COF increase (%)"); ax[1].bar(np.arange(6) + .2, -wp, .4, label="Wear reduction (%)")
ax[1].set_xticks(range(6)); ax[1].set_xticklabels(FORMS[1:]); ax[1].legend()
save(fig, "fig24_composition_vs_response")

# Fig 25/26 companion
comp = pd.read_csv(RAW / "companion_MR6_81.csv")
fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
for T, c in zip([175, 250, 350], ["tab:blue", "tab:orange", "tab:red"]):
    s = comp[comp.temp_setpoint_C == T]
    ax[0].scatter(s.tribofilm_coverage_pct, s.mean_COF, c=c, s=10, label=f"{T}C")
    ax[1].scatter(s.tribofilm_coverage_pct, s.fade_pct, c=c, s=10)
    ax[2].scatter(s.tribofilm_coverage_pct, s["specific_wear_rate_e-6"], c=c, s=10)
ax[0].set(xlabel="Tribofilm coverage (%)", ylabel="COF"); ax[1].set(xlabel="Coverage (%)", ylabel="Fade (%)")
ax[2].set(xlabel="Coverage (%)", ylabel="k (1e-6 mm3/(N m))"); ax[0].legend(fontsize=7)
save(fig, "fig26_companion_tribofilm")
try:
    t31 = pd.read_csv(TAB / "T31_companion_benchmark.csv")
    fig, ax = plt.subplots(figsize=(7, 3.4)); x = np.arange(len(t31))
    ax.bar(x - .2, t31.holdout_R2, .4, label="hold-out"); ax.bar(x + .2, t31.groupedCV_R2, .4, label="grouped CV")
    ax.set_xticks(x); ax.set_xticklabels(t31.model, rotation=20); ax.set_ylabel("R$^2$"); ax.legend()
    save(fig, "fig25_companion_benchmark")
except FileNotFoundError:
    pass
print("figures:", len(list(FIG.glob("*.png"))))
