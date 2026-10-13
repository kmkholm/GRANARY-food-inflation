"""Figures 2-9 for the GRANARY paper (all numbers from runs/ and results/). Writes figures/FigureN.pdf/.png."""
import os, json, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from evaluate import targets, metrics, period_mask
from calibrate import static as cal_static, rolling as cal_rolling, aci as cal_aci

HERE = os.path.dirname(os.path.abspath(__file__))
R, D, RES, FIG = [os.path.join(HERE, "..", x) for x in ("runs", "data", "results", "figures")]
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False, "figure.dpi": 150})
C = {"Chronos zero-shot": "#c8811a", "TimesFM zero-shot": "#3b6ea5", "Zero-shot ensemble": "#8a8a8a", "GRANARY": "#1f7a4d",
     "w/o drivers": "#b23a48", "w/o history": "#e0a458", "w/o identity": "#7a5195", "Chronos only": "#d9a441", "TimesFM only": "#5c8fc4"}
CAL = {"raw": "#b23a48", "static CQR": "#e0a458", "rolling CQR": "#5b8e7d", "ACI (GRANARY)": "#1f7a4d"}
DS = {"fao": dict(file="d1_fao.parquet", H=12, te=("2020-01-01", "2025-03-01"), title="FAOSTAT consumer food price index"),
      "wfp": dict(file="d2_wfp.parquet", H=6, te=("2020-01-01", "2026-04-01"), title="WFP retail market prices")}
SURGE = ("2021-07-01", "2022-12-01")


def save(fig, n):
    fig.savefig(os.path.join(FIG, f"Figure{n}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, f"Figure{n}.png"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def label(ax, s):
    ax.text(-0.16, 1.08, s, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")


# ---------------- Figure 2: data ----------------
W1 = pd.read_parquet(os.path.join(D, "d1_fao.parquet")); W2 = pd.read_parquet(os.path.join(D, "d2_wfp.parquet"))
cov = pd.read_parquet(os.path.join(D, "covariates.parquet"))
fig = plt.figure(figsize=(11, 3.4)); gs = gridspec.GridSpec(1, 3, wspace=0.32)
ax = fig.add_subplot(gs[0]); y = (W1 - W1.shift(12)) * 100
med, lo, hi = y.median(1), y.quantile(0.25, 1), y.quantile(0.75, 1)
ax.fill_between(y.index, lo, hi, color="#1f7a4d", alpha=0.18, lw=0, label="interquartile range")
ax.plot(y.index, med, color="#1f7a4d", lw=1.6, label="median country")
y2 = (W2 - W2.shift(12)) * 100
ax.plot(y2.index, y2.median(1), color="#c8811a", lw=1.1, label="median WFP series")
ax.axvspan(pd.Timestamp(SURGE[0]), pd.Timestamp(SURGE[1]), color="#b23a48", alpha=0.08)
ax.set_ylabel("12-month food price change (%)"); ax.set_ylim(-5, 40); ax.legend(loc="upper left", fontsize=8)
ax.set_title("Food price inflation, 198 countries"); label(ax, "a")
ax = fig.add_subplot(gs[1])
for k, c in (("oil", "#2d2d2d"), ("gas_eu", "#7a5195"), ("wheat", "#c8811a"), ("maize", "#e0a458"), ("urea", "#3b6ea5"), ("potash", "#5b8e7d")):
    s = cov[k].loc["2005-01-01":]; ax.plot(s.index, (s - s.loc["2015-01-01":"2015-12-01"].mean()) * 100, lw=1.1, color=c, label=k.replace("gas_eu", "EU gas"))
ax.axvspan(pd.Timestamp(SURGE[0]), pd.Timestamp(SURGE[1]), color="#b23a48", alpha=0.08)
ax.set_ylabel("log price vs 2015 mean (x100)"); ax.legend(ncol=2, fontsize=7.5, loc="upper left"); ax.set_title("Selected global drivers"); label(ax, "b")
ax = fig.add_subplot(gs[2])
meta = pd.read_csv(os.path.join(D, "series_meta.csv")); mw = meta[meta.dataset == "wfp"]
cat = mw.commodity.str.split(" ").str[0].str.replace(r"[(),]", "", regex=True)
top = cat.value_counts().head(12)
ax.barh(top.index[::-1], top.values[::-1], color="#c8811a"); ax.set_xlabel("number of WFP country series")
ax.set_title(f"WFP panel: {len(mw)} series, {mw.country.nunique()} countries"); label(ax, "c")
save(fig, 2)

# ---------------- load aggregated results ----------------
agg = {ds: pd.read_csv(os.path.join(RES, f"agg_{ds}.csv")) for ds in DS}

# ---------------- Figure 3: accuracy by horizon ----------------
fig, axs = plt.subplots(2, 2, figsize=(10, 6.4)); plt.subplots_adjust(hspace=0.38, wspace=0.25)
for j, ds in enumerate(DS):
    a = agg[ds]; a = a[a.period == "test"]
    for i, (met, yl) in enumerate((("relMAE_RW", "MAE relative to no-change forecast"), ("pinball", "mean pinball loss (pp)"))):
        ax = axs[i, j]
        for v, col, ls in (("Seasonal naive", "#bdbdbd", "--"), ("ETS", "#7f6f5f", "--"), ("ARIMA", "#4d4d4d", "--"),
                           ("AR (direct)", "#111111", "--")):
            b = a[a.variant == v].sort_values("h")
            if len(b):
                ax.plot(b.h, b[f"{met}_mean"], ls=ls, marker="s", ms=2.5, lw=1.0, color=col, label=v)
        for v in ("Chronos zero-shot", "TimesFM zero-shot", "Zero-shot ensemble", "GRANARY"):
            b = a[a.variant == v].sort_values("h")
            ax.plot(b.h, b[f"{met}_mean"], marker="o", ms=3.5, lw=1.6 if v == "GRANARY" else 1.1, color=C[v], label=v)
            if v == "GRANARY":
                ax.fill_between(b.h, b[f"{met}_mean"] - b[f"{met}_std"], b[f"{met}_mean"] + b[f"{met}_std"], color=C[v], alpha=0.2, lw=0)
        if met == "relMAE_RW":
            ax.axhline(1, color="#444", lw=0.7, ls=":")
        ax.set_xlabel("forecast horizon h (months)"); ax.set_ylabel(yl)
        if i == 0:
            ax.set_title(DS[ds]["title"], pad=14)
        label(ax, "abcd"[2 * i + j])
axs[0, 0].legend(fontsize=6.5, ncol=2, loc="upper right")
save(fig, 3)

# ---------------- Figure 4: interval coverage by calibrator ----------------
fig, axs = plt.subplots(2, 2, figsize=(10, 6.2)); plt.subplots_adjust(hspace=0.4, wspace=0.25)
names = {"GRANARY (raw intervals)": "raw", "GRANARY + static CQR": "static CQR", "GRANARY + rolling CQR": "rolling CQR", "GRANARY": "ACI (GRANARY)"}
for j, ds in enumerate(DS):
    for i, per in enumerate(("test", "surge")):
        ax = axs[i, j]; a = agg[ds]; a = a[a.period == per]
        z = a[a.variant == "Zero-shot ensemble"].sort_values("h")
        ax.plot(z.h, z.cov80_mean, color=C["Zero-shot ensemble"], lw=1.1, ls="--", label="zero-shot ensemble")
        for v, lab in names.items():
            b = a[a.variant == v].sort_values("h")
            ax.plot(b.h, b.cov80_mean, marker="o", ms=3, lw=1.5 if lab.startswith("ACI") else 1.0, color=CAL[lab], label=lab)
        ax.axhline(0.8, color="#222", lw=0.8, ls=":"); ax.set_ylim(0.3, 1.0)
        ax.set_xlabel("forecast horizon h (months)"); ax.set_ylabel("coverage of the 80 % interval")
        ax.set_title(("Test 2020 onwards" if per == "test" else "Surge 2021-07 to 2022-12") + f", {ds.upper()}", pad=10)
        label(ax, "abcd"[2 * i + j])
axs[0, 0].legend(fontsize=7.5, loc="lower left")
save(fig, 4)

# ---------------- Figure 5: performance through time (seed 0) ----------------
fig, axs = plt.subplots(2, 2, figsize=(11, 5.8)); plt.subplots_adjust(hspace=0.35, wspace=0.22)
for j, ds in enumerate(DS):
    cfg = DS[ds]; W = pd.read_parquet(os.path.join(D, cfg["file"])); H = cfg["H"]; hsel = H - 1
    Zc = np.load(os.path.join(R, f"zs_{ds}_chronos.npz"), allow_pickle=True); Zt = np.load(os.path.join(R, f"zs_{ds}_timesfm.npz"), allow_pickle=True)
    orig = Zc["origins"]; act, last = targets(W, orig, H); o = pd.to_datetime([str(x) for x in orig])
    F = np.load(os.path.join(R, "main", f"{ds}_s0", "forecasts.npz"))
    zs = (Zc["q"] + Zt["q"]) / 2
    for q, lab, c in ((zs, "Zero-shot ensemble", C["Zero-shot ensemble"]), (F["q"], "GRANARY", C["GRANARY"])):
        e = np.abs(q[:, :, hsel, 4] - last - act[:, :, hsel]) * 100
        axs[0, j].plot(o, np.nanmean(e, 0), color=c, lw=1.3, label=lab)
    rw = np.nanmean(np.abs(act[:, :, hsel]) * 100, 0)
    axs[0, j].plot(o, rw, color="#444", lw=0.9, ls=":", label="no-change forecast")
    axs[0, j].set_ylabel(f"MAE of {H}-month change (pp)"); axs[0, j].set_title(f"{cfg['title']}, h = {H}")
    for q, lab, c in ((F["q"], "raw", CAL["raw"]), (F["qconf"], "ACI (GRANARY)", CAL["ACI (GRANARY)"]), (zs, "zero-shot ensemble", C["Zero-shot ensemble"])):
        a_ = act[:, :, hsel] + last
        cv = np.nanmean(np.where(np.isnan(a_), np.nan, (a_ >= q[:, :, hsel, 0]) & (a_ <= q[:, :, hsel, 8])), 0)
        cv = pd.Series(cv, index=o).rolling(6, min_periods=3).mean()
        axs[1, j].plot(o, cv, color=c, lw=1.3 if lab.startswith("ACI") else 1.0, label=lab)
    axs[1, j].axhline(0.8, color="#222", lw=0.8, ls=":"); axs[1, j].set_ylim(0, 1.02)
    axs[1, j].set_ylabel("80 % coverage (6-origin mean)")
    for ax in axs[:, j]:
        ax.axvspan(pd.Timestamp(cfg["te"][0]), o.max(), color="#1f7a4d", alpha=0.04)
        ax.axvspan(pd.Timestamp(SURGE[0]), pd.Timestamp(SURGE[1]), color="#b23a48", alpha=0.08)
    label(axs[0, j], "ab"[j]); label(axs[1, j], "cd"[j])
axs[0, 0].legend(fontsize=8); axs[1, 0].legend(fontsize=8, loc="lower left")
save(fig, 5)

# ---------------- Figure 6: ablations ----------------
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6)); plt.subplots_adjust(wspace=0.3)
order = ["GRANARY", "w/o drivers", "w/o history", "w/o identity", "Chronos only", "TimesFM only", "Zero-shot ensemble"]
for j, ds in enumerate(DS):
    a = agg[ds]; a = a[a.period == "test"]; hs = [3, 6, 12] if ds == "fao" else [1, 3, 6]
    w = 0.12
    for k, v in enumerate(order):
        b = a[(a.variant == v) & a.h.isin(hs)].sort_values("h")
        axs[j].bar(np.arange(len(hs)) + (k - 3) * w, b.pinball_mean, w, yerr=b.pinball_std.fillna(0), color=C.get(v, "#8a8a8a"),
                   label=v, error_kw=dict(lw=0.6, capsize=1.5))
    axs[j].set_xticks(range(len(hs))); axs[j].set_xticklabels([f"h = {h}" for h in hs]); axs[j].set_ylabel("mean pinball loss (pp)")
    axs[j].set_title(DS[ds]["title"]); label(axs[j], "ab"[j])
axs[1].legend(fontsize=7.5, ncol=2, loc="upper left")
save(fig, 6)

# ---------------- Figure 7: driver attribution ----------------
fig = plt.figure(figsize=(11, 4.6)); gs = gridspec.GridSpec(1, 3, width_ratios=[1.1, 1.1, 1.3], wspace=0.55)
PER = {"2020-01..2021-06": ("2020-01-01", "2021-06-01"), "2021-07..2022-12": SURGE, "2023": ("2023-01-01", "2023-12-01"), "2024 onwards": ("2024-01-01", "2026-12-01")}
for j, ds in enumerate(DS):
    A = [np.load(f, allow_pickle=True) for f in sorted(glob.glob(os.path.join(R, "main", f"{ds}_s[0-9]", "attribution.npz")))]
    attr = np.mean([a["attr"] for a in A], 0); a0 = A[0]; drivers = [d.replace("gas_eu", "EU gas").replace("_", " ") for d in a0["drivers"]]
    o = pd.to_datetime([str(x) for x in a0["origins"]])[a0["pairs"][:, 1]]; h = DS[ds]["H"] - 1
    M = np.array([[attr[(o >= p0) & (o <= p1), h, k].mean() for p0, p1 in PER.values()] for k in range(len(drivers))])
    ax = fig.add_subplot(gs[j]); vm = np.abs(M).max()
    im = ax.imshow(M, cmap="BrBG_r", vmin=-vm, vmax=vm, aspect="auto")
    ax.set_yticks(range(len(drivers))); ax.set_yticklabels(drivers); ax.set_xticks(range(len(PER))); ax.set_xticklabels(PER.keys(), rotation=35, ha="right", fontsize=8)
    for (r, c), v in np.ndenumerate(M):
        ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color="#111")
    ax.set_title(f"{ds.upper()}: mean contribution, h = {h + 1} (pp)"); label(ax, "ab"[j])
reg = pd.read_csv(os.path.join(RES, "region_fao.csv")); reg = reg[reg.h.isin([3, 6, 12])]
ABBR = {"Middle East, North Africa, Afghanistan & Pakistan": "MENAAP", "Europe & Central Asia": "Europe & C. Asia", "Latin America & Caribbean": "Latin Am. & Carib.", "Sub-Saharan Africa": "Sub-Saharan Africa", "East Asia & Pacific": "E. Asia & Pacific", "Other": "Other territories"}
reg["region"] = reg["region"].map(lambda r: ABBR.get(r, r))
ax = fig.add_subplot(gs[2]); rr = reg.pivot(index="region", columns="h", values="relMAE_granary"); rz = reg.pivot(index="region", columns="h", values="relMAE_zs")
yy = np.arange(len(rr))
for k, h in enumerate([3, 6, 12]):
    ax.scatter(rz[h], yy + (k - 1) * 0.2, marker="|", s=60, color="#8a8a8a")
    ax.scatter(rr[h], yy + (k - 1) * 0.2, s=18, color=["#e0a458", "#5b8e7d", "#1f7a4d"][k], label=f"GRANARY h = {h}")
    for y_, a_, b_ in zip(yy + (k - 1) * 0.2, rz[h], rr[h]):
        ax.plot([a_, b_], [y_, y_], color="#bbbbbb", lw=0.7, zorder=0)
ax.set_yticks(yy); ax.set_yticklabels(rr.index, fontsize=8); ax.axvline(1, color="#444", lw=0.7, ls=":")
ax.set_xlabel("relMAE (grey tick: zero-shot ensemble)"); ax.set_title("FAO: gain by World Bank region"); ax.legend(fontsize=7.5); label(ax, "c")
save(fig, 7)

# ---------------- Figure 8: example fan charts ----------------
W = W1; Zc = np.load(os.path.join(R, "zs_fao_chronos.npz"), allow_pickle=True); orig = Zc["origins"]; o = pd.to_datetime([str(x) for x in orig])
F = np.load(os.path.join(R, "main", "fao_s0", "forecasts.npz"))
cands = ["Egypt", "Türkiye", "Kenya", "Germany", "Pakistan", "Brazil"]
cands = [c for c in cands if c in W.columns][:4]
fig, axs = plt.subplots(1, len(cands), figsize=(12, 3.0)); plt.subplots_adjust(wspace=0.3)
for ax, c in zip(np.atleast_1d(axs), cands):
    s = list(W.columns).index(c); oi = int(np.where(o == pd.Timestamp("2021-12-01"))[0][0])
    hist_ = W[c].loc["2019-01-01":"2023-06-01"]; base = W[c].loc["2021-12-01"]
    ax.plot(hist_.index, (hist_ - base) * 100, color="#222", lw=1.2, label="observed")
    fx = pd.date_range("2022-01-01", periods=12, freq="MS")
    for q, col, lab in ((F["qconf"][s, oi], C["GRANARY"], "GRANARY + ACI"), ((Zc["q"][s, oi] + np.load(os.path.join(R, "zs_fao_timesfm.npz"))["q"][s, oi]) / 2, C["Zero-shot ensemble"], "zero-shot ensemble")):
        ax.fill_between(fx, (q[:, 0] - base) * 100, (q[:, 8] - base) * 100, color=col, alpha=0.22, lw=0)
        ax.plot(fx, (q[:, 4] - base) * 100, color=col, lw=1.4, label=lab)
    ax.axvline(pd.Timestamp("2021-12-01"), color="#888", lw=0.7, ls=":"); ax.set_title(c); ax.set_ylabel("log change vs Dec-2021 (x100)")
    ax.tick_params(axis="x", labelrotation=30, labelsize=7)
np.atleast_1d(axs)[0].legend(fontsize=7.5, loc="upper left")
save(fig, 8)

# ---------------- Figure 9: training curves, learned model weights, t-SNE ----------------
from sklearn.manifold import TSNE
fig = plt.figure(figsize=(12, 3.6)); gs = gridspec.GridSpec(1, 3, wspace=0.32)
ax = fig.add_subplot(gs[0])
for ds, c in (("fao", "#1f7a4d"), ("wfp", "#c8811a")):
    for s in range(5):
        f = os.path.join(R, "main", f"{ds}_s{s}", "hist.json")
        if os.path.exists(f):
            h = json.load(open(f)); ax.plot(np.array(h["train"]) / h["train"][0], color=c, lw=0.8, alpha=0.6, label=f"{ds.upper()} train" if s == 0 else None)
            ax.plot(np.array(h["val"]) / h["train"][0], color=c, lw=0.8, ls="--", alpha=0.9, label=f"{ds.upper()} validation" if s == 0 else None)
ax.set_xlabel("epoch"); ax.set_ylabel("pinball loss / epoch-0 train loss"); ax.legend(fontsize=7.5); ax.set_title("Training curves (5 seeds)"); label(ax, "a")
ax = fig.add_subplot(gs[1])
for ds, c in (("fao", "#1f7a4d"), ("wfp", "#c8811a")):
    ws = [np.load(os.path.join(R, "main", f"{ds}_s{s}", "forecasts.npz"))["w"] for s in range(5) if os.path.exists(os.path.join(R, "main", f"{ds}_s{s}", "forecasts.npz"))]
    w = np.nanmean(np.stack(ws), 0)[..., 0]                               # weight on Chronos
    m = np.nanmean(w.reshape(-1, w.shape[-1]), 0)
    ax.plot(np.arange(1, len(m) + 1), m, marker="o", ms=3.5, color=c, label=ds.upper())
ax.axhline(0.5, color="#444", lw=0.7, ls=":"); ax.set_ylim(0, 1); ax.set_xlabel("forecast horizon h (months)")
ax.set_ylabel("learned weight on Chronos-Bolt"); ax.set_title("Foundation-model weighting"); ax.legend(fontsize=8); label(ax, "b")
ax = fig.add_subplot(gs[2])
fe = np.load(os.path.join(R, "main", "fao_s0", "features.npz"), allow_pickle=True)
emb = TSNE(2, perplexity=40, random_state=0, init="pca").fit_transform(fe["feats"])
import pycountry
wb = {c["id"]: c["region"]["value"].strip() for c in json.load(open(os.path.join(os.environ.get("GRANARY_RAW", "raw_data"), "worldbank", "countries.json")))[1]}
mm = pd.read_csv(os.path.join(D, "series_meta.csv")); mm = mm[mm.dataset == "fao"].set_index("series").loc[list(W1.columns)]
def reg_of(m49):
    try:
        return wb.get(pycountry.countries.get(numeric=str(int(float(m49))).zfill(3)).alpha_3, "Other")
    except Exception:
        return "Other"
Zc0 = np.load(os.path.join(R, "zs_fao_chronos.npz"), allow_pickle=True)
act0, last0 = targets(W1, Zc0["origins"], 12)
infl = act0[fe["pairs"][:, 0], fe["pairs"][:, 1], 11] * 100
bins = [(-np.inf, 5, "< 5 %"), (5, 10, "5-10 %"), (10, 20, "10-20 %"), (20, np.inf, "> 20 %"), (None, None, "not yet observed")]
pal = ["#3b6ea5", "#5b8e7d", "#e0a458", "#b23a48", "#cccccc"]
for (a_, b_, lab), col in zip(bins, pal):
    m = np.isnan(infl) if a_ is None else (infl >= a_) & (infl < b_)
    ax.scatter(emb[m, 0], emb[m, 1], s=3, color=col, label=lab, alpha=0.75)
ax.set_xticks([]); ax.set_yticks([]); ax.set_title("t-SNE by realised inflation (FAO)"); ax.legend(fontsize=7, markerscale=3, loc="best"); label(ax, "c")
save(fig, 9)
print("figures written")
