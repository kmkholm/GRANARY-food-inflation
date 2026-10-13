"""Real-data thumbnails for Figure 1 (transparent PNGs embedded into the draw.io diagram)."""
import os, json, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); B = os.path.join(HERE, "..")
D, R = os.path.join(B, "data"), os.path.join(B, "runs")
OUT = os.path.join(HERE, "thumbs"); os.makedirs(OUT, exist_ok=True)
INK, GREEN, AMBER, BLUE, RED, PLUM, GREY = "#1d2b36", "#1f7a4d", "#c8811a", "#3b6ea5", "#b23a48", "#7a3b80", "#9a9a9a"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})


def clean(ax, keepx=False):
    for s in ("top", "right", "left") + (() if keepx else ("bottom",)):
        ax.spines[s].set_visible(False)
    ax.set_yticks([])
    if not keepx:
        ax.set_xticks([])
    ax.patch.set_alpha(0)


def save(fig, n):
    fig.savefig(os.path.join(OUT, n), dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.02); plt.close(fig)


W1 = pd.read_parquet(os.path.join(D, "d1_fao.parquet")); W2 = pd.read_parquet(os.path.join(D, "d2_wfp.parquet"))
cov = pd.read_parquet(os.path.join(D, "covariates.parquet"))

# 1. FAOSTAT food CPI, 12-month inflation, three countries
fig, ax = plt.subplots(figsize=(1.9, 0.9))
for c, col in (("Egypt", RED), ("Kenya", AMBER), ("Germany", BLUE)):
    y = ((W1[c] - W1[c].shift(12)) * 100).loc["2016-01-01":]
    ax.plot(y.index, y, color=col, lw=1.0, label=c)
ax.axhline(0, color=GREY, lw=0.4); clean(ax); ax.legend(fontsize=5.5, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, handlelength=1, borderaxespad=0, columnspacing=0.8)
save(fig, "fao.png")

# 2. WFP retail price sparklines
fig, ax = plt.subplots(figsize=(1.9, 0.9))
cols = [c for c in W2.columns if c.startswith(("KEN|Maize", "NGA|Rice", "AFG|Wheat"))][:3]
for c, col in zip(cols, (GREEN, AMBER, PLUM)):
    s = W2[c].loc["2016-01-01":]; ax.plot(s.index, s - s.dropna().iloc[0], color=col, lw=0.9)
clean(ax); save(fig, "wfp.png")

# 3. global drivers
fig, ax = plt.subplots(figsize=(1.9, 0.9))
for k, col in (("gas_eu", PLUM), ("urea", BLUE), ("oil", INK), ("wheat", AMBER)):
    s = cov[k].loc["2016-01-01":]; ax.plot(s.index, s - s.iloc[0], color=col, lw=0.9, label={"gas_eu": "EU gas"}.get(k, k))
clean(ax); ax.legend(fontsize=5.5, frameon=False, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0), handlelength=1, borderaxespad=0, columnspacing=0.6)
save(fig, "drivers.png")

# 4. zero-shot fans (Egypt, origin 2021-12) for each foundation model
Zc = np.load(os.path.join(R, "zs_fao_chronos.npz"), allow_pickle=True); Zt = np.load(os.path.join(R, "zs_fao_timesfm.npz"), allow_pickle=True)
o = pd.to_datetime([str(x) for x in Zc["origins"]]); oi = int(np.where(o == pd.Timestamp("2021-12-01"))[0][0])
s = list(W1.columns).index("Egypt"); base = W1["Egypt"].loc["2021-12-01"]
hist = (W1["Egypt"].loc["2020-06-01":"2021-12-01"] - base) * 100
fx = pd.date_range("2022-01-01", periods=12, freq="MS")
for name, Z, col in (("fan_chronos.png", Zc, AMBER), ("fan_timesfm.png", Zt, BLUE)):
    q = (Z["q"][s, oi] - base) * 100
    fig, ax = plt.subplots(figsize=(1.6, 0.85))
    ax.plot(hist.index, hist, color=INK, lw=0.9)
    for lo, hi, a in ((0, 8, 0.18), (2, 6, 0.3)):
        ax.fill_between(fx, q[:, lo], q[:, hi], color=col, alpha=a, lw=0)
    ax.plot(fx, q[:, 4], color=col, lw=1.1); ax.axvline(pd.Timestamp("2021-12-01"), color=GREY, lw=0.5, ls=":")
    ax.set_ylim(min(hist.min(), q.min()) - 2, max(30, q.max()) + 2); clean(ax); save(fig, name)

# 5. cross-attention map (FAO seed 0, test pairs): horizon x memory tokens
A = np.load(os.path.join(R, "main", "fao_s0", "attribution.npz"), allow_pickle=True)
M = A["attw"].mean(0)                                                    # H x (1 + 12)
fig, ax = plt.subplots(figsize=(1.9, 1.15))
ax.imshow(M.T, cmap="YlGnBu", aspect="auto")
ax.set_yticks(range(13)); ax.set_yticklabels(["history"] + [d.replace("gas_eu", "EU gas").replace("_", " ") for d in A["drivers"]], fontsize=4.5)
ax.set_xticks([0, 5, 11]); ax.set_xticklabels(["h=1", "h=6", "h=12"], fontsize=4.5); ax.tick_params(length=0)
for sp in ax.spines.values():
    sp.set_visible(False)
save(fig, "attention.png")

# 6. GRANARY + ACI fan for Egypt
F = np.load(os.path.join(R, "main", "fao_s0", "forecasts.npz"))
q = (F["qconf"][s, oi] - base) * 100; obs = (W1["Egypt"].loc["2020-06-01":"2022-12-01"] - base) * 100
fig, ax = plt.subplots(figsize=(1.9, 0.95))
ax.plot(obs.index, obs, color=INK, lw=0.9)
ax.fill_between(fx, q[:, 0], q[:, 8], color=GREEN, alpha=0.22, lw=0); ax.plot(fx, q[:, 4], color=GREEN, lw=1.2)
ax.axvline(pd.Timestamp("2021-12-01"), color=GREY, lw=0.5, ls=":"); clean(ax); save(fig, "fan_granary.png")

# 7. ACI coverage through time (FAO, h = 6), raw vs ACI
from evaluate import targets
act, last = targets(W1, Zc["origins"], 12); h = 5
fig, ax = plt.subplots(figsize=(1.9, 0.85))
for qq, col, lw in ((F["q"], RED, 0.8), (F["qconf"], GREEN, 1.2)):
    a_ = act[:, :, h] + last
    cv = np.nanmean(np.where(np.isnan(a_), np.nan, (a_ >= qq[:, :, h, 0]) & (a_ <= qq[:, :, h, 8])), 0)
    cv = pd.Series(cv, index=o).rolling(6, min_periods=3).mean().loc["2019-01-01":]
    ax.plot(cv.index, cv, color=col, lw=lw)
ax.axhline(0.8, color=INK, lw=0.6, ls=":"); ax.set_ylim(0.2, 1.0); clean(ax); save(fig, "coverage.png")

# 8. driver attribution in the surge (FAO, h = 12, mean of 5 seeds)
import glob
AA = [np.load(f, allow_pickle=True) for f in sorted(glob.glob(os.path.join(R, "main", "fao_s[0-9]", "attribution.npz")))]
attr = np.mean([a["attr"] for a in AA], 0); oo = pd.to_datetime([str(x) for x in AA[0]["origins"]])[AA[0]["pairs"][:, 1]]
m = (oo >= "2021-07-01") & (oo <= "2022-12-01"); v = attr[m, 11].mean(0)
names = [d.replace("gas_eu", "EU gas").replace("_", " ") for d in AA[0]["drivers"]]
order = np.argsort(v)[::-1][:6]
fig, ax = plt.subplots(figsize=(1.9, 1.0))
ax.barh(range(len(order))[::-1], v[order], color=[PLUM if names[i] in ("EU gas", "oil") else BLUE if names[i] in ("urea", "dap", "potash") else AMBER for i in order], height=0.7)
ax.set_yticks(range(len(order))[::-1]); ax.set_yticklabels([names[i] for i in order], fontsize=5.5)
ax.tick_params(length=0); ax.set_xticks([])
for sp in ("top", "right", "bottom"):
    ax.spines[sp].set_visible(False)
ax.patch.set_alpha(0); save(fig, "attribution.png")
print("thumbs ok")
