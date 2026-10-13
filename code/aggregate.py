"""Aggregate the grid (runs/main): mean +- sd over 5 seeds per dataset x variant x horizon, Diebold-Mariano tests of
GRANARY vs the zero-shot ensemble (absolute-error loss, per-origin cross-sectional mean, HAC variance with h-1 lags),
and a regional breakdown (World Bank regions; FAO by M49 -> ISO3, WFP by ISO3).
Writes results/agg_<ds>.csv, results/dm_<ds>.csv, results/region_<ds>.csv and results/summary.json
"""
import os, json, glob
import numpy as np
import pandas as pd
import pycountry
from evaluate import targets, metrics, period_mask
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
R, D, OUT = os.path.join(HERE, "..", "runs"), os.path.join(HERE, "..", "data"), os.path.join(HERE, "..", "results")
os.makedirs(OUT, exist_ok=True)
VAR = {"": "GRANARY", "_nodrv": "w/o drivers", "_nohist": "w/o history", "_noid": "w/o identity",
       "_onlychronos": "Chronos only", "_onlytimesfm": "TimesFM only"}
CFG = {"fao": dict(file="d1_fao.parquet", H=12, te=("2020-01-01", "2025-03-01")),
       "wfp": dict(file="d2_wfp.parquet", H=6, te=("2020-01-01", "2026-04-01"))}
wb = {c["id"]: c for c in json.load(open(os.path.join(os.environ.get("GRANARY_RAW", "raw_data"), "worldbank", "countries.json")))[1]}


def region_of(ds, meta):
    out = []
    for _, r in meta.iterrows():
        iso = r["country"] if ds == "wfp" else None
        if ds == "fao":
            try:
                iso = pycountry.countries.get(numeric=str(int(float(r["m49"]))).zfill(3)).alpha_3
            except Exception:
                iso = None
        out.append(wb[iso]["region"]["value"].strip() if iso in wb else "Other")
    return np.array(out)


def dm_test(e1, e2, h):
    """e1, e2: [S,O] absolute errors; loss differential averaged over series per origin, HAC (Newey-West, h-1 lags)."""
    d = np.nanmean(e1 - e2, axis=0); d = d[~np.isnan(d)]
    T = len(d); dbar = d.mean(); g0 = np.mean((d - dbar) ** 2)
    var = g0 + 2 * sum((1 - k / h) * np.mean((d[k:] - dbar) * (d[:-k] - dbar)) for k in range(1, h)) if h > 1 else g0
    stat = dbar / np.sqrt(var / T)
    return stat, 2 * (1 - stats.norm.cdf(abs(stat)))


summary = {}
for ds, cfg in CFG.items():
    W = pd.read_parquet(os.path.join(D, cfg["file"])); H = cfg["H"]
    meta = pd.read_csv(os.path.join(D, "series_meta.csv")); meta = meta[meta.dataset == ds].set_index("series").loc[list(W.columns)].reset_index()
    rows, dms = [], []
    zs = None
    for suf, vname in VAR.items():
        for f in sorted(glob.glob(os.path.join(R, "main", f"{ds}_s[0-9]{suf}", "result.json"))):
            if suf == "" and any(f.endswith(x + os.sep + "result.json") for x in ("_nodrv", "_nohist", "_noid", "_onlychronos", "_onlytimesfm")):
                continue
            r = json.load(open(f)); seed = int(os.path.basename(os.path.dirname(f)).split("_s")[1][0])
            for key, lab in (("granary_conf", vname),) + ((("zs_mean", "Zero-shot ensemble"), ("zs_chronos", "Chronos zero-shot"),
                                                            ("zs_timesfm", "TimesFM zero-shot"), ("granary", "GRANARY (raw intervals)"),
                                                            ("granary_static", "GRANARY + static CQR"), ("granary_roll36", "GRANARY + rolling CQR"))
                                                           if suf == "" else ()):
                for per in ("test", "surge"):
                    for m in r[key][per]:
                        rows.append(dict(variant=lab, seed=seed, period=per, **m))
    # classical statistical benchmarks (single deterministic run each, test origins only)
    Zc = np.load(os.path.join(R, f"zs_{ds}_chronos.npz"), allow_pickle=True); Zt = np.load(os.path.join(R, f"zs_{ds}_timesfm.npz"), allow_pickle=True)
    orig = Zc["origins"]; act, last = targets(W, orig, H)
    BM = {"snaive": "Seasonal naive", "ar": "AR (direct)", "ets": "ETS", "arima": "ARIMA"}
    bmq = {}
    for k, lab in BM.items():
        f = os.path.join(R, f"bm_{ds}_{k}.npz")
        if not os.path.exists(f):
            continue
        bmq[lab] = np.load(f)["q"]
        for per, (a0, a1) in (("test", cfg["te"]), ("surge", ("2021-07-01", "2022-12-01"))):
            for m in metrics(bmq[lab], act, last, period_mask(orig, len(W.columns), H, a0, a1)).to_dict("records"):
                rows.append(dict(variant=lab, seed=0, period=per, **m))
    df = pd.DataFrame(rows)
    agg = df.groupby(["variant", "period", "h"]).agg(["mean", "std"]).drop(columns=["seed", "n"], level=0)
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg.reset_index().to_csv(os.path.join(OUT, f"agg_{ds}.csv"), index=False)

    # DM tests and regional errors (point = median)
    o = pd.to_datetime([str(x) for x in orig]); te = (o >= pd.Timestamp(cfg["te"][0])) & (o <= pd.Timestamp(cfg["te"][1]))
    zsq = (Zc["q"] + Zt["q"]) / 2
    Fs = [np.load(os.path.join(R, "main", f"{ds}_s{s}", "forecasts.npz"))["q"] for s in range(5)
          if os.path.exists(os.path.join(R, "main", f"{ds}_s{s}", "forecasts.npz"))]
    gq = np.nanmean(np.stack(Fs), 0)                                        # seed-averaged GRANARY quantiles
    for h in range(H):
        e_g = np.abs(gq[:, te, h, 4] - last[:, te] - act[:, te, h]) * 100
        e_z = np.abs(zsq[:, te, h, 4] - last[:, te] - act[:, te, h]) * 100
        st, pv = dm_test(e_g, e_z, h + 1)
        row = dict(h=h + 1, mean_diff_pp=float(np.nanmean(e_g - e_z)), DM=float(st), p=float(pv))
        for lab, qb in bmq.items():                                         # GRANARY vs each classical benchmark
            e_b = np.abs(qb[:, te, h, 4] - last[:, te] - act[:, te, h]) * 100
            ok = ~np.isnan(e_b) & ~np.isnan(e_g)
            st2, pv2 = dm_test(np.where(ok, e_g, np.nan), np.where(ok, e_b, np.nan), h + 1)
            key = {"Seasonal naive": "sn", "AR (direct)": "ar", "ETS": "ets", "ARIMA": "arima"}[lab]
            row.update({f"d_{key}": float(np.nanmean(np.where(ok, e_g - e_b, np.nan))), f"DM_{key}": float(st2), f"p_{key}": float(pv2)})
        dms.append(row)
    pd.DataFrame(dms).to_csv(os.path.join(OUT, f"dm_{ds}.csv"), index=False)
    reg = region_of(ds, meta)
    rr = []
    for g in sorted(set(reg)):
        sm = (reg == g)
        for h in ([0, 2, 5, 11] if H == 12 else [0, 2, 5]):
            a = act[sm][:, te, h] * 100
            eg = np.abs(gq[sm][:, te, h, 4] - last[sm][:, te] - act[sm][:, te, h]) * 100
            ez = np.abs(zsq[sm][:, te, h, 4] - last[sm][:, te] - act[sm][:, te, h]) * 100
            rw = np.abs(a)
            rr.append(dict(region=g, n_series=int(sm.sum()), h=h + 1, relMAE_granary=np.nanmean(eg) / np.nanmean(rw),
                           relMAE_zs=np.nanmean(ez) / np.nanmean(rw)))
    pd.DataFrame(rr).to_csv(os.path.join(OUT, f"region_{ds}.csv"), index=False)
    summary[ds] = {"n_series": int(W.shape[1]), "n_seeds_full": len(Fs)}
    print(ds, "done; seeds", len(Fs))
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
