"""Classical statistical benchmarks on the TEST origins (needs statsforecast; a separate virtual environment is recommended).

python benchmarks.py --ds fao --model arima      models: snaive | ar | ets | arima
Every model sees exactly the context the foundation models saw (log level up to origin o, gaps inside the context
linearly interpolated from past values only, at most 512 months) and returns 9 quantiles of the log level for
h = 1..H, stored as runs/bm_<ds>_<model>.npz with the same layout as the zero-shot files (NaN outside the test origins).
  snaive : y_{o+h-12} (seasonal no-change); quantiles = empirical quantiles of its own past h-step errors (same series)
  ar     : direct AR, Delta_{t,h} = a_h + sum_{j=1..12} b_{h,j} dy_{t-j+1} by OLS on the series' past; residual quantiles
  ets    : statsforecast AutoETS (season 12) on log level, Gaussian prediction intervals
  arima  : statsforecast AutoARIMA (season 12) on log level, Gaussian prediction intervals
"""
import argparse, os, time
import numpy as np
import pandas as pd
from multiprocessing import Pool

QL = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
HERE = os.path.dirname(os.path.abspath(__file__))
DATA, RUNS = os.path.join(HERE, "..", "data"), os.path.join(HERE, "..", "runs")
CFG = {"fao": dict(file="d1_fao.parquet", H=12, o0="2010-01-01", o1="2025-03-01", minctx=60, te0="2020-01-01"),
       "wfp": dict(file="d2_wfp.parquet", H=6, o0="2013-01-01", o1="2026-04-01", minctx=36, te0="2020-01-01")}


def context(col, t, minctx):
    c = col[:t + 1]
    if np.isnan(c[-1]) or (~np.isnan(c)).sum() < minctx:
        return None
    fv = np.argmax(~np.isnan(c)); c = c[fv:]
    return pd.Series(c).interpolate(limit_area="inside").values[-512:]


def fit_one(args):
    model, y, H = args
    try:
        if model == "snaive":
            n = len(y)
            out = np.full((H, 9), np.nan)
            for h in range(1, H + 1):
                if n < 13 + h:
                    continue
                point = y[-1] + (y[n - 1 - 12 + h] - y[n - 1 - 12]) if h <= 12 else np.nan
                # past h-step errors of the same rule, origins t with t + h <= n - 1
                ts = np.arange(12, n - h)
                err = (y[ts + h] - y[ts]) - (y[ts + h - 12] - y[ts - 12])
                err = err[~np.isnan(err)][-120:]
                if len(err) < 12:
                    continue
                out[h - 1] = point + np.quantile(err, QL) - np.median(err) * 0
            return out
        if model == "ar":
            dy = np.diff(y); n = len(y); p = 12
            out = np.full((H, 9), np.nan)
            for h in range(1, H + 1):
                ts = np.arange(p, n - h)                                  # t indexes y; uses dy[t-p..t-1]
                if len(ts) < 3 * p:
                    continue
                X = np.stack([dy[t - p:t][::-1] for t in ts]); Y = y[ts + h] - y[ts]
                X = np.column_stack([np.ones(len(ts)), X])
                beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
                res = Y - X @ beta
                x0 = np.concatenate([[1.0], dy[n - 1 - p:n - 1][::-1]])
                out[h - 1] = y[-1] + x0 @ beta + np.quantile(res, QL)
            return out
        from statsforecast.models import AutoARIMA, AutoETS
        if model == "arima_fast":                     # WFP panel: last 120 months + approximate likelihood in the order search
            m = AutoARIMA(season_length=12, approximation=True); y = y[-120:]
        else:
            m = AutoARIMA(season_length=12) if model == "arima" else AutoETS(season_length=12)
        f = m.forecast(y=y.astype(float), h=H, level=[20, 40, 60, 80])
        cols = ["lo-80", "lo-60", "lo-40", "lo-20", "mean", "hi-20", "hi-40", "hi-60", "hi-80"]
        return np.stack([f[c] for c in cols], 1)
    except Exception:
        return np.full((H, 9), np.nan)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ds", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--procs", type=int, default=11)
    a = ap.parse_args(); cfg = CFG[a.ds]
    if a.ds == "wfp" and a.model == "arima":          # full ARIMA took 6,582 s on 12,460 FAO contexts; WFP would take ~7 h
        a.model = "arima_fast"
    W = pd.read_parquet(os.path.join(DATA, cfg["file"]))
    origins = W.loc[cfg["o0"]:cfg["o1"]].index
    Y = W.values; S, O, H = Y.shape[1], len(origins), cfg["H"]
    idx = {d: i for i, d in enumerate(W.index)}
    jobs, keys = [], []
    for oi, o in enumerate(origins):
        if o < pd.Timestamp(cfg["te0"]):
            continue
        t = idx[o]
        for s in range(S):
            c = context(Y[:, s], t, cfg["minctx"])
            if c is not None:
                jobs.append((a.model, c, H)); keys.append((s, oi))
    print(f"{a.ds} {a.model}: {len(jobs)} test contexts", flush=True)
    t0 = time.time()
    q = np.full((S, O, H, 9), np.nan, np.float32)
    with Pool(a.procs) as pool:
        for (s, oi), r in zip(keys, pool.imap(fit_one, jobs, chunksize=64)):
            q[s, oi] = np.sort(r, axis=1)
    print(f"done in {time.time()-t0:.0f}s; failed {int(np.isnan(q[[k[0] for k in keys], [k[1] for k in keys], 0, 4]).sum())}", flush=True)
    np.savez_compressed(os.path.join(RUNS, f"bm_{a.ds}_{a.model.replace('_fast', '')}.npz"), q=q, origins=np.array([str(o.date()) for o in origins]),
                        series=np.array(W.columns))
