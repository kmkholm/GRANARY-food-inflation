"""Zero-shot rolling-origin forecasts of the two time-series foundation models (needs chronos-forecasting and timesfm).

python zeroshot.py --ds fao --model chronos      (models: chronos = amazon/chronos-bolt-base, timesfm = google/timesfm-2.5-200m-pytorch)
For every series s and every forecast origin o (last observed month), the model sees the series up to o (levels, gaps
inside the context filled by linear interpolation of past values only) and returns 9 quantiles for h = 1..H.
Output runs/zs_<ds>_<model>.npz : q [S, O, H, 9] in LOG units, origins, series names.
"""
import argparse, os, time
import numpy as np
import pandas as pd
import torch

p = argparse.ArgumentParser()
p.add_argument("--ds", required=True)
p.add_argument("--model", required=True)
p.add_argument("--bs", type=int, default=1024)
args = p.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
DATA, RUNS = os.path.join(HERE, "..", "data"), os.path.join(HERE, "..", "runs")
os.makedirs(RUNS, exist_ok=True)
QL = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
CFG = {"fao": dict(file="d1_fao.parquet", H=12, o0="2010-01-01", o1="2025-03-01", minctx=60),
       "wfp": dict(file="d2_wfp.parquet", H=6, o0="2013-01-01", o1="2026-04-01", minctx=36)}[args.ds]
W = pd.read_parquet(os.path.join(DATA, CFG["file"]))
origins = W.loc[CFG["o0"]:CFG["o1"]].index
H = CFG["H"]
S, O = W.shape[1], len(origins)
Y = W.values                                                      # log levels, NaN = missing
q_out = np.full((S, O, H, 9), np.nan, np.float32)

# build all contexts (series s, origin o) that have >= minctx observations and the origin month observed
jobs, ctxs = [], []
idx_of = {d: i for i, d in enumerate(W.index)}
for oi, o in enumerate(origins):
    t = idx_of[o]
    for s in range(S):
        c = Y[:t + 1, s]
        fv = np.argmax(~np.isnan(c)) if (~np.isnan(c)).any() else None
        if fv is None or np.isnan(c[-1]):
            continue
        c = c[fv:]
        if (~np.isnan(c)).sum() < CFG["minctx"]:
            continue
        x = pd.Series(c).interpolate(limit_area="inside").values      # past-only gap filling
        jobs.append((s, oi)); ctxs.append(np.exp(x[-512:]).astype(np.float32))
print(f"{args.ds} {args.model}: {S} series x {O} origins -> {len(jobs)} contexts", flush=True)
t0 = time.time()
if args.model == "chronos":
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-base", device_map="cuda", torch_dtype=torch.float32)
    for b in range(0, len(ctxs), args.bs):
        qs, _ = pipe.predict_quantiles([torch.tensor(c) for c in ctxs[b:b + args.bs]], prediction_length=H, quantile_levels=QL)
        qs = np.log(np.clip(qs.numpy(), 1e-6, None))                 # [B, H, 9]
        for (s, oi), qq in zip(jobs[b:b + args.bs], qs):
            q_out[s, oi] = qq
elif args.model == "timesfm":
    import timesfm
    m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    m.compile(timesfm.ForecastConfig(max_context=512, max_horizon=max(16, H), normalize_inputs=True,
                                     use_continuous_quantile_head=True, fix_quantile_crossing=True,
                                     per_core_batch_size=256, infer_is_positive=True))
    for b in range(0, len(ctxs), args.bs):
        point, quant = m.forecast(horizon=H, inputs=ctxs[b:b + args.bs])   # quant [B, H, 10]: mean + q0.1..q0.9
        qs = np.log(np.clip(np.asarray(quant)[:, :H, 1:10], 1e-6, None))
        for (s, oi), qq in zip(jobs[b:b + args.bs], qs):
            q_out[s, oi] = qq
print(f"done in {time.time()-t0:.0f}s", flush=True)
np.savez_compressed(os.path.join(RUNS, f"zs_{args.ds}_{args.model}.npz"), q=q_out,
                    origins=np.array([str(o.date()) for o in origins]), series=np.array(W.columns))
