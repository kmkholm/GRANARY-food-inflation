"""GRANARY: foundation-model food-price forecasting with a global-driver cross-attention adapter and rolling
conformal intervals.

python granary.py --ds fao --seed 0 --tag main            (variants: --no_drivers --no_history --no_identity --single chronos|timesfm)
Per (series s, origin o) the adapter receives
  * H forecast tokens: the 9 zero-shot quantiles of Chronos-Bolt and of TimesFM-2.5 for horizon h (log change, pp/10)
  * one history token: the last 24 monthly log changes of the series (pp, zero-filled + mask)
  * 12 driver tokens: the last 24 monthly log changes of each World Bank global price (known at o)
  * identity: country embedding (+ commodity embedding for WFP), dropped with p = 0.3 during training
Forecast tokens self-attend and cross-attend to history + drivers (2 layers, d = 64).  For every horizon the head
returns a softmax weight between the two foundation models, a location shift and a log-width factor; quantiles are
re-sorted to stay monotone.  Loss = pinball over the 9 quantiles.  After training, rolling conformalised quantile
regression widens/narrows the [q0.1, q0.9] band per horizon using ONLY errors already observed at each origin
(errors of forecasts made at origins <= o - h), window = 36 origins.
Splits (by origin): FAO train <= 2016-12, val 2017-01..2018-12, test 2020-01..2025-03;
                    WFP train <= 2017-12, val 2018-01..2018-12, test 2020-01..2026-04.
"""
import argparse, os, json, time, copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from evaluate import targets, metrics, period_mask, QL
from calibrate import static as cal_static, rolling as cal_rolling, aci as cal_aci

p = argparse.ArgumentParser()
p.add_argument("--ds", required=True)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--tag", default="dev")
p.add_argument("--epochs", type=int, default=40)
p.add_argument("--bs", type=int, default=512)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--d", type=int, default=64)
p.add_argument("--no_drivers", action="store_true")
p.add_argument("--no_history", action="store_true")
p.add_argument("--no_identity", action="store_true")
p.add_argument("--single", default="", help="use only one foundation model (chronos|timesfm)")
p.add_argument("--conf_window", type=int, default=36)
args = p.parse_args()
torch.manual_seed(args.seed); np.random.seed(args.seed)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA, RUNS = os.path.join(HERE, "..", "data"), os.path.join(HERE, "..", "runs")
OUT = os.path.join(RUNS, args.tag, f"{args.ds}_s{args.seed}" + ("_nodrv" if args.no_drivers else "") +
                   ("_nohist" if args.no_history else "") + ("_noid" if args.no_identity else "") +
                   (f"_only{args.single}" if args.single else ""))
os.makedirs(OUT, exist_ok=True)
dev = "cuda"
CFG = {"fao": dict(file="d1_fao.parquet", H=12, tr="2016-12-01", va0="2017-01-01", va1="2018-12-01", te0="2020-01-01", te1="2025-03-01"),
       "wfp": dict(file="d2_wfp.parquet", H=6, tr="2017-12-01", va0="2018-01-01", va1="2018-12-01", te0="2020-01-01", te1="2026-04-01")}[args.ds]
H = CFG["H"]

# ---------------- assemble ----------------
W = pd.read_parquet(os.path.join(DATA, CFG["file"]))
Z = {m: np.load(os.path.join(RUNS, f"zs_{args.ds}_{m}.npz"), allow_pickle=True) for m in ("chronos", "timesfm")}
origins = pd.to_datetime([str(o) for o in Z["chronos"]["origins"]])
act, last = targets(W, Z["chronos"]["origins"], H)
qfm = np.stack([Z["chronos"]["q"], Z["timesfm"]["q"]], 2)                 # S,O,2,H,9 log level
qfm = (qfm - last[:, :, None, None, None]) * 10.0                       # log change in pp/10
if args.single:
    k = 0 if args.single == "chronos" else 1
    qfm[:, :, 1 - k] = qfm[:, :, k]
S, O = last.shape
cov = pd.read_parquet(os.path.join(DATA, "covariates.parquet"))
dcov = cov.diff() * 100.0                                               # monthly log change, pp
idxW = {d: i for i, d in enumerate(W.index)}
Yd = W.diff().values * 100.0                                            # series monthly log change, pp


def window(arr_rows, t, L=24):
    lo = t - L + 1
    out = np.full((L,) + arr_rows.shape[1:], np.nan)
    src = arr_rows[max(lo, 0):t + 1]
    out[L - len(src):] = src
    return out


drv = np.zeros((O, cov.shape[1], 24), np.float32)
for oi, o in enumerate(origins):
    t = cov.index.get_loc(o)
    drv[oi] = np.nan_to_num(window(dcov.values, t).T) / 10.0
meta = pd.read_csv(os.path.join(DATA, "series_meta.csv"))
meta = meta[meta.dataset == args.ds].set_index("series").loc[list(W.columns)]
countries = sorted(meta["country"].unique()); cmap = {c: i + 1 for i, c in enumerate(countries)}
cid = np.array([cmap[c] for c in meta["country"]])
coms = sorted(meta["commodity"].unique()); kmap = {c: i + 1 for i, c in enumerate(coms)}
kid = np.array([kmap[c] for c in meta["commodity"]])

ok = ~np.isnan(qfm[:, :, 0, 0, 4]) & ~np.isnan(last)
pairs = np.argwhere(ok)                                                 # (s, oi)


def build(sel):
    s, oi = sel[:, 0], sel[:, 1]
    hist = np.zeros((len(sel), 24), np.float32); hm = np.zeros((len(sel), 24), np.float32)
    for j, (ss, oo) in enumerate(sel):
        t = idxW[origins[oo]]
        w = window(Yd[:, ss], t)
        hm[j] = ~np.isnan(w); hist[j] = np.nan_to_num(w) / 10.0
    return dict(q=qfm[s, oi].astype(np.float32), hist=np.stack([hist, hm], -1), drv=drv[oi], cid=cid[s], kid=kid[s],
                y=(act[s, oi] * 10.0).astype(np.float32))


def split(a, b):
    m = (origins[pairs[:, 1]] >= pd.Timestamp(a)) & (origins[pairs[:, 1]] <= pd.Timestamp(b))
    return pairs[m]


tr_p, va_p, te_p = split("1900-01-01", CFG["tr"]), split(CFG["va0"], CFG["va1"]), split(CFG["te0"], CFG["te1"])
all_p = pairs
B_all = build(all_p)
pos = {tuple(x): i for i, x in enumerate(map(tuple, all_p))}
ix = lambda P: np.array([pos[tuple(x)] for x in P])
I_tr, I_va, I_te = ix(tr_p), ix(va_p), ix(te_p)
print(f"[{args.ds}] pairs train {len(I_tr)} val {len(I_va)} test {len(I_te)} | S {S} O {O} H {H}", flush=True)
T = {k: torch.tensor(v, device=dev) for k, v in B_all.items()}


# ---------------- model ----------------
class Granary(nn.Module):
    def __init__(self, H, n_drv, n_c, n_k, d=64, heads=4, layers=2):
        super().__init__()
        self.f_tok = nn.Linear(2 * 9, d); self.h_emb = nn.Embedding(H, d)
        self.hist = nn.Linear(2 * 24, d)
        self.drv = nn.Linear(24, d); self.drv_id = nn.Embedding(n_drv, d)
        self.cemb = nn.Embedding(n_c + 1, d); self.kemb = nn.Embedding(n_k + 1, d)
        self.self_att = nn.ModuleList([nn.TransformerEncoderLayer(d, heads, 2 * d, 0.1, batch_first=True, norm_first=True) for _ in range(layers)])
        self.cross = nn.ModuleList([nn.MultiheadAttention(d, heads, dropout=0.1, batch_first=True) for _ in range(layers)])
        self.cn = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.out = nn.Linear(d, 2 + 1 + 1)                              # model logits (2), shift, log-width
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        self.H = H

    def forward(self, q, hist, drv, cid, kid, use_drv=True, use_hist=True, use_id=True, id_drop=0.0, return_att=False):
        Bn = q.shape[0]
        x = self.f_tok(q.permute(0, 2, 1, 3).reshape(Bn, self.H, 18)) + self.h_emb.weight[None]
        if use_id:
            e = (self.cemb(cid) + self.kemb(kid))[:, None]
            if self.training and id_drop > 0:
                e = e * (torch.rand(Bn, 1, 1, device=q.device) >= id_drop).float()
            x = x + e
        mem = []
        if use_hist:
            mem.append(self.hist(hist.reshape(Bn, -1))[:, None])
        if use_drv:
            mem.append(self.drv(drv) + self.drv_id.weight[None])
        mem = torch.cat(mem, 1) if mem else None
        atts = []
        for sa, ca, cn in zip(self.self_att, self.cross, self.cn):
            x = sa(x)
            if mem is not None:
                a, w = ca(cn(x), mem, mem, need_weights=return_att, average_attn_weights=True)
                x = x + a
                atts.append(w)
        return None, self.out(x), atts                                   # B,H,4


def predict_quantiles(model, idx, train=False, **kw):
    q = T["q"][idx]                                                      # B,2,H,9
    qh = q                                                               # B,2,H,9
    base_in = qh.permute(0, 2, 1, 3)                                     # B,H,2,9
    _, o, atts = model(qh, T["hist"][idx], T["drv"][idx], T["cid"][idx], T["kid"][idx], **kw)
    wts = torch.softmax(o[..., :2], -1)                                  # B,H,2
    mix = (wts[..., None] * base_in).sum(2)                              # B,H,9
    med = mix[..., 4:5]
    out = med + o[..., 2:3] + (mix - med) * torch.exp(o[..., 3:4])
    return torch.sort(out, -1).values, wts, atts


QLt = torch.tensor(QL, device=dev, dtype=torch.float32)


def pinball(qp, y):
    e = y[..., None] - qp
    return torch.maximum(QLt * e, (QLt - 1) * e).mean()


model = Granary(H, drv.shape[1], len(countries), len(coms), d=args.d).to(dev)
kw = dict(use_drv=not args.no_drivers, use_hist=not args.no_history, use_id=not args.no_identity)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
hist_log = {"train": [], "val": []}
best, best_state, best_ep = 1e9, None, -1
y_all = T["y"]


def eval_loss(idx):
    model.eval()
    with torch.no_grad():
        ls = []
        for b in range(0, len(idx), 4096):
            ii = torch.tensor(idx[b:b + 4096], device=dev)
            qp, _, _ = predict_quantiles(model, ii, **kw)
            m = ~torch.isnan(y_all[ii])
            e = torch.nan_to_num(y_all[ii])[..., None] - qp
            l = torch.maximum(QLt * e, (QLt - 1) * e).mean(-1)
            ls.append((l * m).sum().item() / max(m.sum().item(), 1) * len(ii))
    return sum(ls) / len(idx)


t0 = time.time()
for ep in range(args.epochs):
    model.train()
    perm = np.random.permutation(I_tr)
    tl = 0.0; nb = 0
    for b in range(0, len(perm), args.bs):
        ii = torch.tensor(perm[b:b + args.bs], device=dev)
        qp, _, _ = predict_quantiles(model, ii, id_drop=0.3, **kw)
        y = y_all[ii]; m = ~torch.isnan(y)
        e = torch.nan_to_num(y)[..., None] - qp
        loss = (torch.maximum(QLt * e, (QLt - 1) * e).mean(-1) * m).sum() / m.sum()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        tl += loss.item(); nb += 1
    vl = eval_loss(I_va)
    hist_log["train"].append(tl / nb); hist_log["val"].append(vl)
    if vl < best:
        best, best_ep, best_state = vl, ep, copy.deepcopy(model.state_dict())
    if ep % 5 == 0 or ep == args.epochs - 1:
        print(f"ep {ep:3d} train {tl/nb:.4f} val {vl:.4f} | {time.time()-t0:.0f}s", flush=True)
model.load_state_dict(best_state)
torch.save(best_state, os.path.join(OUT, "best.pt"))

# ---------------- predict every pair, put back on the [S,O,H,9] grid (log level) ----------------
model.eval()
qg = np.full((S, O, H, 9), np.nan, np.float32); wg = np.full((S, O, H, 2), np.nan, np.float32)
with torch.no_grad():
    for b in range(0, len(all_p), 4096):
        ii = torch.arange(b, min(b + 4096, len(all_p)), device=dev)
        qp, wts, _ = predict_quantiles(model, ii, **kw)
        P = all_p[b:b + 4096]
        qg[P[:, 0], P[:, 1]] = qp.cpu().numpy() / 10.0 + last[P[:, 0], P[:, 1]][:, None, None]
        wg[P[:, 0], P[:, 1]] = wts.cpu().numpy()


# ---------------- interval calibration with revealed outcomes only (main: ACI, gamma = 0.01) ----------------
qconf = cal_aci(qg, act, last, alpha=0.2, gamma=0.01)
q_static = cal_static(qg, act, last, Z["chronos"]["origins"], CFG["va0"], CFG["va1"])
q_roll = cal_rolling(qg, act, last, args.conf_window)
np.savez_compressed(os.path.join(OUT, "forecasts.npz"), q=qg, qconf=qconf, q_static=q_static, q_roll=q_roll, w=wg, origins=Z["chronos"]["origins"])
res = {"args": vars(args), "best_epoch": best_ep, "val_pinball": best, "time_s": time.time() - t0,
       "params": sum(p.numel() for p in model.parameters())}
mt = period_mask(Z["chronos"]["origins"], S, H, CFG["te0"], CFG["te1"])
ms = period_mask(Z["chronos"]["origins"], S, H, "2021-07-01", "2022-12-01")
for name, qq in (("granary", qg), ("granary_conf", qconf), ("granary_static", q_static), ("granary_roll36", q_roll),
                 ("zs_chronos", Z["chronos"]["q"]), ("zs_timesfm", Z["timesfm"]["q"]),
                 ("zs_mean", (Z["chronos"]["q"] + Z["timesfm"]["q"]) / 2)):
    res[name] = {"test": metrics(qq, act, last, mt).to_dict("records"), "surge": metrics(qq, act, last, ms).to_dict("records")}
json.dump(res, open(os.path.join(OUT, "result.json"), "w"), indent=1, default=float)
json.dump(hist_log, open(os.path.join(OUT, "hist.json"), "w"))
hs = [1, 3, 6, 12] if H == 12 else [1, 3, 6]
for name in ("zs_mean", "granary", "granary_conf"):
    r = pd.DataFrame(res[name]["test"]); r = r[r.h.isin(hs)]
    print(name, " ".join(f"h{int(a.h)}: relMAE {a.relMAE_RW:.3f} pin {a.pinball:.3f} cov80 {a.cov80:.3f}" for a in r.itertuples()), flush=True)
print("done", OUT, f"{time.time()-t0:.0f}s")
