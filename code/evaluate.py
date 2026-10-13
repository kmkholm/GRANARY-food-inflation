"""Shared evaluation of quantile forecasts in log units.
Targets: the h-month log change y_{o+h} - y_o (x100 = cumulative food inflation in % points).
Metrics per horizon: RMSE (pp), MAE (pp), relative MAE vs the no-change (random-walk) forecast (= MASE-style
ratio, < 1 beats the random walk), mean pinball loss over 9 quantiles (pp), empirical coverage of the 80 % interval
[q0.1, q0.9], and the directional hit rate of the change.
"""
import numpy as np
import pandas as pd

QL = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


def targets(W, origins, H):
    """actual log change [S, O, H] and last observed log level [S, O] (NaN where unavailable)."""
    idx = {d: i for i, d in enumerate(W.index)}
    Y = W.values
    S, O = Y.shape[1], len(origins)
    act = np.full((S, O, H), np.nan); last = np.full((S, O), np.nan)
    for oi, o in enumerate(origins):
        t = idx[pd.Timestamp(str(o))]
        last[:, oi] = Y[t]
        for h in range(1, H + 1):
            if t + h < len(Y):
                act[:, oi, h - 1] = Y[t + h] - Y[t]
    return act, last


def metrics(q, act, last, mask=None):
    """q [S,O,H,9] log levels; act [S,O,H] log change; last [S,O]. Returns DataFrame by horizon."""
    qc = (q - last[..., None, None]) * 100.0                        # predicted change in pp
    a = act * 100.0
    valid = ~np.isnan(a) & ~np.isnan(qc[..., 4])
    if mask is not None:
        valid &= mask
    rows = []
    for h in range(a.shape[2]):
        v = valid[:, :, h]
        e = qc[:, :, h, 4][v] - a[:, :, h][v]
        rw = np.abs(a[:, :, h][v])                                  # no-change forecast error
        qq = qc[:, :, h][v]; aa = a[:, :, h][v]
        pin = np.mean([np.mean(np.maximum(l * (aa - qq[:, i]), (l - 1) * (aa - qq[:, i]))) for i, l in enumerate(QL)])
        cov80 = np.mean((aa >= qq[:, 0]) & (aa <= qq[:, 8]))
        hit = np.mean(np.sign(qq[:, 4]) == np.sign(aa))
        rows.append(dict(h=h + 1, n=int(v.sum()), RMSE=np.sqrt(np.mean(e ** 2)), MAE=np.mean(np.abs(e)),
                         relMAE_RW=np.mean(np.abs(e)) / np.mean(rw), pinball=pin, cov80=cov80, dir_hit=hit))
    return pd.DataFrame(rows)


def period_mask(origins, S, H, start, end):
    o = pd.to_datetime([str(x) for x in origins])
    m = (o >= pd.Timestamp(start)) & (o <= pd.Timestamp(end))
    return np.broadcast_to(m[None, :, None], (S, len(o), H))
