"""Interval calibrators applied to saved forecasts (no retraining). All use only outcomes revealed at each origin:
the h-step outcome of origin o' is known at origin o iff o' <= o - h.
  static  : split-conformal CQR margin from the validation origins only (fixed afterwards)
  rolling : CQR margin from the last `window` revealed origins (default 36)
  aci     : adaptive conformal inference (Gibbs and Candes, 2021) on the CQR score, one alpha_t per horizon,
            updated with every revealed miss/hit (step gamma), the margin = (1 - alpha_t) quantile of the revealed scores
"""
import numpy as np
import pandas as pd


def _scores(q, act, last):
    a = act + last[..., None]
    return np.maximum(q[..., 0] - a, a - q[..., 8])                         # S,O,H


def _apply(q, k):                                                           # k [O,H]
    qc = q.copy()
    qc[..., 0] = q[..., 0] - k[None]; qc[..., 8] = q[..., 8] + k[None]
    return qc


def static(q, act, last, origins, va0, va1, alpha=0.2):
    sc = _scores(q, act, last)
    o = pd.to_datetime([str(x) for x in origins])
    m = (o >= pd.Timestamp(va0)) & (o <= pd.Timestamp(va1))
    H = q.shape[2]; k = np.zeros((len(o), H))
    for h in range(H):
        s = sc[:, m, h].ravel(); s = s[~np.isnan(s)]
        k[:, h] = np.quantile(s, min(1.0, (1 - alpha) * (1 + 1 / len(s))))
    return _apply(q, k)


def rolling(q, act, last, window=36, alpha=0.2):
    sc = _scores(q, act, last); S, O, H = sc.shape
    k = np.zeros((O, H))
    for h in range(H):
        for oi in range(O):
            av = np.arange(max(0, oi - h - window), oi - h)
            s = sc[:, av, h].ravel() if len(av) else np.array([]); s = s[~np.isnan(s)]
            k[oi, h] = np.quantile(s, min(1.0, (1 - alpha) * (1 + 1 / len(s)))) if len(s) >= 50 else 0.0
    return _apply(q, k)


def aci(q, act, last, alpha=0.2, gamma=0.01, window=36):
    sc = _scores(q, act, last); S, O, H = sc.shape
    k = np.zeros((O, H)); at = np.full(H, alpha)
    for h in range(H):
        a_t = alpha
        for oi in range(O):
            rv = oi - h - 1                                                 # newest origin revealed at oi
            if rv >= 0:
                av = np.arange(max(0, rv - window + 1), rv + 1)
                s = sc[:, av, h].ravel(); s = s[~np.isnan(s)]
                # update alpha with the coverage of the newest revealed origin under the margin it was issued with
                s_new = sc[:, rv, h]; s_new = s_new[~np.isnan(s_new)]
                if len(s_new):
                    err = np.mean(s_new > k[rv, h])
                    a_t = float(np.clip(a_t + gamma * (alpha - err), 0.001, 0.999))
                k[oi, h] = np.quantile(s, 1 - a_t) if len(s) >= 50 else 0.0
    return _apply(q, k)
