"""Tables and number macros for the paper from results/agg_*.csv, dm_*.csv, region_*.csv and attribution files.
Writes paper/tables.tex and paper/numbers.tex (\\newcommand macros)."""
import os, glob, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES, R, P, D = [os.path.join(HERE, "..", x) for x in ("results", "runs", "paper", "data")]
agg = {ds: pd.read_csv(os.path.join(RES, f"agg_{ds}.csv")) for ds in ("fao", "wfp")}
dm = {ds: pd.read_csv(os.path.join(RES, f"dm_{ds}.csv")) for ds in ("fao", "wfp")}
HS = {"fao": [1, 3, 6, 12], "wfp": [1, 3, 6]}
macros = []


def mac(name, val):
    macros.append(f"\\newcommand{{\\{name}}}{{{val}}}")


def g(ds, variant, h, col, per="test"):
    a = agg[ds]; r = a[(a.variant == variant) & (a.h == h) & (a.period == per)]
    if len(r) == 0:
        return float("nan"), float("nan")
    return float(r[f"{col}_mean"].iloc[0]), float(r[f"{col}_std"].iloc[0]) if f"{col}_std" in r else 0.0


def fmt(m, s=None, d=3):
    return f"{m:.{d}f}" if s is None or np.isnan(s) or s == 0 else f"{m:.{d}f}$\\pm${s:.{d}f}"


# ---- Table 2: main accuracy ----
rows = []
ROWS = (("Seasonal naive", "Seasonal naive"), ("AR (direct)", "AR(12), direct"), ("ETS", "ETS (automatic)"),
        ("ARIMA", "ARIMA (automatic, seasonal)"), ("Chronos zero-shot", "Chronos-Bolt, zero-shot"),
        ("TimesFM zero-shot", "TimesFM 2.5, zero-shot"), ("Zero-shot ensemble", "Zero-shot ensemble"), ("GRANARY", "GRANARY"))
for ds in ("fao", "wfp"):
    for v, lab in ROWS:
        cells = []
        MODELS = tuple(r[0] for r in ROWS)
        for h in HS[ds]:
            m1, s1 = g(ds, v, h, "relMAE_RW"); m2, s2 = g(ds, v, h, "pinball"); m3, s3 = g(ds, v, h, "cov80")
            best1 = np.nanmin([round(g(ds, x, h, "relMAE_RW")[0], 3) for x in MODELS])
            best2 = np.nanmin([round(g(ds, x, h, "pinball")[0], 2) for x in MODELS])
            best3 = np.nanmin([abs(round(g(ds, x, h, "cov80")[0], 2) - 0.8) for x in MODELS])
            bf = lambda val, best, txt: f"\\textbf{{{txt}}}" if abs(val - best) < 1e-9 else txt
            if np.isnan(m1):
                cells.append("-- & -- & --"); continue
            cells.append(f"{bf(round(m1, 3), best1, f'{m1:.3f}')} & {bf(round(m2, 2), best2, f'{m2:.2f}')} & "
                         f"{bf(abs(round(m3, 2) - 0.8), best3, f'{m3:.2f}')}")
        rows.append((ds, lab, " & ".join(cells)))
t = ["\\begin{table}[!htbp]\\centering\\small",
     "\\caption{Test accuracy (origins from January 2020). relMAE: MAE of the median relative to the no-change forecast (below 1 is better). Pinball: mean pinball loss in percentage points. Cov: coverage of the 80\\,\\% interval. Best value per column in bold (coverage: closest to 0.80). Statistical benchmarks are deterministic single runs; on WFP, AR(12) has no forecast for the 3.5\\,\\% of test contexts with too short a history, and ARIMA uses the last 120 months. GRANARY values are means over five seeds, with ACI-calibrated intervals; seed standard deviations are in Table~\\ref{tab:ablation}.}",
     "\\label{tab:main}", "\\setlength{\\tabcolsep}{3.5pt}"]
for ds, title in (("fao", "FAOSTAT consumer food price index (198 countries)"), ("wfp", "WFP retail prices (863 series, 66 countries)")):
    hs = HS[ds]
    t += ["\\resizebox{\\textwidth}{!}{%", f"\\begin{{tabular}}{{l{'ccc' * len(hs)}}}", "\\toprule",
          f"\\multicolumn{{{1 + 3 * len(hs)}}}{{l}}{{\\textit{{{title}}}}}\\\\",
          "Model & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{$h={h}$}}" for h in hs) + "\\\\",
          "".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(hs))),
          " & " + " & ".join(["relMAE & Pinball & Cov"] * len(hs)) + "\\\\", "\\midrule"]
    for d_, lab, c in rows:
        if d_ != ds:
            continue
        if lab.startswith("Seasonal"):
            t.append(f"\\multicolumn{{{1 + 3 * len(hs)}}}{{l}}{{\\textit{{Statistical benchmarks}}}}\\\\")
        if lab.startswith("Chronos-Bolt"):
            t.append(f"\\multicolumn{{{1 + 3 * len(hs)}}}{{l}}{{\\textit{{Foundation models}}}}\\\\")
        t.append(f"{lab} & {c}\\\\")
    t += ["\\bottomrule", "\\end{tabular}}", "\\par\\medskip"]
t += ["\\end{table}"]
tables = ["\n".join(t)]

# ---- Table 3: ablations (pinball, mean +- sd) ----
abl = [("GRANARY", "GRANARY (full)"), ("w/o drivers", "w/o driver tokens"), ("w/o history", "w/o history token"),
       ("w/o identity", "w/o identity embeddings"), ("Chronos only", "Chronos-Bolt only"), ("TimesFM only", "TimesFM 2.5 only"),
       ("Zero-shot ensemble", "Zero-shot ensemble (no adapter)")]
t = ["\\begin{table}[!htbp]\\centering\\small",
     "\\caption{Ablations: mean pinball loss (percentage points, mean $\\pm$ standard deviation over five seeds) on the test origins. Lower is better.}",
     "\\label{tab:ablation}", "\\setlength{\\tabcolsep}{4pt}",
     "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{lccccccc}", "\\toprule",
     " & \\multicolumn{4}{c}{FAOSTAT} & \\multicolumn{3}{c}{WFP}\\\\", "\\cmidrule(lr){2-5}\\cmidrule(lr){6-8}",
     "Variant & $h=1$ & $h=3$ & $h=6$ & $h=12$ & $h=1$ & $h=3$ & $h=6$\\\\", "\\midrule"]
for v, lab in abl:
    cells = []
    for ds in ("fao", "wfp"):
        for h in HS[ds]:
            m, s = g(ds, v, h, "pinball"); cells.append(fmt(m, s, 3))
    t.append(f"{lab} & " + " & ".join(cells) + "\\\\")
t += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
tables.append("\n".join(t))

# ---- Table 4: calibrators ----
cal = [("Zero-shot ensemble", "Zero-shot ensemble"), ("GRANARY (raw intervals)", "GRANARY, no calibration"),
       ("GRANARY + static CQR", "GRANARY + static conformal"), ("GRANARY + rolling CQR", "GRANARY + rolling conformal (36)"),
       ("GRANARY", "GRANARY + ACI (proposed)")]
t = ["\\begin{table}[!htbp]\\centering\\small",
     "\\caption{Coverage of the nominal 80\\,\\% interval by calibrator, on all test origins and on the July 2021--December 2022 surge (mean over five seeds).}",
     "\\label{tab:calib}", "\\setlength{\\tabcolsep}{4pt}",
     "\\begin{tabular}{llccccccc}", "\\toprule",
     " & & \\multicolumn{4}{c}{FAOSTAT} & \\multicolumn{3}{c}{WFP}\\\\", "\\cmidrule(lr){3-6}\\cmidrule(lr){7-9}",
     "Period & Intervals & $h=1$ & $h=3$ & $h=6$ & $h=12$ & $h=1$ & $h=3$ & $h=6$\\\\", "\\midrule"]
for per, plab in (("test", "Test"), ("surge", "Surge")):
    for v, lab in cal:
        cells = [f"{g(ds, v, h, 'cov80', per)[0]:.2f}" for ds in ("fao", "wfp") for h in HS[ds]]
        t.append(f"{plab if v == cal[0][0] else ''} & {lab} & " + " & ".join(cells) + "\\\\")
    if per == "test":
        t.append("\\midrule")
t += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
tables.append("\n".join(t))

# ---- Table 5: Diebold-Mariano ----
t = ["\\begin{table}[!htbp]\\centering\\small",
     "\\caption{Diebold--Mariano tests of GRANARY (seed-averaged median) against the zero-shot ensemble, the direct AR(12) model and automatic ARIMA. $\\bar d$: mean difference in absolute error (percentage points; negative favours GRANARY); $p$: two-sided $p$-value with a HAC variance ($h-1$ lags) over test origins.}",
     "\\label{tab:dm}", "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{lcccccccccccc}", "\\toprule",
     " & \\multicolumn{6}{c}{FAOSTAT} & \\multicolumn{6}{c}{WFP}\\\\", "\\cmidrule(lr){2-7}\\cmidrule(lr){8-13}",
     " & \\multicolumn{2}{c}{vs ensemble} & \\multicolumn{2}{c}{vs AR(12)} & \\multicolumn{2}{c}{vs ARIMA} & \\multicolumn{2}{c}{vs ensemble} & \\multicolumn{2}{c}{vs AR(12)} & \\multicolumn{2}{c}{vs ARIMA}\\\\",
     "$h$ & $\\bar d$ & $p$ & $\\bar d$ & $p$ & $\\bar d$ & $p$ & $\\bar d$ & $p$ & $\\bar d$ & $p$ & $\\bar d$ & $p$\\\\", "\\midrule"]


def dmcells(r):
    if r is None:
        return "-- & -- & -- & -- & -- & --"
    c = [f"{r.mean_diff_pp:.3f} & {r.p:.3f}"]
    for k in ("ar", "arima"):
        c.append(f"{getattr(r, 'd_' + k):.3f} & {getattr(r, 'p_' + k):.3f}" if hasattr(r, "d_" + k) else "-- & --")
    return " & ".join(c)


for h in range(1, 13):
    a = dm["fao"][dm["fao"].h == h].iloc[0]
    b = dm["wfp"][dm["wfp"].h == h].iloc[0] if h <= 6 else None
    t.append(f"{h} & {dmcells(a)} & {dmcells(b)}\\\\")
t += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
tables.append("\n".join(t))
open(os.path.join(P, "tables.tex"), "w").write("\n\n".join(tables) + "\n")
for name, tex in zip(("tab_main", "tab_ablation", "tab_calib", "tab_dm"), tables):   # one file per table, placed at first mention
    open(os.path.join(P, f"{name}.tex"), "w").write(tex + "\n")

# ---- macros used in the prose ----
for ds in ("fao", "wfp"):
    D_ = ds.upper()
    for h in HS[ds]:
        for v, k in (("GRANARY", "G"), ("Zero-shot ensemble", "Z"), ("Chronos zero-shot", "C"), ("TimesFM zero-shot", "T")):
            hh = {1: "one", 3: "three", 6: "six", 12: "twelve"}[h]
            mac(f"{D_}{k}rel{hh}", f"{g(ds, v, h, 'relMAE_RW')[0]:.3f}")
            mac(f"{D_}{k}pin{hh}", f"{g(ds, v, h, 'pinball')[0]:.2f}")
            mac(f"{D_}{k}cov{hh}", f"{g(ds, v, h, 'cov80')[0]:.2f}")
            mac(f"{D_}{k}covsurge{hh}", f"{g(ds, v, h, 'cov80', 'surge')[0]:.2f}")
            mac(f"{D_}{k}relsurge{hh}", f"{g(ds, v, h, 'relMAE_RW', 'surge')[0]:.3f}")
        rawc = g(ds, "GRANARY (raw intervals)", h, "cov80")[0]; mac(f"{D_}rawcov{ {1:'one',3:'three',6:'six',12:'twelve'}[h] }".replace(" ", ""), f"{rawc:.2f}")
    # pinball gain at the longest horizon
    hl = HS[ds][-1]
    gp, zp = g(ds, "GRANARY", hl, "pinball")[0], g(ds, "Zero-shot ensemble", hl, "pinball")[0]
    mac(f"{D_}pingainlong", f"{100 * (zp - gp) / zp:.1f}")
    gr, zr = g(ds, "GRANARY", hl, "relMAE_RW")[0], g(ds, "Zero-shot ensemble", hl, "relMAE_RW")[0]
    mac(f"{D_}relgainlong", f"{100 * (zr - gr) / zr:.1f}")
meta = pd.read_csv(os.path.join(D, "series_meta.csv"))
mac("NFAO", str((meta.dataset == "fao").sum())); mac("NWFP", str((meta.dataset == "wfp").sum()))
mac("NWFPcountries", str(meta[meta.dataset == "wfp"].country.nunique()))
r0 = json.load(open(os.path.join(R, "main", "fao_s0", "result.json"))); r1 = json.load(open(os.path.join(R, "main", "wfp_s0", "result.json")))
mac("ParamsFAO", f"{r0['params']/1e3:.0f}"); mac("ParamsWFP", f"{r1['params']/1e3:.0f}")
open(os.path.join(P, "numbers.tex"), "w").write("\n".join(macros) + "\n")
print(len(macros), "macros;", len(tables), "tables")
