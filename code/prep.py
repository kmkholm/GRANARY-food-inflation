"""Build the two forecasting panels and the global-driver covariates.

D1  FAOSTAT  : monthly consumer food price index (2015 = 100) per country -> log index, 2000-01 .. 2026-03
D2  WFP      : national monthly median retail price (local currency) per country x commodity -> log price,
               2008-01 .. 2026-10 (markets aggregated by median; 'actual' and 'aggregate' price flags kept)
X   Pink Sheet (World Bank CMO, 2026-10-02 release): global monthly prices -> log level, 1960-01 .. 2026-09
Outputs: data/d1_fao.parquet, data/d2_wfp.parquet (wide: rows = month, columns = series), data/covariates.parquet,
data/series_meta.csv
"""
import os, glob, re
import numpy as np
import pandas as pd

RAW = os.environ.get("GRANARY_RAW", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "raw_data"))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
os.makedirs(OUT, exist_ok=True)
MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
          'November', 'December']

# ---------------- D1 FAOSTAT -----------------
fa = pd.read_csv(f"{RAW}/faostat/ConsumerPriceIndices_E_All_Data_(Normalized).csv", encoding="latin1", low_memory=False)
fa = fa[(fa["Item"] == "Consumer Prices, Food Indices (2015 = 100)") & (fa["Area Code"] < 5000)]   # countries only
fa = fa[fa["Months"].isin(MONTHS)]
fa["date"] = pd.to_datetime(dict(year=fa["Year"], month=fa["Months"].map({m: i + 1 for i, m in enumerate(MONTHS)}), day=1))
fa = fa[fa["Value"] > 0]
w1 = fa.pivot_table(index="date", columns="Area", values="Value", aggfunc="first").sort_index()
w1 = w1.loc["2000-01-01":]
# keep countries with >= 180 valid months and data in 2025
keep = [c for c in w1 if w1[c].notna().sum() >= 180 and w1[c].loc["2025-01-01":].notna().any()]
w1 = np.log(w1[keep])
w1.to_parquet(f"{OUT}/d1_fao.parquet")
iso = fa.drop_duplicates("Area").set_index("Area")["Area Code (M49)"].astype(str).str.strip("'")
m1 = pd.DataFrame({"series": keep, "dataset": "fao", "country": keep, "m49": [iso.get(c, "") for c in keep],
                   "commodity": "food CPI", "n_obs": [int(w1[c].notna().sum()) for c in keep]})
print("D1 FAOSTAT:", w1.shape, "range", w1.index.min().date(), w1.index.max().date())

# ---------------- D2 WFP -----------------
cols = ["countryiso3", "date", "market_id", "commodity", "unit", "pricetype", "currency", "price", "priceflag"]
wf = pd.concat([pd.read_csv(f, skiprows=[1], usecols=cols, low_memory=False)
                for f in sorted(glob.glob(f"{RAW}/wfp/prices_20*.csv"))])
wf = wf[(wf["pricetype"] == "Retail") & (wf["price"] > 0) & wf["priceflag"].isin(["actual", "aggregate", "actual,aggregate"])]
wf["date"] = pd.to_datetime(wf["date"]).dt.to_period("M").dt.to_timestamp()
wf["series"] = wf["countryiso3"] + "|" + wf["commodity"] + "|" + wf["unit"].astype(str) + "|" + wf["currency"]
nat = wf.groupby(["series", "date"])["price"].median().reset_index()
w2 = nat.pivot(index="date", columns="series", values="price").sort_index()
w2 = w2.loc["2008-01-01":]
# keep series with >= 84 months, data in 2025-2026, and at most 25 % gaps inside their span
keep2 = []
for c in w2:
    s = w2[c]; fv, lv = s.first_valid_index(), s.last_valid_index()
    span = s.loc[fv:lv]
    if span.notna().sum() >= 84 and lv >= pd.Timestamp("2025-06-01") and span.isna().mean() <= 0.25:
        keep2.append(c)
w2 = np.log(w2[keep2])
w2.to_parquet(f"{OUT}/d2_wfp.parquet")
parts = [c.split("|") for c in keep2]
m2 = pd.DataFrame({"series": keep2, "dataset": "wfp", "country": [p[0] for p in parts], "m49": "",
                   "commodity": [p[1] for p in parts], "n_obs": [int(w2[c].notna().sum()) for c in keep2]})
print("D2 WFP:", w2.shape, "countries", m2.country.nunique(), "commodities", m2.commodity.nunique(),
      "range", w2.index.min().date(), w2.index.max().date())

# ---------------- covariates: Pink Sheet -----------------
ps = pd.read_excel(f"{RAW}/worldbank/CMO-Historical-Data-Monthly.xlsx", "Monthly Prices", header=None)
names = ps.iloc[4].tolist(); body = ps.iloc[6:].copy()
body.columns = ["month"] + [str(n).strip() for n in names[1:]]
body = body[body["month"].astype(str).str.match(r"^\d{4}M\d{2}$")]
body.index = pd.to_datetime(body["month"].str.replace("M", "-") + "-01")
WANT = {"Crude oil, average": "oil", "Natural gas, Europe": "gas_eu", "Wheat, US HRW": "wheat",
        "Maize": "maize", "Rice, Thai 5%": "rice", "Soybean oil": "soy_oil", "Palm oil": "palm_oil",
        "Sugar, world": "sugar", "Urea ": "urea", "DAP": "dap", "Potassium chloride **": "potash", "Beef **": "beef"}
cov = {}
for k, v in WANT.items():
    hit = [c for c in body.columns if c.strip().lower().startswith(k.strip().lower().rstrip("*").strip())]
    if not hit:
        print("missing covariate", k); continue
    cov[v] = pd.to_numeric(body[hit[0]], errors="coerce")
cov = np.log(pd.DataFrame(cov).astype(float)).loc["1990-01-01":]
ix = pd.read_excel(f"{RAW}/worldbank/CMO-Historical-Data-Monthly.xlsx", "Monthly Indices", header=None)
print("indices header rows:", ix.iloc[4:8, :8].fillna("").values.tolist())
cov.to_parquet(f"{OUT}/covariates.parquet")
print("covariates:", cov.shape, cov.columns.tolist(), cov.index.min().date(), cov.index.max().date())
pd.concat([m1, m2]).to_csv(f"{OUT}/series_meta.csv", index=False)
