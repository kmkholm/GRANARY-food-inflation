# GRANARY

Cross-country forecasting of food price inflation with two frozen time-series foundation models (Chronos-Bolt and
TimesFM 2.5), a global-driver cross-attention adapter, and adaptive conformal prediction intervals.

This repository holds the code that produces every number, table and figure of the accompanying manuscript.

## Data (all public)

| Source | Content | Licence |
|---|---|---|
| FAOSTAT consumer price indices | monthly food CPI (2015 = 100), 198 countries, 2000-2026 | FAO open data |
| WFP Global Food Prices Database (HDX) | monthly retail prices, 863 country-commodity series, 66 countries, 2008-2026 | CC BY-IGO |
| World Bank commodity prices (Pink Sheet) | 12 monthly global prices: oil, EU gas, wheat, maize, rice, soybean oil, palm oil, sugar, urea, DAP, potassium chloride, beef | World Bank terms |

`python get_data.py` downloads everything into `raw_data/` (or the folder in `$GRANARY_RAW`).

## Pipeline

```
python get_data.py
python code/prep.py                                   # panels + covariates -> data/
python code/zeroshot.py --ds fao --model chronos      # repeat for timesfm and for --ds wfp
python code/run_grid.py                               # GRANARY + ablations, 2 panels x 5 seeds x 6 variants
python code/granary_explain.py --ds fao --seed 0 --tag main   # driver attribution (repeat per seed / panel)
bash code/run_bm.sh                                   # seasonal naive, AR(12), ETS, ARIMA benchmarks
python code/aggregate.py                              # metrics, Diebold-Mariano tests, regions -> results/
python code/build_tables.py && python code/fig_results.py
```

`results/` holds the aggregated metrics reported in the paper (accuracy by horizon, Diebold-Mariano tests, regional errors).

## Requirements

Python 3.11+ with the packages in `requirements.txt`. The foundation models (Chronos-Bolt, TimesFM 2.5) and the
statistical benchmarks (`statsforecast`) are best installed in a separate virtual environment. A single 8 GB GPU is
enough. One adapter run takes under a minute.

## Licence

MIT (code). The data remain under the licences of their providers.
