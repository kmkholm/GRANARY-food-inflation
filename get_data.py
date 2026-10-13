"""Download the public inputs into raw_data/ (or $GRANARY_RAW).
FAOSTAT consumer price indices (bulk), World Bank Pink Sheet (monthly prices), World Bank country metadata,
WFP Global Food Prices yearly files from the Humanitarian Data Exchange (2008 onwards)."""
import json, os, urllib.request, zipfile

RAW = os.environ.get("GRANARY_RAW", os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw_data"))
for d in ("faostat", "worldbank", "wfp"):
    os.makedirs(os.path.join(RAW, d), exist_ok=True)


def get(url, path):
    print("downloading", url)
    urllib.request.urlretrieve(url, path)


z = os.path.join(RAW, "faostat", "cpi.zip")
get("https://bulks-faostat.fao.org/production/ConsumerPriceIndices_E_All_Data_(Normalized).zip", z)
zipfile.ZipFile(z).extractall(os.path.join(RAW, "faostat"))
# the Pink Sheet link changes with each monthly release; update it from https://www.worldbank.org/en/research/commodity-markets
get("https://thedocs.worldbank.org/en/doc/74e8be41ceb20fa0da750cda2f6b9e4e-0050012026/related/CMO-Historical-Data-Monthly.xlsx",
    os.path.join(RAW, "worldbank", "CMO-Historical-Data-Monthly.xlsx"))
get("https://api.worldbank.org/v2/country?format=json&per_page=400", os.path.join(RAW, "worldbank", "countries.json"))
meta = json.load(urllib.request.urlopen("https://data.humdata.org/api/3/action/package_show?id=global-wfp-food-prices"))
for r in meta["result"]["resources"]:
    name = r["name"].replace("Global WFP food prices ", "prices_").replace("Global WFP ", "").replace(" ", "_")
    if name.startswith("prices_") and int(name.split("_")[1]) < 2008:
        continue
    get(r["url"], os.path.join(RAW, "wfp", name + ".csv"))
print("done:", RAW)
