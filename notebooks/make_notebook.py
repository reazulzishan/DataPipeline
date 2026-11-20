"""Builds pipeline.ipynb: one real, messy dataset taken through every stage of an end-to-end data science pipeline.
Data: Inside Airbnb, New York City, snapshot 2026-09-14 (CC BY 4.0). Every number in the video comes from running this
notebook (visual data -> ../reports/viz.json)."""
import nbformat as nbf

C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# The End-to-End Data Science Pipeline, on real messy data
**Question:** what is a fair nightly price for an Airbnb listing in New York City?
Data: Inside Airbnb, NYC snapshot 2026-09-14, licensed CC BY 4.0. Nothing is pre-cleaned.
""")
code("""
import json, time, re, numpy as np, pandas as pd
from pathlib import Path
np.random.seed(42)
DATA, REP, MOD = Path("../data"), Path("../reports"), Path("../models")
for d in (DATA / "clean", REP, MOD): d.mkdir(parents=True, exist_ok=True)
VIZ = {}                                          # everything the video draws, exported at the end
r = lambda x, n=3: float(np.round(x, n))
print("ready")
""")

# ------------------------------------------------------------------ 1 problem
md("## Stage 1 · Define the problem")
code("""
GOAL   = "predict the nightly price of a NYC listing"
TARGET = "price"
METRIC = "MAE in dollars (average miss per listing)"
RULE   = "the model must beat a simple baseline, on listings it has never seen"
print(GOAL, METRIC, RULE, sep="\\n")
""")

# ------------------------------------------------------------------ 2 ingest
md("## Stage 2 · Ingest: load the raw file exactly as it arrives")
code("""
RAW = DATA / "raw/listings.csv.gz"                # never edited: the raw layer is read-only
t = time.time(); raw = pd.read_csv(RAW, low_memory=False); t_csv = time.time() - t
kinds = raw.dtypes.map(lambda d: "number" if pd.api.types.is_numeric_dtype(d) else "text").value_counts()
print(f"{len(raw):,} rows × {raw.shape[1]} columns · {RAW.stat().st_size / 1e6:.1f} MB compressed · loaded in {t_csv:.2f}s")
print(kinds.to_string())
VIZ["ingest"] = {"rows": len(raw), "cols": raw.shape[1], "mb": r(RAW.stat().st_size / 1e6, 1), "kinds": kinds.to_dict()}
""")
code("""
raw[["name", "neighbourhood_group_cleansed", "room_type", "price", "bathrooms_text", "minimum_nights"]].head(4)
""")

# ------------------------------------------------------------------ 3 profile & validate
md("## Stage 3 · Profile and validate: what is wrong with this data?")
code("""
miss = raw.isna().mean().sort_values(ascending=False)
print(f"columns 100% empty: {(miss == 1).sum()}")
print(f"price missing:      {raw.price.isna().mean():.1%}")
print(f"price stored as:    text, e.g. {raw.price.dropna().iloc[0]!r}")
cols = miss[miss > 0].index[:24]                                       # the missing-value map the video draws
VIZ["missing"] = {"cols": list(cols), "share": [r(miss[c]) for c in cols],
                  "matrix": raw[cols].sample(60, random_state=1).isna().astype(int).values.tolist(), "empty": int((miss == 1).sum())}
""")
code("""
price_num = pd.to_numeric(raw.price.str.replace(r"[$,]", "", regex=True), errors="coerce")
rules = {                                                              # validation rules: each row must pass all of them
    "price present and > 0":        ~(price_num > 0),
    "inside New York City":         ~(raw.latitude.between(40.45, 40.95) & raw.longitude.between(-74.3, -73.65)),
    "accommodates at least 1":      ~(raw.accommodates >= 1),
    "minimum stay ≤ 365 nights":    ~(raw.minimum_nights <= 365),
    "listing id unique":            raw.id.duplicated(),
}
fails = {k: int(v.sum()) for k, v in rules.items()}
for k, v in fails.items(): print(f"{'✗' if v else '✓'} {k:28s} {v:>6,} rows fail")
VIZ["rules"] = fails
""")
code("""
# Is the price missing at random? No: listings that require 30+ nights hide it far more often.
long_stay = raw.minimum_nights >= 30
gap = raw.price.isna().groupby(long_stay).mean()
print(f"price missing · stays under 30 nights: {gap[False]:.1%} · 30+ nights: {gap[True]:.1%}")
print(f"listings that require 30+ nights: {long_stay.mean():.1%}   (NYC's 2023 short-term rental law)")
VIZ["mnar"] = {"short": r(gap[False]), "long": r(gap[True]), "long_share": r(long_stay.mean())}
""")

# ------------------------------------------------------------------ 4 clean
md("## Stage 4 · Clean: one documented step at a time")
code("""
log = []                                                               # every step: what changed, rows and columns after
def step(name, d): log.append({"step": name, "rows": len(d), "cols": d.shape[1]}); print(f"{name:42s} → {len(d):>6,} rows × {d.shape[1]:>2} cols"); return d

df = step("raw", raw.copy())
df = step("drop 100%-empty columns", df.dropna(axis=1, how="all"))
KEEP = ["id", "host_id", "neighbourhood_group_cleansed", "neighbourhood_cleansed", "latitude", "longitude", "property_type",
        "room_type", "accommodates", "bathrooms_text", "bedrooms", "beds", "amenities", "price", "price_quote_price_per_night",
        "minimum_nights", "availability_365", "number_of_reviews", "reviews_per_month", "review_scores_rating", "first_review",
        "last_review", "last_scraped", "host_is_superhost", "calculated_host_listings_count"]
df = step("keep 25 useful columns (drop urls, ids, free text)", df[KEEP])
""")
code("""
before = df.price.dropna().iloc[:3].tolist()
df["price"] = pd.to_numeric(df.price.str.replace(r"[$,]", "", regex=True), errors="coerce")
print("price text → number:", before, "→", df.price.dropna().iloc[:3].tolist())
df = step("drop rows with no price (no label to learn)", df[df.price.notna()])
""")
code("""
def baths(t):
    if pd.isna(t): return np.nan
    if "half" in t.lower(): return 0.5
    return float(re.findall(r"[\\d.]+", t)[0])
print({t: baths(t) for t in ["1 bath", "1.5 shared baths", "Half-bath", "2 private baths"]})
df["baths"] = df.bathrooms_text.map(baths)
df["bath_shared"] = df.bathrooms_text.str.contains("shared", case=False, na=False).astype(int)
""")
code("""
am = df.amenities.map(json.loads)                                       # a JSON list hidden inside one text cell
df["n_amenities"] = am.map(len)
for name, word in [("wifi", "Wifi"), ("kitchen", "Kitchen"), ("aircon", "Air conditioning"), ("washer", "Washer"), ("workspace", "Dedicated workspace")]:
    df[name] = am.map(lambda a: int(word in a))
df["superhost"] = (df.host_is_superhost == "t").astype(int)            # 't' / 'f' text → 1 / 0
scraped = pd.to_datetime(df.last_scraped)
df["days_since_review"] = (scraped - pd.to_datetime(df.last_review)).dt.days   # NaN = never reviewed; filled later, inside the pipeline
df = step("parse baths, amenities, flags and dates", df.drop(columns=["bathrooms_text", "amenities", "host_is_superhost"]))
""")
code("""
df = step("drop impossible minimum stays (> 365 nights)", df[df.minimum_nights <= 365])
cap = 2000
print(f"price 99th percentile: ${df.price.quantile(.99):,.0f} · max: ${df.price.max():,.0f}")
print(f"listings above ${cap:,}/night: {(df.price > cap).sum()} → a different (luxury) market, kept out of this model")
df = step(f"drop prices above ${cap:,} per night", df[df.price <= cap])
""")
code("""
exact = raw.drop(columns=["id", "listing_url"]).duplicated().sum()          # checked on the full raw rows
twins = df.duplicated(subset=["host_id", "latitude", "longitude", "room_type", "accommodates"], keep=False).sum()
print(f"exact duplicate rows (all 88 other columns identical): {exact}")
print(f"look-alike rows (same host, same spot, same room type and size): {twins:,}")
print("→ not errors: one host often rents several identical units in one building. Kept, but they must not leak across the split.")
VIZ["clean_log"] = log; VIZ["dupes"] = {"exact": int(exact), "twins": int(twins)}
VIZ["price_examples"] = before
""")

# ------------------------------------------------------------------ 5 store
md("## Stage 5 · Store: raw layer → clean layer")
code("""
CLEAN = DATA / "clean/listings_2026-09-14.parquet"                       # versioned by snapshot date
df.to_parquet(CLEAN, index=False)
back = pd.read_parquet(CLEAN)
print(f"raw   CSV.gz  {RAW.stat().st_size / 1e6:5.1f} MB · {len(raw):,} rows · {raw.shape[1]} cols · price is text")
print(f"clean Parquet {CLEAN.stat().st_size / 1e6:5.1f} MB · {len(back):,} rows · {back.shape[1]} cols · price is {back.price.dtype}")
VIZ["store"] = {"raw_mb": r(RAW.stat().st_size / 1e6, 1), "clean_mb": r(CLEAN.stat().st_size / 1e6, 1), 
                "raw_cols": raw.shape[1], "clean_cols": df.shape[1], "rows": len(df)}
""")

# ------------------------------------------------------------------ 6 explore
md("## Stage 6 · Explore: let the data answer questions")
code("""
print(df.price.describe(percentiles=[.5]).round(0).to_string())
h_raw = np.histogram(df.price, bins=40); h_log = np.histogram(np.log(df.price), bins=40)
VIZ["hist"] = {"raw": [h_raw[0].tolist(), h_raw[1].round(1).tolist()], "log": [h_log[0].tolist(), h_log[1].round(3).tolist()]}
print(f"skew: raw {df.price.skew():.2f} → log {np.log(df.price).skew():.2f}")
""")
code("""
by = df.groupby(["neighbourhood_group_cleansed", "room_type"]).price.median().unstack().round(0)
print(by.to_string())
VIZ["by_borough"] = {"rows": by.index.tolist(), "cols": by.columns.tolist(), "vals": by.fillna(-1).values.tolist()}
""")
code("""
s = df.sample(4000, random_state=7)
VIZ["map"] = {"lat": s.latitude.round(4).tolist(), "lon": s.longitude.round(4).tolist(), "logp": np.log(s.price).round(3).tolist()}
med = df.groupby(df.minimum_nights >= 30).price.median()
print(f"median price · under 30 nights ${med[False]:.0f} · 30+ nights ${med[True]:.0f}")
VIZ["stay_price"] = {"short": r(med[False], 0), "long": r(med[True], 0)}
""")

# ------------------------------------------------------------------ 7 features
md("## Stage 7 · Feature engineering: turn knowledge into columns")
code("""
TSQ = (40.7580, -73.9855)                                                # Times Square
lat1, lon1, lat2, lon2 = map(np.radians, (df.latitude, df.longitude, TSQ[0], TSQ[1]))
a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
df["km_to_times_sq"] = 6371 * 2 * np.arcsin(np.sqrt(a))
df["long_stay"] = (df.minimum_nights >= 30).astype(int)
top = df.property_type.value_counts().index[:12]
df["property_group"] = df.property_type.where(df.property_type.isin(top), "other")
c = np.corrcoef(df.km_to_times_sq, np.log(df.price))[0, 1]
print(f"distance to Times Square vs log price: correlation {c:.2f}")
NUM = ["accommodates", "bedrooms", "beds", "baths", "bath_shared", "n_amenities", "wifi", "kitchen", "aircon", "washer", "workspace",
       "latitude", "longitude", "km_to_times_sq", "minimum_nights", "long_stay", "availability_365", "number_of_reviews",
       "reviews_per_month", "review_scores_rating", "days_since_review", "superhost", "calculated_host_listings_count"]
CAT = ["neighbourhood_group_cleansed", "neighbourhood_cleansed", "room_type", "property_group"]
print(f"{len(NUM)} numeric + {len(CAT)} categorical features")
VIZ["features"] = {"num": NUM, "cat": CAT, "corr_dist": r(c, 2)}
""")

# ------------------------------------------------------------------ 8 split
md("## Stage 8 · Split before anything learns")
code("""
from sklearn.model_selection import GroupShuffleSplit, train_test_split
new = df[pd.to_datetime(df.first_review) >= "2026-01-01"]               # listings that appeared in 2026: kept for Stage 15
pool = df.drop(new.index)
tr_i, te_i = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(pool, groups=pool.host_id))
train, test = pool.iloc[tr_i], pool.iloc[te_i]
r_tr, r_te = train_test_split(pool, test_size=0.2, random_state=42)
key = ["host_id", "latitude", "longitude", "room_type", "accommodates"]
twin_in = lambda a, b: b.merge(a[key].drop_duplicates(), on=key).shape[0]
print(f"train {len(train):,} · test {len(test):,} · held back (new in 2026) {len(new):,}")
print(f"test rows with a look-alike twin in train · random split: {twin_in(r_tr, r_te):,} · split by host: {twin_in(train, test):,}")
VIZ["split"] = {"train": len(train), "test": len(test), "new": len(new), "twins_random": twin_in(r_tr, r_te), "twins_group": twin_in(train, test)}
""")

# ------------------------------------------------------------------ 9 pipeline
md("## Stage 9 · One pipeline: preprocessing learns from the training set only")
code("""
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
import lightgbm as lgb

def prep(scale):
    num = [SimpleImputer(strategy="median")] + ([StandardScaler()] if scale else [])
    return ColumnTransformer([("num", make_pipeline(*num), NUM),
                              ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore", min_frequency=20)), CAT)])
Xtr, Xte, ytr, yte = train[NUM + CAT], test[NUM + CAT], np.log(train.price), np.log(test.price)   # learn log price: the skew is gone
p = prep(True).fit(Xtr); Z = p.transform(Xtr); Z = Z.toarray() if hasattr(Z, "toarray") else Z
print(f"missing values in train before: {int(Xtr.isna().sum().sum()):,} · after the imputer: {int(np.isnan(Z).sum())}")
print(f"{len(NUM + CAT)} input columns → {Z.shape[1]} model columns after one-hot encoding")
print(f"bedrooms median learned from TRAIN only: {p.named_transformers_['num'][0].statistics_[NUM.index('bedrooms')]}")
""")

# ------------------------------------------------------------------ 10 leakage
md("## Stage 10 · The leakage trap")
code("""
leak_cols = NUM + CAT + ["price_quote_price_per_night"]
lq = lambda d: d.assign(price_quote_price_per_night=pd.to_numeric(d.price_quote_price_per_night.astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce"))
NUMQ = NUM + ["price_quote_price_per_night"]
leaky = Pipeline([("prep", ColumnTransformer([("num", SimpleImputer(strategy="median"), NUMQ), ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CAT)])),
                  ("m", lgb.LGBMRegressor(n_estimators=300, random_state=42, verbose=-1))]).fit(lq(train)[leak_cols], ytr)
r2_leak = r2_score(yte, leaky.predict(lq(test)[leak_cols]))
print(f"with 'price_quote_price_per_night': R² {r2_leak:.3f}  ← too good to be true")
print(f"correlation with the real price: {np.corrcoef(lq(test).price_quote_price_per_night.fillna(0), test.price)[0, 1]:.3f} → it IS the price. Removed.")
VIZ["leak"] = {"r2": r(r2_leak)}
""")

# ------------------------------------------------------------------ 11 model
md("## Stage 11 · Model: baseline first, then better models")
code("""
base = train.groupby(["neighbourhood_group_cleansed", "room_type"]).price.median()
p_base = test.set_index(["neighbourhood_group_cleansed", "room_type"]).index.map(base).fillna(train.price.median()).values
lin = Pipeline([("prep", prep(True)), ("m", Ridge(alpha=1.0))]).fit(Xtr, ytr)
gbm = Pipeline([("prep", prep(False)), ("m", lgb.LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=63, subsample=0.8, subsample_freq=1,
                                                                colsample_bytree=0.8, random_state=42, verbose=-1))]).fit(Xtr, ytr)
res = {}
for name, pred in [("Baseline: median of borough × room type", p_base), ("Ridge regression", np.exp(lin.predict(Xte))), ("LightGBM", np.exp(gbm.predict(Xte)))]:
    res[name] = {"mae": r(mean_absolute_error(test.price, pred), 1), "r2": r(r2_score(yte, np.log(pred)))}
    print(f"{name:42s} MAE ${res[name]['mae']:>6.1f}   R² (log) {res[name]['r2']:.3f}")
VIZ["models"] = res
""")
code("""
# What the random split would have told us (look-alike twins on both sides):
gbm_r = Pipeline([("prep", prep(False)), ("m", lgb.LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=63, subsample=0.8, subsample_freq=1,
                                                                  colsample_bytree=0.8, random_state=42, verbose=-1))]).fit(r_tr[NUM + CAT], np.log(r_tr.price))
mae_r = mean_absolute_error(r_te.price, np.exp(gbm_r.predict(r_te[NUM + CAT])))
print(f"LightGBM MAE · random split ${mae_r:.1f} (looks better) · split by host ${res['LightGBM']['mae']:.1f} (honest)")
VIZ["split"]["mae_random"] = r(mae_r, 1)
""")

# ------------------------------------------------------------------ 12 evaluate
md("## Stage 12 · Evaluate: where does it go wrong?")
code("""
pred = np.exp(gbm.predict(Xte)); err = (pred - test.price).abs()
by_b = pd.DataFrame({"b": test.neighbourhood_group_cleansed, "e": err, "p": test.price}).groupby("b").agg(mae=("e", "mean"), median_price=("p", "median"), n=("e", "size")).round(1)
print(by_b.to_string())
print(f"within ±25% of the real price: {(err / test.price <= .25).mean():.1%} of test listings")
k = np.random.RandomState(3).choice(len(test), 1500, replace=False)
VIZ["eval"] = {"by_borough": by_b.reset_index().values.tolist(), "within25": r((err / test.price <= .25).mean()),
               "scatter": [test.price.values[k].round(0).tolist(), pred[k].round(0).tolist()]}
ex = test.assign(pred=pred.round(0)).iloc[[3, 50, 400]][["neighbourhood_cleansed", "room_type", "accommodates", "price", "pred"]]
print(ex.to_string(index=False)); VIZ["examples"] = ex.values.tolist()
""")

# ------------------------------------------------------------------ 13 explain
md("## Stage 13 · Explain: which features matter?")
code("""
from sklearn.inspection import permutation_importance
smp = np.random.RandomState(0).choice(len(Xte), 2000, replace=False)
pi = permutation_importance(gbm, Xte.iloc[smp], yte.iloc[smp], n_repeats=3, random_state=0, scoring="neg_mean_absolute_error")
imp = pd.Series(pi.importances_mean, index=NUM + CAT).sort_values(ascending=False).head(8)
print("shuffle one column, measure how much worse the model gets:"); print(imp.round(3).to_string())
VIZ["importance"] = [[k_, r(v_)] for k_, v_ in imp.items()]
""")

# ------------------------------------------------------------------ 14 deploy
md("## Stage 14 · Deploy: save the whole pipeline, serve predictions")
code("""
import joblib
PATH = MOD / "price_pipeline.joblib"
joblib.dump(gbm, PATH); model = joblib.load(PATH)                       # cleaning rules + encoder + model in one file
print(f"saved {PATH.name}: {PATH.stat().st_size / 1e6:.1f} MB")
""")
code("""
from fastapi import FastAPI
import warnings
with warnings.catch_warnings():                                           # the test client warns about its own httpx dependency
    warnings.simplefilter("ignore"); from fastapi.testclient import TestClient
app = FastAPI()
@app.post("/predict")
def predict(listing: dict):
    X = pd.DataFrame([listing]).reindex(columns=NUM + CAT)              # missing fields become NaN → the pipeline's imputers fill them
    return {"price_per_night": round(float(np.exp(model.predict(X)[0])), 2)}

one = test.iloc[50][NUM + CAT].replace({np.nan: None}).to_dict()
t = time.time(); out = TestClient(app).post("/predict", json=one).json(); ms = (time.time() - t) * 1000
print(f"POST /predict → {out} in {ms:.0f} ms   (real price ${test.iloc[50].price:.0f})")
VIZ["deploy"] = {"mb": r(PATH.stat().st_size / 1e6, 1), "out": out["price_per_night"], "real": r(test.iloc[50].price, 0), "ms": r(ms, 0),
                 "listing": {k_: one[k_] for k_ in ["neighbourhood_cleansed", "room_type", "accommodates", "bedrooms", "baths", "n_amenities"]}}
""")

# ------------------------------------------------------------------ 15 monitor
md("## Stage 15 · Monitor: does the model still hold on new data?")
code("""
def psi(a, b, bins=10):                                                  # population stability index: > 0.2 = real shift
    q = np.unique(np.quantile(a.dropna(), np.linspace(0, 1, bins + 1)))
    pa = np.histogram(a.dropna(), q)[0] / a.notna().sum() + 1e-6; pb = np.histogram(b.dropna().clip(q[0], q[-1]), q)[0] / b.notna().sum() + 1e-6
    return float(np.sum((pa - pb) * np.log(pa / pb)))
drift = {c_: r(psi(train[c_], new[c_])) for c_ in ["price", "accommodates", "n_amenities", "number_of_reviews", "km_to_times_sq", "minimum_nights"]}
for c_, v_ in drift.items(): print(f"{c_:20s} PSI {v_:.3f} {'⚠ shift' if v_ > .2 else 'ok'}")
mae_new = mean_absolute_error(new.price, np.exp(model.predict(new[NUM + CAT])))
print(f"MAE on {len(new):,} listings new in 2026 (never seen): ${mae_new:.1f} vs test ${res['LightGBM']['mae']:.1f}")
VIZ["monitor"] = {"psi": drift, "mae_new": r(mae_new, 1), "n_new": len(new)}
""")

md("## Save the visual data")
code("""
json.dump(VIZ, open(REP / "viz.json", "w"), default=float)
print("saved", ", ".join(VIZ))
""")

nb = nbf.v4.new_notebook(); nb.cells = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "pipeline.ipynb"); print(len(C), "cells")
