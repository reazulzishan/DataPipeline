"""Real examples for the video, read straight from the raw file (no cleaning applied): -> reports/examples.json"""
import json, numpy as np, pandas as pd
from pathlib import Path

R = Path(__file__).resolve().parents[1]
raw = pd.read_csv(R / "data/raw/listings.csv.gz", low_memory=False)
V = json.loads((R / "reports/viz.json").read_text())
OUT = {}

cols = ["name", "neighbourhood_group_cleansed", "room_type", "price", "bathrooms_text", "minimum_nights", "price_quote_price_per_night"]
OUT["head"] = {"cols": cols, "rows": raw[cols].head(4).astype(object).where(raw[cols].head(4).notna(), None).values.tolist()}

p = pd.to_numeric(raw.price.str.replace(r"[$,]", "", regex=True), errors="coerce").dropna()
OUT["prices"] = {"n": int(len(p)), "max": float(p.max()), "above_2000": int((p > 2000).sum()),
                 "sample": np.sort(p.sample(1500, random_state=4).values).round(2).tolist(), "top": np.sort(p.values)[-12:].round(2).tolist()}

am = raw.amenities.iloc[1]
OUT["amenities"] = {"text": am, "n": len(json.loads(am))}

d = raw[raw.price.notna()]
key = ["host_id", "latitude", "longitude", "room_type", "accommodates"]
g = d[d.duplicated(subset=key, keep=False)].sort_values(key)
pair = g.groupby(key).head(2).iloc[:2]
OUT["twins"] = pair[["id", "name", "neighbourhood_cleansed", "room_type", "accommodates", "price"]].astype(object).values.tolist()
assert pair[key].nunique().max() == 1, "twin pair must share host, location, room type and size"
assert len(set(pair.id)) == 2

(R / "reports/examples.json").write_text(json.dumps(OUT, default=str))
print("prices above $2,000:", OUT["prices"]["above_2000"], "· max", OUT["prices"]["max"])
print("twins:", OUT["twins"])
print("amenities:", OUT["amenities"]["n"], OUT["amenities"]["text"][:120])

# ---- inner-mechanism data: same clean file, same features, same split and model as the notebook (checked below)
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
df = pd.read_parquet(R / "data/clean/listings_2026-09-14.parquet")
TSQ = (40.7580, -73.9855)
lat1, lon1, lat2, lon2 = map(np.radians, (df.latitude, df.longitude, TSQ[0], TSQ[1]))
a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
df["km_to_times_sq"] = 6371 * 2 * np.arcsin(np.sqrt(a)); df["long_stay"] = (df.minimum_nights >= 30).astype(int)
top = df.property_type.value_counts().index[:12]; df["property_group"] = df.property_type.where(df.property_type.isin(top), "other")
NUM, CAT = V["features"]["num"], V["features"]["cat"]
new = df[pd.to_datetime(df.first_review) >= "2026-01-01"]; pool = df.drop(new.index)
tr_i, te_i = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(pool, groups=pool.host_id))
train, test = pool.iloc[tr_i], pool.iloc[te_i]
assert (len(train), len(test), len(new)) == (V["split"]["train"], V["split"]["test"], V["split"]["new"]), "split differs from the notebook"
prep = ColumnTransformer([("num", make_pipeline(SimpleImputer(strategy="median")), NUM),
                          ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore", min_frequency=20)), CAT)])
gbm = Pipeline([("prep", prep), ("m", lgb.LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=63, subsample=0.8, subsample_freq=1,
                                                         colsample_bytree=0.8, random_state=42, verbose=-1))]).fit(train[NUM + CAT], np.log(train.price))
Xt = gbm["prep"].transform(test[NUM + CAT])
stages = [1, 10, 25, 50, 100, 200, 300, 400, 600, 800]
curve = [round(float(mean_absolute_error(test.price, np.exp(gbm["m"].predict(Xt, num_iteration=k)))), 1) for k in stages]
assert abs(curve[-1] - V["models"]["LightGBM"]["mae"]) <= 0.1, f"model differs from the notebook: {curve[-1]}"
OUT["curve"] = {"trees": stages, "mae": curve}
# the imputer and the one-hot encoder at work on real training rows
med = float(gbm["prep"].named_transformers_["num"][0].statistics_[NUM.index("beds")])
rows = train[train.beds.isna()].groupby("room_type").head(1).head(4)
cats = list(gbm["prep"].named_transformers_["cat"][1].categories_[CAT.index("room_type")])
OUT["impute"] = {"median_beds": med, "cats": cats,
                 "rows": [[r_.room_type, None, float(r_.bedrooms) if pd.notna(r_.bedrooms) else None] for r_ in rows.itertuples()]}
# parallel coordinates: 400 real listings
AX = ["accommodates", "bedrooms", "baths", "n_amenities", "km_to_times_sq", "price"]
s_ = df.sample(400, random_state=11)[AX].dropna()
lo, hi = df[AX].quantile(.01), df[AX].quantile(.99)
OUT["parallel"] = {"axes": AX, "lo": lo.round(2).tolist(), "hi": hi.round(2).tolist(), "rows": s_.clip(lo, hi, axis=1).round(2).values.tolist()}
(R / "reports/examples.json").write_text(json.dumps(OUT, default=str))
print("curve", OUT["curve"]); print("impute", OUT["impute"]); print("parallel rows", len(OUT["parallel"]["rows"]))
