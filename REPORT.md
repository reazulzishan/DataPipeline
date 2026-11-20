# The End-to-End Data Science Pipeline, on real messy data — project report

A CS Funda tutorial: one real, uncleaned dataset taken through all 15 stages of a data science pipeline, from the raw
file to a live prediction API and drift monitoring.

**Question:** what is a fair nightly price for an Airbnb listing in New York City?
**Metric:** mean absolute error (MAE) in dollars, on listings the model has never seen, against a simple baseline.

## Contents

| Path | What it is |
|---|---|
| `notebooks/make_notebook.py` | Generates the tutorial notebook |
| `notebooks/pipeline.ipynb` | The notebook (clean) |
| `notebooks/pipeline_live.ipynb` | The same notebook, executed: every number in the video comes from these outputs |
| `reports/viz.json` | Results and plot data saved by the notebook |
| `video/extract_examples.py` → `reports/examples.json` | Real example rows, the learning curve, imputer demo and parallel-coordinates data for the video. It retrains with the same split and model and **stops if the split or the MAE differs** from the notebook |
| `recording/` | Scripts that run the notebook in JupyterLab and capture the cells shown in the video |
| `video/script.md`, `video/make_vo.py`, `video/film/build_dp.py` | Narration script, voice-over and the video build (HyperFrames) |

## The 15 stages and the real results (live run, 2026-11-19)

| # | Stage | Result |
|---|---|---|
| 1 | Question & metric | Predict nightly price; MAE in $; must beat a baseline |
| 2 | Ingest | 29,918 rows × 90 columns, 16.0 MB; the raw file is read-only |
| 3 | Profile & validate | 13 columns 100% empty; price missing for 35.1%. Rules: 10,512 rows fail "price exists and > 0", 24 fail "minimum stay < 365" |
| — | Missing is not random | price missing for 11.0% of short stays vs 40.9% of 30+ night stays; 80.7% of listings require 30+ nights (NYC short-term rental law) |
| 4 | Clean | 90 → 77 → 25 columns; price text → number; 19,406 rows with a price; bathrooms text → number + shared flag; amenities JSON → count + 5 flags; 7 impossible stays and 194 listings above $2,000 removed; final 19,205 × 32 |
| — | Duplicates | 0 exact copies; 2,282 look-alike "twins" (same host, location, room type, size) |
| 5 | Store | Dated Parquet file, 1.1 MB; raw layer untouched |
| 6 | Explore | skew 2.48 → 0.30 after log; private room Manhattan $251 vs Brooklyn $101; median $370 (< 30 nights) vs $146 (30+) |
| 7 | Features | distance to Times Square (corr. −0.42 with log price), long-stay flag, grouped property types: 27 features |
| 8 | Split | grouped by host: train 14,526 / test 3,169, plus 1,510 listings new in 2026 held back. Twins across the split: random 342 → by host 0 |
| 9 | One pipeline | median imputer + one-hot (min frequency 20), fit on train only: 27 columns → 145 |
| 10 | Leakage trap | `price_quote_price_per_night` gives R² = 1.000: it is the price. Removed |
| 11 | Model | baseline (median per borough × room type) $140.3 → Ridge $85.6 → LightGBM $72.8. The random split would report $65.6 (too optimistic) |
| 12 | Evaluate | 61.4% within ±25%; Manhattan MAE $105.0 (highest) |
| 13 | Explain | permutation importance: minimum_nights 0.203, km_to_times_sq 0.093, bath_shared 0.054, accommodates 0.053 |
| 14 | Deploy | whole pipeline in one 4.6 MB joblib file behind FastAPI: Mill Basin private room → $109.29 (real $113) in 20 ms |
| 15 | Monitor | PSI on new 2026 listings: number_of_reviews 4.989, minimum_nights 0.323; MAE on new data $85.0 → retrain |

Three rules: never edit raw data, split before anything learns, always compare with a baseline.

## Honest limits

- One city, one snapshot. The model predicts the *listed* price, not what guests actually pay.
- 194 listings above $2,000 are excluded, so the model does not cover that market.
- Twins are defined by an exact match on host, location, room type and size; looser near-duplicates may remain.
- The deploy timing is one local request through FastAPI's test client, not a load test.

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install pandas pyarrow scikit-learn lightgbm fastapi httpx joblib matplotlib jupyter nbclient
mkdir -p data/raw && curl -L -o data/raw/listings.csv.gz \
  https://data.insideairbnb.com/united-states/ny/new-york-city/2026-09-14/data/listings.csv.gz
python notebooks/make_notebook.py
python recording/run_nb.py notebooks/pipeline.ipynb notebooks/pipeline_live.ipynb
python video/extract_examples.py
```

The notebook writes the clean Parquet file to `data/clean/` and the trained pipeline to `models/`; neither is stored here.

## Data and credits

Data: [Inside Airbnb](https://insideairbnb.com), New York City, snapshot 2026-09-14, licensed CC BY 4.0. The data and the
trained model are not stored in this repository. Tools: pandas, pyarrow, scikit-learn, LightGBM, FastAPI.
