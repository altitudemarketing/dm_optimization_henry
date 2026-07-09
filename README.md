# dm_optimization_algorithm

Forecasting model for the Google Ads recommendation engine. v1 forecasts
**conversion volume** per ad group, 7 days out, using historical performance
already sitting in Snowflake (`HOURLY_STATS_MAT`). This is stage one of a
larger plan: forecast now, rule-based recommendations next, autonomous
bid/budget execution later (see "Where this fits" below).

## Config-driven target metric

`config/config.yaml` -> `model.target_metric` is the only thing that
determines what the model predicts. Supported metrics live in the registry
in `src/metrics.py` (`conversions`, `cpa`, `roas`, `conversion_rate`, `ctr`);
each one just declares which raw columns it needs and how to compute it.

Switching targets later is: edit `target_metric` in the config, then
```
python -m scripts.build_dataset
python -m src.train
```
No changes needed to the Snowflake pull, feature engineering, or training
code — this was verified by actually running the pipeline end-to-end against
both `conversions` and `cpa` during development. What *does* need redoing
after a switch: backtesting/tuning (a different metric has a different error
distribution) and any recommendation thresholds built on top of the forecast.

## Data source

Pulls from `FIVETRAN_DATABASE.GOOGLE_ADS.HOURLY_STATS_MAT` — the agency's
existing curated, multi-client dynamic table (see
`speed_snowflake/snowflake/ddl/google_ads/hourly_stats_mat.sql`), collapsed
from hour/device/network detail down to one row per
(client, campaign, ad_group, day). See `sql/pull_ad_group_daily_performance.sql`
for the reference query.

**Multi-tenant note:** Google Ads `ad_group_id` values are assigned per
Google Ads account and are not guaranteed unique across this agency's
different clients. Every part of this pipeline keys on `entity_id` =
`client_id::campaign_id::ad_group_id`, never `ad_group_id` alone — this
matters, since a naive groupby on `ad_group_id` could silently merge two
different clients' data.

## Setup

```
pip install -r requirements.txt
cp .env.example .env   # fill in Snowflake connection details
```

Snowflake auth uses JWT key-pair signing via the SQL API v2
(`src/snowflake_client.py`) rather than adding a new driver dependency. Use a
dedicated read-only service account/role scoped to only the tables this
pipeline needs.

## Running the pipeline

```
# 1. Build the dataset (pulls from Snowflake, builds features + labels)
python -m scripts.build_dataset

# 2. Train baseline + LightGBM, see backtest metrics
python -m src.train
```

Add `--use-cache` to `build_dataset` to skip the Snowflake pull and reuse
whatever's cached at `data/raw_ad_group_daily_performance.parquet` (useful
while iterating on features/labels).

### Testing without Snowflake access

```
python -m scripts.generate_synthetic_data
python -m scripts.build_dataset --use-cache
python -m src.train
```

This generates synthetic multi-client performance data matching the real
schema, so the pipeline can be validated end-to-end without credentials.
This is exactly how the pipeline was smoke-tested during development —
confirmed a ~59% MAE improvement over the naive baseline on synthetic data
(real numbers will differ once run against actual Snowflake data).

## Structure

```
config/config.yaml       target metric, grain, horizon, rolling windows, split sizes
src/
  config_loader.py        loads + validates config.yaml
  snowflake_client.py      JWT key-pair auth + SQL API v2 query execution
  metrics.py               config-driven metric registry (edit here to add new targets)
  data_loader.py           pulls + aggregates HOURLY_STATS_MAT to daily grain
  features.py              rolling-window, calendar, trend features (metric-agnostic)
  labels.py                leak-safe forward-looking label construction (metric-specific)
  dataset.py               time-aware train/val/test split
  train.py                 baseline + LightGBM training and evaluation
scripts/
  build_dataset.py         orchestrates the pull -> features -> labels -> parquet
  generate_synthetic_data.py   local test data, no Snowflake needed
sql/pull_ad_group_daily_performance.sql   reference copy of the Snowflake query
```

## Why this doesn't include bid/budget change history

An earlier pass tried to build a before/after training set from Snowflake's
bid/budget/bidding-strategy history tables, to eventually support causal
("if we raise budget by X%, conversions change by Y") modeling for autonomous
execution. That data turned out to be too shallow (~14 days of Fivetran
history) and too noisy (sync-artifact timestamps, oscillating duplicate
rows) to be usable yet. That effort is deferred, not abandoned — see task
list item on the feedback/outcome logging loop. Once the recommendation tool
is live and logging its own "recommended X, human did Y, outcome Z" data
going forward, that becomes the clean training set for the causal/autonomous
phase, rather than trying to reconstruct it from imperfect historical logs.

## Where this fits

1. **This repo**: forecast conversion volume per ad group, 7 days out.
2. **Next**: rule-based recommendation layer on top of the forecast
   (e.g. "predicted conversions dropping + budget underutilized -> flag for
   review"), no learned causal effects yet.
3. **Later**: recommendation API + frontend, with human approve/reject.
4. **Later still**: feedback logging of every recommendation and its actual
   outcome, which becomes the training data for autonomous bid/budget
   execution on narrow, low-risk, high-confidence actions.
