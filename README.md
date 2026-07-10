# dm_optimization_algorithm

Forecasting model for the Google Ads recommendation engine. v1 forecasts
**conversion volume** per ad group, 7 days out, using historical performance
already sitting in Snowflake (`HOURLY_STATS_MAT`). This is stage one of a
larger plan: forecast now, rule-based recommendations next, autonomous
bid/budget execution later (see "Where this fits" below).

## Config-driven target metrics

`config/config.yaml` -> `model.target_metrics` (plural) is the list of
metrics `scripts/build_dataset.py` labels and `src/train.py` trains a full
model suite for. Supported metrics live in the registry in `src/metrics.py`
(`conversions`, `clicks`, `cpa`, `cpc`, `roas`, `conversion_rate`, `ctr`);
each one just declares which raw columns it needs, how to compute it, and
whether it's a ratio (`is_ratio`) rather than a directly summable
count/currency total.

These are **independent models, not one joint multi-output model** — each
metric in `target_metrics` gets its own full comparison (LightGBM +
Poisson/Tweedie variants, CatBoost, XGBoost + Poisson/Tweedie variants,
hurdle, SARIMAX + hybrid where applicable) and its own saved models/
`models/<metric>_h<horizon>d_metrics.json`, run in a loop by `src/train.py`'s
`main()`. This was a deliberate choice over a shared-tree joint model
(e.g. CatBoost's `MultiRMSE`) to avoid touching `src/recommend.py`, the
prediction-accuracy Snowflake table, and the frontend tabs, all of which
still assume one scalar prediction per entity — and because specialist models
per metric are generally more accurate than one model splitting its capacity
across differently-shaped targets.

Two model families are metric-conditional, not run unconditionally for
every metric:
- **Hurdle** only runs when the label actually has both zero and nonzero
  rows in train AND val (see `train_hurdle_model`'s guard in `src/train.py`).
  This is automatically true for genuinely zero-inflated metrics
  (`conversions`, and `clicks` at ~30% zero on real data) and automatically
  false for ratio metrics like `cpa`/`cpc` once undefined (zero-denominator)
  rows are dropped — those end up effectively always-nonzero, so a "will
  this convert at all" classifier has nothing to learn. Skipped cleanly with
  a printed explanation rather than erroring.
- **SARIMAX + the SARIMAX/GBM hybrid** only run for non-ratio metrics
  (`MetricDefinition.is_ratio == False`). SARIMAX forecasts a single
  per-entity raw series and sums it over the horizon — that has no direct
  meaning for a ratio label (`cpa`'s label is future-spend-sum /
  future-conversions-sum, not a summable series of its own). Extending this
  to ratio metrics would mean forecasting each raw component separately and
  combining them, not attempted here.

`model.target_metric` (singular) still exists separately and is read by
`src/recommend.py`/`src/evaluate_accuracy.py` — the live recommendation-
scoring model and prediction-accuracy monitoring still act on one scalar
metric (`conversions`) for now. Extending recommendations/accuracy-tracking
to the other 3 metrics is a bigger follow-on task (new guardrail schema,
frontend sections, Snowflake tracking per metric) not attempted here.

Adding a metric to the sweep is: add it to both the registry in
`src/metrics.py` and `target_metrics` in the config, then
```
python -m scripts.build_dataset
python -m src.train
```
No other changes needed to the Snowflake pull or feature engineering code —
features are metric-agnostic by design. What *does* need redoing after
adding a metric: backtesting/tuning (a different metric has a different
error distribution) and any recommendation thresholds built on top of the
forecast, if you extend recommendations to it.

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

# 2. Train baseline, LightGBM, XGBoost, and a hurdle model; see backtest metrics
python -m src.train

# 3. Score the latest data into recommendation records
python -m src.recommend
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
  train.py                 baseline + LightGBM + XGBoost + hurdle model training/evaluation
  recommend.py             scores latest data into guardrailed recommendation records
  evaluate_accuracy.py     scores past recommendations against real outcomes once elapsed
scripts/
  build_dataset.py         orchestrates the pull -> features -> labels -> parquet
  generate_synthetic_data.py   local test data, no Snowflake needed
  tune_hyperparameters.py  Optuna hyperparameter search (separate, weekly-scheduled job)
sql/
  pull_ad_group_daily_performance.sql          reference copy of the Snowflake training query
  grant_ml_recommendations_write.sql           creates ML_RECOMMENDATIONS + narrow write grant
  grant_ml_prediction_accuracy_write.sql       creates ML_PREDICTION_ACCURACY + narrow write grant
```

## Recommendation engine (`src/recommend.py`)

Scores the most recent row of history for every active ad group against the
trained model, compares the forecast to that entity's own trailing actual
performance over the same-length window, and flags the ones worth a
marketer's attention. Config lives under `recommendations:` in
`config/config.yaml`.

Guardrails, and why:
- **Ranked within each client, not a fixed % threshold** — account sizes and
  conversion volume vary enormously, and raw percent-change thresholds are
  unstable on low-count data (see the MAPE discussion in `src/train.py`).
  Only the most extreme `flag_percentile` (default 10%) of predicted-vs-baseline
  swings *within that client's own ad groups* get surfaced.
- **Spend floor** (`min_trailing_spend`) — suppresses recommendations on
  trivial-spend ad groups entirely; there's little at stake, and it's exactly
  where the model's zero-actual segment is least reliable directionally.
- **Confidence segment tag** — every record is tagged `high` or `low`
  confidence based on whether the entity has real trailing conversion volume,
  tying directly back to the segmented eval in `src/train.py` (R² ~0.95 on
  the nonzero segment vs. near-0/undefined on the always-zero segment).
- **`requires_human_review: true` on every record, always** — this is
  recommendation-only. Nothing here writes to Google Ads.

Each run also `INSERT`s its output into
`FIVETRAN_DATABASE.GOOGLE_ADS.ML_RECOMMENDATIONS` (see
`sql/grant_ml_recommendations_write.sql`), which the Worker's
`/api/report/google-ads/ml-recommendations` route reads back for the
frontend's "ML Recommendations" tab. This needed a narrow, one-table
`INSERT`+`SELECT` grant for `SLACK_BOT_RO` — it stays read-only everywhere
else. Never `UPDATE`s or `DELETE`s an old run: every batch is left in place,
tagged with its own `generated_at`, since that history is exactly what the
feedback/outcome logging loop (task on the roadmap below) will need later.

Action taxonomy (v1): `FORECASTED_ZERO_HIGH_SPEND` (red — forecasted ~0
conversions despite real trailing spend), `FORECASTED_DECLINE` (yellow —
among a client's steepest predicted drops), `FORECASTED_GROWTH_OPPORTUNITY`
(green — among a client's strongest predicted gains). Deliberately avoids
prescribing a specific bid/budget delta (e.g. "raise budget 23%") — that
would need the causal/elasticity data this repo doesn't have yet (see
"Why this doesn't include bid/budget change history" below). v1 tells you
*what to look at and which direction*, not *the exact dial to turn*.

## Prediction accuracy monitoring (`src/evaluate_accuracy.py`)

Scores past recommendations against what actually happened, once each
forecast window has genuinely elapsed, and appends the result to
`FIVETRAN_DATABASE.GOOGLE_ADS.ML_PREDICTION_ACCURACY` (see
`sql/grant_ml_prediction_accuracy_write.sql` for the table + the same narrow
`INSERT`+`SELECT`-only grant pattern used for `ML_RECOMMENDATIONS`). Runs
entirely as a single `INSERT ... SELECT` executed in Snowflake — a range join
between `ML_RECOMMENDATIONS` and `HOURLY_STATS_MAT` on `(stat_date,
stat_date + horizon_days]` per entity, so there's no need to pull rows into
pandas for this. A `NOT EXISTS` check guarantees each (entity, prediction
batch) pair is scored exactly once; nothing here ever `UPDATE`s or `DELETE`s
a row.

Scheduled daily (`.github/workflows/evaluate_accuracy.yml`) — no need to run
it more often, since a 7-day horizon means most predictions on any given day
aren't eligible for evaluation yet anyway. The frontend's "Prediction
Accuracy" tab reads this table back via `/api/report/google-ads/ml-accuracy`,
showing MAE over time vs. the naive baseline (is the model's real-world track
record holding up, not just its backtest) and directional accuracy by
recommendation type (when we flagged a decline, did it actually decline).

This is also the foundation for the feedback/outcome logging loop on the
roadmap below — once a table of "predicted X, actual Y" exists across enough
history, it becomes the natural training data for the causal/autonomous
phase.

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

1. **This repo**: forecast conversion volume per ad group, 7 days out
   (`src/train.py`), and turn that forecast into guardrailed recommendation
   records (`src/recommend.py`) — no learned causal effects yet.
2. **Next**: recommendation API + frontend, with human approve/reject.
3. **Later still**: feedback logging of every recommendation and its actual
   outcome, which becomes the training data for autonomous bid/budget
   execution on narrow, low-risk, high-confidence actions.
