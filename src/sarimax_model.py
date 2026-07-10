"""
Two additional forecasting approaches, evaluated alongside LightGBM/XGBoost/
hurdle in src/train.py:

  1. A simple per-entity SARIMAX benchmark -- classic seasonal ARIMA, fit
     independently on each entity's own daily history, with NO engineered
     features and no cross-entity information at all. This is a genuinely
     different kind of baseline than the others: pure autoregressive/
     seasonal structure vs. the tree models' feature-driven approach.
  2. A SARIMAX + LightGBM hybrid: SARIMAX captures each entity's own
     autoregressive/seasonal pattern, then a LightGBM model (using the same
     rolling-window features as everywhere else) is trained on SARIMAX's
     RESIDUALS -- whatever nonlinear or cross-entity structure a purely
     per-series linear model can't capture. Final prediction = SARIMAX's
     forecast + the residual model's correction.

Important simplifications -- this is deliberately a SIMPLE benchmark, not a
production-grade time series system:
  - Fixed (p,d,q)(P,D,Q,s) order for every entity, no per-entity order search
    (that would be slow across hundreds of entities and is overkill for a
    benchmark). s=7 for weekly seasonality, matching the day-of-week features
    already used elsewhere.
  - Each entity is fit ONCE on the training period only, then its forecast is
    extended across the full val+test horizon in a single shot. This means
    SARIMAX is forecasting further and further past its last real observation
    as the test period goes on -- compounding its own forecast error -- unlike
    LightGBM/XGBoost, which get fresh trailing features for every single test
    row. This is a real, known disadvantage of a static single-fit SARIMAX
    vs. a continually-refreshed regression approach; refitting per row would
    be more accurate but far too slow across hundreds of entities run twice a
    day, and isn't what "simple benchmark" was asking for.
  - Entities with too little history (< MIN_HISTORY_DAYS) or a degenerate
    (e.g. constant/all-zero) series fall back to the same naive
    trailing-average prediction used in src/train.py's naive_baseline, rather
    than forcing a fit that would likely fail to converge or fail outright.
  - The residual model's training target uses SARIMAX's in-sample FITTED
    values (one-step-ahead, within the training window) as a stand-in for
    "what SARIMAX would have predicted" there -- these tend to look better
    than true out-of-sample forecasts would, so the residual model's training
    signal runs a little optimistic. Worth knowing if this hybrid's backtest
    looks surprisingly strong.
"""

import time
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

DEFAULT_ORDER = (1, 1, 1)
# Seasonal order deliberately does NOT use seasonal differencing (D=0, not 1)
# -- a double-differenced (regular d=1 AND seasonal D=1) SARIMAX extrapolates
# an estimated trend/seasonal-drift component forward, and over the ~30+ day
# forecast horizon this needs (test + val + horizon_days, extended from a
# single fit point -- see run_sarimax), that trend can compound to wildly
# diverging forecasts on short or noisy real series. D=0 with seasonal AR/MA
# terms still captures weekly seasonality without that instability. Confirmed
# against real Snowflake data: D=1 produced test RMSE=20.8 against MAE=3.17 --
# a handful of entities blowing up to extreme values (RMSE >> MAE is the
# signature of a few huge outliers, not uniformly-bad predictions).
DEFAULT_SEASONAL_ORDER = (1, 0, 1, 7)  # weekly seasonality
MIN_HISTORY_DAYS = 21                   # ~3 weekly cycles -- minimum to even attempt a seasonal fit
# Hard safety net on top of the order change above -- even a well-behaved
# order can occasionally diverge on a pathological series, and the cost of
# checking is negligible. Caps each entity's predicted horizon-day sum at a
# generous multiple of its own best training day -- generous enough to never
# constrain a real forecast, tight enough to catch runaway extrapolation
# before it reaches evaluate()/the hybrid's residual target.
MAX_HISTORICAL_DAILY_MULTIPLE = 5


def _complete_daily_index(series: pd.Series) -> pd.Series:
    """
    Reindexes an entity's daily series to a complete, gap-free date range,
    filling missing days with 0. SARIMAX needs a regularly-spaced index, and
    a missing day here almost always means zero activity that day, not
    missing/uncollected data (see data_loader.py/features.py -- the Snowflake
    pull can simply have no row for a day with no impressions at all).
    """
    if series.empty:
        return series
    full_index = pd.date_range(series.index.min(), series.index.max(), freq="D")
    return series.reindex(full_index, fill_value=0)


def _naive_fallback(daily_series: pd.Series, horizon_days: int) -> float:
    """Same fallback logic as src/train.py's naive_baseline -- trailing daily
    average x horizon -- used when a SARIMAX fit isn't attempted or fails."""
    if len(daily_series) == 0:
        return 0.0
    window = daily_series.tail(7) if len(daily_series) >= 7 else daily_series
    return float(max(window.mean(), 0) * horizon_days)


def _fit_one_entity(train_series: pd.Series, order, seasonal_order):
    """
    Fits SARIMAX on one entity's training-period daily series. Returns None
    (not an exception) for too-short or degenerate (zero-variance) series, or
    if the fit itself fails -- callers fall back to the naive prediction
    rather than letting one badly-behaved entity crash the whole run.
    """
    if len(train_series) < MIN_HISTORY_DAYS or train_series.std() == 0:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # statsmodels is chatty about convergence warnings on noisy count data
            model = SARIMAX(
                train_series.values, order=order, seasonal_order=seasonal_order,
                enforce_stationarity=False, enforce_invertibility=False,
            )
            return model.fit(disp=False)
    except Exception:
        return None


def run_sarimax(df, train_df, val_df, test_df, horizon_days, raw_column="conversions",
                 order=DEFAULT_ORDER, seasonal_order=DEFAULT_SEASONAL_ORDER):
    """
    Fits one SARIMAX per entity on train-period daily `raw_column` values,
    then produces horizon-days-ahead-summed predictions for train (in-sample
    fitted), val, and test rows -- everything evaluate()/evaluate_segmented()
    and the residual hybrid need. `df` should be the full processed dataset
    (same one train.py loads) so each entity's true historical values are
    available; train_df/val_df/test_df are the already-split subsets.

    `raw_column` only makes sense for sum-type metrics (conversions, clicks)
    whose label is a horizon-days sum of a single raw column -- ratio-type
    metrics (cpa, cpc, ...) aren't a single summable per-entity series (their
    label is future-sum-A / future-sum-B), so src/train.py only calls this
    for non-ratio metrics. See src/metrics.py's MetricDefinition.is_ratio.

    Returns a dict with train_pred/val_pred/test_pred (arrays aligned to each
    split's row order) plus fit diagnostics (n_fit, n_fallback).
    """
    start = time.time()
    train_cutoff = pd.to_datetime(train_df["stat_date"]).max()
    # Only need forecasts out to the furthest date any split actually asks
    # for -- decoupled from any single entity's own last observed date, since
    # the label's forward-looking construction already guarantees real
    # historical data exists that far out (see src/labels.py).
    max_needed_date = pd.to_datetime(test_df["stat_date"]).max() + pd.Timedelta(days=horizon_days)

    n_fit, n_fallback = 0, 0
    combined_by_entity = {}     # fitted (train) + forecast (post-train), concatenated for uniform lookup
    history_by_entity = {}      # for the naive fallback, keyed by entity
    cap_by_entity = {}          # hard upper bound on this entity's predicted horizon-day sum

    for entity_id, g in df.groupby("entity_id"):
        s = g.set_index(pd.to_datetime(g["stat_date"]))[raw_column].sort_index()
        s = s.groupby(level=0).sum()  # defensive, in case of any duplicate dates
        full_series = _complete_daily_index(s)
        history_by_entity[entity_id] = full_series

        train_series = full_series[full_series.index <= train_cutoff]
        result = _fit_one_entity(train_series, order, seasonal_order)
        if result is None:
            n_fallback += 1
            continue
        n_fit += 1

        fitted = pd.Series(np.asarray(result.fittedvalues), index=train_series.index)
        steps = (max_needed_date - train_cutoff).days
        forecast_series = pd.Series(dtype=float)
        if steps > 0:
            forecast = result.get_forecast(steps=steps).predicted_mean
            forecast_index = pd.date_range(train_cutoff + pd.Timedelta(days=1), periods=steps, freq="D")
            forecast_series = pd.Series(np.asarray(forecast), index=forecast_index)

        combined_by_entity[entity_id] = pd.concat([fitted, forecast_series])
        # max(..., 1) so an entity whose best day was small (or the rare
        # all-nonzero-but-tiny series) still gets a sane, nonzero cap rather
        # than one that clamps every forecast to ~0.
        cap_by_entity[entity_id] = max(train_series.max(), 1) * horizon_days * MAX_HISTORICAL_DAILY_MULTIPLE

    elapsed = time.time() - start
    print(
        f"SARIMAX: fit {n_fit} entities, fell back to naive for {n_fallback} "
        f"(too little history, zero-variance series, or non-convergence) in {elapsed:.1f}s"
    )

    capped_count = [0]  # mutable box -- closures can't assign to an outer int directly

    def _predict_row(entity_id, stat_date) -> float:
        window_start = stat_date + pd.Timedelta(days=1)
        window_end = stat_date + pd.Timedelta(days=horizon_days)
        combined = combined_by_entity.get(entity_id)
        if combined is not None:
            window = combined[(combined.index >= window_start) & (combined.index <= window_end)]
            if len(window) == horizon_days:  # only trust a complete window
                raw = float(window.sum())
                cap = cap_by_entity.get(entity_id, float("inf"))
                clipped = float(np.clip(raw, 0, cap))
                if clipped != raw:
                    capped_count[0] += 1
                return clipped
        # Fall back: naive trailing average using only data available as of stat_date
        history = history_by_entity.get(entity_id)
        available = history[history.index <= stat_date] if history is not None else pd.Series(dtype=float)
        return _naive_fallback(available, horizon_days)

    def _predict_split(split_df):
        dates = pd.to_datetime(split_df["stat_date"])
        return np.array([_predict_row(eid, d) for eid, d in zip(split_df["entity_id"], dates)])

    train_pred = _predict_split(train_df)
    val_pred = _predict_split(val_df)
    test_pred = _predict_split(test_df)

    if capped_count[0]:
        print(
            f"SARIMAX: clipped {capped_count[0]} runaway prediction(s) back down to a sane "
            f"per-entity cap ({MAX_HISTORICAL_DAILY_MULTIPLE}x that entity's best training day x horizon) "
            f"-- these would otherwise have been extreme extrapolation artifacts, not real forecasts."
        )

    return {
        "train_pred": train_pred,
        "val_pred": val_pred,
        "test_pred": test_pred,
        "n_fit": n_fit,
        "n_fallback": n_fallback,
        "n_capped": capped_count[0],
    }


def train_sarimax_gbm_hybrid(train_df, val_df, test_df, feature_cols, label_col,
                              categorical_features, sarimax_result, gbm_params):
    """
    Trains a LightGBM model on SARIMAX's residuals (actual label - SARIMAX's
    implied prediction for that row), then combines: final prediction =
    SARIMAX's prediction + the residual model's correction. See this module's
    docstring for the caveat on the train-residual target (uses SARIMAX's
    in-sample fitted values, which run a bit optimistic).
    """
    import lightgbm as lgb

    train_residual = train_df[label_col].values - sarimax_result["train_pred"]
    val_residual = val_df[label_col].values - sarimax_result["val_pred"]

    cat_features_present = [c for c in categorical_features if c in feature_cols]
    train_set = lgb.Dataset(
        train_df[feature_cols], label=train_residual, categorical_feature=cat_features_present,
    )
    val_set = lgb.Dataset(
        val_df[feature_cols], label=val_residual,
        reference=train_set, categorical_feature=cat_features_present,
    )

    residual_params = {**gbm_params, "objective": "regression", "metric": "mae"}
    residual_model = lgb.train(
        residual_params,
        train_set,
        valid_sets=[val_set],
        num_boost_round=500,
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)],
    )

    test_residual_pred = residual_model.predict(test_df[feature_cols])
    test_pred = np.clip(sarimax_result["test_pred"] + test_residual_pred, 0, None)

    return {"residual_model": residual_model, "test_pred": test_pred}
