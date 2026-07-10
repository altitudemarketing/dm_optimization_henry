"""
Scores past recommendations against what actually happened, once each
forecast window has fully elapsed, and appends the result to
FIVETRAN_DATABASE.GOOGLE_ADS.ML_PREDICTION_ACCURACY (see
sql/grant_ml_prediction_accuracy_write.sql for the table + narrow grant this
needs).

This runs entirely as a single INSERT ... SELECT executed in Snowflake --
there's no need to pull rows into pandas here. The query:
  1. Finds ML_RECOMMENDATIONS rows whose forecast window (stat_date,
     stat_date + horizon_days] has fully elapsed (with a 1-day buffer past
     the ~12h Fivetran refresh cadence) and haven't been scored yet.
  2. Joins each one against HOURLY_STATS_MAT for that same entity and date
     range to compute what actually happened.
  3. Appends predicted/baseline/actual and their errors as a new row.

Never UPDATEs or DELETEs -- every (entity, prediction batch) pair is scored
at most once, guarded by a NOT EXISTS check, consistent with the
append-only design used for ML_RECOMMENDATIONS itself.

Usage:
    python -m src.evaluate_accuracy [--config path/to/config.yaml]
"""

import argparse

from src.config_loader import load_config
from src.snowflake_client import SnowflakeConfigError, require_snowflake_env, run_query

RECOMMENDATIONS_TABLE = "FIVETRAN_DATABASE.GOOGLE_ADS.ML_RECOMMENDATIONS"
ACCURACY_TABLE = "FIVETRAN_DATABASE.GOOGLE_ADS.ML_PREDICTION_ACCURACY"
RAW_TABLE = "FIVETRAN_DATABASE.GOOGLE_ADS.HOURLY_STATS_MAT"

# SQL form of src/metrics.py's MetricDefinition.compute -- kept as a small,
# separate registry because this evaluation runs entirely inside Snowflake
# (no pandas round-trip), so the ratio math needs a SQL expression, not a
# Python callable. Keep in sync with src/metrics.py if either changes.
SQL_METRIC_AGGREGATES = {
    "conversions":       "SUM(h.CONVERSIONS)",
    "cpa":               "SUM(h.SPEND) / NULLIF(SUM(h.CONVERSIONS), 0)",
    "roas":              "SUM(h.CONVERSIONS_VALUE) / NULLIF(SUM(h.SPEND), 0)",
    "conversion_rate":   "SUM(h.CONVERSIONS) / NULLIF(SUM(h.CLICKS), 0)",
    "ctr":               "SUM(h.CLICKS) / NULLIF(SUM(h.IMPRESSIONS), 0)",
}


def build_evaluation_sql(target_metric: str) -> str:
    if target_metric not in SQL_METRIC_AGGREGATES:
        raise ValueError(
            f"No SQL aggregate defined for target_metric='{target_metric}' -- "
            f"add one to SQL_METRIC_AGGREGATES (mirroring src/metrics.py)."
        )
    actual_expr = SQL_METRIC_AGGREGATES[target_metric]

    return f"""
    INSERT INTO {ACCURACY_TABLE}
        (evaluated_at, generated_at, entity_id, client_id, client_name, campaign_id,
         campaign_name, ad_group_id, ad_group_name, stat_date, target_metric,
         horizon_days, predicted, baseline, actual, abs_error, pct_error,
         baseline_abs_error, action_type, severity, model_used)
    SELECT
        CURRENT_TIMESTAMP()                                            AS evaluated_at,
        r.GENERATED_AT, r.ENTITY_ID, r.CLIENT_ID, r.CLIENT_NAME, r.CAMPAIGN_ID,
        r.CAMPAIGN_NAME, r.AD_GROUP_ID, r.AD_GROUP_NAME, r.STAT_DATE, r.TARGET_METRIC,
        r.HORIZON_DAYS, r.PREDICTED, r.BASELINE,
        COALESCE({actual_expr}, 0)                                     AS actual,
        ABS(r.PREDICTED - COALESCE({actual_expr}, 0))                  AS abs_error,
        CASE WHEN COALESCE({actual_expr}, 0) != 0
             THEN ABS(r.PREDICTED - {actual_expr}) / {actual_expr}
             ELSE NULL END                                             AS pct_error,
        ABS(r.BASELINE - COALESCE({actual_expr}, 0))                   AS baseline_abs_error,
        r.ACTION_TYPE, r.SEVERITY, r.MODEL_USED
    FROM {RECOMMENDATIONS_TABLE} r
    LEFT JOIN {RAW_TABLE} h
        ON h.CLIENT_ID = r.CLIENT_ID
        AND h.CAMPAIGN_ID = r.CAMPAIGN_ID
        AND h.AD_GROUP_ID = r.AD_GROUP_ID
        AND h.STAT_DATE > r.STAT_DATE
        AND h.STAT_DATE <= DATEADD(day, r.HORIZON_DAYS, r.STAT_DATE)
    WHERE r.TARGET_METRIC = '{target_metric}'
        -- forecast window (stat_date, stat_date + horizon_days] must have fully
        -- elapsed, plus a 1-day buffer past Fivetran's ~12h refresh cadence
        AND DATEADD(day, r.HORIZON_DAYS, r.STAT_DATE) <= DATEADD(day, -1, CURRENT_DATE())
        AND NOT EXISTS (
            SELECT 1 FROM {ACCURACY_TABLE} existing
            WHERE existing.ENTITY_ID = r.ENTITY_ID AND existing.GENERATED_AT = r.GENERATED_AT
        )
    GROUP BY
        r.GENERATED_AT, r.ENTITY_ID, r.CLIENT_ID, r.CLIENT_NAME, r.CAMPAIGN_ID,
        r.CAMPAIGN_NAME, r.AD_GROUP_ID, r.AD_GROUP_NAME, r.STAT_DATE, r.TARGET_METRIC,
        r.HORIZON_DAYS, r.PREDICTED, r.BASELINE, r.ACTION_TYPE, r.SEVERITY, r.MODEL_USED
    """


def main(config_path=None):
    config = load_config(config_path)
    target_metric = config["model"]["target_metric"]

    try:
        require_snowflake_env()
    except SnowflakeConfigError as e:
        # Same guard as src/recommend.py's upload step -- there's nothing to
        # evaluate against without real Snowflake access, and this shouldn't
        # block local/synthetic-data development.
        print(f"Skipping accuracy evaluation -- {e}")
        return

    sql = build_evaluation_sql(target_metric)
    result = run_query(sql)
    # Snowflake's SQL API returns a one-row "number of rows inserted" result
    # for INSERT statements -- surface it directly rather than re-deriving a count.
    print(f"Accuracy evaluation complete. Snowflake response: {result.to_dict(orient='records')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    main(args.config)
