"""
Generates recommendation records from the trained forecasting model.

For each active ad group, compares the model's forecast for the next
horizon_days against that same ad group's own trailing actual performance
over an equal-length window, then flags the ones where that comparison is
both large (relative to *this client's* other ad groups) and backed by
enough trailing spend to be worth a marketer's attention.

Design notes (why it works this way):
  - Guardrail, not a fixed % threshold: flags are based on percentile rank of
    predicted-vs-baseline change WITHIN each client's own ad groups, not a
    global percentage cutoff. Account sizes and conversion volumes vary
    enormously across clients, and raw % change is unstable on low-count
    data (see MAPE discussion in src/train.py) -- rank is much more stable.
  - Guardrail, spend floor: recommendations are suppressed entirely below
    `recommendations.min_trailing_spend` -- there isn't enough at stake, and
    it's exactly where the model's zero-actual segment (see
    evaluate_segmented in src/train.py) is least reliable directionally.
  - Confidence tag: ties directly back to the segmented evaluation results --
    the model is reliably strong (R^2 ~0.95 in real runs) on entities with
    real trailing conversion volume, much weaker on the always-zero segment.
    Baseline==0 is the best available proxy at inference time for which
    segment a given entity falls into.
  - This module is recommendation-only BY DESIGN: every record carries
    requires_human_review=True and nothing here writes to Google Ads. See
    README's autonomous-execution roadmap for how this is meant to evolve
    (task: build a write connector, gated on outcome-tracking history).

Usage:
    python -m src.recommend [--config path/to/config.yaml]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb

from src.config_loader import load_config
from src.snowflake_client import SnowflakeConfigError, require_snowflake_env, run_query
from src.train import CATEGORICAL_FEATURES, get_feature_columns

# Column order matches sql/grant_ml_recommendations_write.sql's CREATE TABLE --
# keep these in sync if either changes.
SNOWFLAKE_TABLE = "FIVETRAN_DATABASE.GOOGLE_ADS.ML_RECOMMENDATIONS"
SNOWFLAKE_COLUMNS = [
    "generated_at", "entity_id", "client_id", "client_name", "campaign_id",
    "campaign_name", "ad_group_id", "ad_group_name", "stat_date",
    "target_metric", "horizon_days", "predicted", "baseline",
    "pct_change_vs_baseline", "client_rank_pct", "trailing_spend",
    "action_type", "severity", "confidence_segment", "rationale",
    "requires_human_review", "model_used",
]

# Predictions below this are treated as "the model expects essentially zero".
EPSILON_ZERO = 0.5

ACTION_TAXONOMY = {
    "FORECASTED_ZERO_HIGH_SPEND": {
        "severity": "red",
        "label": "Forecasted zero conversions, meaningful spend",
        "guidance": (
            "Model forecasts ~0 conversions over the next {horizon}d despite "
            "real trailing spend. Review targeting, bids, or pause candidacy."
        ),
    },
    "FORECASTED_DECLINE": {
        "severity": "yellow",
        "label": "Forecasted decline vs. trailing performance",
        "guidance": (
            "Forecast is among this client's steepest predicted drops. "
            "Review recent changes to bids, budget, or targeting."
        ),
    },
    "FORECASTED_GROWTH_OPPORTUNITY": {
        "severity": "green",
        "label": "Forecasted growth opportunity",
        "guidance": (
            "Forecast is among this client's strongest predicted gains. "
            "Consider protecting or increasing budget to capture it."
        ),
    },
}


def _load_model(output_dir: Path, target_metric: str, horizon: int, model_name: str):
    if model_name == "xgboost":
        booster = xgb.Booster()
        booster.load_model(str(output_dir / f"{target_metric}_h{horizon}d_xgboost.json"))
        return "xgboost", booster
    elif model_name == "lightgbm":
        booster = lgb.Booster(model_file=str(output_dir / f"{target_metric}_h{horizon}d_lightgbm.txt"))
        return "lightgbm", booster
    else:
        raise ValueError(
            f"Unknown recommendations.primary_model '{model_name}'. Use 'xgboost' or "
            f"'lightgbm' -- the hurdle model's two-piece (classifier + regressor) design "
            f"isn't wired up for standalone inference here yet."
        )


def _predict(model_kind: str, model, df: pd.DataFrame, feature_cols):
    if model_kind == "xgboost":
        dmatrix = xgb.DMatrix(df[feature_cols], enable_categorical=True)
        return model.predict(dmatrix)
    return model.predict(df[feature_cols])


def _baseline_column(target_metric: str, horizon: int) -> str:
    # Every metric currently in the registry except roas has a directly
    # precomputed trailing rolling column of the same name (see
    # src/features.py) -- reused here as "trailing actual" to compare the
    # forecast against. Extend this mapping if a future metric doesn't.
    if target_metric == "roas":
        raise NotImplementedError(
            "roas has no precomputed rolling column -- compute "
            "conversions_value_rolling_Xd / spend_rolling_Xd if you add roas recommendations."
        )
    return f"{target_metric}_rolling_{horizon}d"


def build_recommendations(config) -> pd.DataFrame:
    target_metric = config["model"]["target_metric"]
    horizon = config["model"]["forecast_horizon_days"]
    label_col = f"label_{target_metric}"

    rec_config = config.get("recommendations", {})
    model_name = rec_config.get("primary_model", "xgboost")
    min_trailing_spend = rec_config.get("min_trailing_spend", 50)
    flag_percentile = rec_config.get("flag_percentile", 0.10)
    high_spend_zero_forecast_spend = rec_config.get("high_spend_zero_forecast_spend", 100)

    processed_path = Path(config["paths"]["processed_dataset"])
    if not processed_path.exists():
        raise FileNotFoundError(f"{processed_path} not found. Run `python -m scripts.build_dataset` first.")
    df = pd.read_parquet(processed_path)

    for cat_col in CATEGORICAL_FEATURES:
        if cat_col in df.columns:
            df[cat_col] = df[cat_col].astype("category")

    feature_cols = get_feature_columns(df, label_col)

    # One row per entity: its most recent day of history. This is what the
    # model forecasts *forward* from -- there's no future label attached to
    # it yet (that's the point of scoring it).
    latest = df.sort_values("stat_date").groupby("entity_id").tail(1).copy()

    output_dir = Path(config["paths"]["model_output"])
    model_kind, model = _load_model(output_dir, target_metric, horizon, model_name)
    latest["predicted"] = np.clip(_predict(model_kind, model, latest, feature_cols), 0, None)

    baseline_col = _baseline_column(target_metric, horizon)
    latest["baseline"] = latest[baseline_col].fillna(0)
    latest["trailing_spend"] = latest[f"spend_rolling_{horizon}d"].fillna(0)
    latest["delta"] = latest["predicted"] - latest["baseline"]
    latest["pct_change_vs_baseline"] = np.where(
        latest["baseline"] > 0, latest["delta"] / latest["baseline"] * 100, np.nan,
    )

    # Guardrail: suppress recommendations for ad groups with too little
    # trailing spend to matter.
    eligible = latest[latest["trailing_spend"] >= min_trailing_spend].copy()

    empty_cols = [
        "generated_at", "entity_id", "client_id", "client_name", "campaign_id", "campaign_name",
        "ad_group_id", "ad_group_name", "stat_date", "target_metric", "horizon_days",
        "predicted", "baseline", "pct_change_vs_baseline", "client_rank_pct",
        "trailing_spend", "action_type", "severity", "confidence_segment",
        "rationale", "requires_human_review", "model_used",
    ]
    if eligible.empty:
        return pd.DataFrame(columns=empty_cols)

    # Guardrail: rank within each client's own ad groups, not by a fixed
    # global threshold -- keeps the bar relative to what's normal for that
    # specific client rather than penalizing naturally smaller accounts.
    eligible["client_rank_pct"] = eligible.groupby("client_id", observed=True)["delta"].rank(pct=True)

    zero_high_spend = (
        (eligible["predicted"] < EPSILON_ZERO)
        & (eligible["trailing_spend"] >= high_spend_zero_forecast_spend)
    )
    decline = (~zero_high_spend) & (eligible["client_rank_pct"] <= flag_percentile)
    growth = (~zero_high_spend) & (eligible["client_rank_pct"] >= (1 - flag_percentile))

    eligible["action_type"] = np.select(
        [zero_high_spend, decline, growth],
        ["FORECASTED_ZERO_HIGH_SPEND", "FORECASTED_DECLINE", "FORECASTED_GROWTH_OPPORTUNITY"],
        default=None,
    )
    flagged = eligible[eligible["action_type"].notna()].copy()
    if flagged.empty:
        return pd.DataFrame(columns=empty_cols)

    flagged["severity"] = flagged["action_type"].map(lambda a: ACTION_TAXONOMY[a]["severity"])

    # Confidence tag ties back to evaluate_segmented(): the model is reliably
    # strong on entities with real trailing conversion volume, much weaker on
    # the always-zero segment. Baseline==0 is the best proxy available at
    # inference time (no future actual to check against yet) for which
    # segment a given entity falls into.
    flagged["confidence_segment"] = np.where(
        flagged["baseline"] > 0, "high (nonzero-history segment)", "low (zero-history segment)",
    )

    def _rationale(row):
        guidance = ACTION_TAXONOMY[row["action_type"]]["guidance"].format(horizon=horizon)
        pct = f"{row['pct_change_vs_baseline']:.0f}%" if pd.notna(row["pct_change_vs_baseline"]) else "n/a (baseline was 0)"
        return (
            f"Forecasted {target_metric} over the next {horizon}d: {row['predicted']:.1f} "
            f"vs. trailing {horizon}d actual: {row['baseline']:.1f} ({pct} change). {guidance}"
        )

    flagged["rationale"] = flagged.apply(_rationale, axis=1)
    flagged["requires_human_review"] = True
    flagged["model_used"] = model_name
    flagged["horizon_days"] = horizon
    flagged["target_metric"] = target_metric
    # Single timestamp for the whole batch -- this is what the Worker later
    # queries MAX(generated_at) against to serve "the latest run"'s
    # recommendations, and what makes every past run's snapshot a distinct,
    # queryable slice (useful for task #10's feedback loop later).
    flagged["generated_at"] = pd.Timestamp.utcnow().tz_localize(None)

    return flagged[empty_cols].sort_values(["severity", "client_rank_pct"]).reset_index(drop=True)


def _sql_literal(value) -> str:
    """Formats a single Python value as a SQL literal for the INSERT below.

    This is building a batch INSERT from values we generated ourselves (not
    user input), but string columns (rationale, names) still get quotes
    escaped defensively rather than assumed safe.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "NULL"
    if isinstance(value, bool) or isinstance(value, np.bool_):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, np.integer, float, np.floating)):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return f"'{value.strftime('%Y-%m-%d %H:%M:%S')}'"
    escaped = str(value).replace("'", "''")
    return f"'{escaped}'"


def upload_to_snowflake(recommendations: pd.DataFrame) -> None:
    """
    Appends this run's recommendations to Snowflake as a new batch (see
    sql/grant_ml_recommendations_write.sql for the table + the narrow
    INSERT-only grant this relies on -- SLACK_BOT_RO otherwise stays
    read-only everywhere else). The Worker's /api/report/google-ads/
    ml-recommendations endpoint reads this table back, filtering to
    MAX(generated_at) for "the latest run".

    Never UPDATEs or DELETEs -- past runs are left in place on purpose,
    since they're the natural training data for the feedback/outcome
    logging loop (task #10) later.
    """
    if recommendations.empty:
        print("No recommendations to upload to Snowflake this run.")
        return

    try:
        require_snowflake_env()
    except SnowflakeConfigError as e:
        # Expected when running locally / against synthetic data (see
        # README's "Testing without Snowflake access") -- don't block that
        # workflow just because there's nowhere real to write to yet.
        print(f"Skipping Snowflake upload -- {e}")
        return

    rows_sql = []
    for _, row in recommendations.iterrows():
        values = ", ".join(_sql_literal(row[col]) for col in SNOWFLAKE_COLUMNS)
        rows_sql.append(f"({values})")

    insert_sql = (
        f"INSERT INTO {SNOWFLAKE_TABLE} ({', '.join(SNOWFLAKE_COLUMNS)}) VALUES\n"
        + ",\n".join(rows_sql)
    )
    run_query(insert_sql)
    print(f"Inserted {len(recommendations)} rows into {SNOWFLAKE_TABLE}.")


def main(config_path=None):
    config = load_config(config_path)
    recommendations = build_recommendations(config)

    out_path = Path(config["paths"].get("recommendations_output", "data/recommendations.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    recommendations.to_json(out_path, orient="records", date_format="iso", indent=2)

    counts = recommendations["action_type"].value_counts().to_dict() if not recommendations.empty else {}
    print(f"{len(recommendations)} recommendations generated: {counts}")
    print(f"Saved to {out_path}")

    upload_to_snowflake(recommendations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    main(args.config)
