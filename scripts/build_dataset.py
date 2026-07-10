"""
Orchestrates the full dataset build: pull raw data from Snowflake (or a cached
parquet), aggregate to daily grain, build rolling-window features, build
labels for every metric in config.yaml's model.target_metrics, and save the
processed dataset.

Usage:
    python -m scripts.build_dataset [--config path/to/config.yaml] [--use-cache]

--use-cache skips the Snowflake pull and reuses whatever's at
paths.raw_data_cache (handy for iterating on features/labels, or for running
against scripts/generate_synthetic_data.py output without credentials).

Builds a label_<metric> column for every entry in model.target_metrics (all
sharing the same metric-agnostic features above) into one processed dataset,
so src/train.py can train/compare models for all of them without rebuilding
the dataset per metric. Falls back to model.target_metric (singular) if
target_metrics isn't set, for backward compatibility.
"""

import argparse
from pathlib import Path

import pandas as pd

from src.config_loader import load_config
from src.data_loader import pull_raw_data
from src.features import build_rolling_features, ensure_daily_grain, make_entity_id
from src.labels import build_labels


def main(config_path=None, use_cache=False):
    config = load_config(config_path)

    raw_cache_path = Path(config["paths"]["raw_data_cache"])
    if use_cache:
        if not raw_cache_path.exists():
            raise FileNotFoundError(
                f"--use-cache set but {raw_cache_path} doesn't exist. "
                f"Run `python -m scripts.generate_synthetic_data` for local testing, "
                f"or omit --use-cache to pull from Snowflake."
            )
        print(f"Loading cached raw data from {raw_cache_path}")
        raw_df = pd.read_parquet(raw_cache_path)
        raw_df["stat_date"] = pd.to_datetime(raw_df["stat_date"])
    else:
        print("Pulling raw data from Snowflake...")
        raw_df = pull_raw_data(config)
        raw_cache_path.parent.mkdir(parents=True, exist_ok=True)
        raw_df.to_parquet(raw_cache_path, index=False)
        print(f"Cached raw data to {raw_cache_path} ({len(raw_df)} rows)")

    df = make_entity_id(raw_df)
    df = ensure_daily_grain(df)
    print(f"{df['entity_id'].nunique()} distinct (client, campaign, ad_group) entities, "
          f"{len(df)} entity-days")

    df = build_rolling_features(df, config["model"]["rolling_windows"])

    target_metrics = config["model"].get("target_metrics") or [config["model"]["target_metric"]]
    horizon = config["model"]["forecast_horizon_days"]
    for target_metric in target_metrics:
        df = build_labels(df, target_metric, horizon)
        label_col = f"label_{target_metric}"
        usable = df[label_col].notna().sum()
        print(f"{target_metric}: {usable} / {len(df)} rows have a complete forward-looking label "
              f"(rest are near the end of the available history, or -- for ratio metrics like "
              f"cpa/cpc -- have an undefined/zero-denominator label over the forecast window)")

    processed_path = Path(config["paths"]["processed_dataset"])
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(processed_path, index=False)
    print(f"Saved processed dataset to {processed_path} ({len(df)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--use-cache", action="store_true")
    args = parser.parse_args()
    main(args.config, args.use_cache)
