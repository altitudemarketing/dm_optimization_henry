"""
Leak-safe label construction for the configured target metric.

This is the ONLY module that needs to change when config.yaml's
model.target_metric changes -- it reads directly from the metric registry
(src/metrics.py). Feature engineering, the Snowflake pull, and training are
all metric-agnostic.
"""

import pandas as pd

from src.metrics import get_metric


def build_labels(daily: pd.DataFrame, target_metric: str, horizon_days: int) -> pd.DataFrame:
    metric_def = get_metric(target_metric)
    daily = daily.sort_values(["entity_id", "stat_date"]).copy()
    grouped = daily.groupby("entity_id")

    future_cols = {}
    for col in metric_def.raw_columns:
        # shift(-1) then rolling(horizon_days) sums the NEXT horizon_days of
        # data, excluding the current row -- this is what prevents "today"
        # from leaking into the label.
        future_cols[col] = grouped[col].transform(
            lambda s: s.shift(-1).rolling(horizon_days, min_periods=horizon_days).sum()
        )

    future_df = pd.DataFrame(future_cols)
    daily[f"label_{target_metric}"] = metric_def.compute(future_df)
    return daily
