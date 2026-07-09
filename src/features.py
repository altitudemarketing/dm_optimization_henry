"""
Feature engineering: rolling-window, calendar, and trend features built from
daily ad-group-level performance. Deliberately metric-agnostic -- these are
generic performance signals and don't need to change when
config.yaml's target_metric changes. Only src/labels.py reads the metric
registry.
"""

import pandas as pd

BASE_COLUMNS = ["impressions", "clicks", "spend", "conversions", "conversions_value"]


def make_entity_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Google Ads ad_group IDs are assigned per Google Ads account and are NOT
    guaranteed unique across different clients' accounts. This agency manages
    many clients in one warehouse (see HOURLY_STATS_MAT), so we always key on
    the full (client, campaign, ad_group) triple -- never ad_group_id alone --
    to avoid silently merging two different clients' data.
    """
    df = df.copy()
    df["entity_id"] = (
        df["client_id"].astype(str)
        + "::" + df["campaign_id"].astype(str)
        + "::" + df["ad_group_id"].astype(str)
    )
    return df


def ensure_daily_grain(df: pd.DataFrame) -> pd.DataFrame:
    """Defensive re-aggregation in case of duplicate rows (e.g. from paginated API responses)."""
    meta_cols = [
        "entity_id", "client_id", "client_name", "campaign_id", "campaign_name",
        "channel_type", "ad_group_id", "ad_group_name", "stat_date",
    ]
    meta_cols = [c for c in meta_cols if c in df.columns]
    return df.groupby(meta_cols, as_index=False)[BASE_COLUMNS].sum()


def build_rolling_features(daily: pd.DataFrame, rolling_windows) -> pd.DataFrame:
    daily = daily.sort_values(["entity_id", "stat_date"]).copy()
    grouped = daily.groupby("entity_id")

    for window in rolling_windows:
        for col in BASE_COLUMNS:
            # shift(1) excludes the current day, so features never leak same-day data
            daily[f"{col}_rolling_{window}d"] = grouped[col].transform(
                lambda s: s.shift(1).rolling(window, min_periods=1).sum()
            )
        daily[f"cpa_rolling_{window}d"] = (
            daily[f"spend_rolling_{window}d"]
            / daily[f"conversions_rolling_{window}d"].replace(0, pd.NA)
        )
        daily[f"ctr_rolling_{window}d"] = (
            daily[f"clicks_rolling_{window}d"]
            / daily[f"impressions_rolling_{window}d"].replace(0, pd.NA)
        )
        daily[f"conversion_rate_rolling_{window}d"] = (
            daily[f"conversions_rolling_{window}d"]
            / daily[f"clicks_rolling_{window}d"].replace(0, pd.NA)
        )

    daily["day_of_week"] = daily["stat_date"].dt.dayofweek
    daily["is_weekend"] = daily["day_of_week"].isin([5, 6]).astype(int)
    daily["day_of_month"] = daily["stat_date"].dt.day

    if 7 in rolling_windows:
        # momentum signal: is the 7-day trailing conversion rate rising or falling
        daily["conversions_rolling_7d_trend"] = (
            daily.groupby("entity_id")["conversions_rolling_7d"].diff()
        )

    return daily
