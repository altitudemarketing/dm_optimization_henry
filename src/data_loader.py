"""
Pulls ad-group-level daily performance data from Snowflake.

Source: FIVETRAN_DATABASE.GOOGLE_ADS.HOURLY_STATS_MAT -- the agency's existing
curated, multi-client dynamic table (see
speed_snowflake/snowflake/ddl/google_ads/hourly_stats_mat.sql), which already
resolves client/campaign/ad-group identity and dedupes Fivetran resync
duplicates. We collapse its hour/device/network detail down to one row per
(client, campaign, ad_group, day), which is the grain this pipeline forecasts at.
"""

import pandas as pd

from src.snowflake_client import run_query

QUERY_TEMPLATE = """
SELECT
    CLIENT_ID,
    CLIENT_NAME,
    CAMPAIGN_ID,
    CAMPAIGN_NAME,
    CHANNEL_TYPE,
    AD_GROUP_ID,
    AD_GROUP_NAME,
    AD_GROUP_STATUS,
    STAT_DATE,
    SUM(IMPRESSIONS)       AS IMPRESSIONS,
    SUM(CLICKS)            AS CLICKS,
    SUM(SPEND)             AS SPEND,
    SUM(CONVERSIONS)       AS CONVERSIONS,
    SUM(CONVERSIONS_VALUE) AS CONVERSIONS_VALUE
FROM FIVETRAN_DATABASE.GOOGLE_ADS.HOURLY_STATS_MAT
WHERE STAT_DATE >= DATEADD(day, -{lookback_days}, CURRENT_DATE())
  AND AD_GROUP_STATUS = 'ENABLED'
GROUP BY
    CLIENT_ID, CLIENT_NAME, CAMPAIGN_ID, CAMPAIGN_NAME, CHANNEL_TYPE,
    AD_GROUP_ID, AD_GROUP_NAME, AD_GROUP_STATUS, STAT_DATE
"""


def pull_raw_data(config: dict) -> pd.DataFrame:
    model_cfg = config["model"]
    lookback_days = (
        model_cfg["lookback_window_days"]
        + model_cfg["forecast_horizon_days"]
        + max(model_cfg["rolling_windows"])
    )
    sql = QUERY_TEMPLATE.format(lookback_days=lookback_days)
    df = run_query(sql)
    df.columns = [c.lower() for c in df.columns]

    numeric_cols = ["impressions", "clicks", "spend", "conversions", "conversions_value"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["stat_date"] = pd.to_datetime(df["stat_date"])
    return df
