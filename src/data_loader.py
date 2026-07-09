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

    # src.snowflake_client already converts DATE/FIXED/REAL columns to proper
    # pandas dtypes based on Snowflake's own column metadata (see
    # _convert_column_types) -- these are just a defensive backstop in case a
    # column ever comes through as plain text.
    numeric_cols = ["impressions", "clicks", "spend", "conversions", "conversions_value"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if not pd.api.types.is_datetime64_any_dtype(df["stat_date"]):
        # This shouldn't happen anymore -- src.snowflake_client should have
        # already converted STAT_DATE based on Snowflake's reported column
        # type. If we get here, that type match failed; check the
        # "[snowflake_client] column types" log line above for what type
        # Snowflake actually reported, rather than let this crash cryptically
        # in dateutil the way it did before (raw epoch-day ints look like
        # nonsense years to pandas' generic string parser).
        sample = df["stat_date"].dropna().iloc[0] if df["stat_date"].notna().any() else None
        try:
            df["stat_date"] = pd.to_datetime(
                pd.to_numeric(df["stat_date"], errors="raise"), unit="D", origin="unix"
            )
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"stat_date came through as dtype {df['stat_date'].dtype} (sample value: "
                f"{sample!r}) instead of a proper datetime -- src.snowflake_client's type "
                f"conversion didn't match this column. Check the '[snowflake_client] column "
                f"types' log line above for the actual type Snowflake reported."
            ) from exc

    print(f"Pulled {len(df)} rows. stat_date range: {df['stat_date'].min()} to {df['stat_date'].max()}")
    print(
        "Sanity check -- spend: min={:.2f} max={:.2f} mean={:.2f} | "
        "conversions: min={:.2f} max={:.2f} mean={:.2f}".format(
            df["spend"].min(), df["spend"].max(), df["spend"].mean(),
            df["conversions"].min(), df["conversions"].max(), df["conversions"].mean(),
        )
    )
    print(
        "If spend/conversions look off by a factor of 10/100/1000 from what "
        "you'd expect, that points to a scale-conversion issue -- flag it "
        "rather than let training run on bad data."
    )

    return df
