-- Reference copy of the query used by src/data_loader.py.
--
-- Pulls ad-group-level daily performance from the curated HOURLY_STATS_MAT
-- dynamic table (speed_snowflake/snowflake/ddl/google_ads/hourly_stats_mat.sql),
-- collapsing its hour/device/network detail down to one row per
-- client x campaign x ad_group x day -- the grain this pipeline forecasts at.
--
-- Replace :lookback_days or the DATEADD literal below to adjust history pulled.

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
WHERE STAT_DATE >= DATEADD(day, -67, CURRENT_DATE())  -- 30 lookback + 7 horizon + 30 max rolling window, per default config.yaml
  AND AD_GROUP_STATUS = 'ENABLED'
GROUP BY
    CLIENT_ID, CLIENT_NAME, CAMPAIGN_ID, CAMPAIGN_NAME, CHANNEL_TYPE,
    AD_GROUP_ID, AD_GROUP_NAME, AD_GROUP_STATUS, STAT_DATE
ORDER BY CLIENT_ID, CAMPAIGN_ID, AD_GROUP_ID, STAT_DATE;
