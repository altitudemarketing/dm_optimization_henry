-- Run this once, as a role with privileges to create tables and grant on this
-- schema (e.g. ACCOUNTADMIN or SYSADMIN -- SLACK_BOT_RO itself cannot run
-- this, it's read-only by design).
--
-- Scope, deliberately narrow, same pattern as
-- sql/grant_ml_recommendations_write.sql: SLACK_BOT_RO gets INSERT + SELECT
-- on this ONE new table only. No UPDATE/DELETE/TRUNCATE -- src/optimize.py
-- only ever appends a new batch each run, never modifies or removes past
-- ones (past runs are the future outcome-tracking feedback loop's training
-- data).

CREATE TABLE IF NOT EXISTS FIVETRAN_DATABASE.GOOGLE_ADS.ML_SPEND_RECOMMENDATIONS (
    generated_at                          TIMESTAMP_NTZ NOT NULL,  -- when this optimization run produced the recommendation
    entity_id                             STRING NOT NULL,          -- client_id::campaign_id::ad_group_id
    client_id                             STRING,
    client_name                           STRING,
    campaign_id                           STRING,
    campaign_name                         STRING,
    ad_group_id                           STRING,
    ad_group_name                         STRING,
    stat_date                             DATE,                     -- most recent day of history this recommendation was computed from
    window_days                           INTEGER,                  -- trailing window the current_* columns are aggregated over
    current_spend                         FLOAT,
    current_conversions                   FLOAT,
    current_cpa                           FLOAT,
    campaign_trailing_spend               FLOAT,                    -- parent campaign's current trailing spend (the grain the response curve was fit at)
    campaign_spend_range_min              FLOAT,                    -- parent campaign's historically observed min chunk spend (response_curve.py panel)
    campaign_spend_range_max              FLOAT,                    -- parent campaign's historically observed max chunk spend
    campaign_n_evidence_chunks            INTEGER,                  -- how many response-curve panel chunks back this campaign's elasticity
    recommended_action                    STRING,                   -- INCREASE_SPEND | DECREASE_SPEND | HOLD | INSUFFICIENT_EVIDENCE
    recommended_pct_change                FLOAT,
    recommended_dollar_change             FLOAT,
    recommended_new_spend                 FLOAT,
    predicted_conversions_at_recommended  FLOAT,
    predicted_conversion_delta            FLOAT,
    predicted_cpa_at_recommended          FLOAT,
    beta_used                             FLOAT,                    -- which CI bound of the elasticity was used (conservative for the recommended direction)
    beta_ci_low                           FLOAT,
    beta_ci_high                          FLOAT,
    confidence_tier                       STRING,                   -- high (stop/resume evidence) | medium (continuous variation only) | insufficient_evidence
    rationale                             STRING,
    requires_human_review                 BOOLEAN,
    model_used                            STRING
);

-- Narrow grant: append + read only, on this table only.
GRANT INSERT, SELECT ON TABLE FIVETRAN_DATABASE.GOOGLE_ADS.ML_SPEND_RECOMMENDATIONS TO ROLE SLACK_BOT_RO;
