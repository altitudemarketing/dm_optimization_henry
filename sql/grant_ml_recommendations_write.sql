-- Run this once, as a role with privileges to create tables and grant on this
-- schema (e.g. ACCOUNTADMIN or whatever owns FIVETRAN_DATABASE.GOOGLE_ADS --
-- SLACK_BOT_RO itself cannot run this, it's read-only by design).
--
-- Scope, deliberately narrow: this grants SLACK_BOT_RO INSERT + SELECT on
-- ONE new table only. It does NOT touch any existing grant, and does NOT
-- grant UPDATE, DELETE, or TRUNCATE -- the pipeline only ever needs to append
-- a new snapshot each run, never modify or remove past ones (past runs are
-- worth keeping: they become the training data for task #10's feedback loop
-- later). If you'd rather this live in a different schema than
-- FIVETRAN_DATABASE.GOOGLE_ADS (that schema is Fivetran-managed/synced --
-- mixing in an agency-authored table there is a little unusual), swap the
-- schema name below for wherever agency-owned tables normally live.

CREATE TABLE IF NOT EXISTS FIVETRAN_DATABASE.GOOGLE_ADS.ML_RECOMMENDATIONS (
    generated_at              TIMESTAMP_NTZ NOT NULL,  -- when this training run produced the recommendation
    entity_id                 STRING NOT NULL,          -- client_id::campaign_id::ad_group_id
    client_id                 STRING,
    client_name                STRING,
    campaign_id                STRING,
    campaign_name              STRING,
    ad_group_id                STRING,
    ad_group_name              STRING,
    stat_date                  DATE,                     -- most recent day of history the forecast was made from
    target_metric               STRING,
    horizon_days                INTEGER,
    predicted                   FLOAT,
    baseline                    FLOAT,
    pct_change_vs_baseline       FLOAT,
    client_rank_pct              FLOAT,
    trailing_spend               FLOAT,
    action_type                  STRING,
    severity                     STRING,
    confidence_segment            STRING,
    rationale                    STRING,
    requires_human_review         BOOLEAN,
    model_used                   STRING
);

-- Narrow grant: append + read only, on this table only.
GRANT INSERT, SELECT ON TABLE FIVETRAN_DATABASE.GOOGLE_ADS.ML_RECOMMENDATIONS TO ROLE SLACK_BOT_RO;
