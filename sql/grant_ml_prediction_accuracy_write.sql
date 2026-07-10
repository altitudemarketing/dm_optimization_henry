-- Run this once, as a role with privileges to create tables and grant on this
-- schema (same as sql/grant_ml_recommendations_write.sql -- SLACK_BOT_RO
-- itself cannot run this, it's read-only by design outside these two tables).
--
-- Scope, deliberately narrow, same philosophy as the recommendations table:
-- INSERT + SELECT only, on this one new table. No UPDATE/DELETE grant --
-- src/evaluate_accuracy.py never touches a row once written; every accuracy
-- check for a given (entity, prediction batch) pair is inserted exactly once,
-- guarded by a NOT EXISTS check in that script's INSERT ... SELECT.

CREATE TABLE IF NOT EXISTS FIVETRAN_DATABASE.GOOGLE_ADS.ML_PREDICTION_ACCURACY (
    evaluated_at        TIMESTAMP_NTZ NOT NULL,  -- when this accuracy check ran
    generated_at        TIMESTAMP_NTZ NOT NULL,  -- which ML_RECOMMENDATIONS batch this scores
    entity_id           STRING NOT NULL,          -- client_id::campaign_id::ad_group_id
    client_id           STRING,
    client_name         STRING,
    campaign_id         STRING,
    campaign_name       STRING,
    ad_group_id         STRING,
    ad_group_name       STRING,
    stat_date           DATE,      -- the "as of" date the original forecast was made from
    target_metric       STRING,
    horizon_days        INTEGER,
    predicted           FLOAT,     -- what the model forecast
    baseline            FLOAT,     -- the naive trailing-actual baseline used at prediction time
    actual               FLOAT,     -- what really happened over (stat_date, stat_date + horizon_days]
    abs_error            FLOAT,     -- |predicted - actual|
    pct_error            FLOAT,     -- |predicted - actual| / actual; NULL when actual = 0 (mirrors the MAPE masking in src/train.py)
    baseline_abs_error    FLOAT,     -- |baseline - actual| -- lets the frontend show "did the model beat naive in production", not just in backtesting
    action_type           STRING,    -- copied from the recommendation, so accuracy can be sliced by "how often was a red flag actually right"
    severity              STRING,
    model_used            STRING
);

-- Narrow grant: append + read only, on this table only.
GRANT INSERT, SELECT ON TABLE FIVETRAN_DATABASE.GOOGLE_ADS.ML_PREDICTION_ACCURACY TO ROLE SLACK_BOT_RO;
