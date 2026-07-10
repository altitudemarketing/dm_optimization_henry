-- Run this once, as a role with privileges to create tables and grant on this
-- schema (e.g. ACCOUNTADMIN or SYSADMIN -- SLACK_BOT_RO itself cannot run
-- this, it's read-only by design).
--
-- Unlike ML_SPEND_RECOMMENDATIONS (written by dm_optimization_algorithm's
-- Python pipeline), this table is written directly by the Cloudflare Worker
-- (speed_snowflake/slack-bot/src/lib/ads/report_google_ads_tabs.js,
-- handleApiGadsMlSpendRecommendationDecision) whenever a human clicks
-- Approve/Reject on a spend recommendation in the ML Recommendations tab's
-- frontend. It logs the human decisions this project's whole design is
-- built around requiring -- every recommendation carries
-- requires_human_review=true, and this is where that review gets recorded.
--
-- Same narrow, append-only grant pattern as every other ML_* table here:
-- INSERT + SELECT only, no UPDATE/DELETE/TRUNCATE. A changed mind adds a
-- newer row (matched back to the same entity_id + recommendation batch via
-- ROW_NUMBER()...ORDER BY DECIDED_AT DESC in the read query) rather than
-- erasing what was originally decided -- this history is exactly what the
-- future outcome-tracking / "recommendation accuracy" feedback loop needs.

CREATE TABLE IF NOT EXISTS FIVETRAN_DATABASE.GOOGLE_ADS.ML_SPEND_RECOMMENDATION_DECISIONS (
    decided_at                     TIMESTAMP_NTZ NOT NULL,  -- when the human clicked approve/reject
    entity_id                      STRING NOT NULL,          -- client_id::campaign_id::ad_group_id, matches ML_SPEND_RECOMMENDATIONS.entity_id
    recommendation_generated_at    TIMESTAMP_NTZ NOT NULL,   -- which batch (ML_SPEND_RECOMMENDATIONS.generated_at) this decision was made against
    decision                       STRING NOT NULL,          -- 'approved' | 'rejected'
    decided_by                     STRING,                   -- email of the signed-in user who made the decision
    note                           STRING                    -- optional free-text note from the reviewer
);

-- Narrow grant: append + read only, on this table only.
GRANT INSERT, SELECT ON TABLE FIVETRAN_DATABASE.GOOGLE_ADS.ML_SPEND_RECOMMENDATION_DECISIONS TO ROLE SLACK_BOT_RO;
