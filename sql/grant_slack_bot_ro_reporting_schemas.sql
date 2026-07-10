-- Read-only visibility grant for SLACK_BOT_RO into two schemas it currently
-- cannot see at all -- confirmed via `SHOW SCHEMAS IN DATABASE
-- FIVETRAN_DATABASE`, which returns nothing for these (not just empty
-- tables -- no USAGE grant on the schema itself exists yet).
--
-- Why these two specifically:
--   AD_REPORTING_REPORTS / AD_REPORTING_STAGING -- sounds like a curated
--     reporting layer that could already model bid/budget history in some
--     form; worth checking before assuming we need to build that ourselves.
--   AIRBYTE_SCHEMA -- a second, separate ingestion pipeline from Fivetran's.
--     May sync Google Ads objects (ad_group_criterion, campaign,
--     campaign_budget history) that the current Fivetran connector doesn't.
--
-- This is READ-ONLY (USAGE + SELECT) -- no write/modify privileges granted,
-- consistent with SLACK_BOT_RO's existing role. Run as ACCOUNTADMIN,
-- SYSADMIN, or whichever role owns these schemas.

GRANT USAGE ON SCHEMA FIVETRAN_DATABASE.AD_REPORTING_REPORTS TO ROLE SLACK_BOT_RO;
GRANT SELECT ON ALL TABLES IN SCHEMA FIVETRAN_DATABASE.AD_REPORTING_REPORTS TO ROLE SLACK_BOT_RO;
GRANT SELECT ON FUTURE TABLES IN SCHEMA FIVETRAN_DATABASE.AD_REPORTING_REPORTS TO ROLE SLACK_BOT_RO;

GRANT USAGE ON SCHEMA FIVETRAN_DATABASE.AD_REPORTING_STAGING TO ROLE SLACK_BOT_RO;
GRANT SELECT ON ALL TABLES IN SCHEMA FIVETRAN_DATABASE.AD_REPORTING_STAGING TO ROLE SLACK_BOT_RO;
GRANT SELECT ON FUTURE TABLES IN SCHEMA FIVETRAN_DATABASE.AD_REPORTING_STAGING TO ROLE SLACK_BOT_RO;

GRANT USAGE ON SCHEMA FIVETRAN_DATABASE.AIRBYTE_SCHEMA TO ROLE SLACK_BOT_RO;
GRANT SELECT ON ALL TABLES IN SCHEMA FIVETRAN_DATABASE.AIRBYTE_SCHEMA TO ROLE SLACK_BOT_RO;
GRANT SELECT ON FUTURE TABLES IN SCHEMA FIVETRAN_DATABASE.AIRBYTE_SCHEMA TO ROLE SLACK_BOT_RO;
