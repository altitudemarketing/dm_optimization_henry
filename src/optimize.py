"""
Optimization/guardrail layer v1: turns the campaign-level response-curve
model (src/response_curve.py) into an exact dollar spend-change
recommendation per AD GROUP, for a human to approve/reject in the frontend.

This is the "recommend an exact amount to increase/decrease spend by" system
discussed in this project's system-design conversation -- distinct from
src/recommend.py, which flags forecasted performance swings but never
proposes a dollar amount. The two modules are complementary and both keep
running: recommend.py flags "something's off/promising here" from the
forecasting models; this module answers "here's the spend change to
consider" from the causal response curve, for ad groups that clear its
guardrails.

How a campaign-level curve becomes an ad-group-level dollar recommendation
(read this before changing the math below):
  The fitted curve is log1p(conversions) = FE_campaign + beta*log1p(spend) +
  coef*log1p(impressions), estimated on CAMPAIGN totals (see
  response_curve.py's docstring for why). Applying it to one ad group's own
  spend change only requires the RATIO of predicted conversions at a new vs.
  current spend level:

      predicted_new / predicted_current
        = exp[beta * (log1p(new_spend) - log1p(current_spend))]

  because FE_campaign and the impressions term (held fixed -- see below)
  cancel out of the ratio. This sidesteps ever needing an ad-group-level
  fixed effect or an absolute-level prediction (which the campaign-grain fit
  isn't equipped to produce for a sub-campaign slice) at the cost of an
  explicit, stated assumption: **the campaign's fitted elasticity applies
  uniformly to each of its ad groups** -- not separately validated, since
  ad-group-level history can't support its own fixed-effects fit (see
  response_curve.py's grain discussion). Guardrails below are deliberately
  conservative to compensate for this.

  Impressions are held FIXED at the ad group's current trailing level in the
  counterfactual -- i.e. this assumes a spend change buys more competitiveness
  in the same auction opportunity, not more distinct auctions. This is the
  more conservative of the two readings of the fitted beta (it doesn't credit
  the recommendation with any assumed impression uplift), and avoids
  compounding uncertainty from a second, unestimated spend->impressions
  relationship.

Guardrails (why each exists):
  - min_trailing_spend: suppress tiny ad groups -- too little at stake, same
    philosophy as recommendations.min_trailing_spend in src/recommend.py.
  - min_evidence_chunks: a campaign needs a minimum amount of its own history
    in the response-curve panel before its elasticity is trusted for a dollar
    recommendation at all. Campaigns with no evidence (too new, or filtered
    out of the panel entirely) get INSUFFICIENT_EVIDENCE, not a fallback
    guess -- "no evidence, no recommendation" rather than silently
    extrapolating from a global average.
  - spend_range_tolerance: caps how far the recommended CAMPAIGN-level total
    spend (current + this ad group's proposed dollar change) can land outside
    that campaign's own historically observed chunk-spend range. Directly
    enforces this project's stated limitation that the curve is "most
    trustworthy within each campaign's own historically observed spend range,
    and increasingly speculative extrapolating beyond it."
  - Asymmetric confidence bounds, not the point estimate: increases are
    evaluated using the LOWER bound of beta's 95% CI (the most pessimistic
    plausible elasticity -- an increase only clears the bar if it looks good
    even under the weakest defensible effect size). Decreases are evaluated
    using the UPPER bound (the most optimistic/steepest plausible elasticity
    -- a decrease only clears the bar if the conversion loss stays small even
    under the strongest defensible effect size). This bakes the model's own
    estimation uncertainty directly into the guardrail instead of trusting
    the point estimate at face value.
  - Zero-conversion-history ad groups are skipped outright (not scored):
    the ratio-based counterfactual (predicted_new / predicted_current) is
    undefined/meaningless when current conversions are 0, and this project's
    own segmented evaluation of the forecasting models already found the
    zero-history segment to be the least reliable one generally.
  - confidence_tier: "high" if the ad group's parent campaign had an actual
    stop/resume event in its history (the specific subset this project's
    validation found meaningfully more accurate); "medium" otherwise
    (elasticity inferred only from continuous day-to-day variation, the
    weaker-validated case).

The increase guardrail compares predicted CPA to an EXTERNAL target, not the
ad group's own average CPA -- read this before changing it back:
  v1 originally required a spend increase to *improve* CPA relative to the
  ad group's own recent average. That bar turns out to be structurally
  unclearable: since FE cancels out of the ratio-based counterfactual (see
  above), whether a spend increase improves or worsens CPA depends only on
  beta vs. 1 -- and this project's fitted beta (~0.13, 95% CI up to ~0.20) is
  far below 1 for every campaign, meaning conversions always grow slower than
  spend under this curve. That makes "does average CPA improve" mathematically
  unclearable by ANY ad group, regardless of its own evidence quality --
  which is exactly why the first production run recommended zero increases.
  A real marketer doesn't need average CPA to improve to justify more spend;
  they need the MARGINAL CPA to stay under whatever they'd consider an
  acceptable/breakeven cost -- an external number, not the campaign's own
  historical average. pull_campaign_cpa_targets() resolves the best available
  such number per campaign, in priority order:
    1. The campaign's own Target CPA, if it's actually on Target CPA bidding
       with a nonzero value set (STG_GOOGLE_ADS__CAMPAIGN_BIDDING_STRATEGY_HISTORY).
       Real audit finding: only ~1% of campaigns are on this strategy with a
       genuine nonzero target -- Maximize Conversions campaigns technically
       have a TARGET_CPA column but it's populated with 0 (no cap was set),
       so a 0 is treated as "no target", not "target of $0".
    2. The CPA implied by the campaign's Target ROAS setting (target_roas =
       conversion value / spend), converted to a CPA-equivalent using that
       campaign's own trailing average value-per-conversion.
    3. The client's own blended trailing CPA across their whole account --
       still an external, non-circular anchor ("would this money work at
       least as well here as it does for this client on average elsewhere"),
       used when neither 1 nor 2 is available (the large majority of
       campaigns, mostly on Maximize Conversions with no explicit ceiling).
  If none of these resolve for a campaign (e.g. a brand-new client with no
  trailing conversions anywhere), increases are never evaluated for its ad
  groups -- reintroducing the ad group's own CPA as a last-resort fallback
  would just reintroduce the original circular, unclearable bar.

Every record is recommendation-only: requires_human_review is always True.
Nothing here writes to Google Ads.

Usage:
    python -m src.optimize [--config path/to/config.yaml]

Requires, in order: `python -m scripts.build_dataset` (ad-group data) and
`python -m scripts.train_response_curve` (the response curve + campaign
evidence this module loads).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config_loader import load_config
from src.snowflake_client import SnowflakeConfigError, require_snowflake_env, run_query

SNOWFLAKE_TABLE = "FIVETRAN_DATABASE.GOOGLE_ADS.ML_SPEND_RECOMMENDATIONS"
SNOWFLAKE_COLUMNS = [
    "generated_at", "entity_id", "client_id", "client_name", "campaign_id",
    "campaign_name", "ad_group_id", "ad_group_name", "stat_date", "window_days",
    "current_spend", "current_conversions", "current_cpa",
    "cpa_target", "cpa_target_source",
    "campaign_trailing_spend", "campaign_spend_range_min", "campaign_spend_range_max",
    "campaign_n_evidence_chunks", "recommended_action", "recommended_pct_change",
    "recommended_dollar_change", "recommended_new_spend",
    "predicted_conversions_at_recommended", "predicted_conversion_delta",
    "predicted_cpa_at_recommended", "beta_used", "beta_ci_low", "beta_ci_high",
    "confidence_tier", "rationale", "requires_human_review", "model_used",
]

OUTPUT_COLUMNS = SNOWFLAKE_COLUMNS  # same shape locally and in Snowflake

MODEL_NAME = "response_curve_v1"

# ── External CPA target resolution (see module docstring) ──────────────────────
# CLIENT_BLENDED_CPA is computed EXCLUDING the campaign it's being attached to
# (client total minus this campaign's own spend/conversions) -- real-data audit
# during this project caught the un-excluded version being circular in
# practice: one client had exactly one campaign in the trailing window, so its
# "external" blended CPA was just that campaign's own noisy 1-conversion
# average relabeled; another client's target campaign alone supplied over half
# the spend/conversions feeding its own "external" comparison. Excluding self
# fixes both -- and correctly yields NULL (no resolvable target) for a
# single-campaign client, rather than a fake external number.
CPA_TARGET_QUERY_TEMPLATE = """
WITH campaign_trailing AS (
    SELECT CLIENT_ID, CAMPAIGN_ID,
           SUM(SPEND) AS SPEND, SUM(CONVERSIONS) AS CONVERSIONS,
           SUM(CONVERSIONS_VALUE) AS CONVERSIONS_VALUE
    FROM FIVETRAN_DATABASE.GOOGLE_ADS.CAMPAIGNS_MAT
    WHERE STAT_DATE >= DATEADD(day, -{history_days}, CURRENT_DATE())
    GROUP BY CLIENT_ID, CAMPAIGN_ID
),
client_totals AS (
    SELECT CLIENT_ID,
           SUM(SPEND) AS CLIENT_TOTAL_SPEND,
           SUM(CONVERSIONS) AS CLIENT_TOTAL_CONVERSIONS
    FROM campaign_trailing
    GROUP BY CLIENT_ID
),
bidding AS (
    SELECT CAST(CAMPAIGN_ID AS STRING) AS CAMPAIGN_ID, TARGET_CPA, TARGET_ROAS, BIDDING_STRATEGY_TYPE
    FROM FIVETRAN_DATABASE.AD_REPORTING_STAGING.STG_GOOGLE_ADS__CAMPAIGN_BIDDING_STRATEGY_HISTORY
    WHERE IS_MOST_RECENT_RECORD = TRUE
)
SELECT
    ct.CAMPAIGN_ID, ct.CLIENT_ID, ct.CONVERSIONS, ct.CONVERSIONS_VALUE,
    b.TARGET_CPA, b.TARGET_ROAS, b.BIDDING_STRATEGY_TYPE,
    (ctot.CLIENT_TOTAL_SPEND - ct.SPEND)
        / NULLIF(ctot.CLIENT_TOTAL_CONVERSIONS - ct.CONVERSIONS, 0) AS CLIENT_BLENDED_CPA
FROM campaign_trailing ct
JOIN client_totals ctot ON ctot.CLIENT_ID = ct.CLIENT_ID
LEFT JOIN bidding b ON b.CAMPAIGN_ID = ct.CAMPAIGN_ID
"""

CPA_TARGET_SOURCE_LABELS = {
    "client_target_cpa": "this campaign's own Target CPA bid strategy setting",
    "client_target_roas_implied": "the CPA implied by this campaign's Target ROAS setting",
    "client_blended_avg_cpa": "this client's own blended average CPA across their account",
}


def pull_campaign_cpa_targets(history_days: int = 90) -> dict:
    """
    Resolves the best available EXTERNAL CPA anchor per campaign, in the
    priority order documented in this module's docstring. Returns
    {campaign_id: {"cpa_target": float, "source": str}} -- campaigns with no
    resolvable target (no real bidding-strategy target AND no trailing
    conversions anywhere in the client's account) are simply absent, which is
    the signal build_spend_recommendations() uses to skip evaluating
    increases for their ad groups entirely.
    """
    df = run_query(CPA_TARGET_QUERY_TEMPLATE.format(history_days=history_days))
    df.columns = [c.lower() for c in df.columns]

    targets = {}
    for _, row in df.iterrows():
        campaign_id = str(row["campaign_id"])
        strategy = row.get("bidding_strategy_type")
        target_cpa = row.get("target_cpa")
        target_roas = row.get("target_roas")
        conversions = row.get("conversions") or 0
        conversions_value = row.get("conversions_value") or 0
        client_blended_cpa = row.get("client_blended_cpa")

        cpa_target, source = None, None

        # Priority 1: a genuine, nonzero Target CPA bid strategy setting.
        # MAXIMIZE_CONVERSIONS campaigns also populate this column but with
        # 0 (no ceiling was ever set) -- treated as "no target", not "a
        # target of $0", which would make every increase fail trivially.
        if strategy == "TARGET_CPA" and pd.notna(target_cpa) and target_cpa > 0:
            cpa_target, source = float(target_cpa), "client_target_cpa"

        # Priority 2: Target ROAS, converted to a CPA-equivalent using this
        # campaign's own trailing average value per conversion.
        elif (
            strategy == "MAXIMIZE_CONVERSION_VALUE"
            and pd.notna(target_roas) and target_roas > 0
            and conversions > 0 and conversions_value > 0
        ):
            avg_value_per_conversion = conversions_value / conversions
            cpa_target, source = float(avg_value_per_conversion / target_roas), "client_target_roas_implied"

        # Priority 3: this client's own blended CPA across their account --
        # still external to the specific ad group being scored, just not to
        # the client. Covers the large majority of campaigns (mostly on
        # Maximize Conversions, with no explicit ceiling of any kind).
        if cpa_target is None and pd.notna(client_blended_cpa) and client_blended_cpa > 0:
            cpa_target, source = float(client_blended_cpa), "client_blended_avg_cpa"

        if cpa_target is not None:
            targets[campaign_id] = {"cpa_target": cpa_target, "source": source}

    print(
        f"Resolved an external CPA target for {len(targets)} / {df['campaign_id'].nunique()} campaigns "
        f"({sum(1 for t in targets.values() if t['source'] == 'client_target_cpa')} from Target CPA, "
        f"{sum(1 for t in targets.values() if t['source'] == 'client_target_roas_implied')} from Target ROAS, "
        f"{sum(1 for t in targets.values() if t['source'] == 'client_blended_avg_cpa')} from client blended CPA)."
    )
    return targets


def _load_response_curve_artifacts(output_dir: Path) -> dict:
    metrics_path = output_dir / "response_curve_metrics.json"
    fe_path = output_dir / "campaign_fixed_effects.json"
    evidence_path = output_dir / "campaign_evidence.json"

    for p in (metrics_path, fe_path, evidence_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Run `python -m scripts.train_response_curve` first."
            )

    with open(metrics_path) as f:
        metrics = json.load(f)
    with open(evidence_path) as f:
        evidence = json.load(f)

    return {
        "beta": metrics["beta_elasticity"],
        "beta_ci_low": metrics["beta_ci_95"][0],
        "beta_ci_high": metrics["beta_ci_95"][1],
        "campaign_evidence": evidence,
    }


def _rolling_col(base: str, window_days: int) -> str:
    return f"{base}_rolling_{window_days}d"


def _predicted_ratio(beta: float, current_spend: float, new_spend: float) -> float:
    """
    exp[beta * (log1p(new_spend) - log1p(current_spend))] -- see module
    docstring for why this ratio is all that's needed (campaign FE and the
    impressions control both cancel out of it).
    """
    return float(np.exp(beta * (np.log1p(new_spend) - np.log1p(current_spend))))


def _evaluate_candidate(
    pct_change: float,
    current_spend: float,
    current_conversions: float,
    campaign_trailing_spend: float,
    evidence: dict,
    beta_low: float,
    beta_high: float,
    spend_range_tolerance: float,
):
    """
    Evaluates one candidate ad-group spend change. Returns None if it fails
    the campaign-level historical spend-range guardrail (infeasible, not
    just unattractive). Otherwise returns a dict of the candidate's predicted
    effects, computed with the conservative CI bound appropriate to its
    direction (see module docstring).
    """
    new_spend = current_spend * (1 + pct_change)
    dollar_delta = new_spend - current_spend
    new_campaign_spend = campaign_trailing_spend + dollar_delta

    range_min = evidence["min_chunk_spend"] * (1 - spend_range_tolerance)
    range_max = evidence["max_chunk_spend"] * (1 + spend_range_tolerance)
    if not (range_min <= new_campaign_spend <= range_max):
        return None

    beta_used = beta_low if pct_change > 0 else beta_high
    ratio = _predicted_ratio(beta_used, current_spend, new_spend)
    predicted_conversions = current_conversions * ratio
    predicted_delta = predicted_conversions - current_conversions
    predicted_cpa = new_spend / predicted_conversions if predicted_conversions > 0 else float("inf")

    return {
        "pct_change": pct_change,
        "new_spend": new_spend,
        "dollar_delta": dollar_delta,
        "beta_used": beta_used,
        "predicted_conversions": predicted_conversions,
        "predicted_delta": predicted_delta,
        "predicted_cpa": predicted_cpa,
    }


def build_spend_recommendations(config) -> pd.DataFrame:
    opt_config = config.get("optimization", {})
    window_days = opt_config.get("window_days", 7)
    min_trailing_spend = opt_config.get("min_trailing_spend", 50)
    min_evidence_chunks = opt_config.get("min_evidence_chunks", 8)
    candidate_pct_changes = opt_config.get("candidate_pct_changes", [-0.20, -0.10, 0.10, 0.20])
    max_pct_change_per_cycle = opt_config.get("max_pct_change_per_cycle", 0.20)
    spend_range_tolerance = opt_config.get("spend_range_tolerance", 0.15)
    cpa_target_margin_pct = opt_config.get("cpa_target_margin_pct", 0.05)
    cpa_target_lookback_days = opt_config.get("cpa_target_lookback_days", 90)
    max_acceptable_conversion_loss_pct = opt_config.get("max_acceptable_conversion_loss_pct", 0.05)

    candidate_pct_changes = [p for p in candidate_pct_changes if abs(p) <= max_pct_change_per_cycle + 1e-9]

    response_curve_dir = Path(config["paths"].get("response_curve_output", "models/response_curve"))
    artifacts = _load_response_curve_artifacts(response_curve_dir)
    beta_low = artifacts["beta_ci_low"]
    beta_high = artifacts["beta_ci_high"]
    campaign_evidence = artifacts["campaign_evidence"]

    print("Resolving external CPA targets (Target CPA / Target ROAS / client blended CPA)...")
    cpa_targets = pull_campaign_cpa_targets(cpa_target_lookback_days)

    processed_path = Path(config["paths"]["processed_dataset"])
    if not processed_path.exists():
        raise FileNotFoundError(f"{processed_path} not found. Run `python -m scripts.build_dataset` first.")
    df = pd.read_parquet(processed_path)

    spend_col = _rolling_col("spend", window_days)
    conv_col = _rolling_col("conversions", window_days)
    if spend_col not in df.columns or conv_col not in df.columns:
        raise ValueError(
            f"{spend_col}/{conv_col} not found in the processed dataset -- "
            f"optimization.window_days ({window_days}) must be one of config.yaml's "
            f"model.rolling_windows."
        )

    # One row per ad group: its most recent day of history -- same convention
    # as src/recommend.py's `latest`.
    latest = df.sort_values("stat_date").groupby("entity_id").tail(1).copy()
    latest["current_spend"] = latest[spend_col].fillna(0)
    latest["current_conversions"] = latest[conv_col].fillna(0)
    latest["current_cpa"] = np.where(
        latest["current_conversions"] > 0, latest["current_spend"] / latest["current_conversions"], np.nan,
    )

    # Current CAMPAIGN-level trailing spend, needed for the spend-range
    # guardrail (fit at campaign grain -- see module docstring). Approximates
    # "campaign total right now" as the sum of its ad groups' own trailing
    # windows, each as of that ad group's own latest date.
    campaign_trailing_spend = latest.groupby("campaign_id")["current_spend"].transform("sum")
    latest["campaign_trailing_spend"] = campaign_trailing_spend

    generated_at = pd.Timestamp.now("UTC").tz_localize(None)
    records = []

    for _, row in latest.iterrows():
        campaign_id = str(row["campaign_id"])
        target_info = cpa_targets.get(campaign_id)
        base = {
            "generated_at": generated_at,
            "entity_id": row["entity_id"],
            "client_id": row.get("client_id"),
            "client_name": row.get("client_name"),
            "campaign_id": row["campaign_id"],
            "campaign_name": row.get("campaign_name"),
            "ad_group_id": row.get("ad_group_id"),
            "ad_group_name": row.get("ad_group_name"),
            "stat_date": row["stat_date"],
            "window_days": window_days,
            "current_spend": float(row["current_spend"]),
            "current_conversions": float(row["current_conversions"]),
            "current_cpa": float(row["current_cpa"]) if pd.notna(row["current_cpa"]) else None,
            "cpa_target": target_info["cpa_target"] if target_info else None,
            "cpa_target_source": target_info["source"] if target_info else None,
            "campaign_trailing_spend": float(row["campaign_trailing_spend"]),
            "beta_ci_low": beta_low,
            "beta_ci_high": beta_high,
            "requires_human_review": True,
            "model_used": MODEL_NAME,
        }

        # Guardrail: too little spend at stake.
        if row["current_spend"] < min_trailing_spend:
            continue

        evidence = campaign_evidence.get(campaign_id)
        if evidence is None or evidence["n_chunks"] < min_evidence_chunks:
            records.append({
                **base,
                "campaign_spend_range_min": evidence["min_chunk_spend"] if evidence else None,
                "campaign_spend_range_max": evidence["max_chunk_spend"] if evidence else None,
                "campaign_n_evidence_chunks": evidence["n_chunks"] if evidence else 0,
                "recommended_action": "INSUFFICIENT_EVIDENCE",
                "recommended_pct_change": None, "recommended_dollar_change": None,
                "recommended_new_spend": None, "predicted_conversions_at_recommended": None,
                "predicted_conversion_delta": None, "predicted_cpa_at_recommended": None,
                "beta_used": None, "confidence_tier": "insufficient_evidence",
                "rationale": (
                    f"Parent campaign has {evidence['n_chunks'] if evidence else 0} chunks of "
                    f"response-curve history (needs >= {min_evidence_chunks}). Not enough evidence "
                    f"to extrapolate a spend change for this campaign yet."
                ),
            })
            continue

        # Guardrail: the ratio-based counterfactual is undefined/meaningless
        # with zero current conversions, and this is exactly the segment this
        # project's forecasting-model evaluation already found least
        # reliable generally.
        if row["current_conversions"] <= 0:
            records.append({
                **base,
                "campaign_spend_range_min": evidence["min_chunk_spend"],
                "campaign_spend_range_max": evidence["max_chunk_spend"],
                "campaign_n_evidence_chunks": evidence["n_chunks"],
                "recommended_action": "INSUFFICIENT_EVIDENCE",
                "recommended_pct_change": None, "recommended_dollar_change": None,
                "recommended_new_spend": None, "predicted_conversions_at_recommended": None,
                "predicted_conversion_delta": None, "predicted_cpa_at_recommended": None,
                "beta_used": None, "confidence_tier": "insufficient_evidence",
                "rationale": (
                    "This ad group has real trailing spend but 0 trailing conversions -- the "
                    "response curve's spend-change ratio is undefined here and this segment is "
                    "the least reliable in this project's own model evaluation. See "
                    "src/recommend.py's FORECASTED_ZERO_HIGH_SPEND flag for this case instead."
                ),
            })
            continue

        confidence_tier = "high" if (evidence["has_stop_event"] or evidence["has_resume_event"]) else "medium"

        candidates = [
            _evaluate_candidate(
                pct, row["current_spend"], row["current_conversions"], row["campaign_trailing_spend"],
                evidence, beta_low, beta_high, spend_range_tolerance,
            )
            for pct in candidate_pct_changes
        ]
        candidates = [c for c in candidates if c is not None]

        # Increases are evaluated against an EXTERNAL CPA target, not the ad
        # group's own average -- see module docstring for why "does average
        # CPA improve" is structurally unclearable given beta << 1. No
        # increase is ever considered when no external target resolved for
        # this campaign (target_info is None) -- see pull_campaign_cpa_targets.
        increase_candidates = []
        if target_info is not None:
            cpa_target = target_info["cpa_target"]
            increase_candidates = [
                c for c in candidates
                if c["pct_change"] > 0
                and c["predicted_cpa"] < cpa_target * (1 - cpa_target_margin_pct)
            ]
        decrease_candidates = [
            c for c in candidates
            if c["pct_change"] < 0
            and abs(c["predicted_delta"]) <= row["current_conversions"] * max_acceptable_conversion_loss_pct
        ]

        if increase_candidates:
            best = max(increase_candidates, key=lambda c: c["pct_change"])
            action = "INCREASE_SPEND"
            source_label = CPA_TARGET_SOURCE_LABELS.get(target_info["source"], "external CPA target")
            rationale = (
                f"Even at the conservative (lower 95% CI) elasticity estimate of {best['beta_used']:.4f}, "
                f"a {best['pct_change']*100:+.0f}% spend change (${best['dollar_delta']:+,.0f}) is predicted "
                f"to reach a CPA of ${best['predicted_cpa']:.2f} -- at least {cpa_target_margin_pct*100:.0f}% "
                f"under {source_label} of ${target_info['cpa_target']:.2f}."
            )
        elif decrease_candidates:
            best = min(decrease_candidates, key=lambda c: c["pct_change"])
            action = "DECREASE_SPEND"
            rationale = (
                f"Even at the conservative (upper 95% CI) elasticity estimate of {best['beta_used']:.4f}, "
                f"a {best['pct_change']*100:+.0f}% spend change (${best['dollar_delta']:+,.0f}) is predicted "
                f"to cost only {abs(best['predicted_delta']):.2f} conversions "
                f"(<= {max_acceptable_conversion_loss_pct*100:.0f}% of current) -- budget could be reallocated."
            )
        else:
            best = None
            action = "HOLD"
            rationale = (
                "No spend change within the tested range and this campaign's historically "
                "observed spend range clears the conservative CI-bound guardrails in either "
                "direction. No confident recommendation this run."
            )

        records.append({
            **base,
            "campaign_spend_range_min": evidence["min_chunk_spend"],
            "campaign_spend_range_max": evidence["max_chunk_spend"],
            "campaign_n_evidence_chunks": evidence["n_chunks"],
            "recommended_action": action,
            "recommended_pct_change": best["pct_change"] if best else None,
            "recommended_dollar_change": best["dollar_delta"] if best else None,
            "recommended_new_spend": best["new_spend"] if best else None,
            "predicted_conversions_at_recommended": best["predicted_conversions"] if best else None,
            "predicted_conversion_delta": best["predicted_delta"] if best else None,
            "predicted_cpa_at_recommended": (
                best["predicted_cpa"] if best and np.isfinite(best["predicted_cpa"]) else None
            ),
            "beta_used": best["beta_used"] if best else None,
            "confidence_tier": confidence_tier,
            "rationale": rationale,
        })

    if not records:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    result = pd.DataFrame.from_records(records)
    for col in OUTPUT_COLUMNS:
        if col not in result.columns:
            result[col] = None
    return result[OUTPUT_COLUMNS].sort_values(
        ["recommended_action", "confidence_tier"],
    ).reset_index(drop=True)


def _sql_literal(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "NULL"
    if isinstance(value, bool) or isinstance(value, np.bool_):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, np.integer, float, np.floating)):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return f"'{value.strftime('%Y-%m-%d %H:%M:%S')}'"
    escaped = str(value).replace("'", "''")
    return f"'{escaped}'"


def upload_to_snowflake(recommendations: pd.DataFrame) -> None:
    """
    Appends to Snowflake as a new batch, same append-only convention as
    src/recommend.py's upload_to_snowflake (see that function's docstring --
    past runs are kept as the future outcome-tracking feedback loop's
    training data, never updated/deleted here).
    """
    if recommendations.empty:
        print("No spend recommendations to upload to Snowflake this run.")
        return

    try:
        require_snowflake_env()
    except SnowflakeConfigError as e:
        print(f"Skipping Snowflake upload -- {e}")
        return

    rows_sql = []
    for _, row in recommendations.iterrows():
        values = ", ".join(_sql_literal(row[col]) for col in SNOWFLAKE_COLUMNS)
        rows_sql.append(f"({values})")

    insert_sql = (
        f"INSERT INTO {SNOWFLAKE_TABLE} ({', '.join(SNOWFLAKE_COLUMNS)}) VALUES\n"
        + ",\n".join(rows_sql)
    )
    run_query(insert_sql)
    print(f"Inserted {len(recommendations)} rows into {SNOWFLAKE_TABLE}.")


def main(config_path=None):
    config = load_config(config_path)
    recommendations = build_spend_recommendations(config)

    out_path = Path(config["paths"].get("spend_recommendations_output", "data/spend_recommendations.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    recommendations.to_json(out_path, orient="records", date_format="iso", indent=2)

    counts = recommendations["recommended_action"].value_counts().to_dict() if not recommendations.empty else {}
    print(f"{len(recommendations)} spend recommendations generated: {counts}")
    print(f"Saved to {out_path}")

    upload_to_snowflake(recommendations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    main(args.config)
