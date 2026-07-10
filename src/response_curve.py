"""
Campaign-level spend -> outcome response-curve model ("Track A" in this
project's design discussion) -- an observational, panel-data estimate of how
spend has historically related to conversions, built as a legitimate,
guardrail-bounded first version while a real controlled-experiment system
("Track B") is still being designed/built.

Design summary (see project history / README for the full rationale):
  - Grain is CAMPAIGN, not ad group or keyword. This is a direct consequence
    of a data audit: keyword-level manual bids never change historically
    (94% of keywords report a bid of exactly 0 -- automated bidding
    dominates these accounts), and ad-group-level budgets/bids aren't
    tracked at all. Campaign-level budget IS tracked with real historical
    change events (see AD_REPORTING_STAGING.STG_GOOGLE_ADS__CAMPAIGN_BUDGET_HISTORY),
    and campaign-level daily spend has substantial natural variation
    (median coefficient of variation ~0.53) to estimate a response curve
    from even without discrete change events.
  - Two-way fixed effects (campaign + calendar-week period), via the
    standard "within" (demeaning) transformation -- nets out both each
    campaign's own baseline quality/scale AND any shock common to all
    campaigns in a given week (holidays, platform-wide auction shifts,
    etc.), leaving each campaign's own idiosyncratic spend deviation to
    identify the elasticity from.
  - log1p (not log) on spend/impressions/conversions -- handles genuine
    zero-spend weeks (stopped campaigns) naturally, without needing to drop
    or special-case them. This matters a lot here since the cleanest
    natural-experiment signal in this data is full stop/resume events.
  - Impressions included as a control. The central causal risk in this
    setting is automated bidding's own anticipation: it predicts a good day
    and spends more, so naive spend-outcome correlation partly reflects "the
    algorithm already knew," not "the extra dollars caused it." Impressions
    is an observable proxy for "more auction opportunity existed that week"
    independent of how much of it we chose to bid on -- controlling for it
    strips out some (not all) of this bias.
  - A genuine time-based train/test split (never random -- same principle
    as src/dataset.py's time_aware_split for the forecasting models), so the
    fitted elasticity's out-of-sample validity is actually checked.
  - A separate, higher-confidence validation pass restricted to clean
    stop/resume (fully off -> on) transitions in the test period. These are
    the cleanest natural experiment available: a paused campaign isn't in
    any auction regardless of what an automated bidding strategy might have
    predicted, which sidesteps the anticipation-bias concern entirely
    (unlike the continuous day-to-day spend variation the main regression
    also relies on).

Important limitation, stated plainly: this remains observational, not a
randomized experiment. It's most trustworthy WITHIN each campaign's own
historically observed spend range, and increasingly speculative
extrapolating beyond it -- recommendation guardrails should reflect that,
not just the model's in-sample confidence.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.snowflake_client import run_query

CAMPAIGN_QUERY_TEMPLATE = """
SELECT
    CLIENT_ID,
    CAMPAIGN_ID,
    CAMPAIGN_NAME,
    CHANNEL_TYPE,
    STAT_DATE,
    SUM(IMPRESSIONS) AS IMPRESSIONS,
    SUM(CLICKS)      AS CLICKS,
    SUM(SPEND)       AS SPEND,
    SUM(CONVERSIONS) AS CONVERSIONS
FROM FIVETRAN_DATABASE.GOOGLE_ADS.CAMPAIGNS_MAT
WHERE STAT_DATE >= DATEADD(day, -{history_days}, CURRENT_DATE())
GROUP BY CLIENT_ID, CAMPAIGN_ID, CAMPAIGN_NAME, CHANNEL_TYPE, STAT_DATE
ORDER BY CLIENT_ID, CAMPAIGN_ID, STAT_DATE
"""

VALUE_COLS = ["impressions", "clicks", "spend", "conversions"]
FEATURE_COL = "log1p_spend"
CONTROL_COLS = ["log1p_impressions"]
TARGET_COL = "log1p_conversions"


def pull_campaign_daily(history_days: int) -> pd.DataFrame:
    sql = CAMPAIGN_QUERY_TEMPLATE.format(history_days=history_days)
    df = run_query(sql)
    df.columns = [c.lower() for c in df.columns]
    for col in VALUE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if not pd.api.types.is_datetime64_any_dtype(df["stat_date"]):
        df["stat_date"] = pd.to_datetime(df["stat_date"])
    print(
        f"Pulled {len(df)} campaign-day rows, {df['campaign_id'].nunique()} campaigns, "
        f"{df['stat_date'].min().date()} to {df['stat_date'].max().date()}"
    )
    return df


def densify_daily(daily: pd.DataFrame) -> pd.DataFrame:
    """
    CAMPAIGNS_MAT doesn't always write an explicit zero row for a day with no
    reported activity -- confirmed directly on a real campaign's stop/resume
    transition, which had a mix of explicit spend=0 rows AND entirely missing
    dates around the same pause. Reindexing each campaign to its own complete
    daily calendar and filling gaps with 0 avoids silently under-counting a
    7-day chunk's true total, or wrongly excluding a genuinely complete week
    just because a quiet day didn't get an explicit row.
    """
    meta_cols = [c for c in ["client_id", "campaign_id", "campaign_name", "channel_type"] if c in daily.columns]
    frames = []

    for campaign_id, g in daily.groupby("campaign_id"):
        g = g.sort_values("stat_date")
        full_index = pd.date_range(g["stat_date"].min(), g["stat_date"].max(), freq="D")
        meta = g[meta_cols].iloc[0].to_dict()
        dense = g.set_index("stat_date")[VALUE_COLS].reindex(full_index, fill_value=0)
        dense = dense.reset_index().rename(columns={"index": "stat_date"})
        for col, val in meta.items():
            dense[col] = val
        frames.append(dense)

    return pd.concat(frames, ignore_index=True)


def build_chunk_panel(daily: pd.DataFrame, window_days: int = 7, min_avg_daily_spend: float = 5.0) -> pd.DataFrame:
    """
    Aggregates densified daily campaign performance into non-overlapping
    window_days chunks. Chunks are anchored to a GLOBAL start date (the
    earliest date across the whole pull), not each campaign's own start --
    this makes `period` a shared calendar-week index across every campaign,
    which is what lets the period fixed effect actually net out shocks
    common to many campaigns in the same week. Non-overlapping (rather than
    a rolling window) avoids the serial-correlation-from-overlap issue that
    would otherwise complicate the cluster-robust standard errors.
    """
    daily = daily.sort_values(["campaign_id", "stat_date"]).copy()
    global_start = daily["stat_date"].min()
    daily["period"] = ((daily["stat_date"] - global_start).dt.days // window_days).astype(int)

    group_cols = [c for c in ["client_id", "campaign_id", "channel_type", "period"] if c in daily.columns]
    agg = daily.groupby(group_cols, as_index=False).agg(
        chunk_start=("stat_date", "min"),
        n_days=("stat_date", "count"),
        spend=("spend", "sum"),
        impressions=("impressions", "sum"),
        clicks=("clicks", "sum"),
        conversions=("conversions", "sum"),
    )

    # Only complete (full window_days) chunks -- a campaign whose observed
    # history starts or ends mid-chunk gets a partial window that would
    # understate totals relative to every other chunk.
    agg = agg[agg["n_days"] == window_days].drop(columns=["n_days"])

    # Excludes campaigns that are negligible-spend THROUGHOUT (near-dead
    # accounts contribute mostly noise) -- computed on each campaign's own
    # overall average so individual near-zero chunks (stop events) within an
    # otherwise-real campaign are kept, since those are exactly what the
    # stop/resume validation needs.
    avg_daily_spend = agg.groupby("campaign_id")["spend"].transform("mean") / window_days
    agg = agg[avg_daily_spend >= min_avg_daily_spend].copy()

    for col in VALUE_COLS:
        agg[f"log1p_{col}"] = np.log1p(agg[col])

    return agg.sort_values(["campaign_id", "period"]).reset_index(drop=True)


def flag_transitions(panel: pd.DataFrame, off_threshold: float = 20.0) -> pd.DataFrame:
    """
    Flags each chunk as a "resume" (previous chunk was ~off, this chunk is
    on) or "stop" (previous chunk was on, this chunk is ~off) transition, per
    campaign. Used for the higher-confidence stop/resume validation --  a
    full off-to-on jump has no forecasting ambiguity (a paused campaign
    isn't bidding in any auction, regardless of what an automated strategy
    might have predicted), unlike the continuous spend variation the main
    regression also relies on.
    """
    panel = panel.sort_values(["campaign_id", "period"]).copy()
    panel["prev_spend"] = panel.groupby("campaign_id")["spend"].shift(1)
    panel["is_resume"] = (panel["prev_spend"] < off_threshold) & (panel["spend"] >= off_threshold)
    panel["is_stop"] = (panel["prev_spend"] >= off_threshold) & (panel["spend"] < off_threshold)
    return panel


def time_based_split(panel: pd.DataFrame, test_fraction: float = 0.2):
    """
    Holds out the most recent test_fraction of periods as test -- never a
    random split, for the same reason src/dataset.py's time_aware_split
    isn't random: test has to represent genuinely unseen time, not
    interpolation.
    """
    periods = sorted(panel["period"].unique())
    cutoff_idx = int(len(periods) * (1 - test_fraction))
    cutoff_period = periods[cutoff_idx]
    train = panel[panel["period"] < cutoff_period].copy()
    test = panel[panel["period"] >= cutoff_period].copy()
    return train, test, cutoff_period


def fit_response_curve(train: pd.DataFrame) -> dict:
    """
    Two-way (campaign + period) fixed-effects regression via the "within"
    (demeaning) transformation -- algebraically equivalent to including a
    dummy for every campaign and every period without needing thousands of
    dummy columns. Standard errors are cluster-robust by campaign_id, since
    each campaign contributes multiple correlated chunks.

    Campaign fixed effects are recovered explicitly afterward (as each
    campaign's mean train-period residual once the fitted slope is removed)
    so they can be reused for out-of-sample prediction. Period fixed effects
    are NOT reused this way -- their whole purpose is to absorb shocks
    specific to a historical calendar week that has no equivalent in a
    future/test period; they're a nuisance control for estimation, not part
    of the predictive equation used afterward.
    """
    cols = [TARGET_COL, FEATURE_COL] + CONTROL_COLS
    df = train.dropna(subset=cols).copy()

    campaign_means = df.groupby("campaign_id")[cols].transform("mean")
    period_means = df.groupby("period")[cols].transform("mean")
    overall_means = df[cols].mean()
    demeaned = df[cols] - campaign_means - period_means + overall_means

    X = sm.add_constant(demeaned[[FEATURE_COL] + CONTROL_COLS])
    y = demeaned[TARGET_COL]
    model = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": df["campaign_id"]})

    beta = float(model.params[FEATURE_COL])
    control_coefs = {c: float(model.params[c]) for c in CONTROL_COLS}

    slope_pred = beta * df[FEATURE_COL] + sum(control_coefs[c] * df[c] for c in CONTROL_COLS)
    residual_for_fe = df[TARGET_COL] - slope_pred
    campaign_fe = residual_for_fe.groupby(df["campaign_id"]).mean().to_dict()
    global_fe_fallback = float(residual_for_fe.mean())

    # For the naive-baseline comparison in evaluate_out_of_sample(): each
    # campaign's own train-period average log1p(conversions), with no
    # spend-sensitivity at all -- "this campaign just keeps doing what it's
    # been doing" is the bar the response curve actually needs to clear.
    train_campaign_mean_log1p = df.groupby("campaign_id")[TARGET_COL].mean().to_dict()
    train_overall_mean_log1p = float(df[TARGET_COL].mean())

    return {
        "model": model,
        "beta": beta,
        "beta_ci": [float(x) for x in model.conf_int().loc[FEATURE_COL]],
        "beta_pvalue": float(model.pvalues[FEATURE_COL]),
        "control_coefs": control_coefs,
        "campaign_fe": campaign_fe,
        "global_fe_fallback": global_fe_fallback,
        "train_campaign_mean_log1p": train_campaign_mean_log1p,
        "train_overall_mean_log1p": train_overall_mean_log1p,
        "n_train_chunks": int(len(df)),
        "n_train_campaigns": int(df["campaign_id"].nunique()),
    }


def _predict(df: pd.DataFrame, fit_result: dict) -> pd.Series:
    beta = fit_result["beta"]
    control_coefs = fit_result["control_coefs"]
    campaign_fe = fit_result["campaign_fe"]
    fallback_fe = fit_result["global_fe_fallback"]

    fe = df["campaign_id"].map(campaign_fe).fillna(fallback_fe)
    pred_log1p = fe + beta * df[FEATURE_COL] + sum(control_coefs[c] * df[c] for c in control_coefs)
    return pred_log1p


def _score(df: pd.DataFrame, pred_log1p: pd.Series, label: str) -> dict:
    pred_conversions = np.expm1(pred_log1p).clip(lower=0)
    actual_log1p = df[TARGET_COL].values
    actual_conversions = df["conversions"].values

    mae_log = float(np.mean(np.abs(actual_log1p - pred_log1p)))
    rmse_log = float(np.sqrt(np.mean((actual_log1p - pred_log1p) ** 2)))
    mae_level = float(np.mean(np.abs(actual_conversions - pred_conversions)))
    rmse_level = float(np.sqrt(np.mean((actual_conversions - pred_conversions) ** 2)))

    print(
        f"[{label}] n={len(df)}  MAE(log1p)={mae_log:.4f}  RMSE(log1p)={rmse_log:.4f}  "
        f"MAE(level)={mae_level:.4f}  RMSE(level)={rmse_level:.4f}"
    )
    return {
        "n": int(len(df)), "mae_log1p": mae_log, "rmse_log1p": rmse_log,
        "mae_level": mae_level, "rmse_level": rmse_level,
    }


def evaluate_out_of_sample(test: pd.DataFrame, fit_result: dict) -> dict:
    """
    Scores the model (fitted on train only) against genuinely unseen test
    periods, and against a naive baseline (each campaign's own train-period
    average conversions, with NO spend-sensitivity at all) -- same
    "is this actually earning its keep" comparison used everywhere else in
    this project's model evaluation.
    """
    cols = [TARGET_COL, FEATURE_COL, "conversions"] + CONTROL_COLS
    df = test.dropna(subset=cols).copy()

    pred_log1p = _predict(df, fit_result)
    model_metrics = _score(df, pred_log1p, "response_curve")

    train_campaign_mean = {k: v for k, v in fit_result.get("train_campaign_mean_log1p", {}).items()}
    baseline_pred = df["campaign_id"].map(train_campaign_mean).fillna(fit_result.get("train_overall_mean_log1p", 0.0))
    baseline_metrics = _score(df, baseline_pred, "naive_baseline (campaign's own train-period average)")

    return {"model": model_metrics, "naive_baseline": baseline_metrics}


def compute_campaign_evidence(panel_with_transitions: pd.DataFrame) -> dict:
    """
    Per-campaign summary of the evidence backing the fitted response curve --
    consumed by src/optimize.py's guardrails, NOT by the regression itself.

    Two things a spend-change recommendation needs to know before trusting
    the fitted beta for a given campaign, that fit_response_curve()'s pooled
    output doesn't carry on its own:
      - Its historically observed spend RANGE (min/max chunk spend). Per this
        module's docstring, the curve is most trustworthy WITHIN a campaign's
        own observed range and increasingly speculative extrapolating beyond
        it -- this is the number a "cap the recommended spend within range"
        guardrail is checked against.
      - Whether it ever had a genuine stop/resume transition. Per the
        out-of-sample validation this module runs, the fitted curve tracked
        actual outcomes meaningfully better on the stop/resume subset than on
        the general population of chunks -- so a campaign with that evidence
        warrants more confidence than one whose elasticity is inferred purely
        from continuous day-to-day variation (the weaker-validated case).

    Uses the FULL panel (train + test), not just train -- this is a
    guardrail bound on what's been observed as of now, not a training input,
    so there's no leakage concern in using all available history for it.
    """
    grouped = panel_with_transitions.groupby("campaign_id")
    evidence = grouped.agg(
        min_chunk_spend=("spend", "min"),
        max_chunk_spend=("spend", "max"),
        mean_chunk_spend=("spend", "mean"),
        n_chunks=("spend", "count"),
    )
    evidence["has_stop_event"] = grouped["is_stop"].any()
    evidence["has_resume_event"] = grouped["is_resume"].any()

    return {
        str(campaign_id): {
            "min_chunk_spend": float(row["min_chunk_spend"]),
            "max_chunk_spend": float(row["max_chunk_spend"]),
            "mean_chunk_spend": float(row["mean_chunk_spend"]),
            "n_chunks": int(row["n_chunks"]),
            "has_stop_event": bool(row["has_stop_event"]),
            "has_resume_event": bool(row["has_resume_event"]),
        }
        for campaign_id, row in evidence.iterrows()
    }


def validate_against_transitions(panel_with_transitions: pd.DataFrame, fit_result: dict, test_cutoff_period: int) -> dict:
    """
    Restricts evaluation to "resume" chunks (off -> on) that fall in the
    held-out test period -- the cleanest available validation of the fitted
    elasticity, since these transitions have no forecasting ambiguity a
    priori (a paused campaign isn't in any auction), unlike the continuous
    spend variation the main regression otherwise relies on.
    """
    resumes = panel_with_transitions[
        panel_with_transitions["is_resume"] & (panel_with_transitions["period"] >= test_cutoff_period)
    ].copy()
    if resumes.empty:
        print("No stop->resume transitions found in the held-out test period -- nothing to validate against yet.")
        return {"n_events": 0}

    cols = [TARGET_COL, FEATURE_COL, "conversions"] + CONTROL_COLS
    resumes = resumes.dropna(subset=cols)
    if resumes.empty:
        return {"n_events": 0}

    pred_log1p = _predict(resumes, fit_result)
    metrics = _score(resumes, pred_log1p, "stop_resume_validation")
    metrics["n_events"] = int(len(resumes))
    return metrics
