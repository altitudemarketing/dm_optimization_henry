"""
Fits and validates the campaign-level spend -> conversions response-curve
model (src/response_curve.py). See that module's docstring for the full
design rationale.

Usage:
    python -m scripts.train_response_curve [--config path/to/config.yaml] [--use-cache]

--use-cache skips the Snowflake pull and reuses whatever's cached at
paths.campaign_raw_cache (handy for iterating without re-querying).
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config_loader import load_config
from src.response_curve import (
    build_chunk_panel,
    compute_campaign_evidence,
    densify_daily,
    fit_response_curve,
    flag_transitions,
    evaluate_out_of_sample,
    pull_campaign_daily,
    time_based_split,
    validate_against_transitions,
)


def main(config_path=None, use_cache=False):
    config = load_config(config_path)
    rc_config = config.get("response_curve", {})
    window_days = rc_config.get("window_days", 7)
    min_avg_daily_spend = rc_config.get("min_avg_daily_spend", 5.0)
    test_fraction = rc_config.get("test_fraction", 0.2)
    off_threshold = rc_config.get("stop_resume_spend_threshold", 20.0)
    history_days = rc_config.get("history_days", 380)

    raw_cache_path = Path(config["paths"].get("campaign_raw_cache", "data/raw_campaign_daily_performance.parquet"))

    if use_cache:
        if not raw_cache_path.exists():
            raise FileNotFoundError(
                f"--use-cache set but {raw_cache_path} doesn't exist. Omit --use-cache "
                f"to pull from Snowflake first."
            )
        print(f"Loading cached campaign data from {raw_cache_path}")
        daily = pd.read_parquet(raw_cache_path)
    else:
        print("Pulling campaign-level daily performance from Snowflake...")
        daily = pull_campaign_daily(history_days)
        raw_cache_path.parent.mkdir(parents=True, exist_ok=True)
        daily.to_parquet(raw_cache_path, index=False)
        print(f"Cached raw campaign data to {raw_cache_path} ({len(daily)} rows)")

    print("Densifying to a complete daily calendar per campaign (fills unreported quiet days with 0)...")
    daily = densify_daily(daily)

    print(f"Building {window_days}-day chunk panel...")
    panel = build_chunk_panel(daily, window_days=window_days, min_avg_daily_spend=min_avg_daily_spend)
    print(f"Panel: {len(panel)} chunks across {panel['campaign_id'].nunique()} campaigns")

    panel = flag_transitions(panel, off_threshold=off_threshold)
    n_resume_events = int(panel["is_resume"].sum())
    n_stop_events = int(panel["is_stop"].sum())
    print(f"Flagged {n_stop_events} stop events and {n_resume_events} resume events across the full panel")

    train, test, cutoff_period = time_based_split(panel, test_fraction=test_fraction)
    print(f"Train: {len(train)} chunks | Test: {len(test)} chunks (cutoff period={cutoff_period})")

    if len(train) == 0 or len(test) == 0:
        raise ValueError(
            "Not enough chunks for a train/test split -- reduce response_curve.test_fraction "
            "or response_curve.min_avg_daily_spend, or increase response_curve.history_days."
        )

    print("Fitting two-way fixed-effects response curve on train...")
    fit_result = fit_response_curve(train)
    print(
        f"Elasticity (beta on log1p(spend)): {fit_result['beta']:.4f} "
        f"(95% CI: [{fit_result['beta_ci'][0]:.4f}, {fit_result['beta_ci'][1]:.4f}], "
        f"p={fit_result['beta_pvalue']:.4g}) -- fit on {fit_result['n_train_chunks']} chunks "
        f"across {fit_result['n_train_campaigns']} campaigns"
    )

    print("Evaluating out-of-sample (test period)...")
    oos_metrics = evaluate_out_of_sample(test, fit_result)

    print("Validating against held-out stop->resume transitions specifically...")
    transition_metrics = validate_against_transitions(panel, fit_result, cutoff_period)

    output_dir = Path(config["paths"].get("response_curve_output", "models/response_curve"))
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "response_curve_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "window_days": window_days,
                "history_days": history_days,
                "beta_elasticity": fit_result["beta"],
                "beta_ci_95": fit_result["beta_ci"],
                "beta_pvalue": fit_result["beta_pvalue"],
                "control_coefs": fit_result["control_coefs"],
                "n_train_chunks": fit_result["n_train_chunks"],
                "n_train_campaigns": fit_result["n_train_campaigns"],
                "n_test_chunks": int(len(test)),
                "n_stop_events_full_panel": n_stop_events,
                "n_resume_events_full_panel": n_resume_events,
                "out_of_sample": oos_metrics,
                "stop_resume_validation": transition_metrics,
            },
            f,
            indent=2,
        )

    campaign_fe_path = output_dir / "campaign_fixed_effects.json"
    with open(campaign_fe_path, "w") as f:
        json.dump(
            {
                "campaign_fe": fit_result["campaign_fe"],
                "global_fe_fallback": fit_result["global_fe_fallback"],
            },
            f,
            indent=2,
        )

    print("Computing per-campaign evidence (historical spend range, stop/resume coverage) for optimization guardrails...")
    campaign_evidence = compute_campaign_evidence(panel)
    evidence_path = output_dir / "campaign_evidence.json"
    with open(evidence_path, "w") as f:
        json.dump(campaign_evidence, f, indent=2)

    print(f"Saved metrics to {metrics_path}")
    print(f"Saved campaign fixed effects to {campaign_fe_path}")
    print(f"Saved campaign evidence to {evidence_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--use-cache", action="store_true")
    args = parser.parse_args()
    main(args.config, args.use_cache)
