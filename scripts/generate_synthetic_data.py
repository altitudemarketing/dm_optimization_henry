"""
Generates synthetic ad-group daily performance data matching the schema
src.data_loader.pull_raw_data returns, so the feature/label/training pipeline
can be smoke-tested locally without a live Snowflake connection.

Usage:
    python -m scripts.generate_synthetic_data [--config path/to/config.yaml]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config_loader import load_config

RNG = np.random.default_rng(42)
CHANNELS = ["SEARCH", "DISPLAY", "SHOPPING", "VIDEO"]


def generate(num_clients=5, campaigns_per_client=3, ad_groups_per_campaign=4, num_days=120) -> pd.DataFrame:
    rows = []
    start_date = pd.Timestamp.today().normalize() - pd.Timedelta(days=num_days)

    for c in range(num_clients):
        client_id = f"CLIENT_{c}"
        client_name = f"Client {c}"
        base_conv_rate = RNG.uniform(0.02, 0.08)

        for cmp in range(campaigns_per_client):
            campaign_id = f"{c}{cmp:02d}0001"
            campaign_name = f"Campaign {c}-{cmp}"
            channel_type = CHANNELS[cmp % len(CHANNELS)]

            for ag in range(ad_groups_per_campaign):
                ad_group_id = f"{c}{cmp:02d}{ag:04d}"
                ad_group_name = f"Ad Group {c}-{cmp}-{ag}"
                base_clicks = RNG.uniform(20, 200)

                for d in range(num_days):
                    date = start_date + pd.Timedelta(days=d)
                    weekday_factor = 0.7 if date.dayofweek >= 5 else 1.0
                    trend_factor = 1.0 + 0.15 * np.sin(d / 20)

                    impressions = max(0, RNG.normal(base_clicks * 12, base_clicks * 2) * weekday_factor)
                    clicks = max(0, RNG.normal(base_clicks, base_clicks * 0.2) * weekday_factor * trend_factor)
                    spend = round(float(clicks * RNG.uniform(1.5, 4.0)), 2)
                    conversions = max(0.0, RNG.normal(clicks * base_conv_rate, clicks * base_conv_rate * 0.3 + 0.01))
                    conversions_value = round(float(conversions * RNG.uniform(50, 200)), 2)

                    rows.append({
                        "client_id": client_id,
                        "client_name": client_name,
                        "campaign_id": campaign_id,
                        "campaign_name": campaign_name,
                        "channel_type": channel_type,
                        "ad_group_id": ad_group_id,
                        "ad_group_name": ad_group_name,
                        "ad_group_status": "ENABLED",
                        "stat_date": date,
                        "impressions": round(float(impressions)),
                        "clicks": round(float(clicks)),
                        "spend": spend,
                        "conversions": round(float(conversions), 2),
                        "conversions_value": conversions_value,
                    })

    return pd.DataFrame(rows)


def main(config_path=None):
    config = load_config(config_path)
    df = generate()
    out_path = Path(config["paths"]["raw_data_cache"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df)} synthetic rows to {out_path}")
    print(f"{df['client_id'].nunique()} clients, {df['campaign_id'].nunique()} campaigns, "
          f"{df['ad_group_id'].nunique()} distinct ad_group_id values")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    main(args.config)
