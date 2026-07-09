"""
Loads the ML pipeline config (config/config.yaml).

Snowflake credentials are intentionally NOT part of this file -- they're read
directly from environment variables by src/snowflake_client.py (see
.env.example), matching the auth pattern already used elsewhere in this
Snowflake project (speed_snowflake/legacy/validate_bot.py). Keeping them
separate means this config can be loaded (e.g. for synthetic-data testing)
without needing real credentials present.
"""

from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


def load_config(config_path: str = None) -> dict:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    _validate(config)
    return config


def _validate(config: dict) -> None:
    required_sections = {"model", "training", "paths"}
    missing = required_sections - config.keys()
    if missing:
        raise ValueError(f"config.yaml is missing required section(s): {missing}")

    required_model_keys = {
        "target_metric", "grain", "forecast_horizon_days",
        "lookback_window_days", "rolling_windows",
    }
    missing_model = required_model_keys - config["model"].keys()
    if missing_model:
        raise ValueError(f"config.yaml 'model' section is missing key(s): {missing_model}")
