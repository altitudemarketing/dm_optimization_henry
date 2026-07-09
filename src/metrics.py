"""
Config-driven metric registry.

Each entry declares (a) which raw performance columns it needs and (b) how to
compute the metric from them. `config.yaml`'s `model.target_metric` selects
one of these by name. To add a new forecasting target in the future, add an
entry here -- feature engineering, the Snowflake pull, and training all read
from this registry rather than hardcoding any particular metric, so nothing
else needs to change.

Raw column names match what src/data_loader.py returns (lowercased):
impressions, clicks, spend, conversions, conversions_value.
"""

from dataclasses import dataclass
from typing import Callable, List

import pandas as pd


@dataclass
class MetricDefinition:
    name: str
    raw_columns: List[str]
    compute: Callable[[pd.DataFrame], pd.Series]
    higher_is_better: bool = True


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, pd.NA)


METRIC_REGISTRY = {
    "conversions": MetricDefinition(
        name="conversions",
        raw_columns=["conversions"],
        compute=lambda df: df["conversions"],
        higher_is_better=True,
    ),
    "cpa": MetricDefinition(
        name="cpa",
        raw_columns=["spend", "conversions"],
        compute=lambda df: _safe_divide(df["spend"], df["conversions"]),
        higher_is_better=False,
    ),
    "roas": MetricDefinition(
        name="roas",
        raw_columns=["conversions_value", "spend"],
        compute=lambda df: _safe_divide(df["conversions_value"], df["spend"]),
        higher_is_better=True,
    ),
    "conversion_rate": MetricDefinition(
        name="conversion_rate",
        raw_columns=["conversions", "clicks"],
        compute=lambda df: _safe_divide(df["conversions"], df["clicks"]),
        higher_is_better=True,
    ),
    "ctr": MetricDefinition(
        name="ctr",
        raw_columns=["clicks", "impressions"],
        compute=lambda df: _safe_divide(df["clicks"], df["impressions"]),
        higher_is_better=True,
    ),
}


def get_metric(name: str) -> MetricDefinition:
    if name not in METRIC_REGISTRY:
        raise ValueError(
            f"Unknown target_metric '{name}'. Available: {sorted(METRIC_REGISTRY.keys())}"
        )
    return METRIC_REGISTRY[name]
