"""
Trains a naive baseline and a LightGBM model for the configured target
metric, using a time-aware split, and reports MAE/MAPE for both so you can
see whether the model is actually earning its keep over a simple heuristic.

Usage:
    python -m src.train [--config path/to/config.yaml]
"""

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score

from src.config_loader import load_config
from src.dataset import time_aware_split

CATEGORICAL_FEATURES = ["client_id", "channel_type"]
ID_AND_META_COLUMNS = {
    "entity_id", "client_id", "client_name", "campaign_id", "campaign_name",
    "ad_group_id", "ad_group_name", "ad_group_status", "stat_date",
}


def _mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def evaluate(y_true, y_pred, label: str) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    # Computed manually rather than via sklearn's mean_squared_error(squared=False)
    # since that param was deprecated/removed across recent sklearn versions --
    # this avoids pinning to a narrow version range just for RMSE.
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    r2 = r2_score(y_true, y_pred)
    mape = _mape(y_true, y_pred)

    print(f"[{label}] MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}  MAPE={mape:.4f}")
    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}


def naive_baseline(train_df: pd.DataFrame, eval_df: pd.DataFrame, horizon_days: int):
    """Predicts the future value as horizon_days x each entity's trailing daily average."""
    last_daily_avg = (
        train_df.sort_values("stat_date")
        .groupby("entity_id")["conversions_rolling_7d"]
        .last()
        / 7.0
    )
    fallback = last_daily_avg.mean()
    preds = eval_df["entity_id"].map(last_daily_avg).fillna(fallback) * horizon_days
    return preds.values


def get_feature_columns(df: pd.DataFrame, label_col: str):
    exclude = set(ID_AND_META_COLUMNS)
    exclude.add(label_col)
    exclude |= {c for c in df.columns if c.startswith("label_")}
    return [c for c in df.columns if c not in exclude]


def describe_label_distribution(df: pd.DataFrame, label_col: str) -> None:
    """
    MAE alone can be misleading on intermittent/low-count data like
    conversions -- a model that predicts near-zero for everything can look
    good on MAE if most entities have zero or near-zero conversions. This
    print makes that visible instead of leaving it to guesswork.
    """
    values = df[label_col]
    zero_rate = float((values == 0).mean())
    print(
        f"Label distribution for '{label_col}': zero_rate={zero_rate:.1%}, "
        f"mean={values.mean():.3f}, median={values.median():.3f}, "
        f"p90={values.quantile(0.9):.3f}, p99={values.quantile(0.99):.3f}, "
        f"max={values.max():.3f}"
    )
    if zero_rate > 0.5:
        print(
            "Over half the labels are exactly zero -- interpret MAE/MAPE "
            "cautiously here, and consider looking at performance specifically "
            "on the nonzero subset (e.g. higher-volume ad groups) separately."
        )


def train_xgboost(train_df, val_df, test_df, feature_cols, label_col, categorical_features):
    """
    Trained alongside LightGBM (not instead of) so accuracy can be compared
    directly on the same split rather than assumed. Uses XGBoost's native
    categorical support (tree_method='hist', enable_categorical=True) to keep
    the comparison apples-to-apples with LightGBM's categorical handling,
    rather than one-hot encoding for just this model.
    """
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    for col in categorical_features:
        if col in feature_cols:
            train_df[col] = train_df[col].astype("category")
            val_df[col] = val_df[col].astype("category")
            test_df[col] = test_df[col].astype("category")

    dtrain = xgb.DMatrix(train_df[feature_cols], label=train_df[label_col], enable_categorical=True)
    dval = xgb.DMatrix(val_df[feature_cols], label=val_df[label_col], enable_categorical=True)
    dtest = xgb.DMatrix(test_df[feature_cols], enable_categorical=True)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "eta": 0.05,
        "max_depth": 6,
        "tree_method": "hist",
    }
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dval, "valid")],
        early_stopping_rounds=30,
        verbose_eval=50,
    )
    preds = booster.predict(dtest, iteration_range=(0, booster.best_iteration + 1))
    return booster, preds


def main(config_path=None):
    config = load_config(config_path)
    target_metric = config["model"]["target_metric"]
    horizon = config["model"]["forecast_horizon_days"]
    label_col = f"label_{target_metric}"

    processed_path = Path(config["paths"]["processed_dataset"])
    if not processed_path.exists():
        raise FileNotFoundError(
            f"{processed_path} not found. Run `python -m scripts.build_dataset` first."
        )

    df = pd.read_parquet(processed_path)
    if label_col not in df.columns:
        raise ValueError(
            f"Column '{label_col}' not found in processed dataset. "
            f"Did config.yaml's target_metric change since the dataset was last built? "
            f"Re-run `python -m scripts.build_dataset`."
        )
    df = df.dropna(subset=[label_col]).copy()
    describe_label_distribution(df, label_col)

    for cat_col in CATEGORICAL_FEATURES:
        if cat_col in df.columns:
            df[cat_col] = df[cat_col].astype("category")

    feature_cols = get_feature_columns(df, label_col)

    train_df, val_df, test_df = time_aware_split(
        df,
        date_col="stat_date",
        test_size_days=config["training"]["test_size_days"],
        validation_size_days=config["training"]["validation_size_days"],
    )
    print(f"Train: {len(train_df)} rows | Val: {len(val_df)} rows | Test: {len(test_df)} rows")

    if min(len(train_df), len(val_df), len(test_df)) == 0:
        raise ValueError(
            "Not enough date range in the dataset for the configured split sizes. "
            "Reduce training.test_size_days / validation_size_days, or pull more history."
        )

    baseline_preds = naive_baseline(train_df, test_df, horizon)
    baseline_metrics = evaluate(test_df[label_col].values, baseline_preds, "baseline")

    cat_features_present = [c for c in CATEGORICAL_FEATURES if c in feature_cols]
    train_set = lgb.Dataset(
        train_df[feature_cols], label=train_df[label_col],
        categorical_feature=cat_features_present,
    )
    val_set = lgb.Dataset(
        val_df[feature_cols], label=val_df[label_col],
        reference=train_set, categorical_feature=cat_features_present,
    )

    params = {
        "objective": "regression",
        "metric": "mae",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "verbosity": -1,
    }

    model = lgb.train(
        params,
        train_set,
        valid_sets=[val_set],
        num_boost_round=500,
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)],
    )

    test_preds = model.predict(test_df[feature_cols])
    model_metrics = evaluate(test_df[label_col].values, test_preds, "lightgbm")

    xgb_booster, xgb_preds = train_xgboost(
        train_df, val_df, test_df, feature_cols, label_col, CATEGORICAL_FEATURES
    )
    xgb_metrics = evaluate(test_df[label_col].values, xgb_preds, "xgboost")

    def _improvement(challenger_mae):
        if not baseline_metrics["mae"]:
            return float("nan")
        return (baseline_metrics["mae"] - challenger_mae) / baseline_metrics["mae"] * 100

    print(f"LightGBM MAE improvement over baseline: {_improvement(model_metrics['mae']):.1f}%")
    print(f"XGBoost MAE improvement over baseline: {_improvement(xgb_metrics['mae']):.1f}%")
    winner = "lightgbm" if model_metrics["mae"] <= xgb_metrics["mae"] else "xgboost"
    print(f"Lower test MAE: {winner}")

    output_dir = Path(config["paths"]["model_output"])
    output_dir.mkdir(parents=True, exist_ok=True)

    lgb_path = output_dir / f"{target_metric}_h{horizon}d_lightgbm.txt"
    model.save_model(str(lgb_path))

    xgb_path = output_dir / f"{target_metric}_h{horizon}d_xgboost.json"
    xgb_booster.save_model(str(xgb_path))

    metrics_path = output_dir / f"{target_metric}_h{horizon}d_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "target_metric": target_metric,
                "horizon_days": horizon,
                "baseline": baseline_metrics,
                "lightgbm": model_metrics,
                "xgboost": xgb_metrics,
                "lower_test_mae": winner,
            },
            f,
            indent=2,
        )

    print(f"Saved LightGBM model to {lgb_path}")
    print(f"Saved XGBoost model to {xgb_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    main(args.config)
