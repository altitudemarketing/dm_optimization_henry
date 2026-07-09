"""
Bayesian hyperparameter search (via Optuna) for the LightGBM and XGBoost
forecasting models.

This is deliberately a SEPARATE script from src.train, meant to be run
occasionally (manually, or on a slower schedule -- see
.github/workflows/tune.yml) rather than on every retrain. Re-running a
Bayesian search from scratch every 12 hours alongside the regular retrain
would be wasteful; retraining frequently with the best-known parameters,
and re-tuning on a much slower cadence, is the more sensible split.

Usage:
    python -m scripts.tune_hyperparameters [--config path/to/config.yaml] [--n-trials 50]

Requires data/processed_dataset.parquet to already exist (run
`python -m scripts.build_dataset` first). Writes config/best_params.json,
which src.train automatically picks up on its next run -- no other wiring
needed.
"""

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

from src.config_loader import load_config
from src.dataset import time_aware_split
from src.train import (
    CATEGORICAL_FEATURES,
    DEFAULT_LGB_PARAMS,
    DEFAULT_XGB_PARAMS,
    get_feature_columns,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _prepare_data(config: dict):
    target_metric = config["model"]["target_metric"]
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
            f"Re-run `python -m scripts.build_dataset` if target_metric changed."
        )
    df = df.dropna(subset=[label_col]).copy()
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")

    feature_cols = get_feature_columns(df, label_col)
    train_df, val_df, _test_df = time_aware_split(
        df,
        date_col="stat_date",
        test_size_days=config["training"]["test_size_days"],
        validation_size_days=config["training"]["validation_size_days"],
    )
    if min(len(train_df), len(val_df)) == 0:
        raise ValueError(
            "Not enough date range for train/val split -- pull more history or "
            "reduce training.test_size_days / validation_size_days first."
        )
    return train_df, val_df, feature_cols, label_col


def _lgb_objective(trial, train_df, val_df, feature_cols, label_col, cat_features_present):
    params = {
        **DEFAULT_LGB_PARAMS,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 0, 7),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
    }
    train_set = lgb.Dataset(
        train_df[feature_cols], label=train_df[label_col], categorical_feature=cat_features_present
    )
    val_set = lgb.Dataset(
        val_df[feature_cols], label=val_df[label_col],
        reference=train_set, categorical_feature=cat_features_present,
    )
    model = lgb.train(
        params, train_set, valid_sets=[val_set], num_boost_round=500,
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )
    preds = model.predict(val_df[feature_cols])
    return mean_absolute_error(val_df[label_col], preds)


def _xgb_objective(trial, train_df, val_df, feature_cols, label_col, categorical_features):
    train_df = train_df.copy()
    val_df = val_df.copy()
    for col in categorical_features:
        if col in feature_cols:
            train_df[col] = train_df[col].astype("category")
            val_df[col] = val_df[col].astype("category")

    params = {
        **DEFAULT_XGB_PARAMS,
        "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
    }
    dtrain = xgb.DMatrix(train_df[feature_cols], label=train_df[label_col], enable_categorical=True)
    dval = xgb.DMatrix(val_df[feature_cols], label=val_df[label_col], enable_categorical=True)
    booster = xgb.train(
        params, dtrain, num_boost_round=500,
        evals=[(dval, "valid")], early_stopping_rounds=30, verbose_eval=False,
    )
    preds = booster.predict(dval, iteration_range=(0, booster.best_iteration + 1))
    return mean_absolute_error(val_df[label_col], preds)


def main(config_path=None, n_trials=50):
    config = load_config(config_path)
    train_df, val_df, feature_cols, label_col = _prepare_data(config)
    cat_features_present = [c for c in CATEGORICAL_FEATURES if c in feature_cols]

    print(f"Tuning LightGBM ({n_trials} trials)...")
    lgb_study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    lgb_study.optimize(
        lambda trial: _lgb_objective(trial, train_df, val_df, feature_cols, label_col, cat_features_present),
        n_trials=n_trials,
    )
    print(f"Best LightGBM val MAE: {lgb_study.best_value:.4f}")
    print(f"Best LightGBM params: {lgb_study.best_params}")

    print(f"Tuning XGBoost ({n_trials} trials)...")
    xgb_study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    xgb_study.optimize(
        lambda trial: _xgb_objective(trial, train_df, val_df, feature_cols, label_col, CATEGORICAL_FEATURES),
        n_trials=n_trials,
    )
    print(f"Best XGBoost val MAE: {xgb_study.best_value:.4f}")
    print(f"Best XGBoost params: {xgb_study.best_params}")

    output = {
        "target_metric": config["model"]["target_metric"],
        "n_trials": n_trials,
        "lightgbm": {"params": lgb_study.best_params, "val_mae": lgb_study.best_value},
        "xgboost": {"params": xgb_study.best_params, "val_mae": xgb_study.best_value},
    }
    out_path = Path("config/best_params.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved tuned hyperparameters to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--n-trials", type=int, default=50)
    args = parser.parse_args()
    main(args.config, args.n_trials)
