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
from src.sarimax_model import run_sarimax, train_sarimax_gbm_hybrid

CATEGORICAL_FEATURES = ["client_id", "channel_type"]
ID_AND_META_COLUMNS = {
    "entity_id", "client_id", "client_name", "campaign_id", "campaign_name",
    "ad_group_id", "ad_group_name", "ad_group_status", "stat_date",
}

# Fixed (non-tuned) + sensible default values for the tunable hyperparameters.
# scripts/tune_hyperparameters.py imports these as the base and overrides the
# tunable keys with Optuna's suggestions, so there's one source of truth for
# which keys are fixed (objective, metric, tree_method) vs. tunable.
DEFAULT_LGB_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "verbosity": -1,
}
DEFAULT_XGB_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "mae",
    "tree_method": "hist",
    "eta": 0.05,
    "max_depth": 6,
}
# Classifier stage of the hurdle/two-stage model (see train_hurdle_model
# below). Not currently tuned by Optuna -- these are reasonable fixed
# defaults for a binary "does this entity convert at all" classifier.
DEFAULT_LGB_CLF_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "verbosity": -1,
}
BEST_PARAMS_PATH = Path("config/best_params.json")


def load_tuned_params(model_name: str, default_params: dict) -> dict:
    """
    Uses hyperparameters from config/best_params.json (produced by
    scripts/tune_hyperparameters.py) if present, falling back to the
    hardcoded defaults otherwise. Tuning is a separate, occasional step (see
    that script's docstring) -- this file won't always exist or be current,
    and that's expected, not an error.
    """
    if not BEST_PARAMS_PATH.exists():
        print(f"No {BEST_PARAMS_PATH} found -- using default {model_name} hyperparameters.")
        return dict(default_params)

    with open(BEST_PARAMS_PATH) as f:
        tuned = json.load(f)

    model_tuned = tuned.get(model_name)
    if not model_tuned:
        print(f"{BEST_PARAMS_PATH} has no '{model_name}' entry -- using defaults.")
        return dict(default_params)

    merged = {**default_params, **model_tuned["params"]}
    print(
        f"Using tuned {model_name} hyperparameters from {BEST_PARAMS_PATH} "
        f"(val MAE {model_tuned['val_mae']:.4f} at tuning time): {model_tuned['params']}"
    )
    return merged


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


def evaluate_segmented(y_true, y_pred, label: str) -> dict:
    """
    Breaks metrics out for the zero-actual vs. nonzero-actual subsets
    separately. A single pooled MAE/R2 can look strong largely by getting the
    (often majority) zero-conversion cases right, which says little about
    accuracy on the ad groups that actually matter for recommendations --
    the ones with real conversion volume.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    zero_mask = y_true == 0
    nonzero_mask = ~zero_mask

    print(f"-- {label}, segmented by actual value --")
    zero_metrics = (
        evaluate(y_true[zero_mask], y_pred[zero_mask], f"{label} | zero-actual (n={int(zero_mask.sum())})")
        if zero_mask.any() else None
    )
    nonzero_metrics = (
        evaluate(y_true[nonzero_mask], y_pred[nonzero_mask], f"{label} | nonzero-actual (n={int(nonzero_mask.sum())})")
        if nonzero_mask.any() else None
    )
    return {"zero_actual": zero_metrics, "nonzero_actual": nonzero_metrics}


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


def train_xgboost(train_df, val_df, test_df, feature_cols, label_col, categorical_features, params):
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


def train_hurdle_model(train_df, val_df, test_df, feature_cols, label_col, categorical_features, reg_params, clf_params=None):
    """
    Two-stage ("hurdle") model for zero-inflated targets like conversions
    (75%+ zero in our real data). A single regressor has to use the same
    trees to learn both "predict ~0" and "predict the right nonzero
    magnitude" -- the majority-zero rows dominate what the loss function
    optimizes for, which is exactly the failure mode describe_label_distribution()
    warns about.

    This splits the job into two specialized models:
      1. A classifier predicting P(label > 0) -- "will this entity convert at all".
      2. A regressor trained ONLY on rows where the label is actually nonzero --
         "given it converts, how much" -- so it isn't pulled toward zero by
         the majority-zero rows.
    Final prediction blends them: P(nonzero) x E[value | nonzero]. This is
    the standard hurdle-model formulation for count/intermittent-demand data.
    """
    clf_params = clf_params or DEFAULT_LGB_CLF_PARAMS
    cat_features_present = [c for c in categorical_features if c in feature_cols]

    y_train_bin = (train_df[label_col] > 0).astype(int)
    y_val_bin = (val_df[label_col] > 0).astype(int)

    clf_train_set = lgb.Dataset(
        train_df[feature_cols], label=y_train_bin, categorical_feature=cat_features_present,
    )
    clf_val_set = lgb.Dataset(
        val_df[feature_cols], label=y_val_bin,
        reference=clf_train_set, categorical_feature=cat_features_present,
    )
    classifier = lgb.train(
        clf_params,
        clf_train_set,
        valid_sets=[clf_val_set],
        num_boost_round=500,
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)],
    )

    train_nonzero = train_df[train_df[label_col] > 0]
    val_nonzero = val_df[val_df[label_col] > 0]
    if len(train_nonzero) == 0 or len(val_nonzero) == 0:
        raise ValueError(
            "Hurdle model needs nonzero-label rows in both the train and "
            "validation splits -- check describe_label_distribution() output; "
            "zero_rate may be too close to 100% for this split size."
        )

    reg_train_set = lgb.Dataset(
        train_nonzero[feature_cols], label=train_nonzero[label_col], categorical_feature=cat_features_present,
    )
    reg_val_set = lgb.Dataset(
        val_nonzero[feature_cols], label=val_nonzero[label_col],
        reference=reg_train_set, categorical_feature=cat_features_present,
    )
    regressor = lgb.train(
        reg_params,
        reg_train_set,
        valid_sets=[reg_val_set],
        num_boost_round=500,
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)],
    )

    def predict(df):
        p_nonzero = classifier.predict(df[feature_cols])
        # The magnitude model never sees zero-labeled rows during training and
        # can occasionally extrapolate slightly negative -- clip before blending.
        magnitude = np.clip(regressor.predict(df[feature_cols]), 0, None)
        return p_nonzero * magnitude

    return {
        "classifier": classifier,
        "regressor": regressor,
        "val_pred": predict(val_df),
        "test_pred": predict(test_df),
    }


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

    lgb_params = load_tuned_params("lightgbm", DEFAULT_LGB_PARAMS)

    model = lgb.train(
        lgb_params,
        train_set,
        valid_sets=[val_set],
        num_boost_round=500,
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)],
    )

    test_preds = model.predict(test_df[feature_cols])
    model_metrics = evaluate(test_df[label_col].values, test_preds, "lightgbm")
    model_segmented = evaluate_segmented(test_df[label_col].values, test_preds, "lightgbm")

    xgb_params = load_tuned_params("xgboost", DEFAULT_XGB_PARAMS)
    xgb_booster, xgb_preds = train_xgboost(
        train_df, val_df, test_df, feature_cols, label_col, CATEGORICAL_FEATURES, xgb_params
    )
    xgb_metrics = evaluate(test_df[label_col].values, xgb_preds, "xgboost")
    xgb_segmented = evaluate_segmented(test_df[label_col].values, xgb_preds, "xgboost")

    # Hurdle/two-stage model: reuses the (possibly tuned) LightGBM regression
    # hyperparameters for its magnitude stage, since that's already a LightGBM
    # regressor solving essentially the same conditional-magnitude problem --
    # just fit on a nonzero-only subset instead of the full data.
    hurdle_result = train_hurdle_model(
        train_df, val_df, test_df, feature_cols, label_col, CATEGORICAL_FEATURES, reg_params=lgb_params,
    )
    hurdle_preds = hurdle_result["test_pred"]
    hurdle_metrics = evaluate(test_df[label_col].values, hurdle_preds, "hurdle")
    hurdle_segmented = evaluate_segmented(test_df[label_col].values, hurdle_preds, "hurdle")

    # SARIMAX benchmark + SARIMAX/LightGBM hybrid -- see src/sarimax_model.py
    # for the full design and its simplifications. These need each entity's
    # raw historical conversions series (not just engineered features), so
    # the pre-split `df` is passed in alongside the train/val/test splits.
    sarimax_result = run_sarimax(df, train_df, val_df, test_df, horizon)
    sarimax_metrics = evaluate(test_df[label_col].values, sarimax_result["test_pred"], "sarimax")
    sarimax_segmented = evaluate_segmented(test_df[label_col].values, sarimax_result["test_pred"], "sarimax")

    hybrid_result = train_sarimax_gbm_hybrid(
        train_df, val_df, test_df, feature_cols, label_col, CATEGORICAL_FEATURES, sarimax_result, lgb_params,
    )
    hybrid_preds = hybrid_result["test_pred"]
    hybrid_metrics = evaluate(test_df[label_col].values, hybrid_preds, "sarimax_gbm_hybrid")
    hybrid_segmented = evaluate_segmented(test_df[label_col].values, hybrid_preds, "sarimax_gbm_hybrid")

    def _improvement(challenger_mae):
        if not baseline_metrics["mae"]:
            return float("nan")
        return (baseline_metrics["mae"] - challenger_mae) / baseline_metrics["mae"] * 100

    print(f"LightGBM MAE improvement over baseline: {_improvement(model_metrics['mae']):.1f}%")
    print(f"XGBoost MAE improvement over baseline: {_improvement(xgb_metrics['mae']):.1f}%")
    print(f"Hurdle MAE improvement over baseline: {_improvement(hurdle_metrics['mae']):.1f}%")
    print(f"SARIMAX MAE improvement over baseline: {_improvement(sarimax_metrics['mae']):.1f}%")
    print(f"SARIMAX+GBM hybrid MAE improvement over baseline: {_improvement(hybrid_metrics['mae']):.1f}%")

    candidates = {
        "lightgbm": model_metrics["mae"],
        "xgboost": xgb_metrics["mae"],
        "hurdle": hurdle_metrics["mae"],
        "sarimax": sarimax_metrics["mae"],
        "sarimax_gbm_hybrid": hybrid_metrics["mae"],
    }
    winner = min(candidates, key=candidates.get)
    print(f"Lower test MAE: {winner}")

    output_dir = Path(config["paths"]["model_output"])
    output_dir.mkdir(parents=True, exist_ok=True)

    lgb_path = output_dir / f"{target_metric}_h{horizon}d_lightgbm.txt"
    model.save_model(str(lgb_path))

    xgb_path = output_dir / f"{target_metric}_h{horizon}d_xgboost.json"
    xgb_booster.save_model(str(xgb_path))

    hurdle_clf_path = output_dir / f"{target_metric}_h{horizon}d_hurdle_classifier.txt"
    hurdle_result["classifier"].save_model(str(hurdle_clf_path))
    hurdle_reg_path = output_dir / f"{target_metric}_h{horizon}d_hurdle_regressor.txt"
    hurdle_result["regressor"].save_model(str(hurdle_reg_path))

    # Note: the per-entity SARIMAX models themselves aren't saved -- there can
    # be hundreds of them (one per active ad group), they're cheap to refit,
    # and every other model in this pipeline is already retrained from scratch
    # each run rather than persisted, so this stays consistent with that.
    # Only the hybrid's residual GBM (a single model) is saved.
    hybrid_path = output_dir / f"{target_metric}_h{horizon}d_sarimax_gbm_hybrid.txt"
    hybrid_result["residual_model"].save_model(str(hybrid_path))

    metrics_path = output_dir / f"{target_metric}_h{horizon}d_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "target_metric": target_metric,
                "horizon_days": horizon,
                "baseline": baseline_metrics,
                "lightgbm": model_metrics,
                "lightgbm_segmented": model_segmented,
                "xgboost": xgb_metrics,
                "xgboost_segmented": xgb_segmented,
                "hurdle": hurdle_metrics,
                "hurdle_segmented": hurdle_segmented,
                "sarimax": sarimax_metrics,
                "sarimax_segmented": sarimax_segmented,
                "sarimax_fit_diagnostics": {"n_fit": sarimax_result["n_fit"], "n_fallback": sarimax_result["n_fallback"]},
                "sarimax_gbm_hybrid": hybrid_metrics,
                "sarimax_gbm_hybrid_segmented": hybrid_segmented,
                "lower_test_mae": winner,
            },
            f,
            indent=2,
        )

    print(f"Saved LightGBM model to {lgb_path}")
    print(f"Saved XGBoost model to {xgb_path}")
    print(f"Saved hurdle classifier to {hurdle_clf_path}")
    print(f"Saved hurdle regressor to {hurdle_reg_path}")
    print(f"Saved SARIMAX+GBM hybrid residual model to {hybrid_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    main(args.config)
