"""
Time-aware train/validation/test split.

Never use a random split for this problem -- validation and test must
represent genuinely future, unseen time periods, or evaluation metrics will
be optimistic and misleading (the model would effectively be tested on
interpolation, not forecasting).
"""

import pandas as pd


def time_aware_split(df: pd.DataFrame, date_col: str, test_size_days: int, validation_size_days: int):
    df = df.sort_values(date_col)
    max_date = df[date_col].max()

    test_cutoff = max_date - pd.Timedelta(days=test_size_days)
    val_cutoff = test_cutoff - pd.Timedelta(days=validation_size_days)

    train = df[df[date_col] <= val_cutoff]
    val = df[(df[date_col] > val_cutoff) & (df[date_col] <= test_cutoff)]
    test = df[df[date_col] > test_cutoff]

    return train, val, test
