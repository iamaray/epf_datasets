import pandas as pd
import numpy as np


def strategic_fill(df):
    """
    Fills missing values in the DataFrame based on the following policy:
    - For each missing hour, check the same hour from the previous day.
      If it exists, use it to fill the missing value.
    - If the previous day's same hour is also missing, check the next day's same hour.
      If it exists, use it to fill the missing value.
    - If both previous and next day are missing, look further back and forward
      up to 7 days in both directions.
    - Finally, use forward and backward fill for any remaining missing values.
    """
    df_filled = df.copy()

    for time in df_filled.index[df_filled.isnull().any(axis=1)]:
        missing_cols = df_filled.columns[df_filled.loc[time].isnull()]

        previous_day = time - pd.Timedelta(days=1)
        if previous_day in df_filled.index:
            for col in missing_cols:
                if not pd.isnull(df_filled.loc[previous_day, col]):
                    df_filled.loc[time, col] = df_filled.loc[previous_day, col]

        still_missing = df_filled.columns[df_filled.loc[time].isnull()]

        if len(still_missing) > 0:
            next_day = time + pd.Timedelta(days=1)
            if next_day in df_filled.index:
                for col in still_missing:
                    if not pd.isnull(df_filled.loc[next_day, col]):
                        df_filled.loc[time, col] = df_filled.loc[next_day, col]

    for time in df_filled.index[df_filled.isnull().any(axis=1)]:
        missing_cols = df_filled.columns[df_filled.loc[time].isnull()]

        for days in range(2, 8):  # Try 2 to 7 days back
            if len(missing_cols) == 0:
                break

            prev_day = time - pd.Timedelta(days=days)
            if prev_day in df_filled.index:
                for col in missing_cols:
                    if not pd.isnull(df_filled.loc[prev_day, col]):
                        df_filled.loc[time, col] = df_filled.loc[prev_day, col]

            missing_cols = df_filled.columns[df_filled.loc[time].isnull()]

            if len(missing_cols) == 0:
                break

            next_day = time + pd.Timedelta(days=days)
            if next_day in df_filled.index:
                for col in missing_cols:
                    if not pd.isnull(df_filled.loc[next_day, col]):
                        df_filled.loc[time, col] = df_filled.loc[next_day, col]

            missing_cols = df_filled.columns[df_filled.loc[time].isnull()]

    df_filled = df_filled.ffill().bfill()

    return df_filled


def shift_forecast_columns(df, forecast_cols, shift_hours=-24):
    """
    Applies df[col].shift(-24) to each column in forecast_cols.
    In other words, for any given index t, the forecast becomes
    the forecast that was originally at t+24 in the raw data.
    """
    df_shifted = df.copy()
    for col in forecast_cols:
        if col in df_shifted.columns:
            df_shifted[col] = df_shifted[col].shift(shift_hours)
    df_shifted.dropna(subset=forecast_cols, inplace=True)
    return df_shifted
