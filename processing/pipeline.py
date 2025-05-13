import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, TensorDataset, DataLoader
from datetime import datetime
import os
import argparse
import json
import pytz

from .utils import generate_data_report
from .transforms import StandardScaleNorm, MinMaxNorm, TransformSequence, DataTransform
from cfgs.cfg import Config


def formPairs(
        x_tensor: torch.Tensor,
        y_tensor: torch.Tensor,
        x_start_hour: int = 9,
        x_y_gap: int = 15,
        x_window: int = 168,
        y_window: int = 24,
        step_size: int = 24,
        target_vars=1):
    """
    Given a time series tensor, forms sliding windows for inputs (X) and targets (Y).
    Assumes that x_tensor and y_tensor have the same length along dim=0.
    """
    assert x_tensor.shape[0] == y_tensor.shape[0]
    N = x_tensor.shape[0]
    x_start = x_start_hour - 1
    X, Y = [], []

    while (x_start + x_window + x_y_gap + y_window) < N:
        x = x_tensor[x_start: x_start + x_window, :]
        y = y_tensor[x_start + x_window + x_y_gap:
                     x_start + x_window + x_y_gap + y_window, 0]
        X.append(x)
        Y.append(y)
        x_start += step_size

    X = torch.stack(X)  # Shape: [num_samples, x_window, num_features]
    Y = torch.stack(Y)  # Shape: [num_samples, y_window]
    return X, Y


def formPairsAR(
        x_tensor: torch.Tensor,
        x_start_hour: int = 9,
        x_y_gap: int = 15,
        x_window: int = 168,
        y_window: int = 24,
        step_size: int = 24):
    """
    For autoregressive training of DeepAR.
    Returns:
      target_seq: Tensor of shape [num_samples, x_window+y_window] (first column of the combined sequence)
      covariate_seq: Tensor of shape [num_samples, x_window+y_window, num_features-1]
                      (all features except the target column)
      mask_seq: Boolean tensor of shape [num_samples, x_window+y_window] with True for observed (conditioning) and
                False for forecast period.
    """
    N = x_tensor.shape[0]
    num_features = x_tensor.shape[1]
    x_start = x_start_hour - 1
    targets, covariates, masks = [], [], []
    while (x_start + x_window + x_y_gap + y_window) < N:
        # shape (x_window, num_features)
        x_obs = x_tensor[x_start: x_start + x_window, :]
        x_fore = x_tensor[x_start + x_window + x_y_gap: x_start +
                          # shape (y_window, num_features)
                          x_window + x_y_gap + y_window, :]
        # shape (x_window+y_window, num_features)
        combined = torch.cat([x_obs, x_fore], dim=0)
        targets.append(combined[:, 0])
        covariates.append(combined[:, 1:])
        mask = torch.cat([torch.ones(x_window, dtype=torch.bool),
                          torch.zeros(y_window, dtype=torch.bool)], dim=0)
        masks.append(mask)
        x_start += step_size

    # Shape: [num_samples, x_window+y_window]
    target_seq = torch.stack(targets)
    # Shape: [num_samples, x_window+y_window, num_features-1]
    covariate_seq = torch.stack(covariates)
    # Shape: [num_samples, x_window+y_window]
    mask_seq = torch.stack(masks)
    return target_seq, covariate_seq, mask_seq


def convert_date_for_comparison(date_obj, is_tz_aware):
    if is_tz_aware and date_obj.tzinfo is None:
        # If data has timezone but comparison date doesn't, add UTC timezone
        return pytz.UTC.localize(date_obj)
    elif not is_tz_aware and date_obj.tzinfo is not None:
        # If data has no timezone but comparison date does, remove timezone
        return date_obj.replace(tzinfo=None)
    return date_obj


def benchmark_preprocess(
        csv_path="data/ercot_data_cleaned.csv",
        train_start_end=(datetime(2023, 2, 10), datetime(2024, 7, 1)),
        val_start_end=(datetime(2024, 7, 1), datetime(2024, 9, 1)),
        test_start_end=(datetime(2024, 9, 1), datetime(2025, 1, 6)),
        spatial=True,
        x_start_hour: int = 9,
        x_y_gap: int = 15,
        x_window: int = 168,
        y_window: int = 24,
        step_size: int = 24,
        batch_size: int = 64,
        num_workers: int = 1,
        included_feats=None,
        num_transform_cols=2,
        train_transform: DataTransform = None,
        ar_model: bool = False,
        output_dir=None):
    """
    Loads the cleaned CSV data and performs time-series splitting, transformation,
    windowing (pair formation), and creation of DataLoaders. The datasets and loaders
    are saved with a suffix indicating if the data is spatial or non-spatial.

    If 'ar_model' is True, the function creates datasets and loaders suitable for the DeepAR model,
    which returns for each sample a tuple of (target, covariates, mask).

    Returns the list of fitted transform objects.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if train_transform is None:
        train_transform = StandardScaleNorm(device=device)

    raw_df = pd.read_csv(csv_path)

    date_column = 'time'
    assert (date_column in raw_df.columns)
    # Ensure proper datetime conversion with explicit format and error handling
    try:
        date_series = pd.to_datetime(
            raw_df[date_column], utc=True, errors='coerce')
        # Check for NaT values that would indicate conversion failures
        if date_series.isna().any():
            print(
                f"Warning: {date_series.isna().sum()} date values couldn't be parsed")
            # Show a sample of problematic values
            problem_indices = raw_df[date_series.isna()].index[:5]
            if len(problem_indices) > 0:
                print(
                    f"Sample problematic values: {raw_df.loc[problem_indices, date_column].tolist()}")
    except Exception as e:
        print(f"Error converting dates: {e}")
        # Fallback approach with more explicit parsing
        date_series = pd.to_datetime(
            raw_df[date_column], format='ISO8601', utc=True, errors='coerce')

    raw_df = raw_df.drop(date_column, axis=1)
    print(raw_df.head())
    # raw_df = raw_df.astype(float)

    print(f"Date series type: {type(date_series)}")
    print(f"Date series dtype: {date_series.dtype}")
    print(f"First few dates: {date_series[:5]}")

    # Handle timezone differences
    is_tz_aware = date_series.dt.tz is not None

    # Convert all date ranges for comparison with consistent timezone handling
    periods = {
        'train': train_start_end,
        'val': val_start_end,
        'test': test_start_end
    }

    dates = {}
    for period, (start, end) in periods.items():
        dates[f'{period}_start'] = convert_date_for_comparison(
            start, is_tz_aware)
        dates[f'{period}_end'] = convert_date_for_comparison(end, is_tz_aware)

    train_start, train_end = dates['train_start'], dates['train_end']
    val_start, val_end = dates['val_start'], dates['val_end']
    test_start, test_end = dates['test_start'], dates['test_end']

    if included_feats is not None:
        raw_df = raw_df[included_feats]

    for col in raw_df.columns:
        if raw_df[col].dtype == 'object':
            raw_df = pd.get_dummies(raw_df, columns=[col])

    for col in raw_df.columns:
        if not pd.api.types.is_numeric_dtype(raw_df[col]):
            raw_df[col] = pd.to_numeric(raw_df[col], errors='coerce')
            raw_df[col] = raw_df[col].fillna(0)

    for col in raw_df.columns:
        if not pd.api.types.is_numeric_dtype(raw_df[col]):
            raise ValueError(
                f"Column {col} could not be converted to numeric type. Current dtype: {raw_df[col].dtype}")

    train_mask = (date_series >= train_start) & (date_series < train_end)
    val_mask = (date_series >= val_start) & (date_series < val_end)
    test_mask = (date_series >= test_start) & (date_series < test_end)

    train_df = raw_df[train_mask]
    val_df = raw_df[val_mask]
    test_df = raw_df[test_mask]

    train_array = train_df.to_numpy(dtype=np.float32)
    val_array = val_df.to_numpy(dtype=np.float32)
    test_array = test_df.to_numpy(dtype=np.float32)

    train_tensor = torch.tensor(train_array, device=device).float()
    val_tensor = torch.tensor(val_array, device=device).float()
    test_tensor = torch.tensor(test_array, device=device).float()

    print("TRANSFORMED TRAIN MAX:", torch.max(
        train_tensor[:, 0].unsqueeze(-1)))

    if train_transform is not None:
        train_transform.change_transform_cols(num_transform_cols)
        train_transform.fit(train_tensor.unsqueeze(0).to(device))

        print(
            f"TRAIN MEAN, STD: {torch.mean(train_transform.mean)}, {torch.mean(train_transform.std)}")

        train_tensor = train_transform.transform(train_tensor)

        # print("TRAIN TENSOR SHAPE", train_tensor.shape)
        print("TRANSFORMED TRAIN MAX:", torch.max(
            train_tensor[:, 0].unsqueeze(-1)))

        val_tensor = train_transform.transform(val_tensor)

        print("TRANSFORMED TRAIN MAX:", torch.max(
            train_tensor[:, 0].unsqueeze(-1)))

        # test_tensor = train_transform.transform(test_tensor)

    suffix = "spatial" if spatial else "non_spatial"

    if output_dir is None:
        output_dir = f"data/{suffix}"

    print(f"Saving data to directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    torch.save(train_tensor, os.path.join(
        output_dir, f"train_tensor_{suffix}.pt"))

    # Save train_tensor to its own file
    # output_dir = f"data/{suffix}"
    os.makedirs(output_dir, exist_ok=True)
    torch.save(train_tensor, os.path.join(
        output_dir, f"train_tensor_{suffix}.pt"))

    if ar_model:
        suffix = f"{suffix}_AR"

        X_train_target, X_train_cov, X_train_mask = formPairsAR(
            train_tensor, x_start_hour, x_y_gap, x_window, y_window, step_size)
        X_val_target, X_val_cov, X_val_mask = formPairsAR(
            val_tensor, x_start_hour, x_y_gap, x_window, y_window, step_size)
        X_test_target, X_test_cov, X_test_mask = formPairsAR(
            test_tensor, x_start_hour, x_y_gap, x_window, y_window, step_size)

        train_dataset = TensorDataset(
            X_train_target, X_train_cov, X_train_mask)
        val_dataset = TensorDataset(X_val_target, X_val_cov, X_val_mask)
        test_dataset = TensorDataset(X_test_target, X_test_cov, X_test_mask)
    else:
        X_train, Y_train = formPairs(
            train_tensor, train_tensor, x_start_hour, x_y_gap, x_window, y_window, step_size)
        X_val, Y_val = formPairs(
            val_tensor, val_tensor, x_start_hour, x_y_gap, x_window, y_window, step_size)
        X_test, Y_test = formPairs(
            test_tensor, test_tensor, x_start_hour, x_y_gap, x_window, y_window, step_size)
        train_dataset = TensorDataset(X_train, Y_train)
        val_dataset = TensorDataset(X_val, Y_val)
        test_dataset = TensorDataset(X_test, Y_test)

    pin_memory = True if device == 'cuda' else False
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)

    print(f"{suffix} Train loader shape: {next(iter(train_loader))[0].shape}")
    print(f"{suffix} Val loader shape: {next(iter(val_loader))[0].shape}")
    print(f"{suffix} Test loader shape: {next(iter(test_loader))[0].shape}")

    train_dataset_path = os.path.join(output_dir, f"train_dataset_{suffix}.pt")
    print(f"Saving train dataset to: {train_dataset_path}")
    torch.save(train_dataset, train_dataset_path)

    val_dataset_path = os.path.join(output_dir, f"val_dataset_{suffix}.pt")
    print(f"Saving validation dataset to: {val_dataset_path}")
    torch.save(val_dataset, val_dataset_path)

    test_dataset_path = os.path.join(output_dir, f"test_dataset_{suffix}.pt")
    print(f"Saving test dataset to: {test_dataset_path}")
    torch.save(test_dataset, test_dataset_path)

    train_loader_path = os.path.join(output_dir, f"train_loader_{suffix}.pt")
    print(f"Saving train loader to: {train_loader_path}")
    torch.save(train_loader, train_loader_path)

    val_loader_path = os.path.join(output_dir, f"val_loader_{suffix}.pt")
    print(f"Saving validation loader to: {val_loader_path}")
    torch.save(val_loader, val_loader_path)

    test_loader_path = os.path.join(output_dir, f"test_loader_{suffix}.pt")
    print(f"Saving test loader to: {test_loader_path}")
    torch.save(test_loader, test_loader_path)

    if train_transform is not None:
        transform_path = os.path.join(output_dir, f"transform_{suffix}.pt")
        print(f"Saving transform to: {transform_path}")
        torch.save(train_transform, transform_path)
    else:
        print("Train transform not given.")

    return train_transform


def preprocess_on_cfg(config: Config, usable_transforms: dict):
    """
    Wrapper function that runs preprocessing on a Config dataclass.

    Args:
        config: Config dataclass containing preprocessing parameters
        transform: Optional transform to apply to the data
        output_dir: Directory to save processed data
        suffix: Optional suffix to add to output filenames

    Returns:
        The fitted transform used for training data
    """
    train_start_end = (pd.to_datetime(config.train_start),
                       pd.to_datetime(config.train_end))
    val_start_end = (pd.to_datetime(config.val_start),
                     pd.to_datetime(config.val_end))
    test_start_end = (pd.to_datetime(config.test_start),
                      pd.to_datetime(config.test_end))

    transform = None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if config.transforms:
        if len(config.transforms) == 1:
            transform_name = config.transforms[0]
            assert (transform_name in usable_transforms.keys()
                    ), f"Transform {transform_name} not found"
            transform = usable_transforms[transform_name](device=device)
        else:
            transform_names = config.transforms
            assert (set(transform_names).issubset(
                set(usable_transforms.keys()))), "One or more transforms not found"
            transform = TransformSequence(
                transforms=[usable_transforms[k](device=device) for k in transform_names])

    return benchmark_preprocess(
        csv_path=config.csv_path,
        train_start_end=train_start_end,
        val_start_end=val_start_end,
        test_start_end=test_start_end,
        spatial=config.spatial,
        x_start_hour=config.x_start_hour,
        x_y_gap=config.x_y_gap,
        x_window=config.x_window,
        y_window=config.y_window,
        step_size=config.step_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        included_feats=config.included_feats,
        num_transform_cols=config.num_transform_cols,
        train_transform=transform,
        ar_model=config.ar_model,
        output_dir=config.output_dir
    )
