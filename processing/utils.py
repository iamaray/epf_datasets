import os
import torch
from datetime import datetime
from typing import Dict, List, Optional, Union, Any


def generate_data_report(
        output_dir,
        suffix,
        config,
        train_tensor,
        val_tensor,
        test_tensor,
        X_train,
        Y_train,
        X_val,
        Y_val,
        X_test,
        Y_test,
        train_transforms,
        included_feats=None,
        ar_model=False):
    """Generate a report summarizing the data processing and save it to a text file."""
    report_path = os.path.join(output_dir, f"data_report_{suffix}.txt")

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"DATA PREPROCESSING REPORT - {suffix.upper()}\n")
        f.write("=" * 80 + "\n\n")

        # Dataset info
        f.write("DATASET INFORMATION\n")
        f.write("-" * 50 + "\n")
        f.write(f"Source CSV: {config.get('csv_path', 'Unknown')}\n")
        f.write(f"Date column: {config.get('date_column', 'Unknown')}\n")

        # Date ranges
        f.write("\nDATE RANGES\n")
        f.write("-" * 50 + "\n")
        f.write(
            f"Training period: {config.get('train_start', 'Unknown')} to {config.get('train_end', 'Unknown')}\n")
        f.write(
            f"Validation period: {config.get('val_start', 'Unknown')} to {config.get('val_end', 'Unknown')}\n")
        f.write(
            f"Testing period: {config.get('test_start', 'Unknown')} to {config.get('test_end', 'Unknown')}\n")

        # Data dimensions
        f.write("\nDATA DIMENSIONS\n")
        f.write("-" * 50 + "\n")
        f.write(f"Raw train tensor shape: {tuple(train_tensor.shape)}\n")
        f.write(f"Raw validation tensor shape: {tuple(val_tensor.shape)}\n")
        f.write(f"Raw test tensor shape: {tuple(test_tensor.shape)}\n")

        # Sliding window parameters
        f.write("\nSLIDING WINDOW PARAMETERS\n")
        f.write("-" * 50 + "\n")
        f.write(
            f"Input window length (x_window): {config.get('x_window', 'Unknown')} hours\n")
        f.write(
            f"Gap between input and forecast (x_y_gap): {config.get('x_y_gap', 'Unknown')} hours\n")
        f.write(
            f"Forecast horizon (y_window): {config.get('y_window', 'Unknown')} hours\n")
        f.write(
            f"Sliding window step size: {config.get('step_size', 'Unknown')} hours\n")
        f.write(
            f"Starting hour offset: {config.get('x_start_hour', 'Unknown')}\n")

        # Dataset samples
        f.write("\nDATASET SAMPLES\n")
        f.write("-" * 50 + "\n")
        if ar_model:
            f.write(f"Number of training samples: {X_train.shape[0]}\n")
            f.write(f"Number of validation samples: {X_val.shape[0]}\n")
            f.write(f"Number of test samples: {X_test.shape[0]}\n")
            f.write(f"AR sequence length: {X_train.shape[1]}\n")
        else:
            f.write(f"Number of training samples: {X_train.shape[0]}\n")
            f.write(f"Number of validation samples: {X_val.shape[0]}\n")
            f.write(f"Number of test samples: {X_test.shape[0]}\n")
            f.write(f"Input sequence length (X): {X_train.shape[1]}\n")
            if Y_train is not None:
                f.write(f"Target sequence length (Y): {Y_train.shape[1]}\n")

        # Feature information
        f.write("\nFEATURE INFORMATION\n")
        f.write("-" * 50 + "\n")
        if included_feats:
            f.write(f"Number of features: {len(included_feats)}\n")
            f.write("Selected features:\n")
            for feat in included_feats:
                f.write(f"  - {feat}\n")
        else:
            f.write(f"Using all available features in the dataset.\n")

        # Model type info
        f.write("\nMODEL TYPE INFORMATION\n")
        f.write("-" * 50 + "\n")
        f.write(f"Prepared for autoregressive model: {ar_model}\n")
        f.write(f"Spatial data: {config.get('spatial', False)}\n")

        # Normalization information
        f.write("\nNORMALIZATION INFORMATION\n")
        f.write("-" * 50 + "\n")
        f.write(
            f"Number of transformed columns: {config.get('num_transform_cols', 'Unknown')}\n")

        # Describe transforms
        if isinstance(train_transforms, list):
            transform = train_transforms[0]
        else:
            transform = train_transforms

        transform_type = transform.__class__.__name__
        f.write(f"Transformation type: {transform_type}\n")

        try:
            if transform_type == "StandardScaleNorm":
                if hasattr(transform, 'mean') and hasattr(transform, 'std'):
                    f.write("\nMEAN AND STD VALUES\n")
                    for i in range(min(5, config.get('num_transform_cols', 3))):
                        f.write(
                            f"Feature {i}: Mean={transform.mean[0, 0, i].item():.4f}, Std={transform.std[0, 0, i].item():.4f}\n")
                    if config.get('num_transform_cols', 3) > 5:
                        f.write("(Only showing first 5 features)\n")
            elif transform_type == "MinMaxNorm":
                if hasattr(transform, 'min_val') and hasattr(transform, 'max_val'):
                    f.write("\nMIN AND MAX VALUES\n")
                    for i in range(min(5, config.get('num_transform_cols', 3))):
                        f.write(
                            f"Feature {i}: Min={transform.min_val[0, 0, i].item():.4f}, Max={transform.max_val[0, 0, i].item():.4f}\n")
                    if config.get('num_transform_cols', 3) > 5:
                        f.write("(Only showing first 5 features)\n")
        except (IndexError, AttributeError) as e:
            f.write(
                f"\nCould not extract detailed transform statistics: {str(e)}\n")

        # Output information
        f.write("\nOUTPUT INFORMATION\n")
        f.write("-" * 50 + "\n")
        f.write(f"Output directory: {output_dir}\n")
        f.write(
            f"DataLoader batch size: {config.get('batch_size', 'Unknown')}\n")
        f.write(
            f"DataLoader workers: {config.get('num_workers', 'Unknown')}\n\n")

        f.write("Generated files:\n")
        f.write(f"  - train_tensor_{suffix}.pt\n")
        f.write(f"  - train_dataset_{suffix}.pt\n")
        f.write(f"  - val_dataset_{suffix}.pt\n")
        f.write(f"  - test_dataset_{suffix}.pt\n")
        f.write(f"  - train_loader_{suffix}.pt\n")
        f.write(f"  - val_loader_{suffix}.pt\n")
        f.write(f"  - test_loader_{suffix}.pt\n")
        f.write(f"  - transforms_{suffix}.pt\n")
        f.write(f"  - data_report_{suffix}.txt (this file)\n\n")

        # Timestamp
        f.write(
            f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"Data report saved to: {report_path}")
    return report_path
