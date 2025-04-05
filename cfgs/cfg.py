from dataclasses import dataclass
from typing import Optional, List


@dataclass
class Config:
    csv_path: str
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    test_start: str
    test_end: str
    ar_model: bool
    x_start_hour: int
    x_y_gap: int
    x_window: int
    y_window: int
    step_size: int
    batch_size: int
    num_workers: int
    included_feats: List[str]
    num_transform_cols: int
    spatial: bool
    output_dir: str
    transforms: List[str]
