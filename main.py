import torch
import json
import argparse

from processing.pipeline import benchmark_preprocess, preprocess_on_cfg
from processing.transforms import StandardScaleNorm, MinMaxNorm, TransformSequence
from cfgs.cfg import Config

usable_transforms = {
    'standard_scale': StandardScaleNorm, 'min_max': MinMaxNorm}


def main(cfg_path):
    """Takes .csv files from data/cleaned and processes them into dataloaders to be stored in data/torch."""

    with open(cfg_path, 'r') as f:
        config_dict = json.load(f)
    config = Config(**config_dict)

    _ = preprocess_on_cfg(config, usable_transforms)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Data preprocessing for power consumption datasets')
    parser.add_argument('--config_path', type=str, default='cfgs/data_proc/spain_data/spain_dataset_non_spatial_ar.json',
                        help='Path to the config file containing preprocessing parameters')
    args = parser.parse_args()

    main(args.config_path)
