"""Return configuration for grid search."""
from functools import partial

import torch
import torchvision

from audiofakedetect.models import DCNN


def get_config() -> dict:
    """Return config dictonary for grid search.

    Note: The keys that adhere to the initial command line args will be
    overridden by this configuration. Any other key is also welcome and
    can be used in "modules"-architecure for example.
    """
    model = "modules"
    if model == "gridmodel":
        model_data = [
            [
                {
                    "layers": [
                        [torchvision.ops, "Permute 0,1,3,2"],
                        "Conv2d 1 [64,32,128] 2 1 2",
                        "MaxFeatureMap2D",
                        "MaxPool2d 2 2",
                        "Conv2d [32,16,64] 64 1 1 0",
                        "MaxFeatureMap2D",
                        "SyncBatchNorm 32",
                        "Conv2d 32 96 3 1 1",
                        "MaxFeatureMap2D",
                        "MaxPool2d 2 2",
                        "SyncBatchNorm 48",
                        "Conv2d 48 96 1 1 0",
                        "MaxFeatureMap2D",
                        "SyncBatchNorm 48",
                        "Conv2d 48 128 3 1 1",
                        "MaxFeatureMap2D",
                        "MaxPool2d 2 2",
                        "Conv2d 64 128 1 1 0",
                        "MaxFeatureMap2D",
                        "SyncBatchNorm 64",
                        "Conv2d 64 64 3 1 1",
                        "MaxFeatureMap2D",
                        "SyncBatchNorm 32",
                        "Conv2d 32 64 1 1 0",
                        "MaxFeatureMap2D",
                        "SyncBatchNorm 32",
                        "Conv2d 32 64 3 1 1",
                        "MaxFeatureMap2D",
                        "MaxPool2d 2 2",
                        "Dropout 0.7",
                    ],
                    "input_shape": (1, 256, 101),
                    "transforms": [partial(transf)],
                },
                {
                    "layers": [
                        "BLSTMLayer 512 512",
                        "BLSTMLayer 512 512",
                        "Dropout 0.1",
                        "Linear 512 2",
                    ],
                    "input_shape": (1, 512),
                    "transforms": [partial(torch.Tensor.mean, dim=1)],
                },
            ]
        ]
    else:
        model_data = []

    wf_config = {
        "transform": ["packets"],
        "learning_rate": [0.0004],
        "weight_decay": [0.001],
        "save_path": ["./data/run3"],
        "data_path": ["./data/fake"],
        "only_use": [["ljspeech", "fbmelgan"]],
        "limit_train": [(55504, 7504, 15504)],
        "cross_data_path": ["./data/fake"],
        "cross_limit": [(55500, 7304, 14600)],
        "only_test_folders": [["conformer", "jsutmbmelgan", "jsutpwg"]],
        "file_type": ["wav"],
        "dropout_cnn": [0.6],
        "dropout_lstm": [0.2],
        "num_of_scales": [256],
        "seconds": [1],
        "sample_rate": [22050],
        "cross_sources": [
            [
                "ljspeech",
                "melgan",
                "lmelgan",
                "mbmelgan",
                "pwg",
                "waveglow",
                "avocodo",
                "hifigan",
                "conformer",
                "jsutmbmelgan",
                "jsutpwg",
                "lbigvgan",
                "bigvgan",
            ],
            [
                "ljspeech",
                "melgan",
                "lmelgan",
                "mbmelgan",
                "pwg",
                "waveglow",
                "hifigan",
                "conformer",
                "jsutmbmelgan",
                "jsutpwg",
            ],
            ["ljspeech", "avocodo"],
            ["ljspeech", "lbigvgan", "bigvgan"],
        ],
        "epochs": [10],
        "validation_interval": [10],
        "block_norm": [False],
        "batch_size": [128],
        "aug_contrast": [False],
        "model": ["modules"],
        "model_data": model_data,
        "module": [DCNN],
        "kernel1": [3],
        "num_devices": [4],
        "ochannels1": [64],
        "ochannels2": [64],
        "ochannels3": [96],
        "ochannels4": [128],
        "ochannels5": [32],
        "hop_length": [220],
        "only_testing": [True],
        "only_ig": [False]
        # "target": [0, 1, None],
    }

    # Alternatively, the config for the In The Wild dataset could look like this:
    # _itw_config = {
    #     "learning_rate": [0.0001],
    #     "weight_decay": [0.001],
    #     "save_path": [
    #         "./data/run2"
    #     ],
    #     "data_path": [
    #         "./data/fake"
    #     ],
    #     "only_use": [["ljspeech", "fbmelgan"]],
    #     "limit_train": [(55504, 7504, 15504)],
    #     "cross_data_path": [
    #         "./data/inthewild/set"
    #     ],
    #     "seconds": [1],
    #     "cross_limit": [(38968, 5568, 11136)],
    #     "file_type": ["wav"],
    #     "dropout_cnn": [0.6],
    #     "dropout_lstm": [0.2],
    #     "num_of_scales": [256],
    #     "wavelet": ["sym8"],
    #     "cross_sources": [["inthewildReal", "inthewildFake"]],
    #     "epochs": [10],
    #     "validation_interval": [10],
    #     "sample_rate": [16000],
    #     "block_norm": [False],
    #     "batch_size": [128],
    #     "aug_contrast": [False],
    #     "model": ["modules"],
    #     "model_data": model_data,
    #     "module": [DCNNxDropout],
    #     "kernel1": [3],
    #     "num_devices": [4],
    #     "ochannels1": [64],
    #     "ochannels2": [64],
    #     "ochannels3": [96],
    #     "ochannels4": [128],
    #     "ochannels5": [32],
    #     "only_testing": [False],
    #     "only_ig": [True],
    # }

    # Alternatively, the config for the ASV spoof dataset could look like this:
    # _asv_config = {
    #     "learning_rate": [0.0001],
    #     "weight_decay": [0.001],
    #     "save_path": [
    #         "./data/run2"
    #     ],
    #     "data_path": [
    #         "./data/asv"
    #     ],
    #     "cross_limit": [(7472, 7672, 21320)],
    #     "cross_sources": [["asv2019real", "asv2019fake"]],
    #     "asvspoof_name_cross": ["LA"],  # or DF_E or None
    #     "cross_data_path": [
    #         "./data/asv"
    #     ],
    #     "limit_train": [(44368, 6336, 12672)],
    #     "file_type": ["flac"],
    #     "asvspoof_name": ["DF_E"],
    #     "sample_rate": [16000],
    #     "dropout_cnn": [0.6],
    #     "dropout_lstm": [0.2, 0.1],
    #     "num_of_scales": [256],
    #     "wavelet": ["sym8"],
    #     "only_use": [
    #         ["asv2021real", "asv2021fake"]
    #     ],
    #     "epochs": [10],
    #     "validation_interval": [10],
    #     "block_norm": [False],
    #     "batch_size": [128],
    #     "aug_contrast": [False],
    #     "model": ["modules"],
    #     "model_data": model_data,
    #     "module": [DCNN],
    #     "kernel1": [3],
    #     "num_devices": [4],
    #     "ochannels1": [64],
    #     "ochannels2": [64],
    #     "ochannels3": [96],
    #     "ochannels4": [128],
    #     "ochannels5": [32, 64, 16],
    #     "only_testing": [False],
    # }

    config = wf_config

    return config


def transf(x: torch.Tensor) -> torch.Tensor:
    """Permutes dimensions 2 and 3 of x and flattens it.

    Args:
        x (torch.Tensor): Input tensor, e.g. from a layer before.

    Returns:
        torch.Tensor: permuted, contiguous, flattend input tensor.
    """
    x = x.permute(0, 2, 1, 3)
    x = x.contiguous()
    return x.view(x.shape[0], x.shape[1], -1)
