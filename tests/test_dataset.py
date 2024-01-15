"""Test custom dataset."""

import os
import sys
import unittest
from unittest.mock import patch

import torch

from audiofakedetect.data_loader import get_costum_dataset
from audiofakedetect.utils import DotDict, get_input_dims

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)


class TestCustomDataset(unittest.TestCase):
    """Test various functionality of the CustomDataset."""

    @patch("audiofakedetect.data_loader.CustomDataset", autospec=True)
    def test_get_costum_dataset(self, mock_custom_dataset):
        """Test get_custom_dataset."""
        # Mock the CustomDataset class
        mock_custom_dataset_instance = mock_custom_dataset.return_value
        mock_custom_dataset_instance.__len__.return_value = 42

        # Call the get_costum_dataset function
        result = get_costum_dataset(
            data_path="./tests/new_data/",
            save_path="./tests/new_data/",
            ds_type="train",
        )

        # Assert that the CustomDataset class was called with the correct arguments
        mock_custom_dataset.assert_called_once_with(
            paths=[
                "tests/new_data/B_fullbandmelgan",
                "tests/new_data/A_ljspeech",
                "tests/new_data/F_parallelwavegan",
                "tests/new_data/C_hifigan",
                "tests/new_data/G_waveglow",
                "tests/new_data/D_melgan",
                "tests/new_data/E_multibandmelgan",
            ],
            labels=[1, 0, 5, 2, 6, 3, 4],
            save_path="./tests/new_data/",
            abort_on_save=False,
            seconds=1,
            resample_rate=22050,
            verbose=False,
            limit=55504,
            ds_type="train",
            only_test_folders=None,
            asvspoof_name=None,
            train_ratio=0.7,
            val_ratio=0.1,
            filetype="wav",
        )

        # Assert that the result is the mocked instance of CustomDataset
        self.assertEqual(result, mock_custom_dataset_instance)

    @patch("audiofakedetect.data_loader.CustomDataset", autospec=True)
    def test_custom_dataset_get_input_dim(self, mock_custom_dataset):
        """Test get_input_dim."""
        mock_custom_dataset_instance = mock_custom_dataset.return_value
        mock_custom_dataset_instance.__len__.return_value = 42

        mock_custom_dataset_instance.__getitem__.return_value = {
            "audio": torch.zeros((2, 1, 22050)),
            "label": torch.tensor(0),
        }

        class TestTransform(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x * 2, None

        transforms = torch.nn.Sequential(TestTransform())

        new_args = DotDict()
        new_args.only_use = ["ljspeech", "fullbandmelgan"]
        new_args.limit_train = [100]
        new_args.save_path = "./tests/new_data/"
        new_args.data_path = "./tests/new_data/"
        new_args.batch_size = 32

        shape = get_input_dims(new_args, transforms)

        self.assertEqual(shape, [32, 2, 1, 22050])
