# -*- coding: utf-8 -*-
"""Different model tests."""
import unittest
import os
import sys
from torchsummary import summary

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

from src.train_classifier import get_model
from src.ptwt_continuous_transform import get_diff_wavelet


class TestModels(unittest.TestCase):
    """Testing different classes and functions of models."""

    def test_dimensions(self) -> None:
        models = ["learndeepnet", "onednet", "lcnn"]
        f_sizes = [9600, 2304, 0]
        # Todo: make this test a bit more universal...
        for i in range(len(models)):
            model = get_model(
                models[i], flattend_size=f_sizes[i], num_of_scales=256, channels=256
            )
            summary(model, (1, 256, 101), verbose=0)
