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
        wavelet = get_diff_wavelet("cmor4.6-0.87")
        sample_rate = 22050
        window_size = 11025

        models = ["learndeepnet", "onednet", "learnnet"]
        f_sizes = [21888, 5440, 39168]
        # Todo: make this test a bit more universal...
        for i in range(len(models)):
            model = get_model(wavelet, models[i], sample_rate=sample_rate, flattend_size=f_sizes[i])
            summary(model, (1, window_size), verbose=0)
