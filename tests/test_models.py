# -*- coding: utf-8 -*-

"""Different dataloader tests."""
import unittest

from src.learn_direct_train_classifier import get_model
from src.ptwt_continuous_transform import get_diff_wavelet

from torchsummary import summary


class TestModels(unittest.TestCase):
    """Testing different classes and functions of models."""

    def test_dimensions(self) -> None:
        wavelet = get_diff_wavelet("cmor4.6-0.87")

        models = ["learndeepnet", "onednet", "learnnet"]
        for model_name in models:
            model = get_model(wavelet, model_name)

            summary(model, (1, 11025), verbose=0)