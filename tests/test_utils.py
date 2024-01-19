"""Test utility functions and classes."""

import os
import sys
import unittest

import torch

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

from audiofakedetect.utils import (
    DotDict,
    _Griderator,
    add_noise,
    build_new_grid,
    contrast,
)


class TestDotDict(unittest.TestCase):
    """Test DotDict."""

    def test_create(self):
        """Test create and get."""
        new_dict = DotDict()
        new_dict.test = "a test"

        self.assertDictEqual(new_dict, {"test": "a test"})


class TestAudioFunctions(unittest.TestCase):
    """Test audio augmentation functions."""

    def test_contrast(self):
        """Test contrast output dim."""
        waveform = torch.randn(2, 16000)
        enhanced_waveform = contrast(waveform)

        self.assertEqual(enhanced_waveform.shape, waveform.shape)

    def test_add_noise(self):
        """Test noise add output dim."""
        waveform = torch.randn(2, 16000)
        noisy_waveform = add_noise(waveform)

        self.assertEqual(noisy_waveform.shape, waveform.shape)


class TestGriderator(unittest.TestCase):
    """Test griderator class for grid search."""

    def setUp(self):
        """Set up the used test instance."""
        self.config = {"param1": [1, 2], "param2": [3, 4]}
        self.init_seeds = [0, 1]
        self.num_exp = 2

    def test_init_with_dict_config(self):
        """Test general initialization."""
        griderator = _Griderator(self.config)
        self.assertIsInstance(griderator, _Griderator)

    def test_init_with_non_dict_config_raises_type_error(self):
        """Test error raising."""
        with self.assertRaises(TypeError):
            _Griderator("invalid_config")

    def test_init_with_seeds(self):
        """Test all parameters."""
        griderator = _Griderator(
            self.config, init_seeds=self.init_seeds, num_exp=self.num_exp
        )
        self.assertIsInstance(griderator, _Griderator)

        griderator = build_new_grid(self.config, seeds=[1, 2, 3], random_seeds=False)
        self.assertEqual(griderator.get_len(), 12)

    def test_get_keys(self):
        """Test get_keys method."""
        griderator = _Griderator(self.config)
        keys = griderator.get_keys()
        self.assertEqual(set(keys), set(["seed", "param1", "param2"]))

    def test_get_len(self):
        """Test different lens of the grid."""
        griderator = _Griderator(self.config)
        self.assertEqual(griderator.get_len(), 20)  # 2 * 2 * 5 (default)

        griderator = _Griderator(self.config, init_seeds=self.init_seeds)
        self.assertEqual(griderator.get_len(), 8)  # 2 * 2 * 2

        # check wrapper
        griderator = build_new_grid(self.config, seeds=self.init_seeds)
        self.assertEqual(griderator.get_len(), 8)

        griderator = _Griderator(self.config, num_exp=3)
        self.assertEqual(griderator.get_len(), 12)  # 2 * 2 * 3

    def test_iteration(self):
        """Test the iterations."""
        griderator = _Griderator(self.config, init_seeds=self.init_seeds)
        values = list(griderator)
        self.assertListEqual(
            values,
            [
                (0, 1, 4),
                (0, 2, 3),
                (0, 2, 4),
                (1, 1, 3),
                (1, 1, 4),
                (1, 2, 3),
                (1, 2, 4),
            ],
        )

        griderator = _Griderator(self.config, init_seeds=self.init_seeds)
        first = griderator.next()
        self.assertEqual(first, (0, 1, 4))

    def test_reset(self):
        """Test reseting the iterable."""
        griderator = _Griderator(self.config)
        list(griderator)  # iterate once
        griderator.reset()
        self.assertEqual(griderator.current, 0)

    def test_update_args(self):
        """Test update_args method."""
        griderator = _Griderator(self.config, init_seeds=self.init_seeds)
        args = DotDict()
        new_args = griderator.update_args(args)
        self.assertEqual(new_args, {"seed": 0, "param1": 1, "param2": 3})

    def test_update_step(self):
        """Test update_step method."""
        griderator = _Griderator(self.config, init_seeds=self.init_seeds)
        args = DotDict()
        new_args, new_step = griderator.update_step(args)
        self.assertEqual(new_args, {"seed": 0, "param1": 1, "param2": 3})
        self.assertEqual(new_step, (0, 1, 4))  # Next step values

        # Consume all steps
        for _ in range(griderator.get_len() - 1):
            new_args, new_step = griderator.update_step(args)
        self.assertEqual(new_args, {"seed": 1, "param1": 2, "param2": 4})
        self.assertEqual(
            new_step, StopIteration
        )  # StopIteration after all steps consumed
