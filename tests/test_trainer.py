"""Test Trainer functions."""

import os
import sys
import unittest
import torch

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

from audiofakedetect.train_classifier import Trainer


class TestAccuracyCalculation(unittest.TestCase):
    """Test accuracy calculation."""

    def test_key_error(self):
        """Test if KeyError is raised if not all dicts have the same keys."""
        count_dict_gathered = [{1: 1}, {2: 1}]
        ok_dict_gathered = [{1: [torch.tensor(False)], 2: []}]
        with self.assertRaises(KeyError):
            Trainer.calculate_acc_label(count_dict_gathered, ok_dict_gathered, key=2)

    def test_result_type(self):
        """Test if return type is correct in multiple scenarios."""
        count_dict_gathered = [{1: 1}]
        ok_dict_gathered = [{1: [], 2: []}]
        acc = Trainer.calculate_acc_label(count_dict_gathered, ok_dict_gathered, key=1)

        self.assertIsInstance(acc, float)

        count_dict_gathered = [{1: 1}]
        ok_dict_gathered = [{1: [True], 2: []}]
        acc = Trainer.calculate_acc_label(count_dict_gathered, ok_dict_gathered, key=1)

        self.assertIsInstance(acc, float)

    def test_accuracy_calculation(self):
        """Test accuracy for one label calculation for a ddp setting.

        Two gpus used in this example.
        """
        count_dict_gathered = [{1: 3, 3: 2, 2: 1, 0: 1}, {1: 3, 3: 1, 2: 1, 0: 2}]
        ok_dict_gathered = [
            {
                1: [torch.tensor(True), torch.tensor(False), torch.tensor(False)],
                3: [torch.tensor(True), torch.tensor(True)],
                2: [torch.tensor(True)],
                0: [torch.tensor(False)],
            },
            {
                1: [torch.tensor(True), torch.tensor(True), torch.tensor(False)],
                3: [torch.tensor(True)],
                2: [torch.tensor(True)],
                0: [torch.tensor(False), torch.tensor(False)],
            },
        ]

        accuracy = Trainer.calculate_acc_label(
            count_dict_gathered, ok_dict_gathered, key=1
        )

        self.assertAlmostEqual(accuracy, (1 + 2) / (3 + 3))

        accuracy = Trainer.calculate_acc_label(
            count_dict_gathered, ok_dict_gathered, key=0
        )

        self.assertAlmostEqual(accuracy, (0 + 0) / (1 + 2))

    def test_accuracy_dict_calc(self):
        """Test accuracy per label calculation for a ddp setting using the dataloader.

        Two gpus used in this example.
        """
        count_dict_gathered = [{1: 3, 3: 2, 2: 1, 0: 1}, {1: 3, 3: 1, 2: 1, 0: 2}]
        ok_dict_gathered = [
            {
                1: [torch.tensor(True), torch.tensor(False), torch.tensor(False)],
                3: [torch.tensor(True), torch.tensor(True)],
                2: [torch.tensor(True)],
                0: [torch.tensor(False)],
            },
            {
                1: [torch.tensor(True), torch.tensor(True), torch.tensor(False)],
                3: [torch.tensor(False)],
                2: [torch.tensor(True)],
                0: [torch.tensor(False), torch.tensor(False)],
            },
        ]

        class DataLoaderCustom:
            def __init__(self):
                class CustomDS:
                    def get_label_name(self, key):
                        return {0: "Zero", 1: "First", 2: "Second", 3: "Third"}.get(
                            key, ""
                        )

                self.dataset = CustomDS()

        dl = DataLoaderCustom()
        common_keys = {0, 1, 2, 3}

        accuracies_dict = Trainer.caculate_acc_dict(
            dl, common_keys, ok_dict_gathered, count_dict_gathered
        )

        self.assertEqual(
            accuracies_dict,
            [
                ("Zero", 0.0),
                ("First", 0.5),
                ("Second", 1.0),
                ("Third", 0.6666666865348816),
            ],
        )
