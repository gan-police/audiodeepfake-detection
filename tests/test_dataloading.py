# -*- coding: utf-8 -*-

"""Different dataloader tests."""

import os
import sys
import unittest

import torchaudio

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)
from src.util import get_frames_list


class TestDataLoader(unittest.TestCase):
    """Testing different classes and functions of Dataloading."""

    def test_pathing(self) -> None:
        """Test correct path lengths."""
        frame_size = 224
        path_list, offset_list = get_frames_list(f"{BASE_PATH}/tests/data/ljspeech_melgan", frame_size=frame_size)
        self.assertEqual(len(path_list), len(offset_list))
        self.assertEqual(len(path_list), (torchaudio.info(f"{BASE_PATH}/tests/data/ljspeech_melgan/LJ008-0217_gen.wav").num_frames // frame_size) * 2)