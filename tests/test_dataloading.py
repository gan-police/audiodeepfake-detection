# -*- coding: utf-8 -*-

"""Different dataloader tests."""

import os
import sys
import unittest

import torch
import torchaudio

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

import src.data_loader as dl


class TestDataLoader(unittest.TestCase):
    """Testing different classes and functions of Dataloading."""

    def test_pathing(self) -> None:
        """Test correct path lengths."""
        frame_size = 224
        path_list, offset_list = dl.get_frames_list(f"{BASE_PATH}/tests/data/ljspeech_melgan", frame_size=frame_size)
        self.assertEqual(len(path_list), len(offset_list))
        path = f"{BASE_PATH}/tests/data/ljspeech_melgan/LJ008-0217_gen.wav"
        self.assertEqual(
            len(path_list),
            (torchaudio.info(path).num_frames // frame_size) * 2
        )

    def test_transform(self) -> None:
        "Test if cwt transform returns correct data shape."
        frame_size = 30
        scales = 20
        waveform = torch.randn(frame_size)
        from src.old.cwt import CWT
        transform = CWT(n_lin=scales)
        waveform_t = transform(waveform)
        self.assertEqual(waveform_t.shape, torch.Size([1, scales, frame_size]))
