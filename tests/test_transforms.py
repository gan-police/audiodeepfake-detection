"""Test utility functions and classes."""

import os
import sys
import unittest
import pywt

import torch

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

from audiofakedetect.wavelet_math import (
    Packets,
    STFTLayer,
    compute_pytorch_packet_representation,
)


class TestSTFTLayer(unittest.TestCase):
    """Test the STFT module."""

    def test_forward(self):
        """Test forward output dimensions."""
        stft_layer = STFTLayer(
            n_fft=512, hop_length=2, log_offset=1e-12, log_scale=True, power=2.0
        )

        input_tensor = torch.randn(2, 1, 22050)

        try:
            output, _ = stft_layer.forward(input_tensor)
        except Exception as e:
            self.fail(f"Forward pass raised an exception: {str(e)}")

        self.assertEqual(output.shape, (2, 1, 257, 11026))

    def test_default_configuration(self):
        """Test forward on default values."""
        stft_layer_default = STFTLayer()

        input_tensor = torch.randn(2, 1, 22050)

        try:
            output, _ = stft_layer_default.forward(input_tensor)
        except Exception as e:
            self.fail(
                f"Forward pass with default configuration raised an exception: {str(e)}"
            )

        self.assertEqual(output.shape, (2, 1, 256, 101))


class TestComputePyTorchPacketRepresentation(unittest.TestCase):
    """Test WPT."""

    def test_compute_packet_representation(self):
        """Test packet computation."""
        pt_data = torch.randn(2, 22050)
        wavelet = pywt.Wavelet("db8")

        try:
            (
                packet_representation,
                block_norm_dict,
            ) = compute_pytorch_packet_representation(
                pt_data,
                wavelet,
                max_lev=7,
                log_scale=True,
                loss_less=False,
                power=2.0,
                block_norm=True,
                compute_welford=True,
            )
        except Exception as e:
            self.fail(f"Function raised an exception: {str(e)}")

        self.assertEqual(packet_representation.shape, (2, 1, 187, 128))

        try:
            (
                packet_representation,
                block_norm_dict,
            ) = compute_pytorch_packet_representation(
                pt_data,
                wavelet,
                max_lev=7,
                log_scale=True,
                loss_less=True,
                power=2.0,
                block_norm=True,
                compute_welford=True,
            )
        except Exception as e:
            self.fail(f"Function raised an exception: {str(e)}")

        self.assertEqual(packet_representation.shape, (2, 2, 187, 128))
        self.assertTrue(block_norm_dict is not None)


class TestPacketsModule(unittest.TestCase):
    """Test Wavelet Packet module."""

    def test_forward(self):
        """Test forward output dimension of packet module."""
        packets_module = Packets(
            wavelet_str="sym8",
            max_lev=7,
            log_scale=True,
            loss_less=False,
            power=2.0,
            block_norm=False,
            compute_welford=True,
        )

        input_tensor = torch.randn(2, 22050)

        try:
            output, block_norm_dict = packets_module.forward(input_tensor)
        except Exception as e:
            self.fail(f"Forward pass raised an exception: {str(e)}")

        self.assertEqual(output.shape, (2, 1, 128, 187))
        self.assertTrue(block_norm_dict is not None)

        packets_module = Packets(
            wavelet_str="sym8",
            max_lev=7,
            log_scale=True,
            loss_less=True,
            power=2.0,
            block_norm=False,
            compute_welford=True,
        )

        try:
            output, _ = packets_module.forward(input_tensor)
        except Exception as e:
            self.fail(f"Forward pass raised an exception: {str(e)}")

        self.assertEqual(output.shape, (2, 2, 128, 187))
