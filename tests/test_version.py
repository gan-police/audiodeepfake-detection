# -*- coding: utf-8 -*-

"""Trivial version test."""

import unittest
import os
import sys

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)
from src.version import get_version


class TestVersion(unittest.TestCase):
    """Trivially test a version."""

    def test_version_type(self) -> None:
        """Test the version is a string.
        This is only meant to be an example test.
        """
        version = get_version()
        self.assertIsInstance(version, str)