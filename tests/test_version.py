"""Trivial version test."""

import os
import sys
import unittest

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

from audiofakedetect.version import VERSION, get_version


class TestVersion(unittest.TestCase):
    """Trivially test the version."""

    def test_version(self) -> None:
        """Test the version is a string."""
        version = get_version()
        self.assertEqual(version, VERSION)
