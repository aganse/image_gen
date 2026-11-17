"""Shared helpers for the test suite (not a test module itself)."""

import os
import time
import unittest

import numpy as np
from PIL import Image


class TimedTestCase(unittest.TestCase):
    """TestCase that prints each test's wall time (flow_models test style)."""

    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("\n%s: %.3fs" % (self.id(), t))


def write_random_images(directory, n, size=(40, 48), subdir=None, ext=".png"):
    """Write n random RGB images of (height, width)=size; returns their paths."""
    if subdir:
        directory = os.path.join(directory, subdir)
    os.makedirs(directory, exist_ok=True)
    rng = np.random.default_rng(0)
    paths = []
    for i in range(n):
        array = rng.integers(0, 256, size=(size[0], size[1], 3), dtype=np.uint8)
        path = os.path.join(directory, f"img_{i:03d}{ext}")
        Image.fromarray(array).save(path)
        paths.append(path)
    return paths
