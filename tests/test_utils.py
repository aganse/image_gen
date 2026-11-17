"""
Tests of the image-output helpers in utils.py.
"""

import os
import tempfile
import unittest

import torch
from PIL import Image

import utils
from tests._helpers import TimedTestCase


class TestUtils(TimedTestCase):

    def setUp(self):
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.images = torch.rand(6, 3, 8, 8) * 2 - 1

    def tearDown(self):
        self.tmp.cleanup()
        super().tearDown()

    def test_print_run_params_writes_file(self):
        path = utils.print_run_params(output_dir=self.tmp.name, num_epochs=3)
        self.assertTrue(os.path.exists(path))
        with open(path) as f:
            self.assertIn("num_epochs", f.read())

    def test_print_run_params_requires_output_dir(self):
        with self.assertRaises(ValueError):
            utils.print_run_params(num_epochs=3)

    def test_default_nrow(self):
        self.assertEqual(utils.default_nrow(64), 8)
        self.assertEqual(utils.default_nrow(10), 4)
        self.assertEqual(utils.default_nrow(0), 1)

    def test_image_grid_range(self):
        grid = utils.image_grid(self.images)
        self.assertEqual(grid.shape[0], 3)
        self.assertGreaterEqual(float(grid.min()), 0.0)
        self.assertLessEqual(float(grid.max()), 1.0)

    def test_save_image_grid(self):
        path = os.path.join(self.tmp.name, "sub", "grid.png")
        utils.save_image_grid(self.images, path, nrow=3)
        with Image.open(path) as image:
            self.assertEqual(image.mode, "RGB")
            self.assertGreater(image.size[0], 8 * 3)

    def test_save_individual_images(self):
        out_dir = os.path.join(self.tmp.name, "sims")
        paths = utils.save_individual_images(self.images, out_dir, prefix="sim")
        self.assertEqual(len(paths), 6)
        self.assertTrue(paths[0].endswith("sim_00000.png"))
        with Image.open(paths[0]) as image:
            self.assertEqual(image.size, (8, 8))

    def test_side_by_side_grid(self):
        path = os.path.join(self.tmp.name, "pair.png")
        utils.side_by_side_grid(self.images, self.images, path)
        self.assertTrue(os.path.exists(path))
        with self.assertRaises(ValueError):
            utils.side_by_side_grid(self.images, self.images[:2], path)


if __name__ == "__main__":
    unittest.main()
