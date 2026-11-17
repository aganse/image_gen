"""
Tests of the differentiable augmentations in augment.py.  CPU only.
"""

import unittest

import torch

import augment
from tests._helpers import TimedTestCase


class TestDiffAugment(TimedTestCase):

    def setUp(self):
        super().setUp()
        torch.manual_seed(0)
        self.x = torch.rand(4, 3, 16, 16) * 2 - 1

    def test_parse_policy(self):
        self.assertEqual(augment.parse_policy("color, cutout"), ["color", "cutout"])
        self.assertEqual(augment.parse_policy(""), [])
        self.assertEqual(augment.parse_policy(None), [])
        self.assertEqual(augment.parse_policy("none"), [])
        with self.assertRaises(ValueError):
            augment.parse_policy("color,rotate")

    def test_empty_policy_is_identity(self):
        out = augment.DiffAugment("")(self.x)
        self.assertTrue(torch.equal(out, self.x))

    def test_each_policy_preserves_shape(self):
        for policy in ("color", "translation", "cutout", "color,translation,cutout"):
            out = augment.DiffAugment(policy)(self.x)
            self.assertEqual(out.shape, self.x.shape, msg=policy)
            self.assertTrue(torch.isfinite(out).all(), msg=policy)

    def test_gradients_flow_through_augmentation(self):
        x = self.x.clone().requires_grad_(True)
        out = augment.DiffAugment("color,translation,cutout")(x)
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertGreater(float(x.grad.abs().sum()), 0.0)

    def test_translation_is_whole_pixel_shift(self):
        # A whole-pixel shift only moves or zero-pads values, so every output
        # value must be (numerically) one of the input values or zero.
        x = torch.arange(1.0, 17.0).view(1, 1, 4, 4).repeat(2, 1, 1, 1)
        out = augment.rand_translation(x, ratio=0.25)
        allowed = torch.cat([x.flatten(), torch.zeros(1)])
        distance = (out.flatten()[:, None] - allowed[None, :]).abs().min(dim=1).values
        self.assertLess(float(distance.max()), 1e-3)

    def test_cutout_zeroes_some_pixels(self):
        ones = torch.ones(8, 3, 16, 16)
        out = augment.rand_cutout(ones)
        self.assertTrue(((out == 0) | (out == 1)).all())
        self.assertLess(float(out.mean()), 1.0)


if __name__ == "__main__":
    unittest.main()
