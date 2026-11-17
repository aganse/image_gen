"""
Tests of image_metrics.py.

The conversion/validation tests run anywhere.  The tests that actually
compute FID/KID/IS need the Inception-v3 weights (a ~100MB download on first
use, cached under TORCH_HOME), so they only run when IMAGE_GEN_TEST_METRICS=1
is set - e.g. inside the Docker image, where the weights are already baked in:

    IMAGE_GEN_TEST_METRICS=1 python -m unittest -v tests.test_image_metrics
"""

import os
import unittest

import torch

import image_metrics
from tests._helpers import TimedTestCase

RUN_METRICS = os.environ.get("IMAGE_GEN_TEST_METRICS") == "1"


class TestConversions(TimedTestCase):

    def test_to_uint8(self):
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        out = image_metrics.to_uint8(x)
        self.assertEqual(out.dtype, torch.uint8)
        self.assertEqual(out.tolist(), [0, 0, 128, 255, 255])

    def test_to_uint8_passthrough(self):
        x = torch.zeros(2, dtype=torch.uint8)
        self.assertIs(image_metrics.to_uint8(x), x)

    def test_real_images_must_be_uint8(self):
        with self.assertRaises(ValueError):
            image_metrics.GenerativeImageMetrics(torch.zeros(2, 3, 8, 8))

    def test_unknown_metric_raises(self):
        evaluator = image_metrics.GenerativeImageMetrics(
            torch.zeros(4, 3, 8, 8, dtype=torch.uint8)
        )
        with self.assertRaises(ValueError):
            evaluator.evaluate(torch.zeros(4, 3, 8, 8, dtype=torch.uint8), metrics=["psnr"])


@unittest.skipUnless(RUN_METRICS, "set IMAGE_GEN_TEST_METRICS=1 to run (needs Inception weights)")
class TestMetricComputation(TimedTestCase):

    @classmethod
    def setUpClass(cls):
        generator = torch.Generator().manual_seed(0)
        cls.real = torch.randint(0, 256, (48, 3, 32, 32), dtype=torch.uint8, generator=generator)
        cls.similar = torch.randint(0, 256, (48, 3, 32, 32), dtype=torch.uint8, generator=generator)
        cls.different = torch.full((48, 3, 32, 32), 255, dtype=torch.uint8)
        cls.different[:, :, ::2, :] = 0  # horizontal stripes, unlike the noise images
        cls.evaluator = image_metrics.GenerativeImageMetrics(
            cls.real, kid_subset_size=24, kid_subsets=10, batch_size=16
        )

    def test_all_metrics_returned_and_finite(self):
        results = self.evaluator.evaluate(self.similar)
        self.assertEqual(set(results), {"fid", "kid", "kid_std", "is", "is_std"})
        for name, value in results.items():
            self.assertTrue(torch.isfinite(torch.tensor(value)), msg=name)

    def test_metric_subset(self):
        results = self.evaluator.evaluate(self.similar, metrics=["kid"])
        self.assertEqual(set(results), {"kid", "kid_std"})

    def test_same_distribution_scores_better_than_different(self):
        near = self.evaluator.evaluate(self.similar, metrics=["fid", "kid"])
        far = self.evaluator.evaluate(self.different, metrics=["fid", "kid"])
        self.assertLess(near["fid"], far["fid"])
        self.assertLess(near["kid"], far["kid"])

    def test_repeat_evaluation_reuses_real_statistics(self):
        first = self.evaluator.evaluate(self.similar, metrics=["fid"])
        second = self.evaluator.evaluate(self.similar, metrics=["fid"])
        self.assertAlmostEqual(first["fid"], second["fid"], places=2)

    def test_kid_subset_size_shrinks_to_fit_few_fakes(self):
        results = self.evaluator.evaluate(self.similar[:10], metrics=["kid"])
        self.assertIn("kid", results)

    def test_one_shot_wrapper(self):
        results = image_metrics.compute_image_metrics(
            self.real, self.similar, metrics=["kid"], kid_subset_size=24, kid_subsets=5
        )
        self.assertIn("kid", results)


if __name__ == "__main__":
    unittest.main()
