"""
Checks that PyTorch can see and use a GPU.  Skipped on machines without one,
so the default test run stays green on CPU-only dev machines; run it inside
the GPU Docker image / on the GPU instance to verify the CUDA setup.
"""

import unittest

import torch

from tests._helpers import TimedTestCase


@unittest.skipUnless(torch.cuda.is_available(), "no CUDA GPU available")
class TestGPU(TimedTestCase):

    def test_gpu_is_visible(self):
        self.assertGreater(torch.cuda.device_count(), 0)
        print("GPU:", torch.cuda.get_device_name(0))

    def test_tensor_math_on_gpu(self):
        x = torch.randn(64, 64, device="cuda")
        y = x @ x.T
        self.assertEqual(y.device.type, "cuda")
        self.assertTrue(torch.isfinite(y).all())

    def test_conv_backward_on_gpu(self):
        # Exercises cuDNN, which a plain matmul does not.
        conv = torch.nn.Conv2d(3, 8, 3, padding=1).cuda()
        out = conv(torch.randn(2, 3, 16, 16, device="cuda"))
        out.mean().backward()
        self.assertIsNotNone(conv.weight.grad)


if __name__ == "__main__":
    unittest.main()
