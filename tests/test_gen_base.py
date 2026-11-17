"""
Tests of the shared pieces in gen_base.py: learning-rate handling, the EMA
wrapper, and the common sampling interface.  CPU only, tiny tensors.
"""

import unittest

import torch

import gen_base
from tests._helpers import TimedTestCase


class _ToyGenerative(gen_base.GenerativeLitModule):
    """Minimal subclass: 'images' are the noise scaled by a parameter."""

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(10.0))

    def make_noise(self, n, generator=None):
        return torch.randn(n, 3, 4, 4, generator=generator)

    def sample_from_noise(self, noise):
        return noise * self.scale


class TestLearningRate(TimedTestCase):

    def test_initial_learning_rate(self):
        self.assertEqual(gen_base.initial_learning_rate(1e-3), 1e-3)
        self.assertEqual(gen_base.initial_learning_rate([1e-4, 100, 0.5]), 1e-4)
        with self.assertRaises(ValueError):
            gen_base.initial_learning_rate([1e-4, 100])
        with self.assertRaises(ValueError):
            gen_base.initial_learning_rate("fast")

    def test_exp_decay_lambda_is_staircase(self):
        fn = gen_base.exp_decay_lambda(10, 0.5)
        self.assertEqual(fn(0), 1.0)
        self.assertEqual(fn(9), 1.0)
        self.assertEqual(fn(10), 0.5)
        self.assertEqual(fn(25), 0.25)

    def test_optimizer_config(self):
        param = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.Adam([param], lr=1e-3)
        self.assertIs(gen_base.optimizer_config(opt, 1e-3), opt)
        config = gen_base.optimizer_config(opt, [1e-3, 10, 0.5])
        self.assertEqual(config["lr_scheduler"]["interval"], "step")

    def test_chw_from_image_shape(self):
        self.assertEqual(gen_base.chw_from_image_shape([64, 32, 3]), (3, 64, 32))
        with self.assertRaises(ValueError):
            gen_base.chw_from_image_shape([64, 64])


class TestEMAModel(TimedTestCase):

    def test_update_moves_toward_model(self):
        net = torch.nn.Linear(2, 2)
        ema = gen_base.EMAModel(net, decay=0.5, warmup=False)
        before = ema.ema_model.weight.clone()
        with torch.no_grad():
            net.weight.add_(1.0)
        ema.update(net)
        expected = before + 0.5
        self.assertTrue(torch.allclose(ema.ema_model.weight, expected, atol=1e-6))
        self.assertEqual(int(ema.num_updates), 1)

    def test_ema_has_no_trainable_params_and_stays_eval(self):
        net = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2))
        ema = gen_base.EMAModel(net)
        self.assertFalse(any(p.requires_grad for p in ema.parameters()))
        ema.train()
        self.assertFalse(ema.ema_model.training)

    def test_warmup_decay_starts_small(self):
        ema = gen_base.EMAModel(torch.nn.Linear(1, 1), decay=0.999, warmup=True)
        self.assertLess(ema.current_decay(), 0.2)


class TestSamplingInterface(TimedTestCase):

    def test_sample_shape_range_and_mode_restored(self):
        model = _ToyGenerative()
        model.train()
        images = model.sample(n=5, batch_size=2)
        self.assertEqual(tuple(images.shape), (5, 3, 4, 4))
        self.assertLessEqual(float(images.max()), 1.0)
        self.assertGreaterEqual(float(images.min()), -1.0)
        self.assertTrue(model.training)

    def test_sample_is_reproducible_with_seeded_generator(self):
        model = _ToyGenerative()
        a = model.sample(n=3, generator=torch.Generator().manual_seed(7))
        b = model.sample(n=3, generator=torch.Generator().manual_seed(7))
        self.assertTrue(torch.equal(a, b))

    def test_sample_requires_n_or_noise(self):
        with self.assertRaises(ValueError):
            _ToyGenerative().sample()


if __name__ == "__main__":
    unittest.main()
