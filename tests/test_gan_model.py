"""
Tests of the GAN networks, losses and LightningModule for both the DCGAN
("bce") and WGAN-GP ("wgan-gp") configurations.  CPU only, tiny shapes.
"""

import unittest

import torch

from gan_model import Discriminator, GANLitModule, Generator, gradient_penalty
from tests._helpers import TimedTestCase

SMALL_ARCH = dict(image_shape=(16, 16, 3), latent_dim=8, gen_base_channels=4,
                  disc_base_channels=4)


class TestNetworks(TimedTestCase):

    def test_generator_output(self):
        for size in (16, 32, 64):
            gen = Generator((size, size, 3), latent_dim=8, base_channels=4)
            out = gen(torch.randn(2, 8))
            self.assertEqual(tuple(out.shape), (2, 3, size, size), msg=size)
            self.assertLessEqual(float(out.abs().max()), 1.0)

    def test_discriminator_output(self):
        for norm in ("batch", "instance", "none"):
            for size in (16, 64):
                disc = Discriminator((size, size, 3), base_channels=4, norm=norm)
                out = disc(torch.randn(3, 3, size, size))
                self.assertEqual(tuple(out.shape), (3,), msg=f"{norm} {size}")

    def test_invalid_sizes_and_norm_raise(self):
        with self.assertRaises(ValueError):
            Generator((48, 48, 3))  # not a power of two
        with self.assertRaises(ValueError):
            Generator((32, 64, 3))  # not square
        with self.assertRaises(ValueError):
            Discriminator((16, 16, 3), norm="group")

    def test_gradient_penalty_is_differentiable_scalar(self):
        disc = Discriminator((16, 16, 3), base_channels=4, norm="none")
        real = torch.randn(4, 3, 16, 16)
        fake = torch.randn(4, 3, 16, 16)
        penalty = gradient_penalty(disc, real, fake)
        self.assertEqual(penalty.dim(), 0)
        self.assertGreaterEqual(float(penalty), 0.0)
        penalty.backward()  # needs second-order gradients through the critic
        grads = [p.grad for p in disc.parameters() if p.grad is not None]
        self.assertGreater(len(grads), 0)


class TestGANLitModule(TimedTestCase):

    def setUp(self):
        super().setUp()
        torch.manual_seed(0)
        self.real = torch.rand(4, 3, 16, 16) * 2 - 1

    def _make(self, **kwargs):
        return GANLitModule(**SMALL_ARCH, **kwargs)

    def test_invalid_configs_raise(self):
        with self.assertRaises(ValueError):
            self._make(gan_loss="hinge")
        with self.assertRaises(ValueError):
            self._make(gan_loss="wgan-gp", disc_norm="batch")
        with self.assertRaises(ValueError):
            self._make(learning_rate=[2e-4, 100, 0.9])

    def test_two_optimizers_and_manual_optimization(self):
        model = self._make()
        self.assertFalse(model.automatic_optimization)
        self.assertEqual(len(model.configure_optimizers()), 2)

    def test_losses_bce(self):
        model = self._make(gan_loss="bce", disc_norm="batch")
        fake = model.generator(torch.randn(4, 8))
        d_loss, extras = model.discriminator_loss(self.real, fake.detach())
        g_loss = model.generator_loss(fake)
        self.assertTrue(torch.isfinite(d_loss))
        self.assertTrue(torch.isfinite(g_loss))
        self.assertIn("d_real", extras)
        d_loss.backward()
        g_loss.backward()

    def test_losses_wgan_gp(self):
        model = self._make(gan_loss="wgan-gp", disc_norm="instance",
                           adam_betas=(0.0, 0.9), n_critic=5)
        fake = model.generator(torch.randn(4, 8))
        d_loss, extras = model.discriminator_loss(self.real, fake.detach())
        g_loss = model.generator_loss(fake)
        self.assertIn("wasserstein_est", extras)
        self.assertIn("grad_penalty", extras)
        d_loss.backward()
        g_loss.backward()
        self.assertTrue(torch.isfinite(d_loss))

    def test_sample_uses_ema_generator(self):
        model = self._make(ema_decay=0.5)
        self.assertIsNotNone(model.ema)
        images = model.sample(n=3)
        self.assertEqual(tuple(images.shape), (3, 3, 16, 16))
        no_ema = self._make(ema_decay=0)
        self.assertIsNone(no_ema.ema)
        self.assertEqual(tuple(no_ema.sample(n=2).shape), (2, 3, 16, 16))

    def test_diffaugment_can_be_disabled(self):
        model = self._make(diffaugment="")
        self.assertTrue(torch.equal(model.augment(self.real), self.real))


if __name__ == "__main__":
    unittest.main()
