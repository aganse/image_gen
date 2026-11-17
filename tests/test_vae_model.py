"""
Tests of the VAE networks and LightningModule.  CPU only, tiny shapes; the
train-step test exercises the forward + backward path and optimizer update.
"""

import unittest

import torch

from tests._helpers import TimedTestCase
from vae_model import ConvVAE, VAELitModule, vae_loss_terms

SMALL = dict(image_shape=(16, 16, 3), latent_dim=8, base_channels=4, channel_mults=(1, 2))


class TestConvVAE(TimedTestCase):

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.net = ConvVAE(**SMALL)
        cls.x = torch.rand(4, 3, 16, 16) * 2 - 1

    def test_forward_shapes(self):
        recon, mu, logvar = self.net(self.x)
        self.assertEqual(recon.shape, self.x.shape)
        self.assertEqual(tuple(mu.shape), (4, 8))
        self.assertEqual(tuple(logvar.shape), (4, 8))

    def test_decode_output_range(self):
        out = self.net.decode(torch.randn(3, 8))
        self.assertEqual(tuple(out.shape), (3, 3, 16, 16))
        self.assertLessEqual(float(out.abs().max()), 1.0)

    def test_non_square_images(self):
        net = ConvVAE(image_shape=(16, 32, 3), latent_dim=8, base_channels=4,
                      channel_mults=(1, 2))
        recon, _, _ = net(torch.zeros(2, 3, 16, 32))
        self.assertEqual(tuple(recon.shape), (2, 3, 16, 32))

    def test_indivisible_size_raises(self):
        with self.assertRaises(ValueError):
            ConvVAE(image_shape=(18, 18, 3), channel_mults=(1, 2, 4))

    def test_loss_terms(self):
        mu = torch.zeros(4, 8)
        logvar = torch.zeros(4, 8)
        recon_term, kl_term = vae_loss_terms(self.x, self.x, mu, logvar)
        self.assertAlmostEqual(float(recon_term), 0.0)
        self.assertAlmostEqual(float(kl_term), 0.0)  # q(z|x) == N(0, I)
        _, kl_term = vae_loss_terms(self.x, self.x, mu + 1.0, logvar)
        self.assertAlmostEqual(float(kl_term), 4.0, places=5)  # 0.5 * 8 dims
        with self.assertRaises(ValueError):
            vae_loss_terms(self.x, self.x, mu, logvar, recon_loss="huber")


class TestVAELitModule(TimedTestCase):

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.model = VAELitModule(**SMALL, learning_rate=1e-3, kl_warmup_epochs=0)
        cls.x = torch.rand(4, 3, 16, 16) * 2 - 1

    def test_trainable_parameters_nonempty(self):
        self.assertGreater(sum(p.numel() for p in self.model.parameters()), 0)

    def test_sample_interface(self):
        images = self.model.sample(n=5, batch_size=2)
        self.assertEqual(tuple(images.shape), (5, 3, 16, 16))
        self.assertLessEqual(float(images.abs().max()), 1.0)

    def test_reconstruct(self):
        recon = self.model.reconstruct(self.x)
        self.assertEqual(recon.shape, self.x.shape)

    def test_configure_optimizers_constant_and_schedule(self):
        self.assertIsInstance(self.model.configure_optimizers(), torch.optim.Optimizer)
        scheduled = VAELitModule(**SMALL, learning_rate=[1e-3, 10, 0.9])
        self.assertIn("lr_scheduler", scheduled.configure_optimizers())

    def test_train_step_reduces_loss(self):
        model = VAELitModule(**SMALL, learning_rate=1e-2)
        optimizer = model.configure_optimizers()
        losses = []
        for _ in range(20):
            recon, mu, logvar = model.vae(self.x)
            recon_term, kl_term = vae_loss_terms(recon, self.x, mu, logvar)
            loss = recon_term + kl_term
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss))
        self.assertTrue(all(torch.isfinite(torch.tensor(losses))))
        self.assertLess(losses[-1], losses[0])


if __name__ == "__main__":
    unittest.main()
