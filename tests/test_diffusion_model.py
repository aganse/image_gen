"""
Tests of the diffusers-based diffusion module: component lookup by class
name, a training step, and sampling.  CPU only, with a tiny UNet.
"""

import unittest

import torch

from diffusion_model import DiffusionLitModule, build_hf_component
from tests._helpers import TimedTestCase

TINY_UNET = {
    "layers_per_block": 1,
    "block_out_channels": [32, 64],
    "down_block_types": ["DownBlock2D", "AttnDownBlock2D"],
    "up_block_types": ["AttnUpBlock2D", "UpBlock2D"],
}


def _make(**kwargs):
    defaults = dict(
        image_shape=(16, 16, 3),
        hf_model_config=dict(TINY_UNET),
        num_inference_steps=3,
        ema_decay=0.99,
    )
    defaults.update(kwargs)
    return DiffusionLitModule(**defaults)


class TestBuildHFComponent(TimedTestCase):

    def test_unknown_class_raises(self):
        with self.assertRaises(ValueError):
            build_hf_component("NoSuchDiffusersClass")

    def test_build_schedulers_by_name(self):
        ddpm = build_hf_component("DDPMScheduler", {"num_train_timesteps": 100})
        self.assertEqual(ddpm.config.num_train_timesteps, 100)
        ddim = build_hf_component("DDIMScheduler", {"num_train_timesteps": 100})
        self.assertEqual(type(ddim).__name__, "DDIMScheduler")

    def test_build_autoencoder_kl_by_name(self):
        # Demonstrates swapping in another diffusers model class by name.
        vae = build_hf_component(
            "AutoencoderKL",
            {
                "in_channels": 3,
                "out_channels": 3,
                "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D"],
                "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D"],
                "block_out_channels": [32, 64],
                "latent_channels": 4,
            },
        )
        x = torch.zeros(1, 3, 16, 16)
        latents = vae.encode(x).latent_dist.mode()
        self.assertEqual(latents.shape[1], 4)
        self.assertEqual(tuple(vae.decode(latents).sample.shape), (1, 3, 16, 16))


class TestDiffusionLitModule(TimedTestCase):

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.model = _make()
        cls.x = torch.rand(2, 3, 16, 16) * 2 - 1

    def test_denoiser_built_from_image_shape(self):
        config = self.model.denoiser.config
        self.assertEqual(config.sample_size, 16)
        self.assertEqual(config.in_channels, 3)
        self.assertEqual(config.out_channels, 3)

    def test_forward_shape(self):
        timesteps = torch.tensor([10, 500])
        out = self.model(self.x, timesteps)
        self.assertEqual(out.shape, self.x.shape)

    def test_invalid_sampler_name_raises(self):
        with self.assertRaises(ValueError):
            _make(sample_scheduler_class="NoSuchScheduler")

    def test_train_step_and_ema_update(self):
        model = _make()
        optimizer = model.configure_optimizers()
        noise = torch.randn_like(self.x)
        timesteps = torch.randint(0, 1000, (2,))
        noisy = model.train_scheduler.add_noise(self.x, noise, timesteps)
        loss = torch.nn.functional.mse_loss(model(noisy, timesteps), noise)
        self.assertTrue(torch.isfinite(loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.on_train_batch_end(None, self.x, 0)
        self.assertEqual(int(model.ema.num_updates), 1)

    def test_sample_ddim(self):
        images = self.model.sample(n=2)
        self.assertEqual(tuple(images.shape), (2, 3, 16, 16))
        self.assertTrue(torch.isfinite(images).all())

    def test_sample_is_deterministic_for_fixed_noise(self):
        noise = self.model.make_noise(2, generator=torch.Generator().manual_seed(3))
        a = self.model.sample(noise=noise)
        b = self.model.sample(noise=noise)
        self.assertTrue(torch.allclose(a, b, atol=1e-5))

    def test_swap_sampler_by_name(self):
        model = _make(sample_scheduler_class="DDPMScheduler", num_inference_steps=2)
        self.assertEqual(tuple(model.sample(n=1).shape), (1, 3, 16, 16))

    def test_no_ema(self):
        model = _make(ema_decay=0)
        self.assertIsNone(model.ema)
        self.assertEqual(tuple(model.sample(n=1).shape), (1, 3, 16, 16))


if __name__ == "__main__":
    unittest.main()
