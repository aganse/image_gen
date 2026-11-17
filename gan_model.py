"""Generative adversarial networks (GANs): DCGAN and WGAN-GP in one module.

A GAN (Goodfellow et al. 2014) trains two networks against each other: a
generator G mapping latent noise z ~ N(0, I) to images, and a discriminator D
trying to tell real images from generated ones.  Two classic variants share
the networks defined here and differ only in loss and a few settings:

    gan_loss="bce"      DCGAN (Radford et al. 2016): D outputs a real/fake
                        logit trained with binary cross-entropy; G uses the
                        "non-saturating" loss.  BatchNorm in D, Adam betas
                        (0.5, 0.999), one D step per G step.
    gan_loss="wgan-gp"  WGAN-GP (Gulrajani et al. 2017): D is a "critic"
                        estimating the Wasserstein distance, kept ~1-Lipschitz
                        by a gradient penalty.  No BatchNorm in the critic
                        (use disc_norm "instance" or "none"), Adam betas
                        (0.0, 0.9), n_critic=5 critic steps per G step.

Both variants optionally use DiffAugment (see augment.py), which is what makes
GAN training viable on only ~5000 images, and an EMA copy of the generator
for sampling.

Classes:
    Generator, Discriminator  plain torch.nn.Modules (DCGAN-style conv nets)
    GANLitModule              LightningModule with the adversarial training loop
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from augment import DiffAugment
from gen_base import (
    EMAModel,
    GenerativeLitModule,
    chw_from_image_shape,
    initial_learning_rate,
)

MAX_CHANNEL_MULT = 8  # widest layers have base_channels * 8 feature maps


def _n_resample_stages(image_shape):
    """Number of stride-2 stages between a 4x4 feature map and the image."""
    _, height, width = chw_from_image_shape(image_shape)
    if height != width:
        raise ValueError(f"GAN networks need square images; got {height}x{width}.")
    n_stages = int(round(math.log2(height))) - 2
    if 2 ** (n_stages + 2) != height or n_stages < 2:
        raise ValueError(
            f"GAN image size must be a power of two >= 16; got {height}."
        )
    return n_stages


def _init_weights(module):
    """DCGAN weight init: N(0, 0.02) convs, N(1, 0.02) norm scales."""
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d)):
        if module.weight is not None:
            nn.init.normal_(module.weight, 1.0, 0.02)
            nn.init.zeros_(module.bias)


class Generator(nn.Module):
    """DCGAN generator: latent vector -> 4x4 feature map -> upsample to image.

    For 64x64 images with base_channels=64 the feature maps are
    4x4(512) -> 8x8(256) -> 16x16(128) -> 32x32(64) -> 64x64(3).
    """

    def __init__(self, image_shape=(64, 64, 3), latent_dim=128, base_channels=64):
        super().__init__()
        channels, _, _ = chw_from_image_shape(image_shape)
        n_stages = _n_resample_stages(image_shape)
        self.latent_dim = int(latent_dim)
        widths = [
            base_channels * min(2 ** (n_stages - 1 - i), MAX_CHANNEL_MULT)
            for i in range(n_stages)
        ]
        layers = [
            nn.ConvTranspose2d(self.latent_dim, widths[0], 4, 1, 0, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
        ]
        for in_ch, out_ch in zip(widths[:-1], widths[1:]):
            layers += [
                nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
        layers += [
            nn.ConvTranspose2d(widths[-1], channels, 4, 2, 1),
            nn.Tanh(),  # images are scaled to [-1, 1]
        ]
        self.net = nn.Sequential(*layers)
        self.apply(_init_weights)

    def forward(self, z):
        return self.net(z.view(z.size(0), self.latent_dim, 1, 1))


class Discriminator(nn.Module):
    """DCGAN discriminator / WGAN critic: image -> single unbounded score.

    norm: "batch" (DCGAN), "instance" or "none" (WGAN-GP; the gradient penalty
    is computed per sample, which BatchNorm's cross-sample coupling violates).
    """

    def __init__(self, image_shape=(64, 64, 3), base_channels=64, norm="batch"):
        super().__init__()
        channels, _, _ = chw_from_image_shape(image_shape)
        n_stages = _n_resample_stages(image_shape)
        if norm not in ("batch", "instance", "none"):
            raise ValueError(
                f"disc_norm must be 'batch', 'instance' or 'none'; got {norm!r}"
            )
        widths = [
            base_channels * min(2 ** i, MAX_CHANNEL_MULT) for i in range(n_stages)
        ]
        # No normalization on the first layer (standard DCGAN practice).
        layers = [
            nn.Conv2d(channels, widths[0], 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        for in_ch, out_ch in zip(widths[:-1], widths[1:]):
            layers.append(nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=(norm == "none")))
            if norm == "batch":
                layers.append(nn.BatchNorm2d(out_ch))
            elif norm == "instance":
                layers.append(nn.InstanceNorm2d(out_ch, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Conv2d(widths[-1], 1, 4, 1, 0))  # 4x4 -> 1x1 score
        self.net = nn.Sequential(*layers)
        self.apply(_init_weights)

    def forward(self, x):
        return self.net(x).flatten(start_dim=1).squeeze(1)


def gradient_penalty(discriminator, real, fake):
    """WGAN-GP penalty: E[(||grad_x D(x_hat)||_2 - 1)^2] on real/fake blends."""
    eps = torch.rand(real.size(0), 1, 1, 1, device=real.device, dtype=real.dtype)
    x_hat = (eps * real + (1.0 - eps) * fake).detach().requires_grad_(True)
    scores = discriminator(x_hat)
    grads = torch.autograd.grad(
        outputs=scores.sum(), inputs=x_hat, create_graph=True
    )[0]
    grad_norm = grads.flatten(start_dim=1).norm(2, dim=1)
    return ((grad_norm - 1.0) ** 2).mean()


class GANLitModule(GenerativeLitModule):
    """LightningModule training Generator vs Discriminator.

    Architecture args (image_shape, latent_dim, gen_base_channels,
    disc_base_channels, disc_norm) come from model_arch_params; the rest from
    training_params.

    GAN training alternates two optimizers, so this module uses Lightning's
    manual optimization.  One training_step = one discriminator update, plus
    a generator update every n_critic-th step.

    Note the logged losses (d_loss, g_loss) do NOT indicate sample quality the
    way a VAE or diffusion loss does - watch the periodic KID metric and the
    sample grids in MLflow instead.  For WGAN-GP, -d_loss (minus the penalty,
    logged as wasserstein_est) approximates the Wasserstein distance and
    should trend toward zero.
    """

    def __init__(
        self,
        image_shape=(64, 64, 3),
        latent_dim=128,
        gen_base_channels=64,
        disc_base_channels=64,
        disc_norm="batch",
        gan_loss="bce",
        learning_rate=2e-4,
        disc_learning_rate=None,
        adam_betas=(0.5, 0.999),
        n_critic=1,
        gp_weight=10.0,
        diffaugment="color,translation,cutout",
        ema_decay=0.999,
        grad_norm_thresh=None,
    ):
        super().__init__()
        if gan_loss not in ("bce", "wgan-gp"):
            raise ValueError(f"gan_loss must be 'bce' or 'wgan-gp'; got {gan_loss!r}")
        if gan_loss == "wgan-gp" and disc_norm == "batch":
            raise ValueError(
                "WGAN-GP's gradient penalty is incompatible with BatchNorm in "
                "the critic; set disc_norm to 'instance' or 'none'."
            )
        for name, lr in (("learning_rate", learning_rate),
                         ("disc_learning_rate", disc_learning_rate)):
            if lr is not None and isinstance(lr, (list, tuple)):
                raise ValueError(
                    f"{name} must be a float for GAN training (learning-rate "
                    "schedules are not supported with the two-optimizer loop)."
                )
        self.automatic_optimization = False

        self.generator = Generator(image_shape, latent_dim, gen_base_channels)
        self.discriminator = Discriminator(image_shape, disc_base_channels, disc_norm)
        self.augment = DiffAugment(diffaugment)
        self.ema = (
            EMAModel(self.generator, decay=ema_decay)
            if ema_decay and ema_decay > 0 else None
        )

        self.latent_dim = int(latent_dim)
        self.gan_loss = gan_loss
        self.learning_rate = initial_learning_rate(learning_rate)
        self.disc_learning_rate = (
            self.learning_rate if disc_learning_rate is None
            else initial_learning_rate(disc_learning_rate)
        )
        self.adam_betas = (float(adam_betas[0]), float(adam_betas[1]))
        self.n_critic = max(1, int(n_critic))
        self.gp_weight = float(gp_weight)
        self.grad_norm_thresh = grad_norm_thresh

    def forward(self, z):
        return self.generator(z)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=self.learning_rate, betas=self.adam_betas
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.disc_learning_rate,
            betas=self.adam_betas,
        )
        return [opt_g, opt_d]

    # -- losses -------------------------------------------------------------

    def discriminator_loss(self, real, fake):
        """Return (loss, dict of extra scalars to log) for the D/critic step."""
        real_aug = self.augment(real)
        fake_aug = self.augment(fake)
        d_real = self.discriminator(real_aug)
        d_fake = self.discriminator(fake_aug)
        if self.gan_loss == "bce":
            loss = (
                F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real))
                + F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
            )
            extras = {}
        else:
            wasserstein = d_real.mean() - d_fake.mean()
            penalty = gradient_penalty(self.discriminator, real_aug, fake_aug)
            loss = -wasserstein + self.gp_weight * penalty
            extras = {"wasserstein_est": wasserstein, "grad_penalty": penalty}
        extras["d_real"] = d_real.mean()
        extras["d_fake"] = d_fake.mean()
        return loss, extras

    def generator_loss(self, fake):
        d_fake = self.discriminator(self.augment(fake))
        if self.gan_loss == "bce":
            # Non-saturating loss: G tries to make D label its fakes as real.
            return F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))
        return -d_fake.mean()

    # -- training loop ------------------------------------------------------

    def _clip(self, optimizer):
        if self.grad_norm_thresh:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=float(self.grad_norm_thresh),
                gradient_clip_algorithm="norm",
            )

    def training_step(self, batch, batch_idx):
        real = batch
        batch_size = real.size(0)
        opt_g, opt_d = self.optimizers()
        log_kwargs = dict(on_step=False, on_epoch=True, batch_size=batch_size)

        # --- discriminator / critic update ---
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        with torch.no_grad():
            fake = self.generator(z)
        d_loss, extras = self.discriminator_loss(real, fake)
        opt_d.zero_grad()
        self.manual_backward(d_loss)
        self._clip(opt_d)
        opt_d.step()
        self.log("d_loss", d_loss, prog_bar=True, **log_kwargs)
        for name, value in extras.items():
            self.log(name, value, **log_kwargs)

        # --- generator update (every n_critic-th step) ---
        if (batch_idx + 1) % self.n_critic == 0:
            z = torch.randn(batch_size, self.latent_dim, device=self.device)
            g_loss = self.generator_loss(self.generator(z))
            opt_g.zero_grad()
            self.manual_backward(g_loss)
            self._clip(opt_g)
            opt_g.step()
            if self.ema is not None:
                self.ema.update(self.generator)
            self.log("g_loss", g_loss, prog_bar=True, **log_kwargs)

    # -- sampling interface -------------------------------------------------

    def make_noise(self, n, generator=None):
        return torch.randn(n, self.latent_dim, generator=generator)

    def sample_from_noise(self, noise):
        net = self.ema.ema_model if self.ema is not None else self.generator
        return net(noise)
