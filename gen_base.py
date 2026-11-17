"""Shared base pieces for the generative Lightning modules in this repo.

Every model module (vae_model.py, gan_model.py, diffusion_model.py) exposes a
LightningModule derived from GenerativeLitModule, so that training scripts,
the periodic evaluation callback, and the final-metrics code can treat VAEs,
GANs and diffusion models identically:

    noise = model.make_noise(16)            # CPU tensor of "latents"
    images = model.sample(noise=noise)      # CPU float tensor in [-1, 1]

Images are always channels-first float tensors scaled to [-1, 1] inside the
models; the user-facing `image_shape` parameter stays (height, width,
channels) to match the flow_models repo convention.
"""

import copy

import lightning as L
import torch


def best_device():
    """Return 'cuda' when a GPU is visible, otherwise 'cpu'."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def chw_from_image_shape(image_shape):
    """Convert an (H, W, C) image_shape param into a (C, H, W) tuple."""
    if len(image_shape) != 3:
        raise ValueError(
            f"image_shape must be (height, width, channels); got {image_shape!r}"
        )
    height, width, channels = (int(v) for v in image_shape)
    return channels, height, width


# ---------------------------------------------------------------------------
# Learning-rate handling (same convention as flow_models)
# ---------------------------------------------------------------------------

def initial_learning_rate(learning_rate):
    """Return the starting learning rate from a float or [init, steps, rate]."""
    if isinstance(learning_rate, (int, float)) and not isinstance(learning_rate, bool):
        return float(learning_rate)
    if isinstance(learning_rate, (list, tuple)) and len(learning_rate) == 3:
        return float(learning_rate[0])
    raise ValueError(
        "learning_rate must be a float or a list of length 3 "
        "[initial_rate, decay_steps, decay_rate]."
    )


def exp_decay_lambda(decay_steps, decay_rate):
    """Staircase exponential decay multiplier: decay_rate ** (step // decay_steps)."""
    decay_steps = int(decay_steps)
    if decay_steps <= 0:
        raise ValueError("decay_steps must be a positive integer.")

    def _lambda(step):
        return float(decay_rate) ** (step // decay_steps)

    return _lambda


def build_lr_scheduler(optimizer, learning_rate):
    """Return a per-step LambdaLR for a 3-element learning_rate, else None.

    learning_rate = float                      -> constant rate, returns None
    learning_rate = [init, decay_steps, rate]  -> staircase exponential decay,
        decayed_lr = init * rate ^ floor(step / decay_steps)
    """
    initial_learning_rate(learning_rate)  # validates the format
    if isinstance(learning_rate, (list, tuple)):
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, exp_decay_lambda(learning_rate[1], learning_rate[2])
        )
    return None


def optimizer_config(optimizer, learning_rate):
    """Package an optimizer (+ optional per-step scheduler) for Lightning."""
    scheduler = build_lr_scheduler(optimizer, learning_rate)
    if scheduler is None:
        return optimizer
    return {
        "optimizer": optimizer,
        "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
    }


# ---------------------------------------------------------------------------
# Exponential moving average of model weights
# ---------------------------------------------------------------------------

class EMAModel(torch.nn.Module):
    """Exponential moving average (EMA) copy of a network.

    Sampling from EMA weights rather than the raw training weights is standard
    practice for GAN generators and diffusion denoisers; it noticeably improves
    sample quality at no training cost.  Being an nn.Module, the EMA copy moves
    with the parent module across devices and is saved in its state_dict.
    """

    def __init__(self, model, decay=0.999, warmup=True):
        super().__init__()
        self.ema_model = copy.deepcopy(model)
        self.ema_model.requires_grad_(False)
        self.ema_model.eval()
        self.decay = float(decay)
        self.warmup = bool(warmup)
        self.register_buffer("num_updates", torch.zeros((), dtype=torch.long))

    def train(self, mode=True):
        # The EMA copy is only ever used for inference; keep it in eval mode
        # even when the parent LightningModule is switched to train mode.
        super().train(mode)
        self.ema_model.eval()
        return self

    def current_decay(self):
        if not self.warmup:
            return self.decay
        n = float(self.num_updates.item())
        return min(self.decay, (1.0 + n) / (10.0 + n))

    @torch.no_grad()
    def update(self, model):
        """Blend the live model's weights into the EMA copy (call once/step)."""
        self.num_updates += 1
        decay = self.current_decay()
        for p_ema, p in zip(self.ema_model.parameters(), model.parameters()):
            p_ema.lerp_(p.detach(), 1.0 - decay)
        # Buffers (e.g. BatchNorm running stats) are copied, not averaged.
        for b_ema, b in zip(self.ema_model.buffers(), model.buffers()):
            b_ema.copy_(b)

    def forward(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)


# ---------------------------------------------------------------------------
# Common LightningModule interface
# ---------------------------------------------------------------------------

class GenerativeLitModule(L.LightningModule):
    """Base class giving all generative models the same sampling interface.

    Subclasses implement:
        make_noise(n, generator=None)  -> CPU tensor of model inputs/latents
        sample_from_noise(noise)       -> images in [-1, 1] on self.device
    """

    def make_noise(self, n, generator=None):
        raise NotImplementedError

    def sample_from_noise(self, noise):
        raise NotImplementedError

    @torch.no_grad()
    def sample(self, n=None, noise=None, batch_size=64, generator=None):
        """Generate images; returns a CPU float tensor (N, C, H, W) in [-1, 1].

        Pass either `n` (fresh noise, optionally from a seeded torch.Generator
        for reproducibility) or an explicit `noise` tensor (e.g. fixed noise
        reused across epochs to visualize training progression).
        """
        if noise is None:
            if n is None:
                raise ValueError("sample() needs either n or noise.")
            noise = self.make_noise(int(n), generator=generator)
        was_training = self.training
        self.eval()
        outputs = []
        try:
            for start in range(0, noise.shape[0], int(batch_size)):
                chunk = noise[start:start + int(batch_size)].to(self.device)
                images = self.sample_from_noise(chunk)
                outputs.append(images.detach().float().clamp(-1.0, 1.0).cpu())
        finally:
            self.train(was_training)
        return torch.cat(outputs, dim=0)
