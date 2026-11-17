"""Differentiable augmentation for GAN training on small datasets.

With only ~5000 training images a GAN discriminator quickly memorizes the
training set, its feedback to the generator stops being useful, and training
collapses.  The standard remedy (Zhao et al. 2020 "Differentiable Augmentation
for Data-Efficient GAN Training"; Karras et al. 2020 "Training GANs with
Limited Data") is to augment *both* real and generated images before they
reach the discriminator, using augmentations that are differentiable so the
generator still receives gradients through them.

The three augmentation families from the DiffAugment paper are implemented:
    color        random brightness, saturation, and contrast
    translation  random integer-pixel shift (zero padded), up to 1/8 of size
    cutout       random square (half image size) set to zero

Usage:
    augment = DiffAugment("color,translation,cutout")
    d_real = discriminator(augment(real_images))
    d_fake = discriminator(augment(fake_images))
"""

import torch
import torch.nn.functional as F


def rand_brightness(x):
    shift = torch.rand(x.size(0), 1, 1, 1, device=x.device, dtype=x.dtype) - 0.5
    return x + shift


def rand_saturation(x):
    mean = x.mean(dim=1, keepdim=True)
    scale = torch.rand(x.size(0), 1, 1, 1, device=x.device, dtype=x.dtype) * 2.0
    return (x - mean) * scale + mean


def rand_contrast(x):
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    scale = torch.rand(x.size(0), 1, 1, 1, device=x.device, dtype=x.dtype) + 0.5
    return (x - mean) * scale + mean


def rand_translation(x, ratio=0.125):
    """Shift each image by a random whole number of pixels, zero padded."""
    batch, _, height, width = x.shape
    max_dy = int(round(height * ratio))
    max_dx = int(round(width * ratio))
    dy = torch.randint(-max_dy, max_dy + 1, (batch,), device=x.device)
    dx = torch.randint(-max_dx, max_dx + 1, (batch,), device=x.device)
    theta = torch.zeros(batch, 2, 3, device=x.device, dtype=x.dtype)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    # Normalized grid coordinates span [-1, 1], so one pixel = 2 / size.
    theta[:, 0, 2] = dx.to(x.dtype) * 2.0 / width
    theta[:, 1, 2] = dy.to(x.dtype) * 2.0 / height
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(
        x, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )


def rand_cutout(x, ratio=0.5):
    """Zero out one random rectangle (ratio * image size) per image."""
    batch, _, height, width = x.shape
    cut_h = int(round(height * ratio))
    cut_w = int(round(width * ratio))
    center_y = torch.randint(0, height, (batch, 1, 1), device=x.device)
    center_x = torch.randint(0, width, (batch, 1, 1), device=x.device)
    ys = torch.arange(height, device=x.device).view(1, height, 1)
    xs = torch.arange(width, device=x.device).view(1, 1, width)
    inside = (
        (ys >= center_y - cut_h // 2) & (ys < center_y - cut_h // 2 + cut_h)
        & (xs >= center_x - cut_w // 2) & (xs < center_x - cut_w // 2 + cut_w)
    )
    mask = (~inside).to(x.dtype).unsqueeze(1)
    return x * mask


AUGMENT_FNS = {
    "color": [rand_brightness, rand_saturation, rand_contrast],
    "translation": [rand_translation],
    "cutout": [rand_cutout],
}


def parse_policy(policy):
    """Turn 'color,translation' (or None / '' / 'none') into a list of names."""
    if policy is None:
        return []
    if isinstance(policy, (list, tuple)):
        names = [str(p).strip().lower() for p in policy]
    else:
        names = [p.strip().lower() for p in str(policy).split(",")]
    names = [n for n in names if n and n != "none"]
    unknown = [n for n in names if n not in AUGMENT_FNS]
    if unknown:
        raise ValueError(
            f"Unknown DiffAugment policy entries {unknown}; "
            f"valid entries are {sorted(AUGMENT_FNS)}."
        )
    return names


class DiffAugment(torch.nn.Module):
    """Apply a comma-separated policy of differentiable augmentations.

    An empty policy ('' / None / 'none') makes this an identity module, so the
    GAN code can call it unconditionally.
    """

    def __init__(self, policy="color,translation,cutout"):
        super().__init__()
        self.policy = parse_policy(policy)

    def forward(self, x):
        for name in self.policy:
            for fn in AUGMENT_FNS[name]:
                x = fn(x)
        return x

    def extra_repr(self):
        return f"policy={','.join(self.policy) or 'none'}"
