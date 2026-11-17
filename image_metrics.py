"""Image-quality metrics for generated images: FID, KID and Inception Score.

All three compare images through the eyes of an ImageNet-trained Inception-v3
network (via torchmetrics / torch-fidelity), so the same numbers can be
computed for any generator - the VAE, GANs and diffusion model here, or images
from another project entirely (see eval_mlflow_run.py).

    FID  Frechet Inception Distance (lower is better).  Distance between
         Gaussians fit to Inception features of real vs generated images.
         The standard headline metric, but biased upward for small sample
         counts - only compare FIDs computed with the same number of images.
    KID  Kernel Inception Distance (lower is better).  An unbiased estimator
         using the same features, so it is the more trustworthy metric with
         only a few thousand real images, and cheap enough to track as a
         time series during training.  Reported as mean and std over subsets.
    IS   Inception Score (higher is better).  Rewards images that are
         individually classifiable and collectively diverse across ImageNet
         classes.  Uses no real images.  On a single-category dataset (e.g.
         only cats) the diversity term is nearly constant, so treat IS as a
         rough sharpness indicator there, not a quality ranking.

The Inception weights are downloaded on first use and cached under TORCH_HOME.
Cloud training jobs in this repo have no internet access, so the Dockerfile
pre-downloads ("bakes") them into the image at build time.

Images passed in are uint8 tensors of shape (N, 3, H, W) with values 0-255;
use to_uint8() to convert model outputs in [-1, 1].
"""

import torch

ALL_METRICS = ("fid", "kid", "is")


def to_uint8(images):
    """Convert float images in [-1, 1] to uint8 in [0, 255]."""
    if images.dtype == torch.uint8:
        return images
    return ((images.clamp(-1.0, 1.0) + 1.0) * 127.5).round().to(torch.uint8)


class GenerativeImageMetrics:
    """Compute FID / KID / IS of generated images against a fixed real set.

    The real images' Inception statistics are computed once (lazily, per
    metric) and reused across calls, so repeated evaluations during training
    only pay for the generated images.

    Args:
        real_images: uint8 tensor (N, 3, H, W) of reference images
        device: device for the Inception network ("cuda" strongly advised)
        kid_subset_size: KID subset size; automatically reduced to the number
            of real/generated images available if that is smaller
        kid_subsets: number of random subsets KID averages over
        batch_size: images per Inception forward pass
    """

    def __init__(
        self,
        real_images,
        device="cpu",
        kid_subset_size=1000,
        kid_subsets=100,
        batch_size=100,
    ):
        if real_images.dtype != torch.uint8:
            raise ValueError("real_images must be a uint8 tensor (use to_uint8).")
        self.real_images = real_images
        self.device = torch.device(device)
        self.kid_subset_size = int(kid_subset_size)
        self.kid_subsets = int(kid_subsets)
        self.batch_size = int(batch_size)
        self._fid = None
        self._kid = None
        self._inception_score = None

    def to(self, device):
        """Move the (already-built) metric networks to another device."""
        self.device = torch.device(device)
        for metric in (self._fid, self._kid, self._inception_score):
            if metric is not None:
                metric.to(self.device)
        return self

    def _feed(self, metric, images, **kwargs):
        for start in range(0, images.shape[0], self.batch_size):
            batch = images[start:start + self.batch_size].to(self.device)
            metric.update(batch, **kwargs)

    def _get_fid(self):
        if self._fid is None:
            from torchmetrics.image.fid import FrechetInceptionDistance
            self._fid = FrechetInceptionDistance(
                feature=2048, reset_real_features=False
            ).to(self.device)
            self._feed(self._fid, self.real_images, real=True)
        return self._fid

    def _get_kid(self):
        if self._kid is None:
            from torchmetrics.image.kid import KernelInceptionDistance
            self._kid = KernelInceptionDistance(
                feature=2048,
                subsets=self.kid_subsets,
                subset_size=min(self.kid_subset_size, self.real_images.shape[0]),
                reset_real_features=False,
            ).to(self.device)
            self._feed(self._kid, self.real_images, real=True)
        return self._kid

    def _get_inception_score(self):
        if self._inception_score is None:
            from torchmetrics.image.inception import InceptionScore
            self._inception_score = InceptionScore().to(self.device)
        return self._inception_score

    @torch.no_grad()
    def evaluate(self, fake_images, metrics=ALL_METRICS):
        """Return a dict of metric values for a uint8 batch of generated images.

        Keys: "fid"; "kid" and "kid_std"; "is" and "is_std" (only those
        requested via `metrics`).
        """
        fake_images = to_uint8(fake_images)
        metrics = [m.lower() for m in metrics]
        unknown = [m for m in metrics if m not in ALL_METRICS]
        if unknown:
            raise ValueError(f"Unknown metrics {unknown}; valid: {ALL_METRICS}.")
        n_fake = fake_images.shape[0]
        results = {}

        if "fid" in metrics:
            fid = self._get_fid()
            self._feed(fid, fake_images, real=False)
            results["fid"] = float(fid.compute())
            fid.reset()  # keeps the real-image statistics

        if "kid" in metrics:
            kid = self._get_kid()
            subset_size = min(
                self.kid_subset_size, self.real_images.shape[0], n_fake
            )
            if subset_size < 2:
                raise ValueError("KID needs at least 2 real and 2 generated images.")
            kid.subset_size = subset_size
            self._feed(kid, fake_images, real=False)
            kid_mean, kid_std = kid.compute()
            results["kid"] = float(kid_mean)
            results["kid_std"] = float(kid_std)
            kid.reset()  # keeps the real-image features

        if "is" in metrics:
            inception_score = self._get_inception_score()
            self._feed(inception_score, fake_images)
            is_mean, is_std = inception_score.compute()
            results["is"] = float(is_mean)
            results["is_std"] = float(is_std)
            inception_score.reset()

        return results


def compute_image_metrics(real_images, fake_images, metrics=ALL_METRICS, **kwargs):
    """One-shot convenience wrapper around GenerativeImageMetrics."""
    evaluator = GenerativeImageMetrics(to_uint8(real_images), **kwargs)
    return evaluator.evaluate(fake_images, metrics=metrics)
