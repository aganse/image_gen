"""Small shared helpers: parameter printing and image-file output."""

import math
import os
import pprint

import torch
from torchvision.utils import make_grid, save_image


def print_run_params(**kwargs):
    """Print the run parameters and dump them to output_dir/run_parameters.txt."""
    if "output_dir" not in kwargs:
        raise ValueError(
            "print_run_params: error: 'output_dir' must be one of the kwargs."
        )

    print("Run params:")
    print(kwargs)

    output_dir = kwargs.pop("output_dir")
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, "run_parameters.txt")
    with open(file_path, "w") as file:
        file.write("Parameters used in this run:\n")
        file.write(pprint.pformat(kwargs))

    print("")
    return file_path


def default_nrow(n_images):
    """Images per grid row giving a roughly square grid."""
    return max(1, int(math.ceil(math.sqrt(max(1, n_images)))))


def image_grid(images, nrow=None, padding=2):
    """Tile images in [-1, 1] into one (3, H, W) float tensor in [0, 1]."""
    images = images.detach().float().cpu()
    nrow = nrow or default_nrow(images.shape[0])
    return make_grid(
        images, nrow=nrow, padding=padding, normalize=True, value_range=(-1.0, 1.0)
    )


def save_image_grid(images, path, nrow=None, padding=2):
    """Save a batch of images in [-1, 1] as one tiled PNG; returns the path."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    save_image(image_grid(images, nrow=nrow, padding=padding), path)
    return path


def save_individual_images(images, output_dir, prefix="sim"):
    """Save each image in [-1, 1] as its own PNG; returns the list of paths.

    Individual files (rather than only a tiled grid) are what image-quality
    metrics need as input, so logging these to MLflow lets FID/KID/IS be
    recomputed later from the run's artifacts (see eval_mlflow_run.py).
    """
    os.makedirs(output_dir, exist_ok=True)
    images = images.detach().float().cpu()
    paths = []
    for i in range(images.shape[0]):
        path = os.path.join(output_dir, f"{prefix}_{i:05d}.png")
        save_image(images[i], path, normalize=True, value_range=(-1.0, 1.0))
        paths.append(path)
    return paths


def side_by_side_grid(top_images, bottom_images, path, padding=2):
    """Save two equal-sized batches as a two-row comparison grid.

    Used for VAE reconstructions: originals on the top row, reconstructions
    directly beneath them.
    """
    if top_images.shape != bottom_images.shape:
        raise ValueError("side_by_side_grid needs two batches of equal shape.")
    both = torch.cat([top_images.detach().cpu(), bottom_images.detach().cpu()], dim=0)
    return save_image_grid(both, path, nrow=top_images.shape[0], padding=padding)
