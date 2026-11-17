"""Image data loading from a local directory or an S3 prefix.

The same code path serves all run platforms:

    local / local Docker   images_path is a filesystem directory
    SageMaker              images_path is /opt/ml/input/data/training, a
                           "FastFile" mount that streams S3 objects on demand
                           (no copy onto the instance's disk)
    AWS Batch / other      images_path may be an s3://bucket/prefix URI, read
                           directly with boto3

Default behavior (cache_in_memory=True) reads every image exactly once at
startup, resizes it to the training resolution, and keeps the whole set in RAM
as one uint8 tensor.  For small datasets this is the common approach: e.g.
5000 images at 64x64x3 is only ~60MB (~250MB at 128x128), it uses no instance
disk, avoids re-reading S3 every epoch, and makes epochs as fast as possible.
For datasets too large for RAM set cache_in_memory=False to read lazily per
item instead (use num_workers > 0 in that case); beyond that scale one would
move to streamed shards (e.g. WebDataset) or the S3 Connector for PyTorch.

All images under images_path are used, recursing into subdirectories, so with
the flow_models data layout (train/cat/, val/cat/, ...) pointing images_path
at ".../train" trains on everything inside train/.
"""

import io
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

_S3_CLIENTS = {}  # one boto3 client per process (clients are not fork-safe)


def is_s3_uri(path):
    return str(path).startswith("s3://")


def parse_s3_uri(uri):
    """Split 's3://bucket/some/prefix' into ('bucket', 'some/prefix')."""
    if not is_s3_uri(uri):
        raise ValueError(f"S3 URI should start with s3://; got {uri!r}")
    bucket, _, prefix = uri[5:].partition("/")
    if not bucket:
        raise ValueError(f"S3 URI has no bucket name: {uri!r}")
    return bucket, prefix


def join_path(base, *parts):
    """Join path components for either a local path or an s3:// URI."""
    parts = [str(p).strip("/") for p in parts if p]
    if is_s3_uri(base):
        return "/".join([str(base).rstrip("/")] + parts)
    return os.path.join(str(base), *parts)


def _s3_client():
    pid = os.getpid()
    if pid not in _S3_CLIENTS:
        import boto3  # imported lazily so local-only runs don't need AWS setup
        _S3_CLIENTS[pid] = boto3.client("s3")
    return _S3_CLIENTS[pid]


def list_image_files(images_path, extensions=IMAGE_EXTENSIONS):
    """Return a sorted list of image file paths (or s3:// URIs) under a root.

    Recurses into subdirectories / key prefixes.  Sorting makes the dataset
    order - and therefore any "first N images" subset - reproducible.
    """
    extensions = tuple(e.lower() for e in extensions)
    files = []
    if is_s3_uri(images_path):
        bucket, prefix = parse_s3_uri(images_path)
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        paginator = _s3_client().get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                if obj["Key"].lower().endswith(extensions):
                    files.append(f"s3://{bucket}/{obj['Key']}")
    else:
        if not os.path.isdir(images_path):
            raise FileNotFoundError(f"Image directory not found: {images_path!r}")
        for root, _, names in os.walk(images_path, followlinks=True):
            for name in names:
                if name.lower().endswith(extensions):
                    files.append(os.path.join(root, name))
    return sorted(files)


def read_file_bytes(path):
    """Read raw bytes from a local file or an s3:// URI."""
    if is_s3_uri(path):
        bucket, key = parse_s3_uri(path)
        return _s3_client().get_object(Bucket=bucket, Key=key)["Body"].read()
    with open(path, "rb") as f:
        return f.read()


def resize_center_crop(image, image_size):
    """Resize a PIL image to cover (height, width), then center-crop to it."""
    target_h, target_w = int(image_size[0]), int(image_size[1])
    src_w, src_h = image.size
    scale = max(target_h / src_h, target_w / src_w)
    new_w = max(target_w, int(round(src_w * scale)))
    new_h = max(target_h, int(round(src_h * scale)))
    if (new_w, new_h) != (src_w, src_h):
        image = image.resize((new_w, new_h), Image.Resampling.BICUBIC)
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    return image.crop((left, top, left + target_w, top + target_h))


def load_image_uint8(path, image_size):
    """Load one image as a uint8 tensor of shape (3, height, width)."""
    with Image.open(io.BytesIO(read_file_bytes(path))) as image:
        image = resize_center_crop(image.convert("RGB"), image_size)
        array = np.array(image, dtype=np.uint8)
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def uint8_to_float(images):
    """Map uint8 [0, 255] images to float32 [-1, 1] (the models' scale)."""
    return images.to(torch.float32) / 127.5 - 1.0


class ImageFolderDataset(Dataset):
    """Unlabeled image dataset from a local directory or S3 prefix.

    Each item is a float32 tensor (3, H, W) scaled to [-1, 1], optionally
    randomly flipped left-right (a label-preserving augmentation that
    effectively doubles a small dataset of faces).

    Args:
        images_path: local directory or s3://bucket/prefix (searched recursively)
        image_size: (height, width) images are resized/center-cropped to
        max_images: use only the first N images (None/0 = all)
        cache_in_memory: load everything once into RAM (see module docstring)
        hflip: apply random horizontal flips in __getitem__
        num_load_threads: parallel readers used when filling the RAM cache
    """

    def __init__(
        self,
        images_path,
        image_size=(64, 64),
        max_images=None,
        cache_in_memory=True,
        hflip=True,
        num_load_threads=16,
    ):
        self.images_path = images_path
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.hflip = bool(hflip)
        self.files = list_image_files(images_path)
        if max_images:
            self.files = self.files[: int(max_images)]
        if not self.files:
            raise ValueError(f"No image files found under {images_path!r}.")
        self.cache = None
        if cache_in_memory:
            print(
                f"Loading {len(self.files)} images from {images_path} into "
                f"memory at {self.image_size[0]}x{self.image_size[1]}...",
                flush=True,
            )
            self.cache = self._load_many(self.files, num_load_threads)
            megabytes = self.cache.numel() / 1e6
            print(f"Done loading images ({megabytes:.0f} MB in RAM).\n", flush=True)

    def _load_many(self, files, num_threads):
        def _load(path):
            return load_image_uint8(path, self.image_size)

        if num_threads and num_threads > 1:
            with ThreadPoolExecutor(max_workers=int(num_threads)) as pool:
                images = list(pool.map(_load, files))
        else:
            images = [_load(path) for path in files]
        return torch.stack(images, dim=0)

    def __len__(self):
        return len(self.files)

    def get_uint8(self, index):
        if self.cache is not None:
            return self.cache[index]
        return load_image_uint8(self.files[index], self.image_size)

    def __getitem__(self, index):
        image = uint8_to_float(self.get_uint8(index))
        if self.hflip and torch.rand(()) < 0.5:
            image = image.flip(-1)
        return image

    def uint8_images(self, n=None, num_load_threads=16):
        """Return the first n images (all if None/0) as uint8 (N, 3, H, W).

        No augmentation is applied; this is the "real" reference set used for
        the FID/KID image-quality metrics.
        """
        n = len(self) if not n else min(int(n), len(self))
        if self.cache is not None:
            return self.cache[:n]
        return self._load_many(self.files[:n], num_load_threads)


def get_data_loader(dataset, batch_size, shuffle=True, num_workers=0):
    """Wrap a dataset in a DataLoader suitable for training.

    Incomplete final batches are dropped (BatchNorm and the GAN losses behave
    better with a constant batch size) unless the dataset is smaller than one
    batch.  num_workers=0 is right for the in-memory cache; use a few workers
    with cache_in_memory=False.
    """
    num_workers = int(num_workers)
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=len(dataset) >= int(batch_size),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def get_training_data(run_params, training_params, model_arch_params):
    """Build the training dataset and loader from the three param dicts.

    Reads images from run_params["images_path"] / run_params["train_subdir"].
    Returns (dataset, loader).
    """
    images_path = join_path(
        run_params["images_path"], run_params.get("train_subdir", "train")
    )
    dataset = ImageFolderDataset(
        images_path,
        image_size=model_arch_params["image_shape"][:2],
        max_images=training_params.get("num_data_input"),
        cache_in_memory=run_params.get("cache_in_memory", True),
        hflip=training_params.get("hflip_augment", True),
    )
    loader = get_data_loader(
        dataset,
        batch_size=training_params["batch_size"],
        num_workers=run_params.get("num_workers", 0),
    )
    return dataset, loader
