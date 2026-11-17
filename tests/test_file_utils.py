"""
Tests of file_utils functionality (local-filesystem paths only by default).

Set IMAGE_GEN_S3URI to an s3://bucket/prefix containing a few images to also
exercise the S3 code path (needs AWS credentials).
"""

import os
import tempfile
import unittest

import torch
from PIL import Image

import file_utils
from tests._helpers import TimedTestCase, write_random_images


class TestPathHelpers(TimedTestCase):
    """Verify the small path/URI helper functions."""

    def test_is_s3_uri(self):
        self.assertTrue(file_utils.is_s3_uri("s3://bucket/prefix"))
        self.assertFalse(file_utils.is_s3_uri("/local/dir"))

    def test_parse_s3_uri(self):
        self.assertEqual(
            file_utils.parse_s3_uri("s3://mybucket/afhq/train"),
            ("mybucket", "afhq/train"),
        )
        self.assertEqual(file_utils.parse_s3_uri("s3://mybucket"), ("mybucket", ""))
        with self.assertRaises(ValueError):
            file_utils.parse_s3_uri("/not/s3")

    def test_join_path(self):
        self.assertEqual(
            file_utils.join_path("s3://b/afhq/", "train"), "s3://b/afhq/train"
        )
        self.assertEqual(
            file_utils.join_path("data", "train"), os.path.join("data", "train")
        )

    def test_uint8_to_float_range(self):
        x = torch.tensor([0, 255], dtype=torch.uint8)
        y = file_utils.uint8_to_float(x)
        self.assertAlmostEqual(float(y[0]), -1.0)
        self.assertAlmostEqual(float(y[1]), 1.0)


class TestImageLoading(TimedTestCase):
    """Verify listing, resizing and dataset behavior on a temp image dir."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        write_random_images(cls.tmp.name, 5, size=(40, 48), subdir="train/cat")
        write_random_images(cls.tmp.name, 3, size=(48, 40), subdir="train/dog", ext=".jpg")
        with open(os.path.join(cls.tmp.name, "train", "notes.txt"), "w") as f:
            f.write("not an image")
        cls.train_dir = os.path.join(cls.tmp.name, "train")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_list_image_files_recursive_sorted(self):
        files = file_utils.list_image_files(self.train_dir)
        self.assertEqual(len(files), 8)
        self.assertEqual(files, sorted(files))

    def test_list_image_files_missing_dir(self):
        with self.assertRaises(FileNotFoundError):
            file_utils.list_image_files(os.path.join(self.tmp.name, "nope"))

    def test_resize_center_crop(self):
        image = Image.new("RGB", (48, 40))  # PIL size is (width, height)
        out = file_utils.resize_center_crop(image, (16, 16))
        self.assertEqual(out.size, (16, 16))
        out = file_utils.resize_center_crop(image, (32, 16))
        self.assertEqual(out.size, (16, 32))

    def test_dataset_cached(self):
        ds = file_utils.ImageFolderDataset(self.train_dir, image_size=(16, 16))
        self.assertEqual(len(ds), 8)
        self.assertEqual(tuple(ds.cache.shape), (8, 3, 16, 16))
        self.assertEqual(ds.cache.dtype, torch.uint8)
        item = ds[0]
        self.assertEqual(tuple(item.shape), (3, 16, 16))
        self.assertEqual(item.dtype, torch.float32)
        self.assertGreaterEqual(float(item.min()), -1.0)
        self.assertLessEqual(float(item.max()), 1.0)

    def test_dataset_lazy_matches_cached(self):
        cached = file_utils.ImageFolderDataset(
            self.train_dir, image_size=(16, 16), hflip=False
        )
        lazy = file_utils.ImageFolderDataset(
            self.train_dir, image_size=(16, 16), hflip=False, cache_in_memory=False
        )
        self.assertIsNone(lazy.cache)
        self.assertTrue(torch.equal(cached[3], lazy[3]))
        self.assertTrue(torch.equal(cached.uint8_images(4), lazy.uint8_images(4)))

    def test_dataset_max_images(self):
        ds = file_utils.ImageFolderDataset(
            self.train_dir, image_size=(16, 16), max_images=3
        )
        self.assertEqual(len(ds), 3)
        self.assertEqual(ds.uint8_images().shape[0], 3)

    def test_dataset_empty_dir_raises(self):
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaises(ValueError):
                file_utils.ImageFolderDataset(empty, image_size=(16, 16))

    def test_data_loader_batches(self):
        ds = file_utils.ImageFolderDataset(self.train_dir, image_size=(16, 16))
        loader = file_utils.get_data_loader(ds, batch_size=4)
        batches = list(loader)
        self.assertEqual(len(batches), 2)
        self.assertEqual(tuple(batches[0].shape), (4, 3, 16, 16))

    def test_data_loader_small_dataset_keeps_partial_batch(self):
        ds = file_utils.ImageFolderDataset(self.train_dir, image_size=(16, 16))
        loader = file_utils.get_data_loader(ds, batch_size=64)
        self.assertEqual(len(list(loader)), 1)

    def test_get_training_data(self):
        run_params = {"images_path": self.tmp.name, "train_subdir": "train"}
        training_params = {"batch_size": 4, "num_data_input": 0}
        model_arch_params = {"image_shape": [16, 16, 3]}
        ds, loader = file_utils.get_training_data(
            run_params, training_params, model_arch_params
        )
        self.assertEqual(len(ds), 8)
        self.assertEqual(tuple(next(iter(loader)).shape), (4, 3, 16, 16))


@unittest.skipUnless(
    os.environ.get("IMAGE_GEN_S3URI"), "set IMAGE_GEN_S3URI to test S3 access"
)
class TestS3Loading(TimedTestCase):
    """Optional: verify listing/loading from a real S3 prefix."""

    def test_s3_dataset(self):
        ds = file_utils.ImageFolderDataset(
            os.environ["IMAGE_GEN_S3URI"], image_size=(16, 16), max_images=4
        )
        self.assertGreater(len(ds), 0)
        self.assertEqual(tuple(ds[0].shape), (3, 16, 16))


if __name__ == "__main__":
    unittest.main()
