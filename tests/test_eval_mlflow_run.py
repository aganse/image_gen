"""
Tests of the standalone eval_mlflow_run.py tool: argument handling always;
the directory-vs-directory metric computation only when
IMAGE_GEN_TEST_METRICS=1 (needs Inception weights; see test_image_metrics.py).
"""

import os
import tempfile
import unittest

import eval_mlflow_run
from tests._helpers import TimedTestCase, write_random_images

RUN_METRICS = os.environ.get("IMAGE_GEN_TEST_METRICS") == "1"


class TestArguments(TimedTestCase):

    def test_run_id_defaults(self):
        args = eval_mlflow_run.parse_args(["--run-id", "abc", "--real-images", "data/train"])
        self.assertEqual(args.artifact_path, "sim_images")
        self.assertEqual(args.image_size, 64)
        self.assertEqual(args.metrics, ["fid", "kid", "is"])
        self.assertFalse(args.log_to_run)

    def test_requires_a_source(self):
        with self.assertRaises(SystemExit):
            eval_mlflow_run.parse_args(["--real-images", "data/train"])

    def test_sources_are_mutually_exclusive(self):
        with self.assertRaises(SystemExit):
            eval_mlflow_run.parse_args(
                ["--run-id", "abc", "--fake-images", "x", "--real-images", "y"]
            )

    def test_log_to_run_requires_run_id(self):
        with self.assertRaises(SystemExit):
            eval_mlflow_run.parse_args(
                ["--fake-images", "x", "--real-images", "y", "--log-to-run"]
            )


@unittest.skipUnless(RUN_METRICS, "set IMAGE_GEN_TEST_METRICS=1 to run (needs Inception weights)")
class TestDirectoryEvaluation(TimedTestCase):

    def test_directory_mode_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_random_images(tmp, 24, size=(32, 32), subdir="real")
            write_random_images(tmp, 24, size=(32, 32), subdir="fake")
            out_json = os.path.join(tmp, "out", "report.json")
            report = eval_mlflow_run.main([
                "--fake-images", os.path.join(tmp, "fake"),
                "--real-images", os.path.join(tmp, "real"),
                "--image-size", "32", "--metrics", "kid",
                "--kid-subset-size", "12", "--device", "cpu",
                "--output-json", out_json,
            ])
            self.assertTrue(os.path.exists(out_json))
        self.assertEqual(report["num_generated_images"], 24)
        self.assertIn("kid", report["metrics"])


if __name__ == "__main__":
    unittest.main()
