"""Standalone tool: compute FID / KID / IS for images logged in an MLflow run.

Points the same metric code used by the train_*.py scripts (image_metrics.py)
at the image artifacts of ANY MLflow run - e.g. a run from another project
such as flow_models - so image quality can be compared across models with
identical preprocessing, reference images and metric settings.

Examples:
    # images logged under artifact folder "sim_images" of a run:
    python eval_mlflow_run.py --run-id 0123abcd... --artifact-path sim_images \
        --real-images s3://mybucket/afhq/train --image-size 64

    # same, and write the results back onto that run as metrics:
    python eval_mlflow_run.py --run-id 0123abcd... --artifact-path sim_images \
        --real-images data/train --image-size 64 --log-to-run

    # no MLflow at all: compare two directories of images
    python eval_mlflow_run.py --fake-images output/vae/sim_images \
        --real-images data/train --image-size 64

Via the project's Docker image (which has the Inception weights baked in):
    make eval-run RUN_ID=0123abcd... REAL_IMAGES=s3://mybucket/afhq/train

Requirements on the run being evaluated:
  * The artifact folder must contain INDIVIDUAL image files (png/jpg), not a
    single tiled grid or matplotlib figure.
  * Many images are needed for meaningful numbers: at least several hundred
    for KID, and ideally several thousand for FID (FID is biased upward for
    small sample counts; only compare FIDs computed from equal counts).
  * MLFLOW_TRACKING_URI (or --tracking-uri) must point at the tracking
    server, and this process needs read access to the run's artifact store
    (e.g. AWS credentials when artifacts live in S3).

Real and generated images are both resized/center-cropped to --image-size
before being passed to the metric network.  Use the resolution the generator
was trained at so that neither set gains or loses detail relative to the other.
"""

import argparse
import json
import os
import sys
import tempfile

from file_utils import ImageFolderDataset
from gen_base import best_device
from image_metrics import ALL_METRICS, GenerativeImageMetrics


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Compute FID/KID/IS for images logged in an MLflow run "
        "(or in a plain directory) against a set of real images."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-id", help="MLflow run ID holding the generated images")
    source.add_argument(
        "--fake-images",
        help="local dir or s3:// URI of generated images (instead of --run-id)",
    )
    parser.add_argument(
        "--artifact-path",
        default="sim_images",
        help="artifact folder within the run containing individual image "
        "files (default: sim_images)",
    )
    parser.add_argument(
        "--real-images",
        required=True,
        help="local dir or s3:// URI of real reference images (searched recursively)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=64,
        help="square size both image sets are resized to (default: 64)",
    )
    parser.add_argument(
        "--max-real", type=int, default=0, help="use at most N real images (0 = all)"
    )
    parser.add_argument(
        "--max-fake", type=int, default=0, help="use at most N generated images (0 = all)"
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(ALL_METRICS),
        choices=list(ALL_METRICS),
        help="metrics to compute (default: fid kid is)",
    )
    parser.add_argument("--kid-subset-size", type=int, default=1000)
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="MLflow tracking URI (default: env var MLFLOW_TRACKING_URI)",
    )
    parser.add_argument(
        "--log-to-run",
        action="store_true",
        help="also log the results as metrics on the run given by --run-id",
    )
    parser.add_argument(
        "--metric-prefix",
        default="eval_",
        help="prefix for metric names when using --log-to-run (default: eval_)",
    )
    parser.add_argument(
        "--device", default=None, help="cuda or cpu (default: cuda if available)"
    )
    parser.add_argument("--output-json", default=None, help="also write results to this file")
    args = parser.parse_args(argv)
    if args.log_to_run and not args.run_id:
        parser.error("--log-to-run requires --run-id")
    return args


def download_run_images(run_id, artifact_path, dst_dir, tracking_uri=None):
    """Download a run's artifact folder; returns the local directory path."""
    import mlflow

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    print(
        f"Downloading artifacts '{artifact_path}' of run {run_id} "
        f"from {mlflow.get_tracking_uri()}...",
        flush=True,
    )
    return mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=artifact_path, dst_path=dst_dir
    )


def evaluate_images(
    fake_path,
    real_path,
    image_size=64,
    metrics=ALL_METRICS,
    max_real=0,
    max_fake=0,
    kid_subset_size=1000,
    device=None,
):
    """Compute metrics between two image locations; returns a results dict."""
    size = (int(image_size), int(image_size))
    fake = ImageFolderDataset(fake_path, image_size=size, max_images=max_fake, hflip=False)
    real = ImageFolderDataset(real_path, image_size=size, max_images=max_real, hflip=False)
    n_fake, n_real = len(fake), len(real)
    if n_fake < 500:
        print(
            f"WARNING: only {n_fake} generated images; FID/KID/IS from so few "
            "images are very noisy (and FID strongly biased).",
            file=sys.stderr,
            flush=True,
        )
    evaluator = GenerativeImageMetrics(
        real.uint8_images(),
        device=device or best_device(),
        kid_subset_size=kid_subset_size,
    )
    results = evaluator.evaluate(fake.uint8_images(), metrics=metrics)
    return {
        "metrics": results,
        "num_generated_images": n_fake,
        "num_real_images": n_real,
        "image_size": int(image_size),
    }


def main(argv=None):
    args = parse_args(argv)
    with tempfile.TemporaryDirectory() as tmp_dir:
        if args.run_id:
            fake_path = download_run_images(
                args.run_id, args.artifact_path, tmp_dir, args.tracking_uri
            )
        else:
            fake_path = args.fake_images
        report = evaluate_images(
            fake_path,
            args.real_images,
            image_size=args.image_size,
            metrics=args.metrics,
            max_real=args.max_real,
            max_fake=args.max_fake,
            kid_subset_size=args.kid_subset_size,
            device=args.device,
        )
    report["real_images"] = args.real_images
    if args.run_id:
        report["run_id"] = args.run_id
        report["artifact_path"] = args.artifact_path
    else:
        report["fake_images"] = args.fake_images

    print(json.dumps(report, indent=2))
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(report, f, indent=2)

    if args.log_to_run:
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=args.tracking_uri)
        for name, value in report["metrics"].items():
            client.log_metric(args.run_id, f"{args.metric_prefix}{name}", value)
        client.set_tag(
            args.run_id,
            f"{args.metric_prefix}settings",
            f"n_fake={report['num_generated_images']}, "
            f"n_real={report['num_real_images']}, image_size={args.image_size}, "
            f"real_images={args.real_images}",
        )
        print(f"Metrics logged to run {args.run_id}.")
    return report


if __name__ == "__main__":
    main()
