"""Evaluation and MLflow logging shared by all the train_*.py scripts.

Kept separate from model_training.py (generic job/tracking orchestration) so
that module stays free of anything image-generation specific.

What lands in MLflow from this module:

    during training (GenEvalCallback, every eval_every_n_epochs epochs)
        artifact  sample_grids/samples_epoch_NNNN.png   grid generated from
                  the same fixed noise each time, to watch quality evolve
        metrics   kid, kid_std   (time series; step = epoch index)

    after training (final_evaluation)
        metrics   fid_final, kid_final, kid_final_std, is_final, is_final_std
        artifacts sim_images/sim_NNNNN.png   num_gen_sims individual images
                  plots/sim_grid.png         grid of generated images
                  plots/real_grid.png        grid of training images
                  metrics/final_metrics.json

    at the very end (finish_run)
        artifacts model/model_summary.txt, model/model_arch.json,
                  run_parameters.txt
"""

import json
import os

import lightning as L
import mlflow
import torch

import utils
from file_utils import uint8_to_float
from image_metrics import GenerativeImageMetrics, to_uint8


def _mlflow_active():
    return mlflow.active_run() is not None


class GenEvalCallback(L.Callback):
    """Periodically sample from the model during training and log to MLflow.

    Losses alone are a poor guide to generative-model quality (for GANs they
    are essentially uninformative), so every eval_every_n_epochs epochs this:
      1. generates a grid of images from *fixed* noise and logs it, and
      2. generates num_eval_sims fresh images and logs their KID against the
         real images as a metric time series alongside the loss curves.

    Args:
        evaluator: GenerativeImageMetrics (or None to skip the metric)
        output_dir: local directory for the grid PNGs
        eval_every_n_epochs: evaluation period; <= 0 disables the callback
        num_eval_sims: generated images per periodic metric evaluation;
            0 logs grids only.  This is the main cost knob for diffusion
            models, whose sampling takes many network evaluations per image.
        num_grid_images: images in the fixed-noise grid
        metrics: which metrics to compute periodically (default KID only)
        sample_batch_size: batch size used when sampling
        seed: seed for the fixed grid noise and the evaluation noise
    """

    def __init__(
        self,
        evaluator,
        output_dir,
        eval_every_n_epochs=10,
        num_eval_sims=1000,
        num_grid_images=64,
        metrics=("kid",),
        sample_batch_size=64,
        seed=1234,
    ):
        super().__init__()
        self.evaluator = evaluator
        self.output_dir = output_dir
        self.eval_every_n_epochs = int(eval_every_n_epochs)
        self.num_eval_sims = int(num_eval_sims)
        self.num_grid_images = int(num_grid_images)
        self.metrics = tuple(metrics)
        self.sample_batch_size = int(sample_batch_size)
        self.seed = int(seed)
        self.fixed_noise = None

    def on_fit_start(self, trainer, pl_module):
        if self.fixed_noise is None and self.num_grid_images > 0:
            generator = torch.Generator().manual_seed(self.seed)
            self.fixed_noise = pl_module.make_noise(
                self.num_grid_images, generator=generator
            )

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1  # number of completed epochs
        if self.eval_every_n_epochs <= 0 or epoch % self.eval_every_n_epochs != 0:
            return
        step = trainer.current_epoch

        if self.fixed_noise is not None:
            images = pl_module.sample(
                noise=self.fixed_noise, batch_size=self.sample_batch_size
            )
            path = os.path.join(
                self.output_dir, "sample_grids", f"samples_epoch_{epoch:04d}.png"
            )
            utils.save_image_grid(images, path)
            if _mlflow_active():
                mlflow.log_artifact(path, artifact_path="sample_grids")

        if self.evaluator is not None and self.num_eval_sims > 0 and self.metrics:
            generator = torch.Generator().manual_seed(self.seed + 1)
            fakes = pl_module.sample(
                n=self.num_eval_sims,
                batch_size=self.sample_batch_size,
                generator=generator,
            )
            self.evaluator.to(pl_module.device)
            results = self.evaluator.evaluate(to_uint8(fakes), metrics=self.metrics)
            summary = ", ".join(f"{k}={v:.5f}" for k, v in results.items())
            print(f"Epoch {epoch} evaluation: {summary}", flush=True)
            if _mlflow_active():
                mlflow.log_metrics(results, step=step)


def build_evaluation(train_dataset, run_params):
    """Create the metrics evaluator and periodic-eval callback from run_params.

    The real reference set for FID/KID is the training set itself (without
    augmentation), the usual convention for small single-split datasets such
    as AFHQ.  Returns (evaluator, callback).
    """
    real_images = train_dataset.uint8_images(run_params.get("num_real_metric_images"))
    evaluator = GenerativeImageMetrics(
        real_images,
        kid_subset_size=run_params.get("kid_subset_size", 1000),
    )
    callback = GenEvalCallback(
        evaluator,
        output_dir=run_params["output_dir"],
        eval_every_n_epochs=run_params.get("eval_every_n_epochs", 10),
        num_eval_sims=run_params.get("num_eval_sims", 1000),
        num_grid_images=run_params.get("num_grid_images", 64),
        sample_batch_size=run_params.get("sample_batch_size", 64),
        seed=run_params.get("seed", 1234),
    )
    return evaluator, callback


def final_evaluation(model, evaluator, run_params, tracking_active):
    """Generate the final images, compute FID/KID/IS, and log it all.

    Generates max(num_final_metric_sims, num_gen_sims) images in one pass;
    the first num_gen_sims are saved individually (and logged to MLflow under
    sim_images/), the first num_final_metric_sims feed the metrics.  Returns
    the dict of final metrics (keys suffixed "_final").
    """
    output_dir = run_params["output_dir"]
    n_save = int(run_params.get("num_gen_sims", 0))
    n_metric = int(run_params.get("num_final_metric_sims", 0))
    metric_names = list(run_params.get("final_metrics", ["fid", "kid", "is"]))
    if not metric_names:
        n_metric = 0
    n_total = max(n_save, n_metric)
    if n_total <= 0:
        return {}

    print(f"Generating {n_total} images for final evaluation...", flush=True)
    generator = torch.Generator().manual_seed(int(run_params.get("seed", 1234)) + 2)
    fakes = model.sample(
        n=n_total,
        batch_size=run_params.get("sample_batch_size", 64),
        generator=generator,
    )

    sim_dir = os.path.join(output_dir, "sim_images")
    plot_paths = []
    if n_save > 0:
        utils.save_individual_images(fakes[:n_save], sim_dir, prefix="sim")
        plot_paths.append(
            utils.save_image_grid(
                fakes[:min(n_save, 64)], os.path.join(output_dir, "sim_grid.png")
            )
        )
        print(f"{n_save} generated images written to {sim_dir}.", flush=True)
    if evaluator is not None:
        real = uint8_to_float(evaluator.real_images[:64])
        plot_paths.append(
            utils.save_image_grid(real, os.path.join(output_dir, "real_grid.png"))
        )

    final_metrics = {}
    metrics_path = None
    if evaluator is not None and n_metric > 0:
        final_metrics, metrics_path = _compute_final_metrics(
            evaluator, fakes[:n_metric], metric_names, model.device, output_dir
        )

    if tracking_active and _mlflow_active():
        if final_metrics:
            mlflow.log_metrics(final_metrics)
        if metrics_path:
            mlflow.log_artifact(metrics_path, artifact_path="metrics")
        if n_save > 0:
            mlflow.log_artifacts(sim_dir, artifact_path="sim_images")
        for path in plot_paths:
            mlflow.log_artifact(path, artifact_path="plots")

    return final_metrics


def _compute_final_metrics(evaluator, fakes, metric_names, device, output_dir):
    """Compute the final metrics; returns (dict with *_final keys, json path)."""
    n_metric = fakes.shape[0]
    print(f"Computing {metric_names} on {n_metric} generated images...", flush=True)
    evaluator.to(device)
    results = evaluator.evaluate(to_uint8(fakes), metrics=metric_names)
    final_metrics = {}
    for key, value in results.items():
        # "kid_std" -> "kid_final_std", "fid" -> "fid_final"
        base, _, suffix = key.partition("_")
        name = f"{base}_final" + (f"_{suffix}" if suffix else "")
        final_metrics[name] = value
    print("Final image metrics:", flush=True)
    for key, value in final_metrics.items():
        print(f"  {key}: {value:.5f}", flush=True)
    metrics_path = os.path.join(output_dir, "final_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "metrics": final_metrics,
                "num_generated_images": n_metric,
                "num_real_images": int(evaluator.real_images.shape[0]),
            },
            f, indent=2,
        )
    return final_metrics, metrics_path


def finish_run(run_params, training_params):
    """Log the model summary / arch / params files and close the MLflow run."""
    if training_params.get("tracking_tool") == "mlflow" and run_params.get(
        "mlflow_run_open"
    ):
        for key in ("model_summary_path", "model_arch_path"):
            path = run_params.get(key)
            if path and os.path.exists(path):
                mlflow.log_artifact(path, artifact_path="model")
        params_path = os.path.join(run_params["output_dir"], "run_parameters.txt")
        if os.path.exists(params_path):
            mlflow.log_artifact(params_path)
        # (model weights are saved to WEIGHTS_PATH rather than logged to MLflow)
        mlflow.end_run()
        run_params["mlflow_run_open"] = False
