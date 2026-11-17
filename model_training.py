"""Training orchestration for the generative models (PyTorch Lightning).

This is the PyTorch/Lightning counterpart of flow_models' model_training.py
and follows the same structure and conventions.  It is intentionally
decoupled from any specific model, cloud platform or experiment-tracking
backend: it accepts any LightningModule, and platform detection
(detect_platform) and checkpoint-directory resolution (resolve_checkpoint_dir)
are the only points of platform awareness; all other code treats the
resulting strings as opaque configuration.

Typical usage:
    from file_utils import get_training_data
    from model_training import default_training_sequence, load_param_overrides

    load_param_overrides(run_params, training_params, model_arch_params)
    dataset, train_loader = get_training_data(
        run_params, training_params, model_arch_params
    )
    model = SomeLitModule(**model_arch_params, ...)
    model, trainer = default_training_sequence(
        model, train_loader, run_params, training_params, model_arch_params
    )

What lands in MLflow from this module (tracking_tool="mlflow"):
    tags     mlflow.user, image_tag, run_env
    params   everything in the three param dicts (+ job_name, images_path)
    metrics  every value the LightningModule logs, once per epoch, plus
             MLflow system metrics (CPU/GPU/memory)
The MLflow run is left open on return so the calling script can add its own
post-training metrics and artifacts (sample images, FID/KID/IS, ...).
"""

import base64
import getpass
import glob
import json
import os
import re
import shutil
import sys
from datetime import datetime

import lightning as L
import mlflow
import torch
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.utilities.model_summary import ModelSummary

_RUN_ID_FILENAME = "mlflow_run_id.txt"
_CKPT_PATTERN = "ckpt-*.ckpt"
_WEIGHTS_FILENAME = "model_weights.pt"
_SUMMARY_FILENAME = "model_summary.txt"
_ARCH_FILENAME = "model_arch.json"


# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------

def detect_platform():
    """Return the current execution platform as a string.

    Returns one of: 'awsbatch', 'sagemaker', 'docker', 'local'.  Platform is
    inferred from environment markers: AWS_BATCH_JOB_ID set indicates AWS
    Batch; SageMaker's TRAINING_JOB_NAME / TRAINING_JOB_ARN env vars or its
    resourceconfig.json file indicate a SageMaker Training Job; /opt/ml
    existing without those indicates this repo's Docker image run locally
    (make local-run); otherwise local.
    """
    if os.environ.get("AWS_BATCH_JOB_ID"):
        return "awsbatch"
    if (
        os.environ.get("TRAINING_JOB_NAME")
        or os.environ.get("TRAINING_JOB_ARN")
        or os.path.exists("/opt/ml/input/config/resourceconfig.json")
    ):
        return "sagemaker"
    if os.path.exists("/opt/ml"):
        return "docker"
    return "local"


def resolve_checkpoint_dir(run_params, platform):
    """Return the appropriate checkpoint directory path for the given platform."""
    if platform == "sagemaker":
        return "/opt/ml/checkpoints"
    return os.path.join(run_params["model_dir"], "checkpoints")


# ---------------------------------------------------------------------------
# Parameter loading
# ---------------------------------------------------------------------------

def load_param_overrides(
    run_params,
    training_params,
    model_arch_params,
    hp_path="/opt/ml/input/config/hyperparameters.json",
):
    """Merge base64-encoded JSON parameter overrides into the three param dicts.

    No-op when the file does not exist (local runs without Docker).  The
    default path is the SageMaker hyperparameter file written by the
    entrypoint; pass a different hp_path for other platforms.
    """
    if not os.path.exists(hp_path):
        return
    with open(hp_path) as f:
        hparams = json.load(f)
    if "params" in hparams:
        overrides = json.loads(base64.b64decode(hparams["params"]))
        run_params.update(overrides.get("run_params", {}))
        training_params.update(overrides.get("training_params", {}))
        model_arch_params.update(overrides.get("model_arch_params", {}))


# ---------------------------------------------------------------------------
# Lightning callbacks
# ---------------------------------------------------------------------------

class MLflowLoggingCallback(L.Callback):
    """Mirror Lightning epoch-end metrics into the active MLflow run.

    Everything a LightningModule logs with self.log(..., on_epoch=True) shows
    up in trainer.callback_metrics; this pushes those values to MLflow once
    per epoch with step = epoch index, so they chart as time series.
    """

    def on_train_epoch_end(self, trainer, pl_module):
        if mlflow.active_run() is None:
            return
        metrics = {}
        for key, value in trainer.callback_metrics.items():
            try:
                metrics[key] = float(value)
            except (TypeError, ValueError):
                continue
        if metrics:
            mlflow.log_metrics(metrics, step=trainer.current_epoch)


class PeriodicCheckpoint(L.Callback):
    """Save a full Lightning checkpoint every N epochs, keeping the last 2.

    Full checkpoints (weights + optimizer state + epoch counters) are what
    allow a spot-interrupted job to resume where it left off.  Only the two
    most recent are kept to bound disk/S3 use.
    """

    def __init__(self, ckpt_dir, every_n_epochs, keep_last=2):
        super().__init__()
        self.ckpt_dir = ckpt_dir
        self.every_n_epochs = int(every_n_epochs)
        self.keep_last = int(keep_last)

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1  # number of completed epochs
        if self.every_n_epochs <= 0 or epoch % self.every_n_epochs != 0:
            return
        os.makedirs(self.ckpt_dir, exist_ok=True)
        path = os.path.join(self.ckpt_dir, f"ckpt-{epoch:04d}.ckpt")
        trainer.save_checkpoint(path)
        print(f"Checkpoint saved: {path}", flush=True)
        files = sorted(glob.glob(os.path.join(self.ckpt_dir, _CKPT_PATTERN)))
        for old in files[:-self.keep_last]:
            os.remove(old)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _flatten_params(params):
    """Flatten a params dict into MLflow-loggable scalars/strings.

    Nested dicts (e.g. hf_model_config) are expanded one level into dotted
    keys ("hf_model_config.layers_per_block") so each setting is individually
    visible and comparable in the MLflow UI; other non-scalars are repr()'d.
    """
    flat_params = {}
    if not params:
        return flat_params
    for key, value in params.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, (int, float, str, bool)) or sub_value is None:
                    flat_params[f"{key}.{sub_key}"] = sub_value
                else:
                    flat_params[f"{key}.{sub_key}"] = repr(sub_value)
        elif isinstance(value, (int, float, str, bool)) or value is None:
            flat_params[key] = value
        else:
            flat_params[key] = repr(value)
    return flat_params


def _capture_and_save_summary(model, output_dir):
    """Generate model summary text, save to file, and return the path."""
    try:
        summary_text = str(ModelSummary(model, max_depth=2))
    except Exception as e:  # summary is informational; never fail a run on it
        n_params = sum(p.numel() for p in model.parameters())
        summary_text = (
            f"{type(model).__name__}: {n_params:,} parameters "
            f"(detailed summary unavailable: {e})"
        )

    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, _SUMMARY_FILENAME)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")
    print(summary_text, flush=True)
    return summary_path


def _find_latest_checkpoint(ckpt_dir):
    """Return (path, epoch) of the newest checkpoint, or (None, 0) if none."""
    ckpt_files = sorted(glob.glob(os.path.join(ckpt_dir, _CKPT_PATTERN)))
    if not ckpt_files:
        return None, 0
    latest = ckpt_files[-1]
    epoch = int(re.search(r"ckpt-(\d+)", os.path.basename(latest)).group(1))
    return latest, epoch


def _build_callbacks(training_params, tracking_tool, ckpt_dir):
    """Build and return the list of Lightning training callbacks."""
    callbacks = []

    if training_params.get("early_stopping_patience", 0) > 0:
        callbacks.append(
            EarlyStopping(
                monitor=training_params.get("early_stopping_monitor", "loss"),
                mode="min",
                patience=training_params["early_stopping_patience"],
                check_on_train_epoch_end=True,
            )
        )

    if tracking_tool == "mlflow":
        callbacks.append(MLflowLoggingCallback())

    if training_params.get("checkpoint_every_n_epochs", 0) > 0:
        os.makedirs(ckpt_dir, exist_ok=True)
        callbacks.append(
            PeriodicCheckpoint(ckpt_dir, training_params["checkpoint_every_n_epochs"])
        )

    return callbacks


def _build_logger(training_params, tracking_tool):
    """Return a Lightning logger (TensorBoard only) or False for none.

    MLflow logging is done by MLflowLoggingCallback with the plain mlflow
    API rather than Lightning's MLFlowLogger, so that this module keeps full
    control of the run lifecycle (resuming a run across spot restarts, and
    leaving the run open for the calling script).
    """
    if tracking_tool != "tensorboard":
        return False
    from lightning.pytorch.loggers import TensorBoardLogger

    log_dir = f"./logs/train/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    tracking_port = training_params.get("tracking_port")
    if tracking_port:
        print(
            f"TensorBoard logs: {log_dir} "
            f"(launch via `tensorboard --logdir {log_dir} --port {tracking_port}`)"
        )
    else:
        print(f"TensorBoard logs: {log_dir}")
    return TensorBoardLogger(save_dir=log_dir, name="", version="")


def _validate_weights_dest(training_params):
    """Fail fast at job start if the configured S3 weights destination is unreachable."""
    if not training_params.get("save_model_weights"):
        return
    weights_dest = os.environ.get("WEIGHTS_PATH", "")
    if weights_dest.startswith("s3://"):
        import boto3
        from urllib.parse import urlparse
        parsed = urlparse(weights_dest)
        try:
            boto3.client("s3").head_bucket(Bucket=parsed.netloc)
        except Exception as e:
            print(
                f"FATAL: cannot access S3 bucket {parsed.netloc!r} "
                f"for model weights upload: {e}",
                flush=True,
            )
            sys.exit(1)


def _save_training_outputs(
    model, run_params, training_params, model_arch_params, ckpt_dir, tracking_active
):
    """Save model weights, clean up checkpoints, and write summary/arch files."""
    os.makedirs(run_params["model_dir"], exist_ok=True)
    _job_name = os.environ.get("JOB_NAME", "")

    if training_params.get("save_model_weights"):
        weights_path = os.path.join(run_params["model_dir"], _WEIGHTS_FILENAME)
        torch.save(model.state_dict(), weights_path)
        print("Model weights saved to file.\n", flush=True)

        weights_dest = os.environ.get("WEIGHTS_PATH")
        if weights_dest:
            if weights_dest.startswith("s3://"):
                import boto3
                from urllib.parse import urlparse
                parsed = urlparse(weights_dest)
                bucket = parsed.netloc
                key_prefix = parsed.path.strip("/")
                if _job_name:
                    key_prefix = f"{key_prefix}/{_job_name}" if key_prefix else _job_name
                filename = os.path.basename(weights_path)
                key = f"{key_prefix}/{filename}" if key_prefix else filename
                boto3.client("s3").upload_file(weights_path, bucket, key)
                print(
                    f"Model weights uploaded to s3://{bucket}/{key}\n", flush=True
                )
            else:
                os.makedirs(weights_dest, exist_ok=True)
                shutil.copy2(weights_path, weights_dest)
                print(f"Model weights copied to {weights_dest}\n", flush=True)

        if training_params.get("checkpoint_every_n_epochs", 0) > 0:
            for f in glob.glob(os.path.join(ckpt_dir, _CKPT_PATTERN)):
                os.remove(f)
            run_id_file = os.path.join(ckpt_dir, _RUN_ID_FILENAME)
            if os.path.exists(run_id_file):
                os.remove(run_id_file)
            print("Checkpoints deleted (final weights saved).\n", flush=True)

    summary_path = _capture_and_save_summary(model, run_params["model_dir"])
    arch_path = os.path.join(run_params["model_dir"], _ARCH_FILENAME)
    with open(arch_path, "w") as f:
        json.dump(
            {"model_class": type(model).__name__, "model_arch_params": model_arch_params},
            f, indent=2, default=repr,
        )
    run_params["model_summary_path"] = summary_path
    run_params["model_arch_path"] = arch_path
    if tracking_active:
        run_params["mlflow_run_open"] = True


def _start_tracking(run_params, training_params, model_arch_params, ckpt_dir, platform):
    """Set up experiment tracking and start (or resume) a run.

    Returns True if an MLflow run was started, False otherwise.
    Currently supports tracking_tool values of 'mlflow' and 'tensorboard';
    None disables tracking.

    MLflow run continuation across spot-instance restarts:
    The checkpoint directory is synced to/from S3 by SageMaker's managed spot
    training between instance replacements.  When checkpointing is enabled we
    persist the run_id there so a restarted instance can resume logging into
    the same MLflow run.  The file is removed after successful job completion
    (in _save_training_outputs) so subsequent independent runs start fresh.
    """
    tracking_tool = training_params.get("tracking_tool")
    valid_tools = {None, "tensorboard", "mlflow"}
    if tracking_tool not in valid_tools:
        raise ValueError(
            f"Unsupported tracking_tool {tracking_tool!r}. "
            f"Expected one of {valid_tools - {None}} or None."
        )

    if tracking_tool != "mlflow":
        return False

    tracking_port = training_params.get("tracking_port")
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    elif tracking_port:
        mlflow.set_tracking_uri(f"http://localhost:{tracking_port}")

    experiment_name = training_params.get(
        "tracking_expt_name", run_params.get("dataset", "image_gen_training")
    )
    try:
        mlflow.set_experiment(experiment_name)
    except Exception as e:
        print(
            f"FATAL: cannot reach MLflow server at "
            f"{mlflow.get_tracking_uri()!r}: {e}",
            flush=True,
        )
        sys.exit(1)

    if mlflow.active_run():
        mlflow.end_run()

    # Check for a persisted run_id from a previous spot-instance attempt.
    # Only relevant (and only written) when checkpointing is enabled, since
    # SageMaker only syncs /opt/ml/checkpoints/ when checkpoint-config is set.
    checkpoint_enabled = training_params.get("checkpoint_every_n_epochs", 0) > 0
    run_id_file = os.path.join(ckpt_dir, _RUN_ID_FILENAME)
    existing_run_id = None
    if checkpoint_enabled and os.path.exists(run_id_file):
        with open(run_id_file) as f:
            existing_run_id = f.read().strip() or None
        if existing_run_id:
            print(
                f"Spot restart detected: resuming MLflow run {existing_run_id}.\n",
                flush=True,
            )

    if existing_run_id:
        try:
            mlflow.start_run(run_id=existing_run_id, log_system_metrics=True)
        except Exception:
            # Run was deleted or is otherwise unavailable; fall through to a fresh run.
            existing_run_id = None

    if not existing_run_id:
        dataset = run_params.get("dataset", "image_gen_run")
        model_name = run_params.get("model_name", "model")
        mlflow.start_run(run_name=f"{model_name}_{dataset}", log_system_metrics=True)

    # Tags are idempotent — safe to re-set when resuming a run.
    mlflow.set_tag("mlflow.user", os.environ.get("HOST_USER", getpass.getuser()))
    mlflow.set_tag("image_tag", os.environ.get("IMAGE_TAG", "[local]"))
    mlflow.set_tag("run_env", platform)
    _job_name = os.environ.get("JOB_NAME", "")
    if _job_name and not existing_run_id:
        mlflow.log_param("job_name", _job_name)

    # Log params only on a fresh run; MLflow raises if you try to overwrite
    # an existing param with a different value.
    if not existing_run_id:
        params_for_logging = {
            **run_params,
            **training_params,
            **model_arch_params,
            "tracking_tool": tracking_tool,
        }
        params_for_logging.pop("output_dir", None)
        _images_path_orig = os.environ.get("IMAGES_PATH_ORIG", "")
        if _images_path_orig:
            params_for_logging["images_path"] = _images_path_orig
        mlflow.log_params(_flatten_params(params_for_logging))

    # Persist run_id so a spot restart can resume this run.
    if checkpoint_enabled:
        os.makedirs(ckpt_dir, exist_ok=True)
        with open(run_id_file, "w") as f:
            f.write(mlflow.active_run().info.run_id)

    run_params["mlflow_run_id"] = mlflow.active_run().info.run_id
    return True


def _load_existing_weights(model, run_params):
    """Load weights from a completed training run for inference."""
    print(
        f"Loading model weights from file in {run_params['model_dir']}.\n",
        flush=True,
    )
    weights_path = os.path.join(run_params["model_dir"], _WEIGHTS_FILENAME)
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    run_params["model_summary_path"] = _capture_and_save_summary(
        model, run_params["output_dir"]
    )


# ---------------------------------------------------------------------------
# Public orchestrator
# ---------------------------------------------------------------------------

def default_training_sequence(
    model,
    train_loader,
    run_params,
    training_params,
    model_arch_params,
    extra_callbacks=None,
):
    """Orchestrate tracking setup, optional training, and weight saving.

    Takes an already-constructed LightningModule, a training DataLoader and
    the three parameter dicts (run_params, training_params,
    model_arch_params), and returns (model, trainer).  trainer is None for
    inference-only runs (do_train=False), in which case weights are loaded
    from run_params["model_dir"] instead.

    extra_callbacks: optional list of additional Lightning callbacks, e.g. a
    model-specific periodic evaluation callback.

    When tracking_tool='mlflow', the MLflow run is left open on return so the
    calling script can log domain-specific post-training metrics and artifacts
    before calling mlflow.end_run() itself.  run_params['mlflow_run_open'] is
    set to True to signal this to the caller.

    Note Lightning moves the model back to the CPU when fit() returns; move
    it to a GPU again (model.to(...)) before heavy post-training sampling.
    """
    platform = detect_platform()
    ckpt_dir = resolve_checkpoint_dir(run_params, platform)
    tracking_tool = training_params.get("tracking_tool")

    tracking_active = _start_tracking(
        run_params, training_params, model_arch_params, ckpt_dir, platform
    )
    _validate_weights_dest(training_params)

    trainer = None
    if run_params["do_train"]:
        print("Training model:", flush=True)

        resume_ckpt = None
        if training_params.get("checkpoint_every_n_epochs", 0) > 0:
            resume_ckpt, resume_epoch = _find_latest_checkpoint(ckpt_dir)
            if resume_ckpt:
                print(
                    f"Resuming from checkpoint epoch {resume_epoch}: {resume_ckpt}\n",
                    flush=True,
                )

        callbacks = _build_callbacks(training_params, tracking_tool, ckpt_dir)
        callbacks += list(extra_callbacks or [])

        # Lightning only supports Trainer-level gradient clipping for modules
        # using automatic optimization; manual-optimization modules (the GAN)
        # clip inside their own training_step.
        grad_clip = training_params.get("grad_norm_thresh")
        if not getattr(model, "automatic_optimization", True):
            grad_clip = None

        # Mixed precision only pays off (and is only well supported) on GPUs.
        precision = training_params.get("precision") or "32-true"
        if not torch.cuda.is_available() and str(precision).startswith("16"):
            print(
                f"Note: precision={precision!r} requested but no GPU is "
                "available; using '32-true' instead.\n",
                flush=True,
            )
            precision = "32-true"

        trainer = L.Trainer(
            max_epochs=training_params["num_epochs"],
            accelerator=training_params.get("accelerator", "auto"),
            devices=1,
            precision=precision,
            gradient_clip_val=grad_clip or None,
            callbacks=callbacks,
            logger=_build_logger(training_params, tracking_tool),
            enable_checkpointing=False,  # handled by PeriodicCheckpoint
            enable_progress_bar=platform in ("local", "docker"),
            enable_model_summary=False,
            num_sanity_val_steps=0,
            log_every_n_steps=10,
        )
        trainer.fit(model, train_dataloaders=train_loader, ckpt_path=resume_ckpt)
        print("Done training model.", flush=True)

        _save_training_outputs(
            model,
            run_params,
            training_params,
            model_arch_params,
            ckpt_dir,
            tracking_active,
        )
    else:
        _load_existing_weights(model, run_params)
        if tracking_active:
            run_params["mlflow_run_open"] = True

    return model, trainer
