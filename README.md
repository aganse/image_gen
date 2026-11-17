# image_gen

[!WARNING]
STILL TESTING THE CODE IN THIS INITIAL COMMIT - W.I.P...

Explore and compare three families of generative image models - **VAEs**,
**GANs** (DCGAN and WGAN-GP), and **diffusion models** - trained from scratch
on the same small dataset, evaluated with the same image-quality metrics
(FID / KID / IS), and logged side by side in MLflow.

This is a PyTorch / Lightning sibling of
[flow_models](https://github.com/aganse/flow_models) (normalizing flows, in
TensorFlow) and purposely matches that repo's form: one module per model,
shared `model_training.py`, `train_<name>.py` entry-point scripts with
matching `params/params_<name>.json` files, and the Docker + `aws/`
job-submission machinery for running on SageMaker or AWS Batch.

| Script | Model | Module | In brief |
|---|---|---|---|
| `train_vae.py` | Convolutional VAE | `vae_model.py` | Stable to train, full coverage of the data, blurry samples - the baseline. |
| `train_dcgan.py` | DCGAN | `gan_model.py` | Classic adversarial training; sharp samples, less stable, can drop modes. |
| `train_wgangp.py` | WGAN-GP | `gan_model.py` | Wasserstein loss + gradient penalty; more stable GAN training, slower. |
| `train_diffusion.py` | DDPM / DDIM | `diffusion_model.py` | Hugging Face `diffusers` UNet + schedulers; best quality, slowest sampling. |

The default experiment generates 64x64 cat faces from the ~5000 cat images of
the Kaggle [animal-faces](https://www.kaggle.com/datasets/andrewmvd/animal-faces)
(AFHQ) dataset - the same data used by flow_models' image script.

> **Status:** the code in this repo was written but not yet tested
> end-to-end.  See [TASKING_STATUS.md](TASKING_STATUS.md) for exactly what has
> and has not been verified and the recommended first-run checklist.


## Modules and their relationships

```
train_vae.py  train_dcgan.py  train_wgangp.py  train_diffusion.py   <- entry points (defaults + main)
      |              \            /                   |
 vae_model.py        gan_model.py              diffusion_model.py   <- one module per model family
      \                  |  augment.py (DiffAugment)  /
       +---------- gen_base.py (common sampling interface, EMA, LR handling)
                         |
 model_training.py   generic Lightning orchestration: platform detection, params
                     overrides, MLflow run handling, checkpoints/resume, weights
 gen_eval.py         periodic sample grids + KID, final FID/KID/IS, MLflow artifacts
 image_metrics.py    FID / KID / IS (torchmetrics)
 file_utils.py       image loading from local dir or S3
 utils.py            params printing, image grids
 eval_mlflow_run.py  standalone tool: metrics for images in ANY MLflow run
 aws/                project-agnostic job-submission makefiles (SageMaker, Batch, CodeBuild/ECR)
```

All four models implement the same small interface (`make_noise`, `sample`),
so the training loop, evaluation, and logging code is shared and the results
are directly comparable.

### Results logged to MLflow

| Kind | What |
|---|---|
| Tags | `mlflow.user`, `image_tag` (Docker image tag), `run_env` (local / docker / sagemaker / awsbatch) |
| Params | every entry of `run_params`, `training_params`, `model_arch_params` (nested configs flattened to dotted keys), plus `job_name` and the original S3 `images_path` |
| Metrics, per epoch | the model's losses (`loss`, `recon_loss`, `kl_loss` / `d_loss`, `g_loss`, `d_real`, `d_fake`, `wasserstein_est`, `grad_penalty` / `loss`), MLflow system metrics (CPU, GPU, memory) |
| Metrics, every `eval_every_n_epochs` | `kid`, `kid_std` - a quality time series to watch alongside the losses |
| Metrics, final | `fid_final`, `kid_final`, `kid_final_std`, `is_final`, `is_final_std` |
| Artifacts | `sample_grids/samples_epoch_NNNN.png` (same fixed noise each time, to watch quality evolve), `sim_images/sim_NNNNN.png` (`num_gen_sims` individual generated images), `plots/sim_grid.png`, `plots/real_grid.png`, `plots/reconstructions.png` (VAE), `metrics/final_metrics.json`, `model/model_summary.txt`, `model/model_arch.json`, `run_parameters.txt` |

Model weights are saved to `WEIGHTS_PATH` (local dir or S3), not to MLflow.

### Reading the metrics

- **KID** (lower is better) is the metric to trust here: it is unbiased for
  small sample counts, which matters with only ~5000 real images.
- **FID** (lower is better) is the usual headline number but is biased upward
  for small sample counts - only compare FIDs computed from the same number of
  images (the default is 5000 generated vs all training images, for all models).
- **IS** (higher is better) rewards diversity across ImageNet classes, so on a
  single-category dataset like cats it is only a rough sharpness indicator.
  To make it meaningful, train on several categories at once: put e.g. the
  cat, dog and wild folders under your `train/` directory (the models are
  unconditional, so no code change is needed - just ~3x the compute per epoch).
- GAN losses do **not** indicate sample quality; use the KID curve and the
  sample grids.

### Is ~5000 images enough?

Yes, with the standard precautions this repo builds in: random horizontal
flips for all models; for the GANs, differentiable augmentation
([DiffAugment](https://arxiv.org/abs/2006.10738)) of everything the
discriminator sees, without which a discriminator memorizes 5000 images and
training collapses; and for GANs and diffusion, sampling from an exponential
moving average (EMA) of the weights.  AFHQ-Cat is itself a standard
"limited data" benchmark.


## Running the Training

To train the models in this repo you have a number of options for platforms,
with CPU or GPU in all:
1. locally in a Python venv without Docker (on own machine or cloud instance)
2. locally in a Docker container (on own machine or cloud instance)
3. cloud training via SageMaker Training Jobs (for full single runs)
4. cloud training via AWS Batch (for queued/parallel job sweeps)

In everything below, `SCRIPT=name` is one of `vae`, `dcgan`, `wgangp`,
`diffusion`, and selects both `train_<name>.py` and `params/params_<name>.json`.

Realistically the VAE and GANs train in a practical time only on a GPU, and
the diffusion model needs one.  For a first smoke test on any machine, set
`num_epochs` to 1-2, `num_data_input` to ~200, `eval_every_n_epochs` to 1,
`num_eval_sims` to ~100 and `num_final_metric_sims` to ~200.

Also note that [doc/all_makefile_targets.md](doc/all_makefile_targets.md)
lists all makefile targets from the top-level, SageMaker, and AWS Batch makefiles,
and [doc/config.md](doc/config.md) describes every parameter.

### Training data

Use the Kaggle [animal-faces](https://www.kaggle.com/datasets/andrewmvd/animal-faces)
dataset (or any images you like), in the same layout as flow_models so the
same `IMAGES_PATH` works for both repos:

```
data/                <-- or s3://mybucket/prefix/     (this is IMAGES_PATH)
    train/
        cat/         <-- everything under train/ is used, subdirectories merged
    val/             <-- not used by this repo
```

- for local runs (options 1-2): set `IMAGES_PATH` to the local directory.
- for cloud runs (options 3-4): upload to S3 and set `IMAGES_PATH=s3://mybucket/prefix`.

By default all training images are read once at startup, resized, and held in
RAM (5000 images at 64x64 is ~60MB), so nothing is copied to the instance's
disk and S3 is not re-read every epoch.  On SageMaker the S3 prefix is mounted
in "FastFile" mode, which streams objects on demand.  See `cache_in_memory`
in [doc/config.md](doc/config.md) for larger datasets.

### Instructions for each platform option

Click to expand your chosen run option:

<details>
<summary><h4>1. Directly in Python (local, in python virtual environment, no Docker)</h4></summary>

1. Create and enter the python virtual environment:
   ```
   make create-env             # creates .venvN and pip-installs requirements.txt
   source .venvN/bin/activate  # enter desired venv (N increments with each create-env call)
   make install-dev            # optional: installs dev/test/linting tools (requirements-dev.txt)
   ```
   (`make create-env` uses `python3.12` by default; override with
   `make create-env PYTHON_BIN=python3.11`.  For a specific CUDA build of
   PyTorch, install torch/torchvision first per https://pytorch.org/get-started/locally/.)
2. Set environment variables for settings that shouldn't be in the repo:
   - `IMAGES_PATH`: training data images location; can be local path or S3 URI.
   - `WEIGHTS_PATH`: location to save final model weights when
     "training_params":"save_model_weights" is true; optional.
   - `AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION` if either
     `IMAGES_PATH` or `WEIGHTS_PATH` is an S3 URI.  (Alternately `~/.aws` credentials
     are used automatically by boto3 if available.)
   - `MLFLOW_TRACKING_URI`: the MLflow tracking server address (if MLflow is used),
     which for these test runs might often be `http://localhost:5000`.
3. Edit the parameter dicts in `get_default_params()` near the top of
   `train_<name>.py` to set desired hyperparameters. Refer to
   [`doc/config.md`](doc/config.md) for a description of every parameter.
   (Note: `params/params_<name>.json` is only read in the Docker image based
   runs, those are NOT USED in this direct-Python mode.)
4. Run `python train_<name>.py`.

The first run that computes metrics downloads the Inception-v3 weights
(~100MB) into `TORCH_HOME` (default `~/.cache/torch`).
</details>
<!-- ─────────────────────────────────────────────────────────────── -->
<details>
<summary><h4>2. Locally in Docker (CPU or GPU, home machine or EC2 instance)</h4></summary>

Note `make local-build` here uses your local working directory as the build
context.  It COPYs whatever .py files exist on disk right now, including
uncommitted changes, because the main purpose of this option is testing.
To configure a GPU-enabled EC2 instance in AWS, you can
follow [these instructions](https://github.com/aganse/py_tf2_gpu_dock_mlflow/blob/main/doc/aws_ec2_install.md).
1. Set environment variables for settings that shouldn't be in the repo:
   - `IMAGES_PATH`: training data images location; can be local path or S3 URI.
   - `WEIGHTS_PATH`: location to save final model weights when
     "training_params":"save_model_weights" is true; optional.
   - `AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION` if either
     `IMAGES_PATH` or `WEIGHTS_PATH` is an S3 URI.  (Alternately note `~/.aws` is
     mapped into the container if it exists.)
   - `MLFLOW_TRACKING_URI`: the MLflow tracking server address (if MLflow is used)
     as seen from INSIDE the Docker container, using the default
     Docker bridge gateway IP, by numeric IP not hostname, as MLflow now
     requires allowList settings for hostnames:
     `export MLFLOW_TRACKING_URI=http://192.168.65.254:5000` (local macOS, i.e.
     host.docker.internal) or
     `export MLFLOW_TRACKING_URI=http://172.17.0.1:5000` (local linux)
2. Edit `params/params_<name>.json` to set the desired hyperparameters. Refer to
   [`doc/config.md`](doc/config.md) for a description of every parameter.
3. Run:
    ```bash
    make local-build DEVICE=cpu            # or =gpu  / installs the cpu or CUDA build of PyTorch
    make local-run SCRIPT=vae DEVICE=cpu   # or =gpu
    ```
See [`aws/job-support-common/README.md`](aws/job-support-common/README.md) for more setup
and running details of local Docker images.
</details>
<!-- ─────────────────────────────────────────────────────────────── -->
<details>
<summary><h4>3. Cloud training via SageMaker Training Jobs (recommended for full single runs)</h4></summary>

Note `make build` here triggers AWS CodeBuild with `--source-version BRANCH`,
which pulls from **GitHub** at that branch to build the Docker image in AWS.  Code
must be pushed to GitHub to be picked up.  No EC2 setup needed — training runs
entirely on independent AWS services.
1. Set environment variables for settings that shouldn't be in the repo:
   - `AWS_ACCT_ID`: your AWS account ID (12-digit number).
   - `AWS_REGION`: AWS region for all resources (e.g. `us-west-2`).
   - `SM_SUBNET`: VPC subnet ID for SageMaker compute instances.
   - `SM_SG`: security group ID for SageMaker compute instances.
   - `IMAGES_PATH`: S3 URI of training data.
   - `WEIGHTS_PATH`: S3 URI for saving final and checkpointed model weights.
   - `MLFLOW_TRACKING_URI`: MLflow tracking server address accessible from within
     your AWS VPC (e.g. `http://10.0.1.50:5000`).
2. One-time only: create this project's ECR repository and CodeBuild project
   (`make create-ecr-repo`, `make create-codebuild-project`), and the SageMaker
   execution IAM role if it doesn't already exist from flow_models
   (`make sm-create-role`).
3. Edit `params/params_<name>.json` to set the desired hyperparameters. Refer to
   [`doc/config.md`](doc/config.md) for a description of every parameter.
4. Run:
    ```bash
    make build BRANCH=myfeature DEVICE=gpu      # default BRANCH=main, default DEVICE=gpu
                                                # (also note build-status and build-logs)
    make sm-submit SCRIPT=diffusion             # uses :latest (points to main-gpu image)
    make sm-submit SCRIPT=dcgan TAG=mybranch-gpu  # use a specific image tag
                   SPOT=1 SM_MAX_WAIT=90000     # optional args:
                                                # SPOT=1 uses a spot instance for the run
                                                # SM_MAX_WAIT is total wall-clock time
                                                #   limit for the job (seconds), including
                                                #   all spot instances and lag between them.
                   SM_INSTANCE_TYPE=ml.g4dn.2xlarge  # override default ml.g4dn.xlarge
                                                # (also note sm-list, sm-status, sm-logs)
    ```
With `SPOT=1` and `checkpoint_every_n_epochs > 0`, an interrupted job resumes
from its last checkpoint and keeps logging into the same MLflow run.

Training jobs run in a private subnet with no internet access, so anything
normally downloaded at run time must be in the Docker image: the Dockerfile
bakes in the Inception weights used by the metrics.

See [`aws/job-support-sagemaker/README.md`](aws/job-support-sagemaker/README.md) for setup
and monitoring details.
</details>
<!-- ─────────────────────────────────────────────────────────────── -->
<details>
<summary><h4>4. Cloud training via AWS Batch (for queued/parallel job sweeps)</h4></summary>

Like option 3, `make build` pulls from GitHub so code must be pushed first.
AWS Batch requires more one-time infrastructure setup than SageMaker — see
[`aws/job-support-awsbatch/README.md`](aws/job-support-awsbatch/README.md) for the full one-time
setup steps before running jobs for the first time.
1. Set environment variables for settings that shouldn't be in the repo:
   - `AWS_ACCT_ID`: your AWS account ID (12-digit number).
   - `AWS_REGION`: AWS region for all resources (e.g. `us-west-2`).
   - `AWSBATCH_SUBNET`: VPC subnet ID for Batch compute instances.
   - `AWSBATCH_SG`: security group ID for Batch compute instances.
   - `IMAGES_PATH`: S3 URI of training data.
   - `WEIGHTS_PATH`: S3 URI for saving final model weights; optional.
   - `MLFLOW_TRACKING_URI`: MLflow tracking server address accessible from within
     your AWS VPC (e.g. `http://10.0.1.50:5000`).
2. Edit the parameter dicts in `get_default_params()` of `train_<name>.py` to
   set desired hyperparameters. (Note: as in flow_models,
   `params/params_<name>.json` override support for Batch is not yet
   implemented; parameters are set inline in the script for now.)
3. Re-run `make register-job-definition` whenever MLFLOW_TRACKING_URI, IMAGES_PATH,
   or WEIGHTS_PATH change, since these are baked into the job definition at
   registration time.
4. Run:
    ```bash
    make build BRANCH=myfeature DEVICE=gpu      # default BRANCH=main, default DEVICE=gpu
                                                # (also note build-status and build-logs)
    make batch-submit SCRIPT=vae                # submit job; prints JOBID immediately
                                                # (also note batch-list, batch-status, batch-logs)
    ```
See [`aws/job-support-awsbatch/README.md`](aws/job-support-awsbatch/README.md) for full setup
and submission instructions.
</details>


### Reinstantiating a model from saved weights
`model_dir` receives `model_weights.pt` (a PyTorch state dict, when
`save_model_weights` is true), `model_arch.json` and `model_summary.txt`.
Rebuild the model with the same parameters, then load the weights:
```python
import json, torch
from train_dcgan import build_model, get_default_params

_, training_params, model_arch_params = get_default_params()
with open("model/dcgan/cats_64x64/model_arch.json") as f:
    model_arch_params.update(json.load(f)["model_arch_params"])
model = build_model(training_params, model_arch_params)
model.load_state_dict(torch.load("model/dcgan/cats_64x64/model_weights.pt", map_location="cpu"))
images = model.sample(n=16)   # float tensor (16, 3, 64, 64) in [-1, 1]
```
Or set `"do_train": false` in the run params to rerun just the generation and
metrics steps of a script from the weights in `model_dir`.


## Scoring images from other MLflow runs (e.g. flow_models)

`eval_mlflow_run.py` computes the same FID / KID / IS for the images logged
in *any* MLflow run, so models from other projects can be compared on equal
terms (same preprocessing, same real images, same metric settings):

```bash
# inside the project's Docker image (metric code + Inception weights baked in):
make local-build DEVICE=cpu
make eval-run RUN_ID=<mlflow-run-id> REAL_IMAGES=s3://mybucket/afhq/train \
     ARTIFACT_PATH=sim_images IMAGE_SIZE=128 LOG=1 DEVICE=cpu

# or directly in a venv; also works on plain directories without MLflow:
python eval_mlflow_run.py --run-id <mlflow-run-id> --artifact-path sim_images \
       --real-images data/train --image-size 128 --log-to-run
python eval_mlflow_run.py --fake-images some/dir --real-images data/train
```

`LOG=1` / `--log-to-run` writes the results back onto that run as
`eval_fid`, `eval_kid`, ... metrics.  The run being scored must have an
artifact folder of **individual** image files (not one tiled grid or plot),
and needs many of them for the numbers to mean anything - several hundred at
the very least for KID, ideally a few thousand for FID.  (flow_models
currently logs only a handful of generated images inside plots, so it needs
a small change to log a large batch of individual images before this tool is
useful on its runs.)


## Tests

```bash
make install-dev
make test          # python -m unittest -v
make lint          # flake8
```
All default tests run on CPU with tiny synthetic data and need no AWS access.
Optional extras: `IMAGE_GEN_TEST_METRICS=1` also runs the tests that compute
real FID/KID/IS (downloads the Inception weights on first use);
`IMAGE_GEN_S3URI=s3://bucket/prefix` also tests S3 image loading; the GPU
tests run automatically when a CUDA GPU is present.


## Versioning

The version lives in `.bumpversion.toml` (`current_version`). Releases are cut
with `make release PART=patch|minor|major`, which bumps, git-tags `vX.Y.Z`,
and publishes a GitHub release; see
[doc/all_makefile_targets.md](doc/all_makefile_targets.md).


## References

### Papers
* VAE
  - [Kingma & Welling 2014, Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
  - [Higgins et al. 2017, beta-VAE](https://openreview.net/forum?id=Sy2fzU9gl)
* GANs
  - [Goodfellow et al. 2014, Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
  - [Radford et al. 2016, DCGAN](https://arxiv.org/abs/1511.06434)
  - [Gulrajani et al. 2017, WGAN-GP](https://arxiv.org/abs/1704.00028)
  - [Zhao et al. 2020, Differentiable Augmentation (DiffAugment)](https://arxiv.org/abs/2006.10738)
  - [Karras et al. 2020, Training GANs with Limited Data (ADA)](https://arxiv.org/abs/2006.06676)
* Diffusion
  - [Ho et al. 2020, DDPM](https://arxiv.org/abs/2006.11239)
  - [Song et al. 2021, DDIM](https://arxiv.org/abs/2010.02502)
* Metrics
  - [Heusel et al. 2017, FID](https://arxiv.org/abs/1706.08500)
  - [Binkowski et al. 2018, KID](https://arxiv.org/abs/1801.01401)
  - [Salimans et al. 2016, Inception Score](https://arxiv.org/abs/1606.03498)
* Dataset
  - [Choi et al. 2020, StarGAN v2 (introduces AFHQ)](https://arxiv.org/abs/1912.01865)

### Libraries
* [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/)
* [Hugging Face diffusers](https://huggingface.co/docs/diffusers) -
  [unconditional training example](https://huggingface.co/docs/diffusers/training/unconditional_training)
* [torchmetrics image metrics](https://lightning.ai/docs/torchmetrics/stable/)
* [PyTorch DCGAN tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
