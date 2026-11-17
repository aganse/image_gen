# TASKING_STATUS — image_gen initial build-out

Written 2026-11-16 at the end of the session that generated this repo, so a
future session (AI or human) can pick up exactly where it left off.

## 1. Bottom line

The full repo was **written but almost none of it has been executed**.  The
build environment had no PyTorch, Lightning, diffusers, torchmetrics, MLflow,
boto3, Docker, AWS CLI or flake8, and could not install packages.

| Check | Status |
|---|---|
| All `.py` files byte-compile (`python -m py_compile`, Python 3.13) | done, passes |
| Home-made AST check: unused imports, undefined names, tabs, trailing whitespace | done, clean |
| `flake8` | **NOT run** (not installed) — expect a few style nits |
| `tests/test_params_files.py` (needs no ML packages) | **run, 8/8 pass** |
| All other unit tests (106 test methods in 12 modules) | **NOT run** |
| `make -n` dry-runs of `sm-submit`, `local-build`, `local-run`, `batch-submit`, `eval-run`, `create-codebuild-project` | done; generated commands inspected and look right |
| `bash -n` on `entrypoint.sh` + its missing-TRAINING_SCRIPT error path | done |
| Any training run, Docker build, AWS submission, MLflow logging | **NOT run** |

So: treat every model/training/metrics code path as unverified until the
checklist in section 5 has been worked through.  API usage of Lightning,
diffusers, torchmetrics and MLflow was written from memory of their
documented interfaces, not checked against installed versions.

## 2. What was built (vs. the agreed spec)

Agreed spec, in brief: a PyTorch/Lightning sibling of `aganse/flow_models`
(reference commit `e3f3e5a`) comparing VAE, DCGAN, WGAN-GP and pixel-space
diffusion, all trained from scratch on ~5000 AFHQ cat images at 64x64; same
repo shape, Dockerfile and `aws/` job submission; everything of interest
logged to MLflow, including FID/KID/IS via torchmetrics; unit tests, README,
makefile; no pretrained models, latent diffusion or fine-tuning.

| Spec item | Delivered |
|---|---|
| One module per model | `vae_model.py`, `gan_model.py` (+ `augment.py` DiffAugment), `diffusion_model.py`; shared interface/EMA/LR code in `gen_base.py` |
| HF module swappable by name | `diffusion_model.build_hf_component(class_name, config, pretrained)` resolves any `diffusers` class; denoiser, training scheduler and sampler are all chosen by name in the params file.  A unit test builds `AutoencoderKL` this way. |
| `model_training.py` mirroring flow_models | Same functions and flow: `detect_platform`, `resolve_checkpoint_dir`, `load_param_overrides`, `_start_tracking` (tags, params, spot-restart run resume via run-id file), `_validate_weights_dest`, `_save_training_outputs` (weights to `WEIGHTS_PATH`, checkpoint cleanup, summary + arch files), `default_training_sequence` leaving the MLflow run open for the caller |
| `train_<name>.py` + `params/params_<name>.json` | `vae`, `dcgan`, `wgangp`, `diffusion`; `train_wgangp.py` is a thin wrapper over `train_dcgan.run_gan_training` |
| MLflow contents | See the table in README ("What lands in MLflow for every run") |
| FID/KID/IS | `image_metrics.py`; KID as a per-`eval_every_n_epochs` time series + fixed-noise sample grids (`gen_eval.GenEvalCallback`); full FID/KID/IS at the end (`gen_eval.final_evaluation`) |
| N sim images to MLflow | `num_gen_sims` individual PNGs under `sim_images/`, plus grids under `plots/` |
| Data loading | RAM cache by default (`file_utils.ImageFolderDataset`), works for local dirs, the SageMaker FastFile mount and `s3://` URIs; lazy mode via `cache_in_memory=false` |
| Dockerfile + `aws/` carried over | Yes, with the changes listed in section 3 |
| Standalone metrics tool (added late in spec discussion) | `eval_mlflow_run.py` + `make eval-run` |
| Tests, README, makefile, docs | `tests/` (13 test modules), `README.md`, `doc/config.md`, `doc/all_makefile_targets.md`, `AGENTS.md`/`CLAUDE.md`, `TODO.md` |

## 3. Where the result differs from the original plan

### 3a. `aws/` is not verbatim (as agreed) — full diff in `doc/aws_changes_vs_flow_models.diff`

Agreed changes (project names lifted into variables so `aws/` is project-agnostic):
- Root `makefile` now defines `PROJECT_NAME`, `GITHUB_REPO_URL`, `DEFAULT_SCRIPT`
  (and optionally `NO_IMAGES_SCRIPTS`); `makefile-common.mk` errors if the first two are missing.
- `ECR_REPO`, `CODEBUILD_PROJ`, `SM_JOB_PREFIX` derive from `PROJECT_NAME` (`?=`, overridable).
- `SCRIPT=<name>` now means `train_<name>.py` + `params/params_<name>.json`
  (was numeric `train_flowmodelsN.py` / `paramsN.json`); `entrypoint.sh` follows the same
  convention and fails clearly if `TRAINING_SCRIPT` is unset or the script is missing.
- SageMaker job names are `<prefix>-<name>-<instance>[-spot]-<timestamp>`, e.g.
  `imagegen-diffusion-g4dn-xl-20261116-190517` (a hyphen was added after the prefix).
- The hard-coded "script 1 needs no images" checks became `NO_IMAGES_SCRIPTS`.

Changes that were NOT discussed beforehand — please review:
1. **Spot checkpoint path (looks like an upstream bug).** `SM_SPOT_FLAGS` was defined
   with `:=`, so `$(JOB_NAME)` expanded to empty at parse time and every spot job
   used the same prefix `$(WEIGHTS_PATH)/checkpoints//`.  With four different models
   here, a stale checkpoint from one interrupted job could be loaded by another.
   Changed to `=` so each job gets `checkpoints/<job-name>/`.  Probably worth porting to flow_models.
2. **Trailing spaces in `BRANCH`/`DEVICE` defaults (looks like an upstream bug).**
   `BRANCH ?= main  # comment` makes the value `"main  "` in GNU make, which breaks
   image tags when `BRANCH` isn't given on the command line.  Comments moved to their own lines.
3. **Params encoding in `sm-submit`.** SageMaker limits a hyperparameter value to 2500
   characters; `params_diffusion.json` base64-encoded raw is ~2300.  The file is now
   minified with a `python3` one-liner before encoding (~1830), with explicit errors for
   a missing/invalid file or an over-limit result.  This also removes the macOS/Linux
   `base64` flag difference.
4. **`DEVICE` build-arg.** `local-build` and `buildspec.yml` now pass
   `--build-arg DEVICE=...` so the Dockerfile can install CPU or CUDA PyTorch wheels.
5. **Batch.** `batch-submit` accepts `SCRIPT=name` (passed as a `TRAINING_SCRIPT`
   override; job name `imagegen-<name>`), and the job definition's default
   `TRAINING_SCRIPT` comes from `DEFAULT_SCRIPT`.  Params-file overrides for Batch are
   still unimplemented, as in flow_models.
6. **Root makefile no longer does `export DEVICE=cpu`.** In flow_models that line
   overrides the documented `DEVICE ?= gpu` default.  Here `DEVICE` defaults to `gpu`
   as the READMEs say; pass `DEVICE=cpu` explicitly for local CPU builds.

### 3b. Other deviations and additions
- **`default_training_sequence` takes an already-built LightningModule** plus the
  DataLoader (flow_models constructs `FlowModel` inside it).  This keeps
  `model_training.py` model-agnostic for the planned extraction.
- **`detect_platform` returns a fourth value, `docker`,** and checks Batch first.
  flow_models reports `sagemaker` for any container because `/opt/ml` exists in the
  image; here SageMaker is identified by `TRAINING_JOB_NAME` / `TRAINING_JOB_ARN` /
  `resourceconfig.json`.  Consequence: local-Docker and Batch runs checkpoint to
  `<model_dir>/checkpoints`, only real SageMaker jobs to `/opt/ml/checkpoints`.
  **Verify on the first SageMaker run that `run_env` is tagged `sagemaker`.**
- **MLflow params:** nested dicts (e.g. `hf_model_config`) are flattened one level to
  dotted keys; `job_name` is only logged on a fresh run (not on spot resume).
- **MLflow run name** is `<model_name>_<dataset>` (flow_models: `<dataset>_<num_gen_sims>`).
- **Checkpoints** are full Lightning `.ckpt` files written by a small custom callback
  (`PeriodicCheckpoint`, keeps last 2); final weights are a state dict `model_weights.pt`;
  `model_arch.json` holds the arch params rather than a Keras-style serialized model.
- **No `restore_best_weights`** on early stopping (Lightning has no equivalent); early
  stopping is off by default for all four models anyway.
- **`augmentation_factor` and `jit_compile` have no counterpart;** one epoch is one pass
  over the data with random flips applied on the fly.
- **Extra modules beyond the plan:** `gen_base.py`, `augment.py`, `gen_eval.py`.
- **GAN learning rates must be scalars** (no schedule with the two-optimizer loop).
- **Precision:** diffusion defaults to `16-mixed` (auto-falls back to `32-true` without
  a GPU); VAE/GANs default to `32-true`.
- **Dockerfile:** same base image; PyTorch pinned via build args
  (`TORCH_VERSION=2.6.0`, `TORCHVISION_VERSION=0.21.0`, `TORCH_CUDA=cu126`) and installed
  before `requirements.txt`; Inception weights baked in with `TORCH_HOME=/opt/torch_home`;
  the XLA-specific `cuda-nvvm` apt package dropped.  **These version pins are from
  memory and unverified — confirm the cu126 wheels exist for that pair, or bump.**
- **`sagemaker-training`** was kept in `requirements.txt` for parity but nothing here
  imports it; drop it if it causes dependency conflicts.
- **`setup.cfg`, `LICENSE`, `.gitignore`, `requirements-dev.txt`** are copied from
  flow_models (`.gitignore` comment renamed); `.bumpversion.toml` starts at `0.1.0`.
- Not carried over: `misc/`, `doc/` images, `FlowModel`-specific analysis (latent
  scatter plots, PCA sampling, interpolation).

## 4. Known risks — the places most likely to need a fix on first run

1. **Lightning checkpoint resume** (`PeriodicCheckpoint` saves in `on_train_epoch_end`;
   resume via `trainer.fit(ckpt_path=...)`).  Expected: resuming from `ckpt-0003` continues
   at the 4th epoch.  `tests/test_model_training.py::test_checkpoints_are_written_pruned_and_resumed`
   checks this; if it fails, the epoch may be repeated or off by one.
2. **GAN manual optimization** (`gan_model.GANLitModule.training_step`): two optimizers,
   `n_critic`, gradient penalty with `create_graph=True`, optional `clip_gradients`.
3. **diffusers details:** `DDIMScheduler.from_config(ddpm.config)`, default `UNet2DModel`
   config validity, and the tiny configs used in `tests/test_diffusion_model.py`.
4. **torchmetrics details:** `reset_real_features=False` reuse, setting
   `kid.subset_size` after construction, and `torch-fidelity` weight caching under `TORCH_HOME`.
5. **MLflow:** `log_system_metrics=True` needs `psutil` (+ `pynvml` for GPU);
   `tests/test_model_training.py::test_mlflow_tracking_...` uses a local `file://` store,
   which recent MLflow versions may warn about or reject (switch to `sqlite:///` if so).
6. **Epoch-end metrics in `trainer.callback_metrics`** being available to
   `MLflowLoggingCallback` at `on_train_epoch_end` (assumed, as for Lightning's own callbacks).
7. **Default hyperparameters are untuned starting points** (epochs: VAE 200, DCGAN 300,
   WGAN-GP 800, diffusion 500; all 64x64, batch 64).  Runtime and quality on
   `ml.g4dn.xlarge` are unmeasured.  Diffusion evaluation cost is roughly
   images x `num_inference_steps` UNet evaluations.
8. **Final-evaluation memory:** 5000 generated 64x64 float images (~250MB) are held in RAM; fine
   at 64x64, reconsider at higher resolution.
9. **flake8** has not been run; `max-complexity = 18` could flag `_start_tracking`
   (it is the same shape as flow_models' version, which presumably passes).

## 5. Next steps, in order

### Step 1 — local, CPU, no AWS (expect to fix small things here)
```bash
make create-env && source .venvN/bin/activate && make install-dev
make lint
make test                                   # fix failures before going further
IMAGE_GEN_TEST_METRICS=1 python -m unittest -v tests.test_image_metrics tests.test_eval_mlflow_run
```

### Step 2 — local smoke training run of each script
With an MLflow server reachable (`MLFLOW_TRACKING_URI`) and `IMAGES_PATH` set, edit
`get_default_params()` temporarily (or use Docker + the params files): `num_epochs` 2,
`num_data_input` 200, `eval_every_n_epochs` 1, `num_eval_sims` 100,
`num_final_metric_sims` 200, `kid_subset_size` 100, `checkpoint_every_n_epochs` 1; for
diffusion also `num_inference_steps` 5.  Then `python train_vae.py`, etc.  Check in MLflow:
params, tags, per-epoch losses, `kid` series, `sample_grids/`, `sim_images/`, `plots/`,
`*_final` metrics, `model/` artifacts.  Also try `"do_train": false` afterwards.

### Step 3 — Docker
```bash
make local-build DEVICE=cpu          # verifies torch pins, requirements, weight baking
make local-run SCRIPT=vae DEVICE=cpu # with smoke-test values in params/params_vae.json
make eval-run RUN_ID=<run from above> REAL_IMAGES=$IMAGES_PATH/train DEVICE=cpu
```
Confirm the run needs no network for the Inception weights (e.g. `docker run --network none ...`
for a directory-mode `eval_mlflow_run.py` call).

### Step 4 — AWS
1. Create the GitHub repo `aganse/image_gen` and push (CodeBuild builds from GitHub).
2. One-time: `make create-ecr-repo`, `make create-codebuild-project` (roles/subnet/SG are
   shared with flow_models if already created).
3. `make build`, `make create-vpc-endpoints VPC_ID=...`, then a short
   `make sm-submit SCRIPT=vae` with smoke-test params; check `run_env=sagemaker` in MLflow,
   weights at `WEIGHTS_PATH/<job-name>/model_weights.pt`.
4. `make sm-submit SCRIPT=vae SPOT=1` to exercise checkpoint sync and (if interrupted) run resume.
5. Full runs of all four; `make delete-vpc-endpoints` afterwards.

### Step 5 — finalize
- Tune defaults from the real runs; record runtimes/costs and a results table + sample
  grids in the README; remove the "Status" note there.
- Decide whether to port section 3a items 1-3 back to flow_models.
- Remaining ideas are in `TODO.md` (CI workflows, Batch params overrides, extracting
  `aws/` + `model_training.py`, latent diffusion / fine-tuning, multi-category IS run,
  flow_models change to log individual sim images for `eval_mlflow_run.py`).
