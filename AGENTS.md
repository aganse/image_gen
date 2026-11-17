# Repository Guidelines

## Project Structure & Module Organization
This repository is a flat Python project of PyTorch / Lightning generative
image models (VAE, DCGAN, WGAN-GP, diffusion), structured like its sibling
repo flow_models. Core modules live at the repo root: one module per model
family (`vae_model.py`, `gan_model.py` with `augment.py`,
`diffusion_model.py`), their shared base (`gen_base.py`), generic training
orchestration (`model_training.py`), evaluation/MLflow logging
(`gen_eval.py`, `image_metrics.py`), and helpers (`file_utils.py`,
`utils.py`). Training entry points are `train_<name>.py`, each paired with
`params/params_<name>.json`; keep new experiment scripts in that pattern, with
an alphanumeric `<name>` (it becomes part of SageMaker job names).
`eval_mlflow_run.py` is a standalone metrics tool. Tests live in `tests/`.
Documentation and parameter notes are in `doc/`. Generated artifacts are
written under `model/` and `output/`. AWS/cloud job support files (CodeBuild,
ECR, SageMaker, Batch) are in `aws/`.

Keep the layering intact: `aws/` must stay free of project-specific names
(they come from variables in the root `makefile`), and `model_training.py`
must stay free of model- or image-specific code (that belongs in
`gen_eval.py` or the model modules). Both are intended to be extracted into
a standalone reusable repo later.

`TASKING_STATUS.md` records what has and has not been verified and the
next steps; read it first and update it when you finish a chunk of work.

## Build, Test, and Development Commands
- `make create-env` — create a new numbered virtualenv (for example `.venv4`) and install runtime dependencies.
- `source .venvN/bin/activate` — activate the latest environment you created.
- `make install-dev` — install developer tools from `requirements-dev.txt` before you run the tests or linting.
- `make test` — run the test suite.
- `make lint` — run the code linting check (flake8).
- `python train_vae.py` — run a local training script; update parameters in the script's `get_default_params()` first.
- `make local-build DEVICE=cpu` then `make local-run SCRIPT=vae DEVICE=cpu` — build and run the Docker image locally.

## Coding Style & Naming Conventions
Use 4-space indentation and follow PEP 8-style Python. Aim for lines of about
80-90 characters (`flake8` in `setup.cfg` does not enforce line length).
Prefer descriptive `snake_case` names for functions, variables, and files; use
`CapWords` for classes. Format with `black`, sort imports with `isort`, and
verify with `flake8` before submitting changes. This is partly a tutorial
project: keep the explanatory module docstrings current.

## Testing Guidelines
Tests use the standard `unittest` framework. Name new files
`tests/test_<feature>.py` and keep test methods as `test_<behavior>()`. All
default tests run on CPU with tiny synthetic tensors. Some tests are opt-in
via environment: `IMAGE_GEN_TEST_METRICS=1` runs real FID/KID/IS computations
(needs the Inception weights), `IMAGE_GEN_S3URI` enables the S3 loading test,
and `tests/test_gpu.py` runs only when a CUDA GPU is present. For routine
changes, run targeted tests first, then `python -m unittest -v`.

## Commit & Pull Request Guidelines
Use short, imperative subjects such as `Fix KID subset size` or
`Add spectral norm option`: one-line summary, present tense, under ~72
characters. Pull requests should explain the training or utility change, note
any config/env requirements, and include sample output paths or screenshots
of sample grids when behavior changes. Avoid committing large generated files
unless they are intentional repository fixtures.
