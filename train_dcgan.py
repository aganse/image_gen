"""Train a DCGAN to generate images (default: 64x64 cat faces).

Local use:   python train_dcgan.py         (edit the dicts below)
Docker/AWS:  make local-run SCRIPT=dcgan / make sm-submit SCRIPT=dcgan
             (params/params_dcgan.json overrides the dicts below)

Requires env var IMAGES_PATH (local dir or s3:// URI containing train/).
See doc/config.md for a description of every parameter.

train_wgangp.py reuses run_gan_training() from this file with different
default parameters; both are thin wrappers over gan_model.py.
"""

import os

import utils
from file_utils import get_training_data
from gan_model import GANLitModule
from gen_base import best_device
from gen_eval import build_evaluation, final_evaluation, finish_run
from model_training import default_training_sequence, load_param_overrides


def get_default_params():
    """Default user parameter settings.

    Change here for local non-Docker usage; or change in
    params/params_dcgan.json for Docker-based runs.
    """
    run_params = {
        "output_dir": "output/dcgan",
        "model_dir": "model/dcgan/cats_64x64",
        "model_name": "dcgan",
        "dataset": "cats",
        "images_path": os.environ.get("IMAGES_PATH", "data"),
        "train_subdir": "train",  # images are read recursively from images_path/train_subdir
        "do_train": True,  # true = training, false = inference w existing model in model_dir
        "do_final_eval": True,  # generate final images + FID/KID/IS after training
        "cache_in_memory": True,  # load all training images into RAM once (best for small datasets)
        "num_workers": 0,  # DataLoader workers; keep 0 with cache_in_memory
        "num_gen_sims": 64,  # number of individual generated images to save/log at the end
        "num_grid_images": 64,  # images in the fixed-noise sample grid logged during training
        "eval_every_n_epochs": 10,  # period of sample grid + KID logging; <=0 disables
        "num_eval_sims": 1000,  # generated images per periodic KID evaluation; 0 = grids only
        "num_final_metric_sims": 5000,  # generated images for the final FID/KID/IS
        "final_metrics": ["fid", "kid", "is"],
        "num_real_metric_images": 0,  # real images used for metrics; 0 = all training images
        "kid_subset_size": 1000,
        "sample_batch_size": 64,
        "seed": 1234,
    }
    training_params = {
        "num_epochs": 300,
        "batch_size": 64,
        "learning_rate": 2e-4,  # generator Adam learning rate (float only for GANs)
        "disc_learning_rate": None,  # discriminator rate; None = same as learning_rate
        "adam_betas": [0.5, 0.999],  # DCGAN paper setting
        "gan_loss": "bce",  # "bce" (DCGAN) or "wgan-gp"
        "n_critic": 1,  # discriminator steps per generator step
        "gp_weight": 10.0,  # gradient-penalty weight (wgan-gp only)
        "diffaugment": "color,translation,cutout",  # "" disables; key for small datasets
        "ema_decay": 0.999,  # EMA of generator weights used for sampling; 0 disables
        "early_stopping_patience": 0,  # keep 0: GAN losses don't track sample quality
        "num_data_input": 0,  # cap on number of training images; 0 = use all
        "hflip_augment": True,  # random horizontal flips
        "grad_norm_thresh": None,  # if not None, clip norm of gradients at this thresh
        "precision": "32-true",  # keep full precision for GANs (esp. with gradient penalty)
        "tracking_tool": "mlflow",  # "tensorboard", "mlflow", or None
        "tracking_port": 5000,  # typ 6006 for tensorboard and 5000 for mlflow
        "tracking_expt_name": "image_gen",
        "save_model_weights": True,
        "checkpoint_every_n_epochs": 10,  # for spot-restart resume; 0 disables
    }
    model_arch_params = {
        "image_shape": [64, 64, 3],  # (height, width, channels); square, power of 2
        "latent_dim": 128,
        "gen_base_channels": 64,
        "disc_base_channels": 64,
        "disc_norm": "batch",  # "batch" (DCGAN), "instance" or "none" (WGAN-GP)
    }
    return run_params, training_params, model_arch_params


def build_model(training_params, model_arch_params):
    return GANLitModule(
        **model_arch_params,
        gan_loss=training_params["gan_loss"],
        learning_rate=training_params["learning_rate"],
        disc_learning_rate=training_params["disc_learning_rate"],
        adam_betas=training_params["adam_betas"],
        n_critic=training_params["n_critic"],
        gp_weight=training_params["gp_weight"],
        diffaugment=training_params["diffaugment"],
        ema_decay=training_params["ema_decay"],
        grad_norm_thresh=training_params["grad_norm_thresh"],
    )


def run_gan_training(run_params, training_params, model_arch_params):
    """Full GAN run: data -> train -> final images/metrics -> MLflow."""
    load_param_overrides(run_params, training_params, model_arch_params)
    os.makedirs(run_params["output_dir"], exist_ok=True)
    os.makedirs(run_params["model_dir"], exist_ok=True)
    # List the param settings:
    print("")
    utils.print_run_params(**run_params, **training_params, **model_arch_params)

    # Get the data
    # ------------
    train_dataset, train_loader = get_training_data(
        run_params, training_params, model_arch_params
    )
    print("train_loader test: shape of one batch: ", next(iter(train_loader)).shape, "\n")

    # Build and train the model
    # -------------------------
    model = build_model(training_params, model_arch_params)
    evaluator, eval_callback = build_evaluation(train_dataset, run_params)
    model, trainer = default_training_sequence(
        model,
        train_loader,
        run_params,
        training_params,
        model_arch_params,
        extra_callbacks=[eval_callback],
    )

    # Generate images, compute metrics, log results
    # ---------------------------------------------
    model.to(best_device())
    tracking_active = bool(
        training_params["tracking_tool"] == "mlflow" and run_params.get("mlflow_run_open")
    )
    if run_params["do_final_eval"]:
        final_evaluation(model, evaluator, run_params, tracking_active)
    finish_run(run_params, training_params)


def main():
    run_gan_training(*get_default_params())


if __name__ == "__main__":
    main()
