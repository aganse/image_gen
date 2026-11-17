"""Train a WGAN-GP to generate images (default: 64x64 cat faces).

Local use:   python train_wgangp.py        (edit the overrides below)
Docker/AWS:  make local-run SCRIPT=wgangp / make sm-submit SCRIPT=wgangp
             (params/params_wgangp.json overrides the settings below)

Requires env var IMAGES_PATH (local dir or s3:// URI containing train/).
See doc/config.md for a description of every parameter.

Same networks and training code as train_dcgan.py (see gan_model.py); only
the default parameters differ, following the WGAN-GP paper: Wasserstein
critic loss with gradient penalty, no BatchNorm in the critic, Adam betas
(0.0, 0.9), learning rate 1e-4, and 5 critic steps per generator step.
"""

import train_dcgan


def get_default_params():
    """DCGAN defaults with the WGAN-GP specific settings swapped in.

    Change here for local non-Docker usage; or change in
    params/params_wgangp.json for Docker-based runs.
    """
    run_params, training_params, model_arch_params = train_dcgan.get_default_params()
    run_params.update({
        "output_dir": "output/wgangp",
        "model_dir": "model/wgangp/cats_64x64",
        "model_name": "wgangp",
    })
    training_params.update({
        # The generator is only updated every n_critic-th step, so more epochs
        # are needed for the same number of generator updates as DCGAN.
        "num_epochs": 800,
        "learning_rate": 1e-4,
        "adam_betas": [0.0, 0.9],
        "gan_loss": "wgan-gp",
        "n_critic": 5,
        "gp_weight": 10.0,
    })
    model_arch_params.update({
        "disc_norm": "instance",  # BatchNorm is incompatible with the gradient penalty
    })
    return run_params, training_params, model_arch_params


def main():
    train_dcgan.run_gan_training(*get_default_params())


if __name__ == "__main__":
    main()
