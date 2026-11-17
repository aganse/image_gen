# Configuration Reference

This guide lists the user-facing parameters exposed by the training scripts:

- **V** &rarr; `train_vae.py`
- **G** &rarr; `train_dcgan.py` and `train_wgangp.py` (same parameters, different defaults)
- **D** &rarr; `train_diffusion.py`

The `Used by` column indicates which scripts consume each parameter.  Default
values are defined in `get_default_params()` inside each script (used for
direct-Python runs) and mirrored in `params/params_<name>.json` (used for
Docker, SageMaker runs), so this document focuses on intent and usage.

Environment-specific settings are passed as environment variables, never in
the params files: `IMAGES_PATH`, `WEIGHTS_PATH`, `MLFLOW_TRACKING_URI`.

Note for SageMaker runs: the params file is minified, base64-encoded and
passed as one hyperparameter, which SageMaker limits to 2500 characters.
`make sm-submit` checks this; the files in this repo are at ~1300-1850.

## Run Parameters

| Parameter | Description | Used by |
| --- | --- | --- |
| `output_dir` | Folder for sample grids, generated images, metrics JSON and other run outputs. | V,G,D |
| `model_dir` | Folder used to save/load model weights, summary, arch file, and (non-SageMaker) checkpoints. | V,G,D |
| `model_name` | Short label used in the MLflow run name (`<model_name>_<dataset>`). | V,G,D |
| `dataset` | Label for the dataset, used in the MLflow run name and logged as a param. | V,G,D |
| `images_path` | Root directory or S3 URI of the image data; taken from env var `IMAGES_PATH`. | V,G,D |
| `train_subdir` | Subdirectory of `images_path` holding the training images (read recursively). Use e.g. `train/cat` to restrict to one class folder. | V,G,D |
| `do_train` | When `true`, train; when `false`, load weights from `model_dir` and only run the generation/metrics steps. | V,G,D |
| `do_final_eval` | Generate final images and compute the final metrics after training. | V,G,D |
| `cache_in_memory` | Read all training images once and hold them in RAM as uint8 (fast, no disk use; ~60MB for 5000 images at 64x64). Set `false` for datasets too large for RAM (then set `num_workers` > 0). | V,G,D |
| `num_workers` | DataLoader worker processes. Keep `0` with `cache_in_memory`. | V,G,D |
| `num_gen_sims` | Number of generated images saved as individual PNGs and logged to MLflow (`sim_images/`) at the end. | V,G,D |
| `num_grid_images` | Number of images in the fixed-noise sample grid logged during training. | V,G,D |
| `eval_every_n_epochs` | Period (epochs) of the sample grid + KID logging during training; `<=0` disables. | V,G,D |
| `num_eval_sims` | Generated images per periodic KID evaluation; `0` logs grids only. The main cost knob for diffusion runs. | V,G,D |
| `num_final_metric_sims` | Generated images used for the final FID/KID/IS. Keep equal across models you want to compare (FID depends on it). | V,G,D |
| `final_metrics` | Which final metrics to compute: any of `"fid"`, `"kid"`, `"is"`; empty list skips. | V,G,D |
| `num_real_metric_images` | Number of training images used as the real reference set for metrics; `0` = all. | V,G,D |
| `kid_subset_size` | KID subset size (automatically reduced if fewer images are available). | V,G,D |
| `sample_batch_size` | Batch size used when generating images. | V,G,D |
| `num_recon_images` | Number of training images shown with their reconstructions in `plots/reconstructions.png`; `0` skips. | V |
| `seed` | Seed for the fixed grid noise and evaluation sampling (not for weight init / data order). | V,G,D |

## Training Parameters

| Parameter | Description | Used by |
| --- | --- | --- |
| `num_epochs` | Number of training epochs (one epoch = one pass over the training images). | V,G,D |
| `batch_size` | Training mini-batch size. | V,G,D |
| `learning_rate` | Optimizer learning rate. A scalar gives a constant rate; a list `[initial_rate, decay_steps, decay_rate]` gives a staircase exponential decay (`initial_rate * decay_rate ^ floor(step / decay_steps)`). GANs accept a scalar only (generator rate). | V,G,D |
| `disc_learning_rate` | Discriminator/critic learning rate; `null` = same as `learning_rate`. | G |
| `adam_betas` | Adam `[beta1, beta2]`: `[0.5, 0.999]` for DCGAN, `[0.0, 0.9]` for WGAN-GP. | G |
| `gan_loss` | `"bce"` (DCGAN, non-saturating) or `"wgan-gp"` (Wasserstein critic + gradient penalty). | G |
| `n_critic` | Discriminator/critic updates per generator update (1 for DCGAN, 5 for WGAN-GP). | G |
| `gp_weight` | Gradient-penalty weight (WGAN-GP only). | G |
| `diffaugment` | Comma-separated DiffAugment policy applied to real and fake images before the discriminator: any of `color`, `translation`, `cutout`; `""` disables. Essential on small datasets. | G |
| `ema_decay` | Decay of the exponential moving average of generator/denoiser weights used for sampling; `0` disables EMA. | G,D |
| `kl_weight` | Weight (beta) on the KL term of the VAE loss. Lower = sharper reconstructions but worse samples. | V |
| `kl_warmup_epochs` | Linearly ramp the KL weight from 0 over this many epochs; `0` disables. | V |
| `recon_loss` | VAE reconstruction loss: `"mse"` or `"l1"`. | V |
| `weight_decay` | Optimizer weight decay. | V,D |
| `early_stopping_patience` | Patience (epochs) for early stopping on the training `loss`; `<=0` disables. Keep disabled for GANs (losses don't track quality) and diffusion (loss plateaus while quality still improves). | V,G,D |
| `num_data_input` | Cap on the number of training images used (first N in sorted order); `0` = all. Handy for smoke tests. | V,G,D |
| `hflip_augment` | Random horizontal flips of training images. | V,G,D |
| `grad_norm_thresh` | If not `null`, clip the gradient norm at this threshold. | V,G,D |
| `precision` | Lightning precision setting: `"32-true"` or `"16-mixed"` (mixed precision; GPU only - falls back to 32-bit on CPU). Keep `"32-true"` for the GANs. | V,G,D |
| `accelerator` | Optional Lightning accelerator override (`"auto"` default, `"cpu"`, `"gpu"`). | V,G,D |
| `tracking_tool` | Experiment tracker: `"mlflow"`, `"tensorboard"`, or `null`. | V,G,D |
| `tracking_port` | Port used to build `http://localhost:<port>` when `MLFLOW_TRACKING_URI` is not set (or printed in the TensorBoard hint). | V,G,D |
| `tracking_expt_name` | MLflow experiment name. Using one experiment for all four scripts makes side-by-side comparison easiest. | V,G,D |
| `save_model_weights` | Save `model_weights.pt` to `model_dir` and copy/upload it to `WEIGHTS_PATH` if set. | V,G,D |
| `checkpoint_every_n_epochs` | Save a full resumable checkpoint every N epochs (last 2 kept; removed after a successful run when weights are saved); `0` disables. Needed for resuming spot-interrupted jobs. | V,G,D |

## Model Architecture Parameters

| Parameter | Description | Used by |
| --- | --- | --- |
| `image_shape` | `[height, width, channels]` images are resized/center-cropped to. GANs need square power-of-two sizes >= 16; the VAE needs sizes divisible by 2^(number of `channel_mults`). | V,G,D |
| `latent_dim` | Dimension of the latent vector z. | V,G |
| `base_channels` | Feature maps in the VAE's first conv stage. | V |
| `channel_mults` | Per-stage channel multipliers; each entry adds one stride-2 stage (e.g. `[1, 2, 4, 8]` takes 64x64 down to 4x4). | V |
| `gen_base_channels` | Base width of the generator (widest layer is 8x this). | G |
| `disc_base_channels` | Base width of the discriminator/critic (widest layer is 8x this). | G |
| `disc_norm` | Normalization in the discriminator: `"batch"` (DCGAN), `"instance"` or `"none"` (required for WGAN-GP). | G |
| `hf_model_class` | Name of the `diffusers` denoiser class, e.g. `"UNet2DModel"`. | D |
| `hf_model_config` | Constructor kwargs for that class (`layers_per_block`, `block_out_channels`, `down_block_types`, `up_block_types`, ...). `sample_size`, `in_channels`, `out_channels` default from `image_shape`. | D |
| `hf_pretrained` | Optional Hugging Face Hub id or local directory to initialize the denoiser from pretrained weights (`null` = train from scratch). Cloud jobs have no internet: bake the weights into the Docker image first. | D |
| `train_scheduler_class` | Name of the `diffusers` scheduler defining the training noise process, e.g. `"DDPMScheduler"`. | D |
| `train_scheduler_config` | Constructor kwargs for it (`num_train_timesteps`, `beta_schedule`, `prediction_type`, ...). | D |
| `sample_scheduler_class` | Name of the `diffusers` scheduler used for sampling, e.g. `"DDIMScheduler"`, `"DDPMScheduler"`, `"DPMSolverMultistepScheduler"`. | D |
| `num_inference_steps` | Denoising steps per generated image. Sampling cost scales linearly with this. | D |

### Changing image resolution

All defaults target 64x64. For 128x128:
- **VAE**: add a stage, e.g. `"channel_mults": [1, 2, 4, 8, 8]`.
- **GANs**: just change `image_shape`; the networks add a stage automatically.
- **Diffusion**: add a level to the UNet, e.g. `"block_out_channels": [128, 128, 256, 256, 512]`
  with `"DownBlock2D"` prepended to `down_block_types` and `"UpBlock2D"`
  appended to `up_block_types`; reduce `batch_size` to fit GPU memory.
- Update `model_dir` so the names stay truthful.
