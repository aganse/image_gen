# Recently-completed and next-up tasks for image_gen
Not comprehensive long-term tracking here, just keeping organized in current work...
(See TASKING_STATUS.md for the detailed state of the initial build-out.)


### Bugs List
- [ ]


### To Do
- [ ] First-run verification checklist in TASKING_STATUS.md (unit tests, lint,
      local smoke runs, Docker build, SageMaker runs)
- [ ] Tune default hyperparameters / epoch counts from real runs
- [ ] Add example result images + metrics table to README once runs exist
- [ ] GitHub unittests workflow
- [ ] GitHub linting workflow
- [ ] Add params-file override support to AWS Batch submission (shared with flow_models)
- [ ] Log AWS Batch job ID as MLflow param (shared with flow_models)
- [ ] Extract aws/ (and possibly model_training.py) into standalone reusable repo;
      port the generalizations made here back to flow_models at the same time
- [ ] flow_models: log a large batch of individual sim images so
      eval_mlflow_run.py can score its runs

### Maybe
- [ ] Latent diffusion (small UNet in the latent space of a pretrained AutoencoderKL)
- [ ] Fine-tuning a pretrained diffusers UNet (hf_pretrained + weights baked into image)
- [ ] Multi-category run (cats+dogs+wild) for a meaningful Inception Score
- [ ] 128x128 configurations
- [ ] Hinge loss / spectral norm options for the GAN
- [ ] Mixed precision for sampling (speeds up diffusion evaluation)
- [ ] Latent-space interpolation plots for VAE/GAN (as in flow_models do_interp)


### Done
- [x] Initial repo build-out (see TASKING_STATUS.md)
