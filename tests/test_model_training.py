"""
Tests of the training orchestration in model_training.py: platform detection,
parameter overrides, param flattening, checkpoint discovery, and short
end-to-end runs of default_training_sequence on a tiny VAE (CPU friendly),
including checkpoint resume and MLflow logging to a temporary local store.
"""

import base64
import json
import os
import tempfile
import unittest
from unittest import mock

import torch
from torch.utils.data import DataLoader

import model_training
from tests._helpers import TimedTestCase
from vae_model import VAELitModule

SMALL_ARCH = dict(image_shape=[16, 16, 3], latent_dim=8, base_channels=4,
                  channel_mults=[1, 2])


def _tiny_loader(n=8, batch_size=4):
    torch.manual_seed(0)
    data = torch.rand(n, 3, 16, 16) * 2 - 1
    return DataLoader(list(data), batch_size=batch_size, shuffle=False)


def _params(tmp_dir, **training_overrides):
    run_params = {
        "output_dir": os.path.join(tmp_dir, "output"),
        "model_dir": os.path.join(tmp_dir, "model"),
        "model_name": "vae",
        "dataset": "unittest",
        "do_train": True,
    }
    training_params = {
        "num_epochs": 1,
        "batch_size": 4,
        "learning_rate": 1e-3,
        "early_stopping_patience": 0,
        "grad_norm_thresh": None,
        "precision": "32-true",
        "accelerator": "cpu",
        "tracking_tool": None,
        "save_model_weights": True,
        "checkpoint_every_n_epochs": 0,
    }
    training_params.update(training_overrides)
    os.makedirs(run_params["output_dir"], exist_ok=True)
    return run_params, training_params, dict(SMALL_ARCH)


class TestPlatformAndParams(TimedTestCase):

    def test_detect_platform_batch(self):
        with mock.patch.dict(os.environ, {"AWS_BATCH_JOB_ID": "abc"}):
            self.assertEqual(model_training.detect_platform(), "awsbatch")

    def test_detect_platform_sagemaker(self):
        env = {k: v for k, v in os.environ.items() if k != "AWS_BATCH_JOB_ID"}
        env["TRAINING_JOB_NAME"] = "imagegen-vae-test"
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(model_training.detect_platform(), "sagemaker")

    def test_resolve_checkpoint_dir(self):
        run_params = {"model_dir": "model/x"}
        self.assertEqual(
            model_training.resolve_checkpoint_dir(run_params, "sagemaker"),
            "/opt/ml/checkpoints",
        )
        self.assertEqual(
            model_training.resolve_checkpoint_dir(run_params, "local"),
            os.path.join("model/x", "checkpoints"),
        )

    def test_load_param_overrides(self):
        overrides = {
            "run_params": {"num_gen_sims": 3},
            "training_params": {"num_epochs": 7},
            "model_arch_params": {"latent_dim": 5},
        }
        blob = base64.b64encode(json.dumps(overrides).encode()).decode()
        with tempfile.TemporaryDirectory() as tmp:
            hp_path = os.path.join(tmp, "hyperparameters.json")
            with open(hp_path, "w") as f:
                json.dump({"params": blob}, f)
            run_params = {"num_gen_sims": 10, "keep": 1}
            training_params = {"num_epochs": 1}
            model_arch_params = {"latent_dim": 128}
            model_training.load_param_overrides(
                run_params, training_params, model_arch_params, hp_path=hp_path
            )
        self.assertEqual(run_params, {"num_gen_sims": 3, "keep": 1})
        self.assertEqual(training_params["num_epochs"], 7)
        self.assertEqual(model_arch_params["latent_dim"], 5)

    def test_load_param_overrides_missing_file_is_noop(self):
        run_params = {"a": 1}
        model_training.load_param_overrides(
            run_params, {}, {}, hp_path="/nonexistent/hyperparameters.json"
        )
        self.assertEqual(run_params, {"a": 1})

    def test_flatten_params(self):
        flat = model_training._flatten_params({
            "a": 1, "b": "x", "c": None, "d": [1, 2],
            "cfg": {"layers": 2, "widths": [8, 16]},
        })
        self.assertEqual(flat["a"], 1)
        self.assertEqual(flat["d"], "[1, 2]")
        self.assertEqual(flat["cfg.layers"], 2)
        self.assertEqual(flat["cfg.widths"], "[8, 16]")
        self.assertNotIn("cfg", flat)

    def test_find_latest_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(model_training._find_latest_checkpoint(tmp), (None, 0))
            for epoch in (2, 10):
                open(os.path.join(tmp, f"ckpt-{epoch:04d}.ckpt"), "w").close()
            path, epoch = model_training._find_latest_checkpoint(tmp)
            self.assertTrue(path.endswith("ckpt-0010.ckpt"))
            self.assertEqual(epoch, 10)

    def test_invalid_tracking_tool_raises(self):
        with self.assertRaises(ValueError):
            model_training._start_tracking(
                {}, {"tracking_tool": "wandb"}, {}, "/tmp/ckpt", "local"
            )


class TestDefaultTrainingSequence(TimedTestCase):

    def setUp(self):
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory()
        # Make sure an ambient WEIGHTS_PATH can't redirect test outputs.
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("WEIGHTS_PATH", None)

    def tearDown(self):
        self.tmp.cleanup()
        super().tearDown()

    def _model(self):
        return VAELitModule(**SMALL_ARCH, learning_rate=1e-3)

    def test_train_saves_weights_summary_and_arch(self):
        run_params, training_params, arch = _params(self.tmp.name)
        model, trainer = model_training.default_training_sequence(
            self._model(), _tiny_loader(), run_params, training_params, arch
        )
        self.assertIsNotNone(trainer)
        for name in ("model_weights.pt", "model_summary.txt", "model_arch.json"):
            self.assertTrue(
                os.path.exists(os.path.join(run_params["model_dir"], name)), msg=name
            )
        self.assertNotIn("mlflow_run_open", run_params)  # tracking disabled

    def test_inference_mode_loads_saved_weights(self):
        run_params, training_params, arch = _params(self.tmp.name)
        trained, _ = model_training.default_training_sequence(
            self._model(), _tiny_loader(), run_params, training_params, arch
        )
        run_params["do_train"] = False
        loaded, trainer = model_training.default_training_sequence(
            self._model(), None, run_params, training_params, arch
        )
        self.assertIsNone(trainer)
        for (name, a), (_, b) in zip(
            trained.state_dict().items(), loaded.state_dict().items()
        ):
            self.assertTrue(torch.equal(a.cpu(), b.cpu()), msg=name)

    def test_checkpoints_are_written_pruned_and_resumed(self):
        # Run 1: 3 epochs, checkpoint every epoch, no final weights -> the
        # checkpoints are left in place (as after a spot interruption).
        run_params, training_params, arch = _params(
            self.tmp.name,
            num_epochs=3, checkpoint_every_n_epochs=1, save_model_weights=False,
        )
        loader = _tiny_loader()
        _, trainer = model_training.default_training_sequence(
            self._model(), loader, run_params, training_params, arch
        )
        ckpt_dir = os.path.join(run_params["model_dir"], "checkpoints")
        ckpts = sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt"))
        self.assertEqual(ckpts, ["ckpt-0002.ckpt", "ckpt-0003.ckpt"])  # last 2 kept
        steps_after_3_epochs = trainer.global_step

        # Run 2: ask for 5 epochs; should resume at epoch 3 and train 2 more.
        training_params["num_epochs"] = 5
        _, trainer = model_training.default_training_sequence(
            self._model(), loader, run_params, training_params, arch
        )
        self.assertEqual(trainer.current_epoch, 5)
        self.assertEqual(trainer.global_step, steps_after_3_epochs * 5 // 3)

    def test_final_weights_remove_checkpoints(self):
        run_params, training_params, arch = _params(
            self.tmp.name, num_epochs=2, checkpoint_every_n_epochs=1
        )
        model_training.default_training_sequence(
            self._model(), _tiny_loader(), run_params, training_params, arch
        )
        ckpt_dir = os.path.join(run_params["model_dir"], "checkpoints")
        self.assertEqual([f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")], [])

    def test_weights_copied_to_local_weights_path(self):
        run_params, training_params, arch = _params(self.tmp.name)
        dest = os.path.join(self.tmp.name, "weights_dest")
        os.environ["WEIGHTS_PATH"] = dest
        model_training.default_training_sequence(
            self._model(), _tiny_loader(), run_params, training_params, arch
        )
        self.assertTrue(os.path.exists(os.path.join(dest, "model_weights.pt")))

    def test_mlflow_tracking_logs_params_tags_and_metrics(self):
        import mlflow

        tracking_uri = "file://" + os.path.join(self.tmp.name, "mlruns")
        os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
        run_params, training_params, arch = _params(
            self.tmp.name, num_epochs=2, tracking_tool="mlflow",
            tracking_expt_name="unittest_expt",
        )
        try:
            model_training.default_training_sequence(
                self._model(), _tiny_loader(), run_params, training_params, arch
            )
            self.assertTrue(run_params["mlflow_run_open"])
            run_id = run_params["mlflow_run_id"]
        finally:
            mlflow.end_run()

        client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
        run = client.get_run(run_id)
        self.assertEqual(run.data.params["num_epochs"], "2")
        self.assertEqual(run.data.params["latent_dim"], "8")
        self.assertNotIn("output_dir", run.data.params)
        self.assertIn("run_env", run.data.tags)
        self.assertIn("image_tag", run.data.tags)
        loss_history = client.get_metric_history(run_id, "loss")
        self.assertEqual([m.step for m in loss_history], [0, 1])
        self.assertIn("kl_loss", run.data.metrics)


if __name__ == "__main__":
    unittest.main()
