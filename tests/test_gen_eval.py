"""
Tests of the evaluation/logging helpers in gen_eval.py, using a stand-in
metrics evaluator so that no Inception weights are needed.
"""

import json
import os
import tempfile
import unittest

import lightning as L
import torch
from torch.utils.data import DataLoader

import gen_eval
from tests._helpers import TimedTestCase
from vae_model import VAELitModule

SMALL_ARCH = dict(image_shape=[16, 16, 3], latent_dim=8, base_channels=4,
                  channel_mults=[1, 2])


class _FakeEvaluator:
    """Mimics GenerativeImageMetrics without running an Inception network."""

    def __init__(self):
        self.real_images = torch.randint(0, 256, (10, 3, 16, 16), dtype=torch.uint8)
        self.calls = []

    def to(self, device):
        return self

    def evaluate(self, fake_images, metrics=("fid", "kid", "is")):
        self.calls.append((tuple(fake_images.shape), tuple(metrics)))
        assert fake_images.dtype == torch.uint8
        values = {"fid": 12.5, "kid": 0.25, "kid_std": 0.05, "is": 2.0, "is_std": 0.1}
        wanted = set(metrics) | {f"{m}_std" for m in metrics}
        return {k: v for k, v in values.items() if k in wanted}


class TestGenEvalCallback(TimedTestCase):

    def test_grids_and_metric_logged_on_schedule(self):
        torch.manual_seed(0)
        data = torch.rand(8, 3, 16, 16) * 2 - 1
        loader = DataLoader(list(data), batch_size=4)
        evaluator = _FakeEvaluator()
        with tempfile.TemporaryDirectory() as tmp:
            callback = gen_eval.GenEvalCallback(
                evaluator, output_dir=tmp, eval_every_n_epochs=2,
                num_eval_sims=6, num_grid_images=4,
            )
            trainer = L.Trainer(
                max_epochs=4, accelerator="cpu", devices=1, logger=False,
                enable_checkpointing=False, enable_progress_bar=False,
                enable_model_summary=False, callbacks=[callback],
            )
            trainer.fit(VAELitModule(**SMALL_ARCH), train_dataloaders=loader)
            grids = sorted(os.listdir(os.path.join(tmp, "sample_grids")))
        self.assertEqual(grids, ["samples_epoch_0002.png", "samples_epoch_0004.png"])
        self.assertEqual(len(evaluator.calls), 2)
        self.assertEqual(evaluator.calls[0], ((6, 3, 16, 16), ("kid",)))

    def test_disabled_when_period_not_positive(self):
        callback = gen_eval.GenEvalCallback(None, output_dir="unused", eval_every_n_epochs=0)
        trainer = type("T", (), {"current_epoch": 0})()
        callback.on_train_epoch_end(trainer, None)  # must be a no-op


class TestFinalEvaluation(TimedTestCase):

    def test_outputs_and_metric_names(self):
        model = VAELitModule(**SMALL_ARCH)
        evaluator = _FakeEvaluator()
        with tempfile.TemporaryDirectory() as tmp:
            run_params = {
                "output_dir": tmp,
                "num_gen_sims": 5,
                "num_final_metric_sims": 12,
                "final_metrics": ["fid", "kid", "is"],
                "sample_batch_size": 8,
                "seed": 1,
            }
            metrics = gen_eval.final_evaluation(model, evaluator, run_params, False)
            sims = sorted(os.listdir(os.path.join(tmp, "sim_images")))
            self.assertEqual(len(sims), 5)
            self.assertEqual(sims[0], "sim_00000.png")
            for name in ("sim_grid.png", "real_grid.png", "final_metrics.json"):
                self.assertTrue(os.path.exists(os.path.join(tmp, name)), msg=name)
            with open(os.path.join(tmp, "final_metrics.json")) as f:
                saved = json.load(f)
        self.assertEqual(
            set(metrics),
            {"fid_final", "kid_final", "kid_final_std", "is_final", "is_final_std"},
        )
        self.assertEqual(saved["num_generated_images"], 12)
        self.assertEqual(evaluator.calls[0][0], (12, 3, 16, 16))

    def test_nothing_requested_returns_empty(self):
        model = VAELitModule(**SMALL_ARCH)
        with tempfile.TemporaryDirectory() as tmp:
            run_params = {"output_dir": tmp, "num_gen_sims": 0, "num_final_metric_sims": 0}
            self.assertEqual(gen_eval.final_evaluation(model, None, run_params, False), {})

    def test_finish_run_without_tracking_is_noop(self):
        gen_eval.finish_run({"output_dir": "unused"}, {"tracking_tool": None})


if __name__ == "__main__":
    unittest.main()
