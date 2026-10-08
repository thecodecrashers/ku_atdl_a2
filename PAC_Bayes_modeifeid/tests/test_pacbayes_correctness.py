"""Regression checks for the corrected PAC-Bayes objective and final evaluation."""

import contextlib
import io
import json
import math
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from snn.core.extra_fn import (discretized_prior_bound, gaussian_kl, inverse_binary_kl,
                               monte_carlo_error_upper)
from snn.core.fc import FC
from snn.experiments.run_pacb import run_pacb
from experiments.recompute_summary import recompute_summary_for_dir


def tiny_model(weights=None):
    rng = np.random.default_rng(19)
    x = rng.normal(size=(20, 2)).astype(np.float32)
    y = np.where(x[:, :1] >= 0, 1, -1).astype(np.float32)
    return FC(x, y, layers=[2, 3, 1], seed=11, initial_weights=weights, device="cpu")


class BoundNumericsTests(unittest.TestCase):
    def test_inverse_endpoints(self):
        for c in (0, 0.001, 1, 100):
            self.assertAlmostEqual(inverse_binary_kl(0, c), -math.expm1(-c), places=14)
            self.assertEqual(inverse_binary_kl(1, c), 1)
        self.assertEqual(inverse_binary_kl(0.23, 0), 0.23)
        for q, c in ((-0.1, 1), (1.1, 1), (0.1, -1), (float('nan'), 1)):
            with self.assertRaises(ValueError):
                inverse_binary_kl(q, c)

    def test_inverse_residual_and_monotonicity(self):
        for q in (0.001, 0.03, 0.5, 0.95):
            previous = q
            for c in (1e-5, 0.01, 0.1, 1):
                p = inverse_binary_kl(q, c)
                self.assertGreaterEqual(p, previous)
                self.assertLessEqual(p, 1)
                if p < 1:
                    kl = q * math.log(q / p) + (1-q) * math.log((1-q)/(1-p))
                    resolution = abs((p - q) / (p * (1 - p))) * math.ulp(p)
                    self.assertAlmostEqual(kl, c, delta=1e-8 + 2 * resolution)
                previous = p
        # A loose Pinsker initialization must not force an unnecessary bound of one.
        self.assertLess(inverse_binary_kl(0.1, 2), 1)

    def test_gaussian_kl_against_torch_distributions(self):
        rng = np.random.default_rng(82)
        w, w0, logs = [rng.normal(size=7) for _ in range(3)]
        rho = -2.3
        q = torch.distributions.Normal(torch.tensor(w), torch.tensor(np.exp(logs)))
        p = torch.distributions.Normal(torch.tensor(w0), torch.full((7,), math.exp(rho), dtype=torch.float64))
        expected = torch.distributions.kl_divergence(q, p).sum().item()
        actual = gaussian_kl([w], [w0], [logs], rho)
        self.assertAlmostEqual(actual, expected, delta=expected * 1e-12)
        self.assertEqual(gaussian_kl([w], [w], [np.full_like(w, rho)], rho), 0)
        close = gaussian_kl([w], [w], [np.full_like(w, rho + 1e-8)], rho)
        self.assertGreaterEqual(close, 0)
        self.assertLess(close, 1e-12)

    def test_prior_grid_and_invalid_legacy_prior(self):
        mean, prior, logs = [np.array([0.2])], [np.array([0.0])], [np.array([-2.0])]
        selected = discretized_prior_bound(mean, prior, logs, -2.2, 100)
        index = selected["prior grid index"]
        self.assertGreaterEqual(index, 1)
        self.assertAlmostEqual(selected["selected prior variance"], 0.1 * math.exp(-index / 100))
        kl = gaussian_kl(mean, prior, logs, selected["selected log prior std"])
        expected = (kl + math.log(math.pi ** 2 * 100 / (6 * 0.025)) + 2 * math.log(index)) / 99
        self.assertAlmostEqual(selected["relative entropy bound"], expected)
        with self.assertRaisesRegex(ValueError, "retraining"):
            discretized_prior_bound(mean, prior, logs, -0.715526342, 100)

    def test_mc_correction_is_conservative(self):
        one = monte_carlo_error_upper(0.01, 1)
        many = monte_carlo_error_upper(0.01, 1000)
        self.assertGreater(one, many)
        self.assertGreater(many, 0.01)
        self.assertAlmostEqual(monte_carlo_error_upper(0, 1), 0.995)
        with self.assertRaises(ValueError):
            monte_carlo_error_upper(0.01, 0)


class ObjectiveAndSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.silence = contextlib.redirect_stdout(io.StringIO())
        self.silence.__enter__()

    def tearDown(self):
        self.silence.__exit__(None, None, None)

    def test_current_mean_kl_value_and_gradient(self):
        model = tiny_model()
        prior = model.get_model_weights()
        objective = model.PACB_objective(prior)
        _, logs, rho = model.PACB_init
        noise = [np.zeros_like(w) for w in prior]
        objective(model._t(model.X), model._t(model.Y), noise)
        old_kl = model.KLdivTimes2.item() / 2
        with torch.no_grad():
            for w in model.params:
                w.add_(0.1)
        objective(model._t(model.X), model._t(model.Y), noise)
        current_kl = model.KLdivTimes2.item() / 2
        expected_increase = sum(w.size for w in prior) * 0.1 ** 2 / (2 * math.exp(2 * rho.item()))
        self.assertAlmostEqual(current_kl - old_kl, expected_increase, delta=2e-3)
        gradients = torch.autograd.grad(model.KLdivTimes2.sum() / 2, model.params)
        for w, w0, grad in zip(model.params, prior, gradients):
            expected = (w.detach() - model._t(w0)) / torch.exp(2 * rho.detach())
            torch.testing.assert_close(grad, expected)
        reference = gaussian_kl(model.get_model_weights(), prior,
                                [v.detach().numpy() for v in logs], rho.item())
        self.assertAlmostEqual(current_kl, reference, delta=reference * 2e-6)
        self.assertAlmostEqual(model.B.item() ** 2 * 2 * 19 - current_kl,
                               math.log(math.pi ** 2 * 20 / (6 * 0.025))
                               + 2 * math.log(100 * (math.log(0.1) - 2 * rho.item())), delta=2e-3)

    def test_frozen_weights_and_no_epoch_evaluation(self):
        model = tiny_model()
        prior = model.get_model_weights()
        with patch.object(model, "_pacb_correct_sum", wraps=model._pacb_correct_sum) as accuracy:
            model.optimize_PACB(prior, epochs=3, batch_size=5, trainWeights=False, eval_interval=0)
            accuracy.assert_not_called()
        for before, after in zip(prior, model.get_model_weights()):
            np.testing.assert_array_equal(before, after)
        self.assertTrue(all(not p.requires_grad for p in model.params))
        self.assertTrue(all(row["stochastic_train_accuracy"] is None for row in model.history))
        reference = gaussian_kl(model.get_model_weights(), prior,
                                [v.detach().numpy() for v in model.PACB_init[1]], float(model.log_prior_std[0]))
        self.assertAlmostEqual(model.history[-1]["KL"], reference, delta=reference * 2e-6)

    def test_prior_projection_and_stable_loss(self):
        model = tiny_model()
        rho = model.PACB_init[2]
        with torch.no_grad():
            rho.fill_(0.5)
        model._project_prior_std(rho)
        self.assertGreaterEqual(100 * (math.log(0.1) - 2 * rho.item()), 1)
        logits = torch.tensor([[-1000.0], [1000.0]], requires_grad=True)
        loss = model.logistic_loss(logits, torch.ones_like(logits))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_unequal_evaluation_batches(self):
        model = tiny_model()
        x, y = model.X[:17], model.Y[:17]
        noise = [np.zeros_like(w) for w in model.get_model_weights()]
        scale = [torch.ones_like(w) for w in model.params]
        batched = model.print_accuracy_in_batches_noise(x, y, noise, scale, no_batches=10)
        with torch.no_grad():
            correct = ((model.model(model._t(x), model.params) >= 0).float() * 2 - 1 == model._t(y))
        self.assertEqual(batched, correct.sum().item() / 17)

    def test_final_snapshot_roundtrip_and_preservation(self):
        model = tiny_model()
        model.X, model.Y = np.tile(model.X, (5, 1)), np.tile(model.Y, (5, 1))
        model.Nsamples = len(model.X)
        prior = model.get_model_weights()
        with tempfile.TemporaryDirectory(prefix="pacbayes_unit_") as tmp:
            run_pacb(prior, model, (model.X[:13], model.Y[:13]), epochs=2,
                     learning_rate=0.001, drop_lr=250, lr_factor=0.1, seed=11,
                     trainw=True, snn_samples=5, output_dir=tmp, eval_interval=0)
            output = Path(tmp)
            summary_bytes = (output / "final_summary.json").read_bytes()
            summary = json.loads(summary_bytes)
            with (output / "model_mean_optTrue_LR0.001_seed11.pickle").open("rb") as stream:
                saved = pickle.load(stream)
            snapshot = saved["final_posterior"]
            self.assertEqual(len(saved["log_post_all"]), len(saved["PACB_weights"]))
            reference = gaussian_kl(snapshot["mean_weights"], snapshot["prior_weights"],
                                    snapshot["log_post_std"], summary["selected log prior std"])
            self.assertAlmostEqual(summary["KL(Q || P)"], reference)
            self.assertGreaterEqual(summary["PAC-Bayes bound"], summary["PAC-Bayes bound without MC correction"])
            self.assertEqual(summary["confidence"], 0.965)
            # Replay only the independent evaluation seed, with the complete saved posterior.
            replay = tiny_model(snapshot["mean_weights"])
            replay.X, replay.Y, replay.Nsamples = model.X, model.Y, model.Nsamples
            with patch("experiments.recompute_summary.Interpreter") as interpreter:
                interpreter.return_value.interpret.return_value = (replay, (model.X[:13], model.Y[:13]), None)
                recompute_summary_for_dir(tmp, snn_samples=5, mc_seed=summary["MC seed"])
            recomputed = json.loads((output / "final_summary_recomputed.json").read_text())
            for key in ("PAC-Bayes bound", "KL(Q || P)", "SNN train error", "SNN test error"):
                self.assertEqual(summary[key], recomputed[key])
            self.assertEqual(summary_bytes, (output / "final_summary.json").read_bytes())
            with self.assertRaises(FileExistsError):
                run_pacb(prior, model, (model.X, model.Y), 0, 0.001, 250, 0.1, 11,
                         True, 1, output_dir=tmp)
            with self.assertRaises(FileExistsError):
                recompute_summary_for_dir(tmp)

    def test_legacy_snapshot_rejected(self):
        with tempfile.TemporaryDirectory(prefix="pacbayes_legacy_") as tmp:
            path = Path(tmp) / "model_mean_legacy.pickle"
            with path.open("wb") as stream:
                pickle.dump({"PACB_weights": [], "log_post_all": []}, stream)
            with self.assertRaisesRegex(ValueError, "complete final posterior"):
                recompute_summary_for_dir(tmp)
            self.assertFalse((Path(tmp) / "final_summary_recomputed.json").exists())

    def test_final_evaluation_does_not_reset_training_rng(self):
        model = tiny_model()
        prior = model.get_model_weights()
        model.optimize_PACB(prior, epochs=0, batch_size=5, eval_interval=0)
        np.random.seed(834)
        expected = np.random.normal(size=4)
        np.random.seed(834)
        model.evaluate_SNN_accuracy(model.X, model.Y, prior, N_SNN_samples=2, mc_seed=2094)
        np.testing.assert_array_equal(np.random.normal(size=4), expected)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
