"""Scientific invariants for paired interventions, rather than accuracy targets."""
import json
import unittest
from pathlib import Path

import numpy as np
import torch

from experiments.damping.run import make_data, model_for, validate_config, data_hash
from RRN_functions import RRNModel

CONFIG = json.loads(Path(__file__).with_name("config.json").read_text())


class ExperimentTests(unittest.TestCase):
    def test_default_model_matches_explicit_original_grid(self):
        torch.manual_seed(77)
        original = RRNModel(n_classes=2, verbose=False)
        torch.manual_seed(77)
        explicit = model_for(CONFIG, 2, "baseline")
        for key, value in original.state_dict().items():
            self.assertTrue(torch.equal(value, explicit.state_dict()[key]), key)
        original.eval()
        explicit.eval()
        x = torch.from_numpy(make_data(CONFIG)["validation"][0][:4])
        with torch.no_grad():
            self.assertTrue(torch.equal(original(x), explicit(x)))

    def test_fixed_count_spacing_and_stable_modes(self):
        validate_config(CONFIG)
        for start in CONFIG["starting_frequencies_hz"]:
            for condition in ("baseline", "low_damped", "high_damped"):
                model = model_for(CONFIG, start, condition)
                self.assertEqual(model.K, 11)
                np.testing.assert_allclose(model.frange[1:] / model.frange[:-1], 1.618)
                for a1, a2 in zip(model.w1.numpy(), model.w2.numpy()):
                    self.assertLess(np.abs(np.roots([1, -float(a1), -float(a2)])).max(), 1)

    def test_matching_trainable_initialization_and_targeted_buffers(self):
        models = []
        for condition in ("baseline", "low_damped", "high_damped"):
            torch.manual_seed(101)
            models.append(model_for(CONFIG, 2, condition))
        for model in models[1:]:
            for (_, left), (_, right) in zip(models[0].named_parameters(), model.named_parameters()):
                self.assertTrue(torch.equal(left, right))
        self.assertTrue(torch.equal(models[0].w2[4:], models[1].w2[4:]))
        self.assertTrue(torch.equal(models[0].w2[:-4], models[2].w2[:-4]))
        expected = np.exp(-1 / (1000 * 0.02))
        np.testing.assert_allclose(np.sqrt(-models[1].w2[:4].numpy()), expected, rtol=1e-6)

    def test_shared_reproducible_independent_splits(self):
        left, right = make_data(CONFIG), make_data(CONFIG)
        self.assertEqual(data_hash(left), data_hash(right))
        for name, (x, y, f) in left.items():
            self.assertEqual(x.shape[1], 100)
            self.assertEqual(set(np.unique(x)), {0, 1})
            self.assertEqual((y == 0).sum(), (y == 1).sum())
            self.assertTrue(np.all(f[y == 0] == 0))
            self.assertTrue(np.all((f[y == 1] >= 10) & (f[y == 1] <= 50)))
        bigger = dict(CONFIG, test_size=4000)
        other = make_data(bigger)
        for name in ("train", "validation"):
            for a, b in zip(left[name], other[name]):
                np.testing.assert_array_equal(a, b)

    def test_valid_gradients_and_diagonal_projection(self):
        torch.set_num_threads(1)
        for condition in ("baseline", "low_damped", "high_damped"):
            model = model_for(CONFIG, 2, condition)
            x = torch.from_numpy(make_data(CONFIG)["train"][0][:8])
            y = torch.from_numpy(make_data(CONFIG)["train"][1][:8])
            loss = torch.nn.functional.cross_entropy(model(x), y)
            loss.backward()
            for p in model.parameters():
                self.assertTrue(torch.isfinite(p.grad).all())
            torch.optim.AdamW(model.parameters()).step()
            model.project_W_no_diag()
            self.assertTrue(torch.equal(model.W_res.diag(), torch.zeros(11)))

    def test_rejects_grid_that_would_drop_nodes(self):
        with self.assertRaises(ValueError):
            model_for(CONFIG, 2.2, "baseline")
        with self.assertRaises(ValueError):
            validate_config(dict(CONFIG, affected_node_count=6))


if __name__ == "__main__":
    unittest.main()
