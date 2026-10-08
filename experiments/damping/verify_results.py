"""Audit all recorded fits and optionally replay one complete training run."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.damping.run import CONDITIONS, data_hash, make_data, paired_rows, train_run


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).with_name("results"))
    parser.add_argument("--replay", action="store_true", help="Retrain original-grid baseline for the first seed.")
    args = parser.parse_args()
    folder = args.results.resolve()
    metadata = json.loads((folder / "metadata.json").read_text())
    c = metadata["config"]
    for relative, expected in metadata["source_sha256"].items():
        # Text files can be checked out with CRLF. Normalize for comparison.
        raw = (ROOT / relative).read_bytes()
        normalized = raw.replace(b"\r\n", b"\n")
        candidates = (raw, normalized, normalized.replace(b"\n", b"\r\n"))
        if not any(hashlib.sha256(b).hexdigest() == expected for b in candidates):
            raise AssertionError(f"Source changed: {relative}")
    data = make_data(c)
    assert data_hash(data) == metadata["data_sha256"], "Generated dataset hash changed"
    rows = []
    for r in read_csv(folder / "runs.csv"):
        rows.append({k: int(v) if k in ("grid", "seed", "best_epoch") else
                     v if k == "condition" else float(v) for k, v in r.items()})
    expected_keys = {(g, s, k) for g in range(len(c["starting_frequencies_hz"]))
                     for s in c["training_seeds"] for k in CONDITIONS}
    actual_keys = [(r["grid"], r["seed"], r["condition"]) for r in rows]
    assert len(actual_keys) == len(set(actual_keys)), "Duplicate fits"
    assert set(actual_keys) == expected_keys, "Missing or unexpected fits"
    for r in rows:
        name = f"grid{r['grid']}_{r['condition']}_seed{r['seed']}"
        history = read_csv(folder / "histories" / f"{name}.csv")
        assert len(history) == c["epochs"], name
        accuracies = np.array([float(e["validation_accuracy"]) for e in history])
        assert np.isfinite(accuracies).all()
        assert r["best_epoch"] == int(np.argmax(accuracies)) + 1, name
        assert r["validation_accuracy"] == accuracies.max(), name
        assert 0 <= r["test_accuracy"] <= 1 and np.isfinite(r["test_loss"]), name
        prediction = folder / "predictions" / f"{name}.npy"
        if prediction.exists():
            p = np.load(prediction)
            assert p.shape == data["test"][1].shape
            assert float((p == data["test"][1]).mean()) == r["test_accuracy"], name
    paired = read_csv(folder / "paired_effects.csv")
    expected_pairs = paired_rows(rows)
    assert len(paired) == len(expected_pairs)
    for recorded, expected in zip(paired, expected_pairs):
        assert float(recorded["delta_accuracy_pp"]) == expected["delta_accuracy_pp"]
    print(f"Verified {len(rows)} fits: sources, regenerated data hash, checkpoint selection, metrics, and paired effects.")
    if args.replay:
        torch.set_num_threads(c["cpu_threads"])
        torch.use_deterministic_algorithms(True)
        seed = c["training_seeds"][0]
        recorded = next(r for r in rows if r["grid"] == 0 and r["seed"] == seed and r["condition"] == "baseline")
        with tempfile.TemporaryDirectory(prefix="rrn_replay_", dir=folder.parent) as temp:
            out = Path(temp)
            for name in ("checkpoints", "predictions", "histories"):
                (out / name).mkdir()
            replay = train_run(c, data, 0, c["starting_frequencies_hz"][0], "baseline", seed, out)
            for key in ("best_epoch", "validation_accuracy", "test_accuracy", "test_loss"):
                assert replay[key] == recorded[key], (key, replay[key], recorded[key])
        print("Exact replay passed: the complete 30-epoch baseline run reproduced all recorded metrics.")


if __name__ == "__main__":
    main()
