"""Run the complete paired experiment; execute from any working directory.

python experiments/damping/run.py --config experiments/damping/config.json
Each condition is trained afresh. Test scores never select checkpoints/settings.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from RRN_functions import RRNModel, rrn_extract_features

CONDITIONS = ("baseline", "low_damped", "high_damped")


def validate_config(c):
    for name in ("train_size", "validation_size", "test_size"):
        if c[name] < 2 or c[name] % 2:
            raise ValueError(f"{name} must be positive and even.")
    if c["epochs"] < 4:
        raise ValueError("At least four epochs are needed for the OneCycle schedule.")
    if not 0 < c["affected_node_count"] <= c["node_count"] // 2:
        raise ValueError("Affected low and high groups must be nonempty and disjoint.")
    if not 0 < c["baseline_r"] < 1:
        raise ValueError("baseline_r must satisfy 0 < r < 1.")
    intervention_r = np.exp(-1 / (c["sampling_rate_hz"] * c["intervention_decay_seconds"]))
    if not 0 < intervention_r < c["baseline_r"]:
        raise ValueError("Intervention must produce faster decay than baseline.")
    if c["golden_ratio"] <= 1:
        raise ValueError("Frequency ratio must exceed one.")
    if len(set(c["starting_frequencies_hz"])) != len(c["starting_frequencies_hz"]):
        raise ValueError("Starting frequencies must be distinct.")
    if len(set(c["training_seeds"])) != len(c["training_seeds"]) or not c["training_seeds"]:
        raise ValueError("Training seeds must be nonempty and distinct.")
    for a in c["starting_frequencies_hz"]:
        freqs = a * c["golden_ratio"] ** np.arange(c["node_count"])
        if a <= 0 or np.any(freqs >= c["sampling_rate_hz"] / 4):
            raise ValueError("Every explicit frequency must be positive and below Fs/4.")


def make_split(n, rng, c):
    """Same per-bin event law as upstream simulated_data.py, with a local RNG.

    Constant versus sinusoidally modulated underlying Poisson rate. A binary
    sample records at least one event, not the number of events in the bin.
    A fixed sinusoid phase matches upstream. Nominal mean rate is shared;
    finite-window spike counts need not match exactly between classes.
    """
    fs = c["sampling_rate_hz"]
    steps = round(c["duration_seconds"] * fs)
    t = np.arange(steps) / fs
    y = np.repeat(np.arange(2), n // 2).astype(np.int64)
    f = np.zeros(n)
    f[n // 2:] = rng.uniform(*c["modulation_range_hz"], n // 2)
    rates = np.full((n, steps), float(c["firing_rate_hz"]))
    rates[n // 2:] *= 1 + np.sin(2 * np.pi * f[n // 2:, None] * t)
    x = (rng.random((n, steps)) < -np.expm1(-rates / fs)).astype(np.float32)
    order = rng.permutation(n)
    return x[order], y[order], f[order]


def make_data(c):
    # Independent split streams: test-size changes do not alter train or val.
    streams = np.random.SeedSequence(c["data_seed"]).spawn(3)
    return {name: make_split(c[size], np.random.default_rng(seed), c)
            for name, size, seed in zip(
                ("train", "validation", "test"),
                ("train_size", "validation_size", "test_size"), streams)}


def data_hash(data):
    h = hashlib.sha256()
    for name in ("train", "validation", "test"):
        for a in data[name]:
            h.update(a.tobytes())
    return h.hexdigest()


def model_for(c, start, condition):
    freqs = start * c["golden_ratio"] ** np.arange(c["node_count"])
    damping = np.full(c["node_count"], c["baseline_r"])
    m = c["affected_node_count"]
    if condition == "low_damped":
        damping[:m] = np.exp(-1 / (c["sampling_rate_hz"] * c["intervention_decay_seconds"]))
    elif condition == "high_damped":
        damping[-m:] = np.exp(-1 / (c["sampling_rate_hz"] * c["intervention_decay_seconds"]))
    elif condition != "baseline":
        raise ValueError(condition)
    return RRNModel(Fs=c["sampling_rate_hz"], n_classes=2,
                    init_scale=c["initial_weight_scale"], verbose=False,
                    node_frequencies=freqs, damping=damping)


def loader(split, c, shuffle=False, seed=0):
    x, y, _ = split
    return DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
                      batch_size=c["batch_size"], shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed), num_workers=0)


@torch.no_grad()
def evaluate(model, batches):
    model.eval()
    losses, predictions, labels = [], [], []
    for x, y in batches:
        logits = model(x)
        losses.append(torch.nn.functional.cross_entropy(logits, y, reduction="sum").item())
        predictions.append(logits.argmax(1).numpy())
        labels.append(y.numpy())
    pred, truth = np.concatenate(predictions), np.concatenate(labels)
    return float((pred == truth).mean()), float(sum(losses) / len(truth)), pred


def train_run(c, data, grid, start, condition, seed, out):
    # Reset identically: every condition/grid receives identical trainable
    # initialization and minibatch order for this seed. Buffers alone differ.
    torch.manual_seed(seed)
    model = model_for(c, start, condition)
    train = loader(data["train"], c, shuffle=True, seed=seed)
    validation = loader(data["validation"], c)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c["learning_rate"],
                                  weight_decay=c["weight_decay"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=c["learning_rate"], epochs=c["epochs"],
        steps_per_epoch=len(train), pct_start=0.15,
        anneal_strategy="cos", div_factor=25.0)
    best_acc, best_epoch, best_state, history = -1, 0, None, []
    run_name = f"grid{grid}_{condition}_seed{seed}"
    t0 = time.perf_counter()
    for epoch in range(1, c["epochs"] + 1):
        model.train()
        loss_total, correct = 0.0, 0
        for x, y in train:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, y, label_smoothing=c["label_smoothing"])
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss in {run_name}, epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), c["gradient_clip_norm"],
                                           error_if_nonfinite=True)
            optimizer.step()
            model.project_W_no_diag()
            scheduler.step()
            loss_total += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
        val_acc, val_loss, _ = evaluate(model, validation)
        history.append(dict(epoch=epoch, train_loss=loss_total / len(train.dataset),
                            train_accuracy=correct / len(train.dataset),
                            validation_loss=val_loss, validation_accuracy=val_acc))
        # Same strict-improvement checkpoint rule as upstream; first wins ties.
        if val_acc > best_acc + 1e-6:
            best_acc, best_epoch = val_acc, epoch
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    test_acc, test_loss, predictions = evaluate(model, loader(data["test"], c))
    torch.save({"model_state": best_state, "config": c, "grid": grid,
                "condition": condition, "seed": seed, "best_epoch": best_epoch},
               out / "checkpoints" / f"{run_name}.pt")
    np.save(out / "predictions" / f"{run_name}.npy", predictions)
    write_csv(out / "histories" / f"{run_name}.csv", history)
    return dict(grid=grid, start_hz=start, condition=condition, seed=seed,
                best_epoch=best_epoch, validation_accuracy=best_acc,
                test_accuracy=test_acc, test_loss=test_loss,
                seconds=time.perf_counter() - t0)


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def paired_rows(rows):
    lookup = {(r["grid"], r["seed"], r["condition"]): r for r in rows}
    pairs = []
    for r in rows:
        if r["condition"] != "baseline":
            b = lookup[(r["grid"], r["seed"], "baseline")]
            pairs.append(dict(grid=r["grid"], start_hz=r["start_hz"], seed=r["seed"],
                              condition=r["condition"],
                              delta_accuracy_pp=100 * (r["test_accuracy"] - b["test_accuracy"])))
    return pairs


def diagnostic(c, data, out):
    # No recurrence isolates the direct filter intervention. No fitting or
    # checkpoint choice here. Report raw features, not normalized classifier input.
    examples = data["test"][0][:100]
    rows = []
    for grid, start in enumerate(c["starting_frequencies_hz"]):
        for condition in CONDITIONS:
            model = model_for(c, start, condition)
            model.W_res.requires_grad_(False)
            with torch.no_grad():
                model.W_res.zero_()
                features = np.stack([rrn_extract_features(model, x).numpy() for x in examples])
            for k, f in enumerate(model.frange):
                rows.append(dict(grid=grid, condition=condition, node=k, frequency_hz=f,
                                  mean_raw_feature=float(features[:, k].mean())))
    write_csv(out / "filter_diagnostic.csv", rows)


def metadata(c, data):
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    except (OSError, subprocess.CalledProcessError):
        commit, status = "unknown", "unknown"
    files = ("RRN_functions.py", "experiments/damping/run.py", "experiments/damping/config.json")
    return dict(config=c, data_sha256=data_hash(data), upstream_or_parent_commit=commit,
                git_status_at_start=status, python=sys.version, platform=platform.platform(),
                numpy=np.__version__, torch=torch.__version__,
                source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in files},
                interpretation="Exploratory paired comparisons on one shared synthetic dataset; no consciousness inference.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results"))
    parser.add_argument("--resume", action="store_true", help="Resume only when settings, data and source hashes match.")
    args = parser.parse_args()
    c = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(c)
    torch.set_num_threads(c["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    data = make_data(c)
    out = args.output.resolve()
    meta = metadata(c, data)
    rows = []
    if (out / "metadata.json").exists():
        if not args.resume:
            raise SystemExit("Output exists. Use --resume or choose a new --output directory.")
        old = json.loads((out / "metadata.json").read_text())
        for key in ("config", "data_sha256", "source_sha256", "numpy", "torch"):
            if old[key] != meta[key]:
                raise SystemExit(f"Cannot resume: {key} changed.")
        if (out / "runs.csv").exists():
            with (out / "runs.csv").open(newline="") as f:
                for r in csv.DictReader(f):
                    rows.append({k: int(v) if k in ("grid", "seed", "best_epoch") else
                                 v if k == "condition" else float(v) for k, v in r.items()})
    else:
        out.mkdir(parents=True, exist_ok=True)
        (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    for folder in ("checkpoints", "predictions", "histories"):
        (out / folder).mkdir(exist_ok=True)
    np.savez_compressed(out / "data.npz", **{f"{name}_{key}": arr
                        for name, arrays in data.items()
                        for key, arr in zip(("x", "y", "modulation_hz"), arrays)})
    done = {(r["grid"], r["condition"], r["seed"]) for r in rows}
    total = len(c["starting_frequencies_hz"]) * len(CONDITIONS) * len(c["training_seeds"])
    for grid, start in enumerate(c["starting_frequencies_hz"]):
        for seed in c["training_seeds"]:
            for condition in CONDITIONS:
                if (grid, condition, seed) in done:
                    continue
                r = train_run(c, data, grid, start, condition, seed, out)
                rows.append(r)
                write_csv(out / "runs.csv", rows)
                print(f"{len(rows):02d}/{total} grid={grid} start={start:.3f} {condition:12s} "
                      f"seed={seed} test={r['test_accuracy']:.3f} "
                      f"best_epoch={r['best_epoch']} time={r['seconds']:.1f}s", flush=True)
    write_csv(out / "paired_effects.csv", paired_rows(rows))
    diagnostic(c, data, out)
    print(f"Complete: {out}", flush=True)


if __name__ == "__main__":
    main()
