"""Generate slide-ready PNG, vector SVG and PDF figures from recorded runs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"baseline": "#526477", "low_damped": "#007C91", "high_damped": "#C44E3B"}
LABELS = {"baseline": "Baseline", "low_damped": "Low-frequency damping", "high_damped": "High-frequency damping"}


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save(fig, folder, stem):
    for extension in ("png", "svg", "pdf"):
        fig.savefig(folder / f"{stem}.{extension}", dpi=220, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).with_name("results"))
    args = parser.parse_args()
    folder = args.results.resolve()
    c = json.loads((folder / "metadata.json").read_text())["config"]
    rows = read_csv(folder / "runs.csv")
    expected = len(c["starting_frequencies_hz"]) * 3 * len(c["training_seeds"])
    keys = {(int(r["grid"]), int(r["seed"]), r["condition"]) for r in rows}
    if len(rows) != expected or len(keys) != expected:
        raise SystemExit("Complete, unique runs are required before plotting results.")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"})
    figs = folder / "figures"
    figs.mkdir(exist_ok=True)
    starts = c["starting_frequencies_hz"]
    labels = [f"{s:.2f} Hz" for s in starts]

    # Method graphic, with no fitted outcomes: frequencies, envelope, task.
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.4))
    fig.subplots_adjust(left=.06, right=.98, bottom=.24, top=.78, wspace=.35)
    fig.suptitle("Selective damping across four golden-ratio grids", fontsize=21, y=.96)
    ax = axes[0]
    m, n = c["affected_node_count"], c["node_count"]
    for i, start in enumerate(starts):
        f = start * c["golden_ratio"] ** np.arange(n)
        ax.scatter(f[:m], np.full(m, i), color=COLORS["low_damped"], s=48)
        ax.scatter(f[m:-m], np.full(n-2*m, i), color="#AAAAAA", s=48)
        ax.scatter(f[-m:], np.full(m, i), color=COLORS["high_damped"], s=48)
    ax.axvspan(10, 50, color="#F1CF75", alpha=.3, zorder=0)
    ax.set_xscale("log")
    ax.set_yticks(range(4), labels)
    ax.set_xlabel("Node frequency (Hz)")
    ax.set_title("11 nodes in every grid", loc="left", fontsize=15)
    ax.text(0, -.29, "Teal: lowest 4   Red: highest 4\nYellow: task modulation range", transform=ax.transAxes, fontsize=10)
    ax = axes[1]
    t = np.linspace(0, c["duration_seconds"], 300)
    tau = -1 / (c["sampling_rate_hz"] * np.log(c["baseline_r"]))
    ax.plot(t*1000, np.exp(-t/tau), color=COLORS["baseline"], lw=3, label="Baseline (~100 s)")
    ax.plot(t*1000, np.exp(-t/c["intervention_decay_seconds"]), color=COLORS["high_damped"], lw=3, label="Intervention (20 ms)")
    ax.set(xlabel="Time (ms)", ylabel="Free-response envelope", ylim=(0, 1.08))
    ax.set_title("Change decay, keep frequency", loc="left", fontsize=15)
    ax.legend(fontsize=10, frameon=False, loc="center right")
    ax.text(0, -.29, "Intervention applied to either group;\nall other nodes retain baseline damping.", transform=ax.transAxes, fontsize=10)
    ax = axes[2]
    rng = np.random.default_rng(c["data_seed"])
    t = np.arange(100) / 1000
    for i, (label, rate) in enumerate((
            ("Constant rate", np.full(100,100.)),
            ("Rhythmically modulated", 100*(1+np.sin(2*np.pi*25*t))))):
        spikes = rng.random(100) < -np.expm1(-rate/1000)
        ax.vlines(t[spikes]*1000, i, i+.6, color="#25374A", lw=2)
        ax.text(0, i+.75, label, fontsize=11)
    ax.set(xlabel="Time (ms)", xlim=(-2,102), ylim=(-.1,2), yticks=[])
    ax.set_title("Classify 100 ms spike trains", loc="left", fontsize=15)
    ax.text(0, -.29, "Modulation sampled from 10–50 Hz.\nSame train / validation / test data.", transform=ax.transAxes, fontsize=10)
    fig.text(.06, .035, "4 grids × 3 damping conditions × 3 matched training seeds = 36 fits.  Models retrained in each condition.", fontsize=12)
    save(fig, figs, "01_method")

    fig, axes = plt.subplots(1, 2, figsize=(15.6, 6.0))
    fig.subplots_adjust(left=.07, right=.98, bottom=.22, top=.78, wspace=.3)
    fig.suptitle("Classification changes after selective damping", fontsize=21, y=.96)
    seeds = c["training_seeds"]
    scores = {(int(r["grid"]), int(r["seed"]), r["condition"]): float(r["test_accuracy"]) for r in rows}
    summary = []
    for offset, condition in zip((-.2, 0, .2), COLORS):
        means = []
        for g in range(4):
            values = np.array([scores[g, s, condition]*100 for s in seeds])
            means.append(values.mean())
            axes[0].scatter(g+offset+np.linspace(-.025,.025,len(values)), values, s=28,
                            alpha=.65, color=COLORS[condition])
            summary.append(dict(grid=g, start_hz=starts[g], condition=condition,
                                mean_test_accuracy=float(values.mean()/100),
                                sd_training_seeds=float(values.std(ddof=1)/100),
                                n_training_seeds=len(seeds)))
        axes[0].plot(np.arange(4)+offset, means, "o-", color=COLORS[condition],
                     label=LABELS[condition], lw=2, ms=8)
    axes[0].axhline(50, color="#BBBBBB", ls="--", lw=1)
    axes[0].set_title("Held-out test accuracy", loc="left", fontsize=16)
    axes[0].set(xticks=range(4), xticklabels=labels, xlabel="Starting frequency", ylabel="Accuracy (%)")
    axes[0].legend(fontsize=10, frameon=False)
    effects = {}
    for condition, offset in (("low_damped", -.08), ("high_damped", .08)):
        values = np.array([[100*(scores[g,s,condition]-scores[g,s,"baseline"]) for s in seeds] for g in range(4)])
        effects[condition] = values
        axes[1].plot(np.arange(4)+offset, values.mean(1), "o-", lw=2, ms=8,
                     color=COLORS[condition], label=LABELS[condition])
        for g in range(4):
            axes[1].scatter(g+offset+np.linspace(-.025,.025,len(seeds)), values[g],
                            color=COLORS[condition], s=28, alpha=.65)
    axes[1].axhline(0, color="#999999", ls="--", lw=1)
    axes[1].set_title("Paired change from each grid's baseline", loc="left", fontsize=16)
    axes[1].set(xticks=range(4), xticklabels=labels, xlabel="Starting frequency", ylabel="Accuracy change (percentage points)")
    axes[1].legend(fontsize=10, frameon=False)
    for ax in axes:
        ax.grid(axis="y", alpha=.15)
    fig.text(.07, .075, "Large points: mean of 3 training seeds. Small points: individual fits. Shared independent test set: 2,000 examples.", fontsize=11)
    fig.text(.07, .035, "Exploratory sensitivity check on one dataset; shifted grids change frequency coverage. No inference about consciousness.", fontsize=11)
    save(fig, figs, "02_results")
    with (folder / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    # Direct-filter check: remove recurrent coupling to isolate oscillator changes.
    diag = read_csv(folder / "filter_diagnostic.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.subplots_adjust(bottom=.24, top=.80, wspace=.3)
    fig.suptitle("Check the intervention independently of learned coupling", fontsize=17)
    for condition in ("low_damped", "high_damped"):
        base = [r for r in diag if int(r["grid"]) == 0 and r["condition"] == "baseline"]
        altered = [r for r in diag if int(r["grid"]) == 0 and r["condition"] == condition]
        ratio = np.array([float(r["mean_raw_feature"]) for r in altered]) / np.array([float(r["mean_raw_feature"]) for r in base])
        axes[0].plot([float(r["frequency_hz"]) for r in base], ratio, "o-", color=COLORS[condition], label=LABELS[condition])
    axes[0].set(xscale="log", yscale="log", xlabel="Node frequency (Hz)", ylabel="Raw feature / baseline")
    axes[0].axhline(1, color="#999999", ls="--")
    axes[0].legend(fontsize=9, frameon=False)
    axes[0].set_title("Uncoupled filter response, original grid", loc="left", fontsize=12)
    for condition in COLORS:
        selected = [r for r in rows if int(r["grid"]) == 0 and r["condition"] == condition]
        histories = [read_csv(folder / "histories" / f"grid0_{condition}_seed{r['seed']}.csv") for r in selected]
        curve = np.mean([[float(r["validation_accuracy"])*100 for r in h] for h in histories], axis=0)
        axes[1].plot(np.arange(1,len(curve)+1), curve, color=COLORS[condition], label=LABELS[condition])
    axes[1].set(xlabel="Epoch", ylabel="Validation accuracy (%)")
    axes[1].set_title("Training check, original grid (seed means)", loc="left", fontsize=12)
    axes[1].legend(fontsize=9, frameon=False)
    fig.text(.08, .055, "Filter check uses W = 0 and 100 fixed test inputs. Raw features combine position and velocity; they are not EEG power.", fontsize=10)
    save(fig, figs, "03_diagnostics")

    lines = ["# Recorded exploratory results", "",
             "Four grids; three freshly trained damping conditions; three matched training seeds per condition.",
             "One shared dataset. No significance tests or claims of clinical validity.", "",
             "| Start (Hz) | Baseline accuracy | Low damping change | High damping change |", "|---|---|---|---|"]
    for g, start in enumerate(starts):
        baseline = np.mean([scores[g,s,"baseline"]*100 for s in seeds])
        lines.append(f"| {start:.2f} | {baseline:.2f}% | {effects['low_damped'][g].mean():+.2f} pp | {effects['high_damped'][g].mean():+.2f} pp |")
    lines += ["", "Changes are paired against baseline with the same grid and training seed.",
              "Individual fits are shown in the figure and recorded in runs.csv; averages alone can hide training variability."]
    (folder / "RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(f"Saved three figures (PNG/SVG/PDF), summary.csv, RESULTS.md to {folder}")


if __name__ == "__main__":
    main()
