# Selective damping and frequency placement in an RRN

This extension asks whether selectively damping low- versus high-frequency
nodes changes spike-train classification, and whether the effect persists
across four shifted golden-ratio frequency grids. It is an exploratory model
sensitivity experiment motivated by brain rhythms, not a model of consciousness.

## Design fixed before evaluation

The starting-frequency interval runs from the original 2 Hz to `2/phi` Hz,
where `phi = 1.618`. Divide that interval into four equal steps in Hz and use
the original start plus the three interior points:

```text
start[j] = 2 - j * (2 - 2/phi) / 4,  j = 0, 1, 2, 3
```

This gives **2.0000, 1.8090, 1.6180, and 1.4271 Hz**, with the lower endpoint
`2/phi = 1.2361 Hz` excluded. The starting frequencies are equally spaced in
Hz; the nodes within each grid remain multiplicatively spaced by 1.618.
Each grid contains exactly 11 nodes with frequencies `start * 1.618**k` for
`k = 0,...,10`.

This rule samples placements across one golden-spacing interval and has a
clearer rationale than the previous small exponential shifts. We exclude the
lower endpoint because a full one-node shift shares ten frequencies with the
original grid, replacing only its highest frequency with a new lowest one.
The original and three interior grids provide four distinct placements.
Downward shifts retain all 11 nodes below the code's `Fs/4 = 250 Hz` limit.
These deterministic shifts check sensitivity to node placement; they do not
compare frequency ratios or constitute randomly sampled placements.
The shifts also change the covered frequency range and extend below the paper's
2 Hz lower bound. This limitation is intentional and must be reported.

Every grid has three conditions:

| Condition | Lowest four nodes | Middle three nodes | Highest four nodes |
|---|---|---|---|
| Baseline | Original damping | Original damping | Original damping |
| Low damped | 20 ms decay | Original damping | Original damping |
| High damped | Original damping | Original damping | 20 ms decay |

"Low" and "high" refer to relative node ranks, not clinical EEG bands.
The lowest four original frequencies are approximately 2, 3.2, 5.2, and 8.5 Hz;
the highest four are 58.1, 93.9, 152.0, and 245.9 Hz. The task's modulation
range is 10–50 Hz, so neither affected group coincides with that range in the
original grid. After shifting, the lowest of the high group can enter the range
(approximately 46.97 Hz in the 1.6180 Hz grid and 41.43 Hz in the 1.4271 Hz grid).
This task alignment is part of the interpretation, not a claim of matched
low- and high-frequency information.

Baseline `r = 0.99999` has a free-response envelope decay time near 100 seconds
at 1000 Hz. Intervention `r = exp(-1/(1000*0.020)) = 0.951229...` has a
20 ms decay time. The 20 ms value is a prespecified illustrative intervention,
not fitted to test accuracy and not a clinical estimate. Damping also changes
gain and resonance width; a classification effect cannot be assigned solely
to memory. Frequencies and node count remain fixed within each grid.

Each of the 12 grid/condition configurations is trained afresh with three
matched training seeds (101, 202, 303): 36 fits. For a given training seed,
all configurations start with the same trainable parameter tensors and receive
the same minibatch ordering. The buffers encoding frequency/damping differ.
All configurations share the exact same training, validation, and test arrays.
Frequency placement is a robustness check; the primary comparisons are paired
accuracy changes from each grid's own baseline. Compare these paired changes
across grids to assess whether the damping effect depends on placement.
Baseline-versus-baseline and the same damped condition across grids are also
valid comparisons of placement sensitivity. Changing placement changes coverage
and task alignment together, so it does not isolate their individual mechanisms.
This is not a 2×2 design.

## Data and training

Data use the upstream simulated-spike law: binary observations of at least one
Poisson event per 1 ms bin, in 100 ms windows. Half the examples have a constant
underlying rate of 100 Hz. Half have rate `100*(1+sin(2*pi*f*t))`, where `f` is
uniform in 10–50 Hz and the phase is fixed at zero, matching upstream.
Classes share a nominal mean rate; finite-window means and observed counts
can differ. Consequently, this is a task using temporal patterns and possibly
count cues, not a pure test of spectral discrimination.

Training: 1000 examples; validation: 200; independent test: 2000. The larger
test set is a deliberate departure from the paper's 200 examples to reduce
test sampling noise. Three independent local RNG streams generate the splits
from data seed 20261007. Changing test size leaves training/validation unchanged.

Training follows the upstream pipeline: CPU PyTorch, 30 epochs, minibatches of
128, AdamW, gradient norm clipping at 1, and cosine OneCycle scheduling.
Learning rate 0.003, weight decay 0, and label smoothing 0 use the published
golden-RRN choices. Recurrent diagonals are projected to zero after updates.
Best validation accuracy selects the checkpoint (first checkpoint wins ties).
Only then is the held-out test set evaluated. No test-driven setting selection.

The original model recurrence, input gating, batch normalization, and
position/velocity amplitude feature calculation are retained. Only explicit
frequency and per-node damping arguments are added to `RRNModel`; default
frequency arrangements and damping are unchanged. The binary classifier has
167 allocated trainable scalars, of which 11 diagonal weights are constrained
to zero: 156 effective free parameters. This differs from the paper's 252
effective parameters for its 10-class MNIST readout.

## Run locally

From the repository root on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r experiments/damping/requirements.txt
.\.venv\Scripts\python.exe -m unittest experiments.damping.test_experiment -v
.\.venv\Scripts\python.exe experiments/damping/run.py
.\.venv\Scripts\python.exe experiments/damping/plot_results.py
.\.venv\Scripts\python.exe experiments/damping/verify_results.py --replay
```

Use `requirements-lock.txt` instead of `requirements.txt` to match the recorded
environment. On Linux/macOS, use `.venv/bin/python` in place of the Windows path.
Results are not overwritten by default. To rerun independently:

```powershell
.\.venv\Scripts\python.exe experiments/damping/run.py --output experiments/damping/results_rerun
.\.venv\Scripts\python.exe experiments/damping/plot_results.py --results experiments/damping/results_rerun
```

`--resume` permits interrupted runs only when configuration, data, core source
hashes, NumPy, and PyTorch match the recorded metadata. The experiment enables
deterministic CPU algorithms and uses one PyTorch CPU thread. Bitwise matching
across different hardware/library versions is not guaranteed.

## Review the code and artifacts

| File | Purpose |
|---|---|
| `../../RRN_functions.py` | Optional explicit frequencies and per-node damping |
| `config.json` | Complete prespecified experiment settings |
| `run.py` | Local data RNG, paired training, checkpoint selection, measurements |
| `test_experiment.py` | Node stability/count, matched parameters, data isolation, finite gradients |
| `plot_results.py` | Figures generated only after all unique fits are complete |
| `verify_results.py` | Audit recorded runs and optionally replay a complete baseline fit |
| `results/metadata.json` | Configuration, versions, source/data hashes, provenance |
| `results/runs.csv` | Every fit's validation/test metrics and selected epoch |
| `results/histories/` | Every epoch's train and validation metrics |
| `results/paired_effects.csv` | Accuracy change against matching baseline |
| `results/filter_diagnostic.csv` | Raw feature check with recurrent coupling set to zero |
| `results/summary.csv` | Per-grid/condition means and seed standard deviations |
| `results/RESULTS.md` | Recorded result table |
| `results/figures/` | Method, result, and diagnostic figures in PNG, SVG, PDF |

Large local checkpoints, predictions, and the generated dataset are excluded
from Git. They can be regenerated using the committed configuration and code.
All small result tables, histories, provenance, and figures are reviewable in Git.
The unmodified upstream notebooks are not dependencies of this extension's
runner, and may need additional packages and datasets to execute.

## Two-slide presentation

1. Method: use `figures/01_method.png` to explain the task, grids, and damping.
2. Results: use `figures/02_results.png` to show accuracy and paired changes.

Use `03_diagnostics.png` for questions about the intervention or training.
PNG is convenient for PowerPoint; SVG/PDF preserve vector detail.

Show each grid and training seed rather than just a grand mean. The fits share
one dataset, and shifts are fixed rather than randomly sampled. Do not treat
36 fits as 36 independent dataset replications. Three training seeds provide
an exploratory view of training variability, not a precise uncertainty estimate.
No significance tests are reported. Findings apply to this task, damping choice,
frequency range, and dataset; they do not establish biological equivalence,
consciousness, or universal superiority of golden spacing.

Reference: Mark A. Kramer, *Brain-inspired, interpretable, resonant recurrent
neural networks*, https://arxiv.org/abs/2506.17083 and the upstream repository
https://github.com/Mark-Kramer/Resonant-Recurrent-Network.
