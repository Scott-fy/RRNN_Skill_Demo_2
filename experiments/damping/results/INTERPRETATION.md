# Interpretation of the revised equal-spacing study

We reran all 36 fits with starting frequencies of 2.0000, 1.8090, 1.6180, 1.4271 Hz.
The rule is `start[j] = 2 - j*(2 - 2/1.618)/4`, for j = 0, 1, 2, 3.
The original 2 Hz grid and three interior points divide the starting-frequency
interval into equal steps in Hz. The lower endpoint 2/phi is excluded.
Golden spacing within each 11-node grid is unchanged.

## Findings

High-frequency damping improved mean accuracy in all four grids. It improved 10 of 12 individual paired fits. Across these fits, its mean paired change was +1.08 percentage points; low-frequency damping's mean change was -0.32 percentage points. These are descriptive averages on a shared dataset.

High-frequency damping's grid-averaged changes, in order from 2.000 to 1.427 Hz,
were +0.83, +2.18, +0.10, +1.22 percentage points. Low-frequency damping's changes
were -0.27, +0.30, -0.22, -1.10 percentage points. Individual fits vary; inspect
the small points in the result figure and the rows in paired_effects.csv.
No significance tests are reported.

## What the comparisons establish

Within a grid, baseline versus an intervention measures the damping effect
at that placement. Across grids, baseline versus baseline (or the same damped
condition versus itself) measures placement sensitivity. Comparing paired
damping-minus-baseline changes across grids tests whether the intervention's
effect depends on placement. This directly addresses whether the original
starting point at 2 Hz was unusually favorable for that effect.

Grid shifts change both alignment and frequency coverage. Therefore, these
comparisons measure placement sensitivity but do not uniquely explain whether
alignment, coverage, or altered learned coupling caused an observed difference.
That qualification does not invalidate the robustness question.

## Physiological motivation and possible explanations

The lowest four and highest four nodes are selected by rank in every grid;
they are not fixed clinical EEG bands. The task modulates spike rate at 10–50 Hz.
Most nodes in the highest group lie above that range, but its lowest node moves
from approximately 58.06 Hz in the original grid to 46.97 Hz and 41.43 Hz in
the final two grids. Thus the group's relation to informative input changes.

Damping changes persistence, gain, resonance width, and feature scaling.
Retraining also changes connections. Filtering unhelpful activity is one
possible explanation for improvements, but this experiment does not isolate
that mechanism. The uncoupled diagnostic (W = 0) checks the direct filter
intervention using raw position/velocity amplitude features. It does not
establish the mechanism of the final trained network.

This is a brain-motivated computational experiment, not a validated simulation
of an awake or unconscious brain. Classification performance cannot establish
equivalence to a consciousness mechanism.

## Suggested two-slide narration

Method: "We kept golden spacing and 11 nodes. We divided the interval from
2 Hz to 2/phi into four equal steps and used the original start plus three
interior points. We shortened decay to 20 ms for either the lowest four or
highest four nodes, retrained with matched initialization and data, and compared
each intervention with the baseline in the same grid."

Results: "High-frequency damping improved mean accuracy in all four grids. It improved 10 of 12 individual paired fits. Across these fits, its mean paired change was +1.08 percentage points; low-frequency damping's mean change was -0.32 percentage points. These are descriptive averages on a shared dataset. This tests how oscillator damping interacts with task
structure and node placement. It does not measure consciousness."

## Validation and scope

Seven scientific-invariant tests passed, including the new quarter-interval
placement rule and endpoint exclusion. The recorded-result audit checks all
36 histories, first-best validation checkpoint selection, prediction accuracy,
regenerated data hash, source hashes, and paired changes. A complete 30-epoch
baseline replay verified reproducibility.

Only the starting-frequency list changed in the configuration. Training,
validation, and test arrays have the same hash as in the previous study.
All nine original-grid fits exactly match the previous checkpoint choices and
evaluation metrics. Previous small-shift results remain available in Git history;
the figures and tables in this directory describe the revised study.

Limits: one dataset, three training seeds, four deterministic grids, one damping
strength, relative node groups, and changing frequency coverage. The fixed-phase
short-window task can contain count cues as well as temporal-pattern cues.
Isolated-mode stability does not prove coupled-network stability. Results do
not establish statistical significance or generalize to other tasks or bands.
