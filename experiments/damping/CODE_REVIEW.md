# Code review guide

Upstream base: `2334cd8e8c0cc6032611adf54627595d128b7bad`.
Local review branch: `experiment/selective-damping`.

## Review scope

1. `RRN_functions.py` adds optional explicit node frequencies and scalar/per-node
   damping. Explicit grids raise on invalid modes instead of silently dropping
   nodes. Existing defaults preserve original behavior, verified by identical
   parameters, buffers, and forward output in the compatibility test.
2. `experiments/damping/config.json` defines all settings before evaluation.
3. `run.py` implements independent data streams, matched initialization and batch
   order, validation-only checkpoint choice, paired comparisons, local artifacts,
   provenance, and resume checks. The optimizer and feature calculation follow
   the original approach. The runner avoids upstream notebook dependencies.
4. `test_experiment.py` exercises meaningful scientific invariants. The CI file
   runs these tests without running the full study. CI configuration is included;
   execution was verified locally, not on a hosted CI service.
5. `verify_results.py` audits measurements and can reproduce a full baseline fit.
6. `plot_results.py` generates three figures from complete unique recorded runs.
7. `results/` includes measurements, histories, provenance, figures, and an
   interpretation. Checkpoints, predictions, and generated arrays stay local
   and are reproducible rather than versioned binary training artifacts.

## Inspect changes

```powershell
git diff 2334cd8e8c0cc6032611adf54627595d128b7bad..HEAD -- RRN_functions.py experiments/damping/run.py experiments/damping/test_experiment.py
git log -1 --stat
.\.venv\Scripts\python.exe -m unittest experiments.damping.test_experiment -v
.\.venv\Scripts\python.exe experiments/damping/verify_results.py
```

Setup and execution commands, design assumptions, deviations from the paper,
and limitations are in `README.md` in this directory. Exact runtime versions
are in `requirements-lock.txt` and `results/metadata.json`.

## Portable repository

The delivered `rrn-damping-review.bundle` contains Git history and the committed
review branch. From the folder containing that bundle:

```powershell
git clone -b experiment/selective-damping rrn-damping-review.bundle RRN-damping-review
```

The clone contains code, tests, settings, small recorded results, and all
figures. Follow the setup instructions to regenerate excluded training artifacts.
The accompanying ZIP contains the same tracked snapshot for convenient browsing.
