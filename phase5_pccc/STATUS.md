# Paper-budget reproduction prepared

The CPU and Intel XPU implementations, source-equivalence checks and exact
restart tests have passed. The planned experiment is DepthAWB at256points,
10000epochs per model and all three official folds. The main GPU suite contains
12jobs; a separate CPU reference covers one fold/model.

Final results are **not complete**. Actual live progress is saved in the local
`results/xpu_full_v1/` and `results/cpu_reference_v1/` directories; run
`python phase5_pccc/status.py` with the project interpreter to inspect it.

After all scheduled jobs and independent checks succeed, this status is replaced
by links to the measured results. The earlier100epoch pilot is preserved in
`phase4_pccc/` and must not be presented as the new10000epoch result.
