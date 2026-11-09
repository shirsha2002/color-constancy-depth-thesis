# Paper-budget reproduction running

The CPU and Intel XPU workers have been launched and actual training progress
has been observed. Source-equivalence checks and exact restart tests passed.
The experiment is DepthAWB at256points,
10000epochs per model and all three official folds. The main GPU suite contains
12jobs; a separate CPU reference covers one fold/model.

Final results are **not complete**. Actual live progress is saved in the local
`results/xpu_full_v1/` and `results/cpu_reference_v1/` directories; run
`python phase5_pccc/status.py` with the project interpreter to inspect it.

After all scheduled jobs and independent checks succeed, this status is replaced
by links to the measured results. The earlier100epoch pilot is preserved in
`phase4_pccc/` and must not be presented as the new10000epoch result.

This local checked-in file is a launch snapshot. `status.py` reads live progress.
The completion publisher updates GitHub in a separate worktree; a local tracked
copy can remain at its launch text until the new remote commit is pulled.
