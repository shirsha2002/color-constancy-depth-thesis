# PCCC paper-budget DepthAWB reproduction

Shirsha Mitra / 2023521460130 / 9 November 2026.

This extends the completed100epoch pilot into **10000epochs/model, all three
official DepthAWB folds,256points/sample**. Read [STATUS.md](STATUS.md) for actual
completion status. Starting these jobs is not completing or reproducing the
entire paper.

## Runs

- **GPU main suite:** Intel Arc140T through PyTorch2.14.1+xpu. Four variants
  (paper-weighted real/flat geometry and released-sum real/flat geometry), three
  folds each, one predeclared seed per fold:12models.
- **CPU reference:** separate10000epoch paper-weighted real-geometry fold0 run.
  It is not mixed into the GPU out-of-fold series.
- **Selection rules:** report predeclared fixed-final results as primary.
  Separately reproduce the released entry point's held-out-selected checkpoint,
  explicitly labelled test-selected/optimistic.

Full specification and source/paper/preprocessing differences:
[PROTOCOL.md](PROTOCOL.md). The earlier pilot stays in `phase4_pccc/`.

## Verified hardware and restartability

Both devices pass actual PCCC forward/backward and Adam checks. The matched
deterministic benchmark is in [benchmark_cpu.json](benchmark_cpu.json) and
[benchmark_xpu.json](benchmark_xpu.json); its GPU speedup is about5x for the
measured batch computation, not an end-to-end training guarantee.

The original CPU environment stays unchanged. Intel XPU is installed in the
isolated `.venv-xpu` environment, sharing existing scientific packages through
an explicit local `.pth` file. [xpu_environment.json](xpu_environment.json)
records the installed versions; [hardware.json](hardware.json) records the GPU.
No graphics-driver change was needed for the checked execution.

Original same-size256point pooling is mathematically an identity, but its XPU
backward has no deterministic implementation. The wrapper bypasses that
redundant pool only for256point input. CPU outputs and gradients match the
original exactly. Both CPU and XPU interrupted/resumed checks reproduce final
weights and histories bitwise; see [CPU](restart_preflight.json) and
[XPU](restart_preflight_xpu.json).

## Monitor without launching another worker

From the project root:

```powershell
& .\.venv\Scripts\python.exe phase5_pccc/status.py
Get-Content phase5_pccc/results/xpu_full_v1/worker.log -Tail 12
Get-Content phase5_pccc/results/cpu_reference_v1/worker.log -Tail 12
```

In VS Code: **Terminal > Run Task > PCCC: paper-budget status**.

While training, the worker requests Windows to stay awake automatically;
display sleep remains allowed. Keep the laptop plugged in. Closing the lid,
rebooting or explicitly suspending Windows may interrupt work. Every50epochs,
model, Adam state and all RNG states are atomically saved for resumption.

## Launch or resume (only when that worker is stopped)

```powershell
& .\.venv\Scripts\python.exe phase5_pccc/launch.py --device xpu --out phase5_pccc/results/xpu_full_v1
& .\.venv\Scripts\python.exe phase5_pccc/launch.py --device cpu --out phase5_pccc/results/cpu_reference_v1 --reference-only
```

The launcher refuses an existing live worker. Training also holds a per-job
process lock. Do not start a second copy in the same output directory.
The two reference/main workers may overlap, so their wall times include shared
CPU/GPU power and memory contention. Initial hardware timing was measured
sequentially before these long runs.

## Completion pipeline

Workers finish their predeclared schedules, aggregate out-of-fold predictions,
independently recompute angular scores, verify checkpoints/source/data hashes
and re-run saved models. They write numerical tables and a readable result
summary only after checks pass. Raw data and large checkpoint files stay local.

After successful audit, the authorised publication step uses a **separate Git
worktree**, checks that GitHub has the same frozen numerical source, and uploads
only the completed numerical outputs/status. It does not checkout, stage or
modify the user's working branch or existing PDF. It refuses divergent pushes
and different experiment identities. CPU results use a separate reference status.
If publication fails, local verified results remain available and the worker log
shows the failure; no fake success or automatic force-push is performed.

This scope excludes the paper's other datasets,4096point setting, all competing
methods and local multi-illuminant validation. Published accuracy is not promised.
