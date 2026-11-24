# Local GPU first: measured readiness

Update 23 November: all 21 pilot models completed on XPU and their audit passed.
The original strict CPU/XPU gradient mismatch below remains recorded. Separate
ten-update repeatability tests passed twice with identical XPU states and initial
prediction angle disagreement 0.000714 degrees, below the predeclared 0.02-degree
bound. The worker verifies that clearance evidence before an original-PCCC XPU
job. See `pccc_xpu_repeatability.json`, `PILOT_PROTOCOL.md` and `AUDIT_RECOVERY.md`.
The following details record the earlier hardware check and initial restriction.

16 November 2026. User preference: use the laptop GPU if practical; use Modal
when local hardware cannot reasonably handle the work. No new cloud spending
or Modal job was needed for these checks. The canceled phase-five run remains
stopped; these are separate prototype implementation checks.

The installed Intel XPU runtime detects an Intel Arc 140T. Its reported 16 GB
is not a claim of 16 GB dedicated VRAM. Both checks used PyTorch 2.14.1 in the
existing XPU environment, with four CPU threads and deterministic algorithms.

## Measurements

One step uses 16 official ETH3D training captures, two forward passes, angular
and consistency losses, backward and Adam. Two warmup updates precede eight
synchronized timed updates. No held-out capture was used.

| Mode | Intel GPU seconds/step | CPU seconds/step |
|---|---:|---:|
| Learned fusion | 0.06293 | 0.03623 |
| Mask fusion | 0.01650 | 0.01404 |
| Ordinary fusion | 0.01679 | 0.01365 |
| RGB control | 0.01508 | 0.01170 |

All four fusion modes passed CPU/GPU prediction and gradient tolerance checks
and ten finite GPU optimizer updates. The learned model's framework peak
allocated memory was about 33.37 MiB. This excludes driver/runtime overhead
and is not total system/GPU memory usage. A separate synthetic two-epoch test
passed on XPU, including checkpoint saving/loading, exact same-device prediction
re-inference, optimizer/test separation and final completion records.

**Decision:** the local GPU can run the fusion prototype and modest pilots.
The CPU was faster for these small batches; GPU availability does not guarantee
acceleration. At the measured step time, a 100-epoch ETH3D fold requires roughly
44 seconds of training compute, and 10,000 epochs about 73 minutes. These are
extrapolations only and exclude validation, I/O, checkpointing, startup, thermal
throttling and other workloads. Do not extrapolate this to a completed suite.

## Original-PCCC baseline is not cleared for XPU

The full XPU hardware check ended with FAILED after the original PCCC baseline
failed the predeclared CPU/GPU gradient comparison (atol 0.002, rtol 0.02).
The four fusion modes passed before that failure. This is retained explicitly
in [hardware_xpu.json](hardware_xpu.json); the baseline was not silently skipped
or accepted with a relaxed tolerance. The CPU check passed all five modes;
original PCCC measured about 0.2231 seconds/step there.

The XPU training worker now permits the four fusion modes and explicitly refuses
an original-PCCC XPU run until that discrepancy is resolved. Do not attribute
the discrepancy to a particular kernel or claim the baseline is invalid without
further investigation. A full matched study still needs its frozen protocol and
an appropriate baseline backend. No large or paid experiment has been launched.

## Evidence and commands

[GPU measurements](hardware_xpu.json), [CPU measurements](hardware_cpu.json),
[XPU worker check](worker_checks_xpu.json), and [benchmark source](benchmark_hardware.py).

```powershell
& .\.venv-xpu\Scripts\python.exe phase9_reliability_fusion/benchmark_hardware.py --device xpu
& .\.venv-xpu\Scripts\python.exe phase9_reliability_fusion/benchmark_hardware.py --device cpu
& .\.venv-xpu\Scripts\python.exe phase9_reliability_fusion/test_worker.py --device xpu
```

The first command currently reports the retained original-PCCC gradient failure.
These commands make no Modal API calls. The worker also accepts `--device xpu`
for explicit new fusion training jobs. Modal remains the fallback when the
declared complete study is impractical locally, with a concrete cost bound
before paid execution.
