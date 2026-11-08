# PCCC: small, genuine-data CPU reproduction

Shirsha Mitra, 2023521460130. Started 9 November 2026.

This study follows **Point Cloud Color Constancy (CVPR 2022)**. Its output is an
illumination chromaticity estimate. It does not substitute monocular depth
accuracy for CC evaluation.

The locked pilot uses all 185 original DepthAWB samples, each containing 256 XYZRGB
points and an illumination label. Official fold 0 supplies 62 test captures;
98 samples are training and 25 are inner validation. There are four variants and
three seeds each, 100 epochs/model, on the laptop's CPU. The final test stage runs
only after all twelve models finish.

The source remains pinned and unmodified locally. One return-unpacking error
is fixed in memory. The released three-channel summed head and the paper's
four-channel weighted-head restoration are separately named. Their augmentation
differences, the preserved invalid-depth mask and all reduced-budget departures
are documented in [the protocol](PROTOCOL.md).

Read [the measured results](RESULTS.md),
[every seed](results/cpu_v1/per_seed_metrics.csv),
[the independent audit](results/cpu_v1/independent_audit.json), and
[the short English professor update](../output/QQ_PCCC_CPU_Pilot_English.txt).
This is an **adapted reproduction pilot**, not a full published-score replication
or a new algorithm claim.

## Run on this configured laptop

From the project root, without another copy of the pilot already running:

```powershell
& .\phase4_pccc\run.ps1
```

In VS Code choose **Terminal > Run Task > PCCC: run CPU pilot**. Completed models
are verified and reused. A changed frozen training configuration is refused;
use a new experiment directory and protocol for a new numeric study.

The existing `.venv` remains the project interpreter. Native point clouds,
labels, input archives and upstream source stay on the laptop. All 12 checkpoint
identities and predictions are retained; the first predeclared seed's four
checkpoints are versioned, while eight additional seed checkpoints remain local
to keep Git lightweight. First-seed publication was fixed before test inspection.

## Prepare a fresh checkout

Python 3.13 was used for this run. From the repository root:

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r phase4_pccc/requirements.txt
git clone https://github.com/xyxingx/Point-Cloud-Color-Constancy.git external/Point-Cloud-Color-Constancy
git -C external/Point-Cloud-Color-Constancy checkout ddc8770e04ef5509a621c5a7ee04e330b9204a93
& .\.venv\Scripts\python.exe phase4_pccc/acquire_inputs.py
& .\phase4_pccc\run.ps1
```

The minimal acquisition fetches only three original DepthAWB assets, about 23 MB,
and verifies their frozen SHA/ZIP identities. It does not download full-resolution
image archives or the other PCCC datasets. A verified full acquisition from the
previous phase also works. Camera registration and radiometry are inherited
from the original release, not independently reconstructed from the sparse data.

## Individual stages

```powershell
& .\.venv\Scripts\python.exe phase4_pccc/prepare.py
& .\.venv\Scripts\python.exe phase4_pccc/checks.py
& .\.venv\Scripts\python.exe phase4_pccc/pilot.py train --epochs 100 --threads 8
& .\.venv\Scripts\python.exe phase4_pccc/finish.py
```

Checkpoint selection uses inner validation only. The default outer test is not
an early-stopping set. All 12 models are required for final test evaluation; all
source/data fingerprints must match. Do not start a duplicate training process
in the same output directory.

The original article's 256-point DepthAWB reference mean is 1.05 degrees, under a
different published budget/evaluation. Do not claim that value has been
reproduced from merely running the source. [Paper](https://arxiv.org/abs/2111.11280),
[authors' repository](https://github.com/xyxingx/Point-Cloud-Color-Constancy).
