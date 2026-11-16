# PCCC camera-frame and depth-scale follow-up

Shirsha Mitra / 2023521460130. Started16November2026. US$8 cap approved in chat.

This tests the prior NYU/DIODE coordinate-convention hypothesis. It does not
implement the proposed reliability-aware fusion. Read [PROTOCOL.md](PROTOCOL.md)
before interpreting any results. Raw data and previous results remain unchanged.

Three new representations (camera-real, camera-unitdepth, camera-flat) x three
official folds = nine released-head10000epoch models. Existing native and
native-flat phase-six models are reused. All RGB/labels are unchanged. Input
transformation assertions and exact sample hashes are in preflight.json and
inputs/*/manifest.json. Cloud budget planning is in budget.json.

Local preparation (does not allocate a cloud GPU):

```powershell
& .\.venv\Scripts\python.exe phase7_depth_format/prepare.py
```

Cloud commands allocate paid T4 resources; do not repeat a completed run:

```powershell
$env:PYTHONUTF8='1'
& .\.venv-modal\Scripts\modal.exe run phase7_depth_format/run_modal.py --confirm-budget 8 --stage smoke
& .\.venv-modal\Scripts\modal.exe run --detach phase7_depth_format/run_modal.py --confirm-budget 8 --stage train
```

Only the already approved bounded protocol is allowed. No automatic retry or
fallback to a more expensive GPU. The run's remote folder is
`pccc-results/nyu_frame_v1`. Completion records are mandatory before scoring.
The cancelled phase-five laptop jobs remain stopped.

After download to `phase7_depth_format/results/nyu_frame_v1`, locally run:

```powershell
& .\.venv\Scripts\python.exe phase7_depth_format/analyze.py
```

The analysis checks complete/config/input/checkpoint identities, independently
recomputes all angles and re-infers final checkpoints. It reports all declared
contrasts, both checkpoint rules, and NYU/DIODE subgroup summaries. Check
RESULTS.md and results/nyu_frame_v1/audit.json for actual completion; preparation
or launch alone is not a result.

