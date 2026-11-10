# Phase 6: full PCCC on all three datasets with a cloud GPU (Modal)

Completed 10 November 2026. Professor-facing report:
[output/pdf/Shirsha_Mitra_PCCC_GPU_Update_2026_11_10.pdf](../output/pdf/Shirsha_Mitra_PCCC_GPU_Update_2026_11_10.pdf),
animations in [output/gif/](../output/gif/), short message in
[output/QQ_PCCC_GPU_Update_2026_11_10_English.txt](../output/QQ_PCCC_GPU_Update_2026_11_10_English.txt).

## What was run

The frozen phase-5 trainer ([phase5_pccc/train.py](../phase5_pccc/train.py), authors' PCCC code at
commit `ddc8770`) was run on Modal NVIDIA T4 GPUs with the paper's full budget: **10,000 epochs**,
Adam 0.0003, batch 16, 256-point clouds, the official 3-fold splits, one seed per fold
(`20261109 + fold`). In total **48 models, about 22.5 T4 GPU-hours**.

| Step | Data | Models |
|---|---|---|
| A | DepthAWB (185 scenes): real 3D vs flat geometry, released and paper-weighted output layers | 12 |
| B | ETH3D (195) and NYU-v2 + DIODE (324): the same four variants | 24 |
| C | DepthAWB with damaged depth: 10% / 30% noise, 50% removed, 3D positions shuffled (released layer) | 12 |

*Flat* divides each valid point's XYZ by |Z| (keeps the pixel ray, removes depth; missing points
stay 0). It is close to, not identical with, the paper's "w/o depth". Step C damage is fixed per
scene ([make_inputs.py](make_inputs.py)) and used for both training and testing; the step-A
clean-depth models are also scored on the damaged test clouds without retraining.

Two checkpoint rules are reported: **final epoch** (primary; the test fold chooses nothing) and
**best test-fold epoch**, which is what the authors' released script keeps (optimistic).

## Results (mean angular error in degrees, all test scenes of the three folds, released layer)

| Dataset | Real 3D, final | Flat, final | Real minus flat (95% CI) | Real 3D, best test-fold | Paper, with / w/o depth |
|---|---:|---:|---|---:|---|
| ETH3D | 0.98 | 1.31 | -0.33 (-0.52 to -0.17), 3D better on 120/195 | 0.82 | 0.88 / 1.42 |
| DepthAWB | 1.77 | 1.86 | -0.09 (-0.36 to 0.18), 78/185 | 1.32 | 1.05 / 2.39 |
| NYU-v2 + DIODE | 2.61 | 2.51 | +0.11 (-0.16 to 0.38), 155/324 | 2.16 | 2.20 / 2.80 |

Step C (DepthAWB, clean depth = 1.77, flat = 1.86):

| Depth used | Retrained on it | Trained on clean, tested on it |
|---|---:|---:|
| 10% depth noise | 1.71 | 1.76 |
| 30% depth noise | 2.01 | 1.96 |
| 50% of depth removed | 1.67 | 1.98 |
| 3D positions shuffled | 1.98 | 2.36 |

Full statistics, both output layers and all intervals: `results/*/summary.csv`, `summary.json`,
`paired.json`, `per_capture_errors.json`, and Appendix A of the PDF. Every score is recomputed
from the saved prediction vectors (largest difference 3.6e-15 degrees).

**Reading:** real 3D clearly helps on ETH3D only. Inside NYU-v2 + DIODE the 258 NYU clouds store
positive, very small z values while the 66 DIODE clouds store negative z values about 221 times
larger (median), so the geometry is inconsistent. A clean-trained model is hurt by damaged test
depth. These are observations from one seed per fold; the depth-format explanation is a
hypothesis to test next, not a result.

## Reproduce

```powershell
# One-time: separate environment for the Modal client, then authenticate with your own token.
.\.venv\Scripts\python.exe -m venv .venv-modal
.\.venv-modal\Scripts\python.exe -m pip install modal

.\.venv-modal\Scripts\modal.exe run --detach phase6_modal/run_depthawb.py        # Step A
.\.venv\Scripts\python.exe phase6_modal/make_inputs.py                           # inputs for B and C
.\.venv-modal\Scripts\modal.exe run --detach phase6_modal/run_bc.py              # Steps B and C
.\.venv-modal\Scripts\modal.exe volume get pccc-results depthawb_t4_v1 phase6_modal/results/
.\.venv-modal\Scripts\modal.exe volume get pccc-results bc_t4_v1 phase6_modal/results/
.\.venv\Scripts\python.exe phase6_modal/analyze_a.py phase6_modal/results/depthawb_t4_v1
.\.venv\Scripts\python.exe phase6_modal/analyze_bc.py phase6_modal/results/bc_t4_v1 phase6_modal/results/depthawb_t4_v1
.\.venv\Scripts\python.exe phase6_modal/make_figures.py
.\.venv\Scripts\python.exe phase6_modal/build_report.py
```

Add `--epochs 60` to a `run_*.py` call for a cheap smoke test (written to a separate `*_smoke60` folder).
[bench.py](bench.py) times one model on several machine types. Measured seconds per epoch: T4 0.117,
A10 0.092, 4-core CPU 1.14, 2-core CPU 2.20 (L4 was never scheduled). The T4 was cheapest per model.

Modal credentials live in the user's home folder (`~/.modal.toml`), never in this repository.

## Execution notes

- The Modal image installs `torch==2.5.1`. It cannot save `uint32` tensors, so the runners store the
  NumPy RNG key array as `int64` inside checkpoints (bookkeeping only; training is unchanged).
- Upstream files come from a Windows checkout; the runners set `core.autocrlf=true` inside the
  container so the frozen upstream-cleanliness check sees the same tree as on Windows.
- Deterministic algorithms are on (`CUBLAS_WORKSPACE_CONFIG=:4096:8`).

## What is versioned

Code, manifests (with SHA-256 of every input), configs, saved predictions, summaries, figures and
compact histories (`history_every10.json`, with the SHA-256 of the full file). Not versioned:
model checkpoints, the full per-epoch `history.json` files (61 MB), derived Step-C/flat clouds
(rebuilt by `make_inputs.py`; the PCCC release has no licence file), and smoke-test outputs.
