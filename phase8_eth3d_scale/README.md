# ETH3D scene-scale follow-up: inputs prepared

16 November 2026. Continues the unfinished local preparation identified in
[the phase-seven next-step plan](../phase7_depth_format/NEXT_STEPS.md).

**195 derived point clouds are prepared and independently verified. No new
model has been trained and no new angular-error result is available.**

This isolates per-capture XYZ scale on ETH3D, where the previous study observed
a geometry advantage. It does not implement fusion or replace measured depth
with RGB-predicted depth. Read the [frozen protocol](PROTOCOL.md) before training.

## Prepared and checked

- All 195 native clouds and labels match their source hashes; the existing flat
  arrays match the declared ray-preserving transformation exactly.
- The new condition divides valid XYZ by median absolute valid Z, retaining
  native axes, point ordering, RGB and subthreshold geometry.
- Official folds remain 130 training / 65 held-out captures each, with each
  capture held out exactly once. No exact duplicate native cloud files were
  found; this does not establish physical scene independence.
- The independent saved-array check confirms reconstruction, relative depth,
  camera rays, labels, masks, signs and RGB preservation. After float32
  conversion the maximum absolute ray-coordinate difference is
  1.7881393432617188e-7, within the declared numerical tolerance.
- The actual existing tensor loader accepted all 195 captures unchanged in
  `real` mode. All six reusable native/flat final-checkpoint hashes, recorded
  prediction membership, configuration and numerical-source hashes passed.

Input inspection found 30 subthreshold nonzero XYZ points in 22 captures.
They were retained exactly. An initial preparation guard assumed that every
subthreshold point was zero and stopped before completing the manifest; that
assumption was corrected using input evidence before any training or scoring.
This is not evidence of dataset corruption. There are 13,047 subthreshold points
in total, including exact zeros. No original data or earlier results were changed.

## Evidence and local commands

[Input manifest](inputs/ETH3D_unitdepth/manifest.json),
[preparation checks](preflight.json), [independent verification](verification.json)
and [baseline/loader readiness](readiness.json) record identities and outcomes.
The protocol and preparer hashes are frozen in the manifest. Preparation refuses
to overwrite different frozen evidence. Derived cloud files stay local.

```powershell
& .\.venv\Scripts\python.exe phase8_eth3d_scale/prepare.py
& .\.venv\Scripts\python.exe phase8_eth3d_scale/verify_inputs.py
& .\.venv\Scripts\python.exe phase8_eth3d_scale/check_readiness.py
```

These commands prepare/check local files; they allocate no GPU and launch no
training. The intended follow-up would train three released-head models at
10,000 epochs, reusing six earlier baselines. A bounded execution runner,
training-only runtime estimate and explicit new spending cap are still needed
before paid execution. The completed phase-seven $8 allowance is not reused.
The canceled laptop experiment remains stopped.

The separate RGB-predicted-depth comparison still needs original full RGBs and
verified image/point correspondence, as recorded in the
[ETH3D feasibility note](../phase7_depth_format/ETH3D_FEASIBILITY.md).
