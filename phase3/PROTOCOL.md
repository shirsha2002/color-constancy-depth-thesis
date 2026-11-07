# Phase three: CC-specific geometry and real multi-light evidence

Recorded 7 November 2026, before running the new method or scoring the new data.
Student: Shirsha Mitra, 2023521460130.

## Research questions

1. With geometry held fixed, how much depth error is attributable to chromatic
   illumination rather than brightness/shading alone?
2. Can a task-internal chromatic mechanism improve on ordinary adaptation with
   the same data, augmentation, losses, parameter budget and training budget?
3. Do results transfer from simulated changes to captured real RGB-D lighting?

## Evidence and prior work

Preserve the original NYU and phase-two experiments. The current module did not
beat ordinary adaptation; do not reframe it as a positive CC finding. The
supervisor identified Point Cloud Color Constancy (CVPR 2022): measured RGB-D
predicts illumination there, whereas this thesis predicts depth from RGB alone.
Ground-truth depth must never enter monocular inference.

LumiDepth (ECCV 2026) is close prior work on stable multi-illumination monocular
depth and supplies ReMID. FiLM, augmentation consistency, illuminant equivariance
and tangent invariance all have prior art. Any new coupling is a hypothesis until
literature comparison and controlled improvements establish its contribution.

## Data acquisition and split lock

- Inspect and acquire ReMID from the download linked by the primary LumiDepth
  paper. Confirm original group IDs, paired lights, RGB/depth registration,
  reference encoding, metric units and provided folds before any model scoring.
- Acquire PCCC illumination annotations and accessible assets where useful;
  reannotated sRGB illumination is distinct from color-checker-calibrated RAW.
- Acquire a new NYU subset: 200 training, 40 validation and 128 evaluation scenes,
  one original frame per scene, seed 20261107. Exclude all 40 scenes inspected in
  the original pilot. Preserve official train/test membership. These are real
  RGB-D captures, but digital color perturbations remain simulated illumination.
- Vari remains the acquired rendered multi-light dataset. It has only two rooms;
  do not use additional views as evidence of scene diversity.

Write hashed manifests and split IDs before training. Select hyperparameters and
checkpoints using training/validation only. Reserve new held-out scenes for a
single frozen-method evaluation. If a source's physical depth convention cannot
be verified, label relative/invariance scores as such; do not invent metric units.

## Required controls and reporting

Keep the frozen original, ordinary task adaptation, matched augmentation plus
consistency, achromatic conditioning, chromatic conditioning and proposed-loss
ablation. A gain over the frozen model alone is insufficient. Apply the same
augmentation schedule and budget to matched learned controls.

Separate original real captures, physically captured changed-light pairs and
simulated diagonal/scalar perturbations in every result table. No fitted
prediction scale or shift against test depth. Report common masks, per-image
depth accuracy, lighting instability and scene-level paired outcomes. Account
for correlated light variants by aggregating or resampling whole scenes.

Record failed hypotheses, harms and acquisition failures. No test-driven model
selection or claim of guaranteed improvement. Method equations, numerical
configuration and the precise split must be frozen before the final run.
