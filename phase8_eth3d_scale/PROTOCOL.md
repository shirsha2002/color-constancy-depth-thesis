# ETH3D scene-scale diagnostic

Frozen before any new training or accuracy evaluation, 16 November 2026.
Status: local input preparation only. No cloud execution or new spending is
authorized by this protocol.

## Question and data

Does removing per-capture absolute coordinate scale change the geometry benefit
observed in the phase-six ETH3D illuminant-estimation experiment?

Use all 195 labeled released ETH3D captures, 256 XYZRGB points each, and the
unchanged official three folds (130 training / 65 held-out per fold). These
benchmarks have been inspected previously: this is development evidence, not
independent confirmation. Official fold membership does not establish physical
scene independence. Retain any duplicate-cloud findings in the input audit.

## Frozen conditions

- Native: reuse phase-six ETH3D_real inputs and released-head models.
- Unit depth: divide valid XYZ by that capture's median absolute valid Z.
  Valid means abs(Z) > 1e-8. Keep native axes, RGB, point ordering and all
  subthreshold XYZ unchanged, including zeros and tiny nonzero residuals.
  Input inspection found such residuals, so do not assume every subthreshold
  point has exact zero XYZ or silently repair the release. Use input geometry
  only, never labels or errors.
- Flat: reuse phase-six ETH3D_flat inputs and released-head models. Divide
  each valid XYZ by its own abs(Z). This preserves rays and missingness and
  removes relative depth. It is not a pure RGB-only model.

The unit-depth transformation retains relative depth and discards absolute
scene scale. Do not add an axes rotation or search alternate transforms after
seeing outcomes. Do not infer sensor units or calibration from this operation.
Derived arrays remain float64 on disk and use the frozen trainer's existing
float32 conversion. Check preservation before and after that conversion.

## Planned training and selection (not launched)

Three new unit-depth released-sum-head models, one per official fold, at
10,000 epochs. Reuse the pinned phase-five numerical trainer and phase-six
native/flat baselines. Same seed 20261109 + fold, Adam LR 0.0003, batch 16,
scalar augmentation SD 0.20, float32, and deterministic backend policy.
The unit-depth manifest must be passed with geometry mode real so it is not
flattened again. Match the prior backend/software policy in an eventual runner;
record any runtime-only accommodation and its source hash explicitly.

Primary checkpoint: fixed final epoch 10,000. If an author-style best-test
checkpoint is retained, report it separately as optimistic, never primary.
No additional head, gate, seeds, loss, or hyperparameter tuning in this study.
Before executing, check all six reused final baseline checkpoint/config hashes,
prepare a bounded runner, training-only timing estimate and a new cost cap.
Do not restart the canceled laptop study or assume the completed phase-seven
$8 allowance applies to this new study.

## Endpoints and reporting

Illuminant recovery angular error in degrees, preserving cosine clamp 0.999999.
Pool each capture's out-of-fold prediction once. Report mean, median, trimean,
best-quarter and worst-quarter errors for all three conditions, with per-fold
summaries and all saved predictions.

Two primary contrasts: unit-depth minus native; unit-depth minus flat.
Negative differences favor unit depth. Use 10,000 paired capture bootstrap
replicates with NumPy default_rng(20261117), sequential contrasts in that order,
and percentile 2.5/97.5 intervals. These two intervals are not multiplicity
adjusted; they are conditional on the trained models/captures, not room-level
or repeated-seed uncertainty. Report both regardless of outcome. Do not make a
broad efficacy claim from these intervals. Independently recompute saved angles
and verify source/config/input/checkpoint identities and final-model inference.

This diagnostic does not evaluate RGB-predicted depth, monocular depth accuracy,
reliability-aware fusion or novelty. Full original RGBs and verified image/point
correspondence remain prerequisites for the separate predicted-depth experiment.

