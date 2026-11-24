# Decision after the camera-frame and scale experiment

16 November 2026. **Design/proposal only: no new experiment or result is reported here.**

## What the completed evidence permits

The [phase-seven results](RESULTS.md) and [independent audit](results/nyu_frame_v1/audit.json)
complete the NYU/DIODE representation diagnostic. Neither primary comparison
establishes a benefit from per-capture median-depth normalization. Keep native,
axes-only, normalized and matched-flat outcomes visible; do not tune another
transform on these same test captures or make normalization mandatory in a new
method. The [protocol](PROTOCOL.md) already identifies these folds as development
evidence, with one seed per fold.

The earlier [phase-six plan](../phase6_modal/report/report_text.txt), Section 7,
also proposed ETH3D scene-scale and predicted-depth diagnostics before broader
fusion experiments. Those remain proposals here. The completed normalization
test does not show that unreliable depth caused the earlier dataset differences.

## Advisable next step: prepare the ETH3D comparison locally

The completed [local readiness check](ETH3D_FEASIBILITY.md) verifies the sampled
clouds, labels and frozen depth checkpoint. Original RGBs and verified sampling
correspondence are missing. Resolve those prerequisites before generating
predicted-depth inputs; scene-scale input preparation is already feasible.

Write and freeze a separate protocol and input manifest before any training.
ETH3D is useful because [phase six](../phase6_modal/README.md) observed a geometry
gain there. Address two questions separately:

| Proposed diagnostic | Comparison and purpose |
|---|---|
| Scene scale | Measured native XYZ versus per-capture median-scaled XYZ, with a matched ray/flat control. Test whether discarding absolute scale changes the observed geometry gain. |
| Predicted depth | Measured-depth, frozen RGB-predicted-depth and ray/flat inputs with identical RGB, labels, sample identities and evaluation rules. Test whether the illuminant estimator can benefit from depth obtainable from RGB. |

First verify access to each corresponding full RGB image, the released point
sampling locations, camera information and valid-depth masks. Audit image/point
correspondence and color processing before replacing XYZ. Record the frozen
depth model/checkpoint, resize/crop, camera-Z versus ray-distance convention,
units and treatment of invalid points. If these cannot be established, document
the missing prerequisite rather than fabricate aligned geometry. Report any
scale alignment explicitly and fit no test-time scale against measured depth.

Use a small local input/inference check to establish feasibility and estimate
runtime; this note does not authorize model training or new cloud expenditure.
RGB-predicted depth connects the pipeline to monocular inference, but its endpoint
remains **illuminant estimation**. It does not demonstrate improved monocular
depth accuracy. Confirm formal thesis scope before final submission.

## Proposed fusion study, after the diagnostic is specified

A color branch, geometry branch and input-derived reliability gate are a
candidate implementation, not an established contribution. Before training,
define matched controls: original PCCC; ray/flat input; ordinary two-branch
fusion; fusion with an explicit validity mask; learned gating; and gating
without depth-damage training. Treat normalization as a separate ablation.
Match training examples, corruption schedules, optimizer, update budget and
capacity as closely as possible; report remaining parameter/compute differences.

The gate may use available input geometry, missingness and neighborhood cues;
it must not receive test errors, illumination labels or hidden reference depth.
Evaluate clean inputs as well as fixed depth-damage conditions. A gate value is
not calibrated uncertainty by definition, and fallback behavior needs measured
evidence. Review relevant fusion/CC work before any novelty claim. The earlier
[method-design review](../phase3/research/METHOD_DESIGN_REVIEW.md) supplies useful
principles for matched controls, inference inputs and scene separation, although
its architecture was for the earlier depth-estimation task.

## Data and decision rules before training

- Freeze scene/capture identities, duplicate checks, training/validation/test
  membership, transformations, checkpoint rules and primary contrasts. Keep
  related views/light captures together where identities permit; disclose any
  limits in the released split metadata.
- Tune gates, loss weights and corruption strengths using training/validation
  data only. Preserve official folds for historical comparison, but label
  already inspected benchmark results as development evidence. Independent
  confirmation requires newly held-out scenes rather than renamed old tests.
- Use fixed-final or validation-selected checkpoints as primary; keep any
  test-selected reference separate. Report all declared comparisons, harms,
  paired uncertainty and subgroup results. Repeated seeds are needed to assess
  training variability; budget them before proposing a final claim.
- Prepare a concrete model count, timing estimate and cost bound before any
  paid execution. The completed phase-seven $8 allowance is not permission for
  another study. Keep the canceled phase-five workers stopped.

Multiple-illuminant estimation is an **optional later extension** requiring
suitable spatial illumination labels, a different output/evaluation protocol
and its own data review. It is not necessary to complete the focused diagnostic
or justify the first fusion comparison.
