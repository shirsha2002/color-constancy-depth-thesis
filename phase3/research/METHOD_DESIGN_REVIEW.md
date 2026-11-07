# Method review for the next integrated CC-depth experiment

Reviewed 7 November 2026. This document proposes experiments; it does not report
new measurements or establish algorithm novelty.

## What must change from phase two

The previous module predicts all feature gains/shifts from three global RGB
means. Those means mix surface reflectance, illuminant color and camera ISP.
There is no supervised illumination estimate, so a depth improvement would not
by itself establish color constancy. In fact, the ordinary task adapter has
AbsRel 0.398801 versus 0.427686 for the chromatic consistency variant, with lower
lighting instability as well. Replacing ordinary adaptation by the chromatic
module therefore removes a useful domain-adaptation control.

Keep a condition-independent task adapter in every learned method. Add a small
chromatic residual to that shared design and ask whether it adds value over a
capacity- and training-matched robustness baseline. Do not select new settings
using the already inspected Vari evaluation room or original NYU evaluation
subset and then describe them as untouched confirmation data.

## Closest primary-source work and novelty boundary

| Work | Relevant established idea | Implication for this project |
|---|---|---|
| [PCCC, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Xing_Point_Cloud_Color_Constancy_CVPR_2022_paper.html) | RGB plus measured depth form colored point clouds for illumination estimation; the paper adds illumination annotations to RGB-D data. | Its direction is depth/geometry to CC. A monocular depth system must not receive measured test depth merely to run PCCC. Test the reverse CC-to-depth contribution explicitly. |
| [FiLM](https://arxiv.org/abs/1709.07871) | Feature-wise affine conditioning. | Calling a FiLM layer an illumination module is not architectural novelty. |
| [md4all, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Gasperini_Robust_Monocular_Depth_Estimation_under_Challenging_Conditions_ICCV_2023_paper.html) | Challenging appearance inputs train against valid supervision from original corresponding scenes. | Ordinary adaptation with the same appearance augmentation and supervision is essential. |
| [Saunders et al., ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Saunders_Self-supervised_Monocular_Depth_Estimation_Lets_Talk_About_The_Weather_ICCV_2023_paper.html) | Augmented/unaugmented images provide pseudo-supervised depth consistency. | Paired depth consistency alone is not a new loss. |
| [LumiDepth, ECCV 2026](https://link.springer.com/chapter/10.1007/978-3-032-37314-4_9) | Multi-illumination RGB groups, disagreement-calibrated pseudo supervision, frequency-aware consistency/distillation and the real ReMID RGB-D benchmark. | This is particularly close. A general multi-light invariance claim or uncertainty-weighted consistency is insufficient differentiation. ReMID is a relevant real-light benchmark candidate. |
| [Offset equivariant networks](https://arxiv.org/abs/2207.00292) | Photometric/illuminant equivariance through network design. | Known gain-equivariance does not become a new concept by adding a depth head. |
| [Tangent Prop](https://proceedings.neurips.cc/paper/1991/hash/65658fde58ab3c2b6e5132a39fae7cb9-Abstract.html) | Regularizing derivatives along specified nuisance transformations. | A finite-difference illumination-Jacobian penalty has established ancestry. |

The useful research hypothesis is narrower: can an explicitly estimated
chromatic nuisance, a task-internal low-rank correction and a physically
specified chromatic intervention loss improve monocular depth more than generic
augmentation/consistency, while preserving geometry and clean-image accuracy?
The combination needs experiments and a more exhaustive prior-art review before
claiming a paper contribution. This search is not proof of novelty.

## Recommended implementation that fits cached CPU features

Use the same frozen DINOv2 encoder and DPT decoder. Cache the four 384-channel
token grids for each actual RGB input and each fixed training intervention. The
core input at inference remains **one RGB image**.

1. **Shared ordinary adaptation.** Per-layer learned constant gains/shifts give
   every method the same domain/scale adaptation path: 4 x 2 x 384 parameters.
2. **Illumination head.** Pool cached semantic tokens and combine them with
   photometric descriptors: log-channel ratios, luminance quantiles and saturated
   pixel fractions. Project the context to 16 channels, then use a 16-32-2 MLP
   to estimate log(R/G), log(B/G). A label-free three-value channel mean should
   not be called a measured illuminant. Exclude reference color-chart regions
   when the dataset protocol requires it.
3. **Task-internal low-rank residual.** For each layer, use 384-8-384 projections.
   Modulate the eight bottleneck channels using the estimated two-dimensional
   illuminant. Add the resulting bounded residual to the shared task-adapted
   tokens before the depth decoder. Let a context-based scalar gate reduce
   unreliable correction. This uses roughly 50,000-60,000 parameters depending
   on the context descriptor, rather than a full encoder finetune.
4. **Identity initialization.** Initialize the final residual projection to zero
   and the gate to a nonzero sigmoid value, or initialize a small nonzero
   projection and a zero residual scale. Do not zero both multiplicative paths:
   that can remove all useful gradients. Verify identity output and nonzero
   adapter gradients before the numerical study.

A possible residual is

```text
F_task[l] = F[l] * (1 + gamma[l]) + beta[l] * std(F[l])
q = h(e_hat) - h(0)                         # correction vanishes at neutral cue
R[l] = std(F[l]) * tanh(U[l](D[l](LN(F[l])) * q))
F_out[l] = F_task[l] + 0.1 * gate(context) * R[l]
```

This remains a proposal using known low-rank/conditioning components. The gate
is a learned reliability mechanism unless its uncertainty is calibrated and
evaluated; do not call it calibrated uncertainty by definition.

Do not feed reference depth, reference normals, test illumination labels or
per-image depth-scale fits into this forward path. If geometry is later used
to guide the illumination estimate, derive it from a detached first-pass
monocular prediction and report its cost. Measured depth is permitted as a
training target and as evaluation reference, not as hidden inference input.
Known camera intrinsics are acceptable only if declared for every method; do
not silently switch from RGB-only to RGB-D inference.

## Losses and what each can actually establish

Use common depth supervision, optimizer, pairs and update counts in all methods.
Record initial weights and change them only using development data.

- **Depth:** the same valid-pixel AbsRel/log-depth objective for every variant.
  If adding gradient/edge preservation, add it to the strong control as well.
- **Illumination supervision:** where valid real illuminant labels exist, use
  log-ratio error or normalized-vector angular loss. Reconstruct a positive
  vector as `[exp(e_R), 1, exp(e_B)]` and normalize it. Compute angular metrics
  in float64 with a clamped cosine; distinguish this auxiliary task from depth.
- **Known-gain equivariance:** for an explicitly linear-RGB diagonal transform
  with gain vector `g`, the expected log-ratio shift is
  `[log(g_R/g_G), log(g_B/g_G)]`. Supervise the difference between the two
  illumination predictions. This is an established equivariance principle.
  If the original illuminant is unknown, this supplies only relative, simulated
  supervision; it does not create real absolute illuminant ground truth.
- **Chromatic tangent sensitivity:** apply symmetric small interventions along
  two zero-sum log-gain directions, separately from scalar exposure. Penalize
  the finite difference of predicted log depth divided by intervention size.
  This is a concrete chromatic-direction invariance test with Tangent Prop
  ancestry, not automatically a novel loss. Cache transformed encoder features
  so gradients flow through the trainable residual and decoder operations.
- **Geometry preservation:** retain supervised depth accuracy and the common
  edge objective while reducing chromatic sensitivity. Feature collapse can
  achieve perfect consistency with wrong depth, so consistency is never the
  sole acceptance metric.

Optional deeper hypothesis: estimate a two-direction feature tangent from
known training interventions and constrain the chromatic residual to that
nuisance subspace. Compare it against a learned unconstrained low-rank residual
at equal rank. A globally shared two-dimensional feature subspace is only a
local approximation to a nonlinear encoder; measure its fit on validation
scenes before treating it as a valid physical model.

## Minimum fair comparison matrix

| Method | Training purpose |
|---|---|
| Frozen original | Establish source-model accuracy and sensitivity. |
| Ordinary task adaptation | Control for domain/depth-scale fitting. |
| Ordinary adaptation + identical augmentations + identical depth consistency | Strong generic appearance-robustness control; mandatory. |
| Capacity-matched achromatic branch + same losses where applicable | Test whether brightness/shading information explains the gain. |
| Generic content-conditioned low-rank branch | Test whether any extra input-conditioned capacity explains the gain. |
| Chromatic branch with depth supervision only | Ablate explicit CC supervision/equivariance. |
| Full chromatic branch | Candidate coupling. |
| Full method without chromatic tangent/equivariance term | Isolate the proposed loss contribution. |

An oracle real illuminant or exact inverse digital gain can be an explicitly
labeled diagnostic, not the primary deployable system. Exact digital inversion
recovers the original input by construction and is therefore a pipeline sanity
check, not evidence of learned real-world CC. Test-time cue swapping/zeroing can
check whether the network uses its cue, but it is an out-of-distribution
intervention and does not alone establish useful physical color constancy.

## Data split, radiometry and evaluation safeguards

1. Freeze independent scene IDs before training. Different views, scans, light
   captures and augmentations of one room/object stay in one split. Detect RGB
   duplicates and repeated source scenes across merged datasets.
2. Report three distinct columns: rendered paired-light data; **unmodified real
   RGB-D captures**; and digitally perturbed real photographs. The last column
   is simulated channel change on real content, not real captured relighting.
3. Where real repeated-light captures exist, preserve camera/geometry alignment
   and depth support. One real capture per scene with an illumination label can
   test real-image depth and CC accuracy, but cannot by itself prove invariance
   under real illumination changes.
4. Read the original camera/ISP protocol. Diagonal gain in camera RAW is not
   interchangeable with a linearized processed sRGB image. PCCC-compatible
   labels require inspecting whether RGBs have already undergone AWB and what
   color space the supplied illuminant describes. Do not silently whiten an
   already balanced image using an incompatible label.
5. Keep spatial depth validity masks and native units. Resize sparse references
   without bilinear interpolation through invalid zeros. Validate camera-Z
   versus radial-distance conventions. Report the exact clipped depth range
   and per-image valid support; never choose it by evaluation model errors.
6. For digital interventions, hold global luminance constant in the chromatic
   arm, use a separate scalar exposure arm, and record quantization/clipping.
   Freeze corruption strengths and a saturation acceptance rule on training
   data. Use the same scoring pixels across methods and paired conditions.
7. Select hyperparameters/checkpoints on scene-disjoint validation data, then
   evaluate once on new held-out scenes. Do not keep tuning on that test result.
8. Report per-scene AbsRel/RMSE/delta1, clean accuracy, average and worst-case
   paired-light instability, illumination angular error where labels are valid,
   and latency/memory. Aggregate and bootstrap whole scenes, not correlated
   light variants as independent samples. Use multiple training seeds for the
   final claim; a single-seed laptop run is development evidence.

## Decision rule for an honest progress update

A CC-specific improvement requires beating the strong matched robust control,
not merely the frozen model. Report absolute scores, paired scene-level changes,
harms and uncertainty, and whether removing CC supervision or replacing the
chromatic cue with achromatic/content conditioning removes the advantage.
If only simulated corruptions improve, state that result and leave real-light
transfer open. If the strong control wins, retain the negative result and revise
the cue or coupling on development data. No method should be announced as
publishably novel or state of the art from this review or an initial pilot.
