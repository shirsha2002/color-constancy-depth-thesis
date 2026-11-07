# Numerical configuration locked before new final evaluation

7 November 2026. Implementation: `learning/model.py` and `learning/study.py`.
The runner records SHA-256 values for both, preprocessing, the upstream depth
transform, all data manifests and the base checkpoint before numerical work.

## Data roles

- NYU: 200 training, 40 validation and 128 evaluation scenes; one frame per
  original scene. Official train/test membership is retained. All 40 scenes
  inspected in the first pilot are excluded. Split IDs were fixed before
  training in `data/nyu_rgbd/selection.json`.
- ReMID: all 735 real lighting images, 50 capture groups and 25 conservative
  category/scene groups are **external evaluation only**. No model or checkpoint
  is selected using these data. Capture IDs include transport shard because
  numeric IDs restart. Physical room identities are unknown.
- Vari: the previously acquired 264 evaluation lighting images from one rendered
  room provide an additional simulated-scene diagnostic. This is previously
  inspected public exploratory data, not a new independent confirmation set.

Measured ReMID depth is explicitly millimeters in the native metadata and is
divided by 1,000. References remain unmodified. All scores use finite depth
strictly between 0.1 and 10 meters and clip predictions to [0.1,10] meters, with
no fitted scale or shift. NYU uses the declared scaled Eigen crop; ReMID and
Vari use the full valid native frame. Original NYU references are the dataset
authors' Kinect-derived, filled depth rather than dense direct measurements.

## Inputs and interventions

The official frozen Depth Anything V2 Metric Hypersim Small encoder and decoder
use input size 224 with preserved aspect ratio and eight CPU threads. Four
384-channel encoder grids and the original model's predicted depth are cached.
An RGB-only inference first pass supplies predicted pseudo geometry; measured
depth, measured normals and test illumination labels never enter the forward
path. Virtual normalized rays are not calibrated world coordinates.

Training pairs contain the original RGB and a deterministic image-specific
chromatic or scalar intervention. Each 200-scene epoch visits all training
scenes once. The same seed, pair schedule and budgets apply to every variant.
Validation averages original, chromatic and exposure conditions in the forty
validation scenes.

The 128-scene synthetic diagnostic has five inputs per scene: unmodified RGB,
a matched-headroom neutral control, a red/blue chromatic change, a green-axis
chromatic change and a scalar exposure change. Chromatic gains are normalized
to preserve source mean linear luminance. A single source-only headroom factor
prevents clipping across all four controlled arms. The native image is scored
separately because this headroom can dim an image. This tests digital channel
changes on real content, not physically captured relighting. Captured ReMID
lighting and rendered Vari lighting are reported separately.

## Architecture and comparisons

The shared class has 59,699 parameters: a constant task adapter, semantic and
photometric context, an RGB/predicted-geometry point-set encoder and per-layer
384-8-384 residuals with bounded conditioning. Every residual output projection
starts at zero. The original depth encoder and decoder remain frozen.

| Variant | Internal conditioning | Relative-chroma loss |
|---|---|---:|
| frozen | Original model | none |
| ordinary_aug | Constant task adapter; matched augmentation/consistency | 0 |
| achromatic | Brightness cue plus conditional residual | 0 |
| content | Semantic cue plus conditional residual | 0 |
| chromatic_noeq | Channel-ratio/context cue | 0 |
| chromatic_eq | Channel-ratio/context cue | 0.2 |
| flat_geometry_eq | Same geometry-branch architecture with constant predicted depth | 0.2 |
| geometry_chromatic_eq | Geometry branch uses detached frozen predicted depth | 0.2 |

All learned variants instantiate the same class, but some control modes disable
branches, so equal total parameter count does not mean equal active capacity.
The full versus flat-geometry comparison has the same active architecture and
losses. Content/achromatic controls and the no-equivariance ablation help
separate cue effects, capacity and training-objective effects.

## Optimization and objectives

600 paired updates (three 200-scene epochs), batch size two, AdamW learning rate
0.001, weight decay 0.0001, gradient-norm cap 1, seed 20261107. Validate every
200 updates; select the lowest mean validation AbsRel. No final test data enter
this decision.

The common depth objective is supervised AbsRel plus 0.1 paired absolute
log-depth disagreement plus 0.001 modulation regularization. The explicit
chromatic term is SmoothL1 between the estimated log-ratio change and the known
intervention log(R/G), log(B/G). Its weight is 0.2 for the declared variants.
Absolute physical illuminants are not supervised in this run. The branch must
therefore be described as a relative chromatic cue, not calibrated illuminant
recovery. Acquired PCCC annotations/point clouds are literature/baseline assets;
they are not silently substituted for dense metric depth or used as extra
training supervision.

## Acceptance and interpretation

Report all variants, including harms. Gains over frozen alone are insufficient.
Compare full conditioning with the ordinary augmented, content, achromatic,
no-equivariance and flat-geometry controls. Report per-scene/cluster outcomes,
original-image accuracy and physical paired-light stability together. A small
gain, failure or tradeoff must not be promoted to broad CC-specific success.

PCCC, FiLM, low-rank adaptation, illuminant equivariance, Tangent Prop, md4all and
LumiDepth establish close prior art. This experiment tests a concrete coupled
prototype; it does not by itself certify a novel paper contribution. Publication
readiness additionally requires competitive baselines, independent repetitions,
adequate compute and a defensible method distinction.
