# Phase two protocol recorded before measuring results

Date: 28 October 2026. Student: Shirsha Mitra, 2023521460130.

## Research sequence

The supervisor requested supporting data, experimental evidence of the role of
color constancy in geometry, and a task-integrated method. Preserve the first NYU
study unchanged as a preprocessing baseline. The main task remains monocular
depth estimation. A calibrated DTU stereo diagnostic provides complementary
evidence about real changes in lighting and geometry estimation.

## Data and claim boundaries

- Vari: physically rendered indoor RGB and depth with eleven lighting conditions
  per fixed camera. The available public archive is the authors' TEST release;
  the official training archive remains unreleased. Its two rooms cannot support
  a broad generalization claim. We create a new, explicitly exploratory split:
  room212 for development, room485 for unseen-room evaluation. This is not an
  official Vari benchmark score.
- Requested selection, seed20261028: twenty training camera groups and four
  validation camera groups in room212, twenty-four evaluation groups in room485,
  all eleven lighting conditions. The final source manifest must document actual
  availability before training begins. No camera group crosses these splits.
- Decode reference depth exactly as the authors' loader specifies and record
  source dtype, quantization, units, ranges and within-group equality. Never use
  a depth model's prediction as reference depth.
- DTU: scenes1 and6 from the original SampleSet, calibrated selected views and
  seven captured lighting setups with structured-light reference surfaces. Its
  lighting changes primarily vary direction/shading. This alone does not establish
  changes in illuminant chromaticity or accurate physical illuminant recovery.

## Monocular baseline and integrated candidate

Use the existing official Depth Anything V2 Metric Hypersim Small checkpoint,
frozen encoder and frozen depth decoder. Use input-size224 and eight CPU threads
for this laptop pilot, preserving aspect ratio and the official model transform.
Cache encoder feature tensors in float32. All feature-based variants use the same
cache and depth decoder. Scores are not directly comparable with the earlier NYU
experiment at input-size518.

Insert a small, identity-initialized feature modulation module BETWEEN the DINOv2
encoder and depth decoder. A small MLP uses an image's relative log-channel means
as an illumination-color cue and predicts bounded feature gains and shifts.
Depth supervision and paired-lighting feature/depth consistency jointly train
this internal module. No corrected RGB image is produced by this method.

This is a CC-inspired engineering prototype using known feature modulation, not
a demonstrated novel color-constancy algorithm. Without reference illuminants,
we measure geometry accuracy and lighting stability, not illuminant angular error.

Fixed comparisons:

1. Frozen original model.
2. Gray-World image preprocessing plus frozen model (explicit old-style baseline).
3. Condition-independent task adapter, depth supervision only.
4. Achromatic illumination-conditioned adapter with consistency losses.
5. Chromatic CC-inspired adapter, depth supervision only.
6. Chromatic CC-inspired adapter with depth/feature consistency.

The achromatic and chromatic variants share module sizes, training pairs,
optimizer settings and losses where applicable. The chromatic supervised-only
variant ablates consistency. Ordinary task adaptation controls for domain fitting.

## Training and evaluation

- Fixed seed20261028, AdamW, learning rate0.001, weight decay0.0001.
- Initial budget150 updates per learned variant, one paired-lighting camera per
  update (two images), gradient norm cap1.0. Validation every50updates; select the
  checkpoint by mean validation AbsRel using only room212 development views.
- Loss: supervised AbsRel plus0.1 paired log-depth consistency plus0.05 normalized
  feature consistency for consistency variants, with0.001 modulation penalty.
- Bounded modulation prevents a consistency-only feature collapse. Model weights
  outside the small adapter stay fixed. Use train-only descriptor statistics.
- Identical reference masks for all methods: finite depth strictly between0.1
  and9.99m, full rendered frame; predictions clipped to[0.1,10]m. No fitted depth
  scale or shift. Record mean per-image AbsRel, RMSE(m), delta1 and per-light scores.
- Lighting instability: for each fixed-camera group, average absolute deviation
  of predictions from the eleven-light mean, divided by its fixed reference depth.
  Low instability by itself does not establish accurate geometry.
- Report the number of rooms, camera groups and images separately. Multiple views
  of one room are not independent scenes. No population-generalization confidence
  claim from a single held-out room. Disclose every protocol change and all harms.

## Sources

- Vari/DCA authors: https://github.com/zzt76/dca
- Vari paper: https://arxiv.org/abs/2112.08006
- DTU original data: https://roboimagedata.compute.dtu.dk/?page_id=36
- Frozen model: https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth
- Feature-wise linear modulation background: https://arxiv.org/abs/1709.07871

GitHub contains source, manifests, configurations, small learned adapters and
measured results. Source images, large archives, feature/prediction caches,
environments and pretrained base-model weights remain local.

## Decoder refinement before training or evaluation

Archive inspection found the depth TIFFs contain sixteen-bit identical RGB
channels. The authors' PIL RGB-to-L loading path reduces these values to eight
bits before applying inverse normalization and the10m scale. Preserve original
TIFFs. The PRIMARY exploratory reference extends the same normalization at the
original precision: depth=(1-uint16/65535)*10m. Also export the authors-compatible
eight-bit reference and measure their difference. This refinement is explicit;
results are not exact reproductions of the official decoder/benchmark protocol.
No depth-model errors were consulted in making this precision-preservation choice.
