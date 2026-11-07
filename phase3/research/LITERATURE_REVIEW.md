# Research directions after the supervisor's November feedback

Checked 7 November 2026. This document records a literature and source-code
review, not new experimental results. Dataset acquisition status belongs in
the separate data manifest; a download link alone is not an acquired dataset.

## The paper the supervisor meant

**Xiaoyan Xing, Yanlin Qian, Sibo Feng, Yuhan Dong and Jiri Matas,
"Point Cloud Color Constancy," CVPR 2022, pp. 19750-19759 (PCCC).**

- [CVF publication record](https://openaccess.thecvf.com/content/CVPR2022/html/Xing_Point_Cloud_Color_Constancy_CVPR_2022_paper.html)
- [Author manuscript with appendix](https://arxiv.org/html/2111.11280)
- [Official code](https://github.com/xyxingx/Point-Cloud-Color-Constancy)
- [Dataset share linked by the authors](https://drive.google.com/drive/folders/1qBw_bvaxIvduIm2vzrYhEPX9khTm1Bo9?usp=sharing)

PCCC estimates illumination from measured RGB-D point clouds. Its input contains
XYZ positions and RGB intensities; a PointNet-style model aggregates local
illumination estimates. The target is an illuminant vector, assessed by angular
error. This direction is **geometry -> color constancy**. It motivates a link
between geometry and color, but does not demonstrate the reverse proposition
needed by this thesis: **color-constancy reasoning -> more accurate monocular
depth**. Sensor depth at inference would change the thesis's monocular task.

The paper's benchmark sizes are 324 combined NYU-v2/DIODE examples, 195 ETH3D
examples and 185 DepthAWB captures. These are useful real-data complements, not
large-scale training sets by current depth-model standards. DepthAWB pairs a
Panasonic GH5S camera with Intel L515 depth. The authors distinguish processed
sRGB-derived labels from RAW-domain color-checker labels; these annotation
qualities should not be treated as interchangeable.

## Data and color-space caveats

The [official release README](https://github.com/xyxingx/Point-Cloud-Color-Constancy/blob/ddc8770e04ef5509a621c5a7ee04e330b9204a93/README.md)
lists illumination labels, 256/4096-point samples, images and cross-validation
folds. It identifies NYU-v2/DIODE images as linearized sRGB, and ETH3D/DepthAWB
images as RAW-linear RGB. Native full-resolution RGB-D data for the public source
datasets must be obtained from their original sites; the point samples alone do
not establish a dense depth benchmark.

PCCC's [appendix and dataset discussion](https://arxiv.org/html/2111.11280)
acknowledge that inverse gamma and approximate color-matrix reversal cannot
recover true sensor RAW. Neutral-surface annotation also has selection error.
Consequently, angular scores on these relabelled images describe their released
color space, not a calibrated physical illuminant in every scene. For DepthAWB,
the checker-containing exposure supplies labels and the checker-free exposure
is used for learning/evaluation. An input checker must not leak its answer.

Any depth test must separately establish camera alignment, invalid masks, scale
and units. Do not mistake a released Z coordinate at sparse sample locations for
a verified dense camera-depth map. Related captures and source-scene identities
must stay in one partition.

## PCCC reproducibility review

Official code inspected at commit
`ddc8770e04ef5509a621c5a7ee04e330b9204a93`. The following are concrete code findings,
not claims that the published algorithm is invalid:

| Source | Finding and consequence |
|---|---|
| [train_main.py](https://github.com/xyxingx/Point-Cloud-Color-Constancy/blob/ddc8770e04ef5509a621c5a7ee04e330b9204a93/train_main.py) | `--datasets` is parsed but not forwarded to either `PcdColor` constructor, so the loader defaults to ETH3D. GPU calls are unconditional. The actual fold flag is `--foldnum`, unlike the README example. |
| [pointnet/model.py](https://github.com/xyxingx/Point-Cloud-Color-Constancy/blob/ddc8770e04ef5509a621c5a7ee04e330b9204a93/pointnet/model.py) | `PointNetfeat.forward` returns three objects, whereas `PCCC.forward` unpacks two. The final convolution returns three channels and the training code sums point predictions; the paper's explicit fourth weighting channel/softmax is not present in this path. |
| [pointnet/dataloader.py](https://github.com/xyxingx/Point-Cloud-Color-Constancy/blob/ddc8770e04ef5509a621c5a7ee04e330b9204a93/pointnet/dataloader.py) | Absolute author-specific data paths need replacement. Fold membership controls train versus held-out examples; the Git tree contains no fold CSVs, so check the data release. CPU-compatible tensors need explicit handling. |
| Training selection | The released training loop repeatedly scores the held-out fold and saves the epoch with its best score. For a new scientific comparison, reserve an inner development split for checkpoint selection and keep the final test untouched. |

Record every adaptation and numerical check. A patched implementation is an
adapted baseline until matching preprocessing, folds, model and published
evaluation have been verified; merely running the repository is not a paper
reproduction. No PCCC accuracy or speed has been reproduced in this review.

## Close prior art that affects the proposed innovation

**LumiDepth, ECCV 2026.** The [publisher's paper](https://link.springer.com/chapter/10.1007/978-3-032-37314-4_9)
by Anqi Cheng et al. was first published online on 16 September 2026. It already
learns stable depth from multi-illumination image groups using disagreement-based
pseudo-supervision and frequency-aware consistency/distillation. It introduces
**ReMID**, a real multi-illumination RGB-D benchmark. This is especially close to
the present thesis and must be compared before claiming innovation in lighting
consistency or uncertainty filtering.

The publisher's Notes, footnote 1, directly links the
[ReMID dataset folder](https://drive.google.com/drive/folders/1-3txE3KiFiscACxs1Xch8UO1Ax82VBNg?usp=sharing).
Thus the source relationship is verified; asset integrity, licenses, camera
registration, depth units and split independence still require inspection. The
paper also notes normalized depth notation; do not infer metric PNG units or
the official alignment protocol from its abstract. The chapter preview was
read; the subscription-only full chapter was not obtained in this review.

Additional primary sources constrain what can count as a new contribution:

| Work | Relevance |
|---|---|
| [Integral Fast Fourier Color Constancy, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Wei_Integral_Fast_Fourier_Color_Constancy_CVPR_2025_paper.html), Wenjun Wei, Yanlin Qian et al. | Qian's newer multi-illuminant work; local illumination estimation is established. It is an AWB method, not a depth architecture. |
| [Uncertainty estimation in color constancy, Pattern Recognition 2025](https://doi.org/10.1016/j.patcog.2024.111175), Marco Buzzelli and Simone Bianco | CC uncertainty and uncertainty-based method cascading already exist. A confidence gate alone is not sufficient novelty. |
| [Structure-Centric Robust Monocular Depth Estimation via Knowledge Distillation, ACCV 2024](https://arxiv.org/abs/2410.06982), Runze Chen et al. | Uses illumination and semantic structure for robust depth, including illumination/depth constraints. Separating structure from texture is not a new principle. |
| [Uncertainty and Self-Supervision in Single-View Depth, 2024 dissertation](https://arxiv.org/abs/2406.14226), Javier Rodriguez-Puigvert | Prior uncertainty-aware depth and illumination-derived supervision; its endoscopic lighting assumptions do not automatically transfer to room scenes. |
| [FiLM, AAAI 2018](https://arxiv.org/abs/1709.07871), Ethan Perez et al. | Feature-wise affine conditioning is established; the phase-two adapter is an implementation choice, not an architecture invention by itself. |
| [Self-Supervised Learning of Color Constancy, ICDL 2024](https://arxiv.org/abs/2404.08127) | Invariance learning across illuminations is existing CC research. Consistency supervision alone cannot be claimed as new. |

This targeted search is not exhaustive and does not establish patentability or
publication novelty. No exact equivalence to every possible proposed system was
established either. A specific architecture/loss needs a fresh comparison once
its equations and inputs are fixed.

## An experiment that would answer the supervisor's causal question

The following are study-design recommendations, not completed results:

1. Test controlled chromatic gains at fixed luminance separately from scalar
   exposure changes, retaining identity inputs. Digital gain tests are simulated
   corruptions even when the starting photograph is real.
2. On real RGB-D scenes with illuminant labels, compare predicted illumination,
   reference-illuminant diagnostics, achromatic compensation and incorrect or
   shuffled illumination controls. Reference labels used at inference form an
   explicitly labelled oracle diagnostic, not a deployable method.
3. Train an RGB-only depth system whose illumination head influences internal
   geometry features. Contrast it with equal-capacity ordinary adaptation and an
   achromatic head. Sensor/reference depth is supervision or evaluation only.
4. Separate an illumination-estimation loss from depth supervision and a
   cross-light geometric constraint. Compare the full method with each loss
   removed, shuffled CC supervision and ordinary augmentation. This tests
   whether gains come from meaningful chromatic reasoning rather than capacity.
5. Report depth accuracy and lighting stability together, grouped by independent
   scene, with identical masks and declared scaling. Select settings only on
   development scenes. Correlation of angular and depth errors is descriptive;
   controlled interventions are needed to support a mechanism.

The promising research hypothesis is a demonstrably useful coupling between an
illumination estimate and monocular geometry. A positive result remains to be
measured against the controls above, and a publication claim additionally needs
broader real-data validation and comparison with close prior art.

## Dataset choice by scientific role

- **ReMID:** first inspect for real paired-light RGB-D geometry validation;
  ownership link is verified above. Its final usable counts and units must come
  from the acquisition audit.
- **PCCC releases:** useful real illumination supervision and RGB-D baselines;
  relatively small and heterogeneous in color-space/annotation quality.
- **Vari:** keep as the simulated paired-light arm, with the existing explicit
  two-room limitation. More views do not create independent rooms.
- [**MIT Multi-Illumination**](https://projects.csail.mit.edu/illumination/): over
  1,000 real scenes with 25 lighting directions; useful for robustness or
  consistency, but its project release does not establish reference depth for a
  depth-accuracy claim.
- The supervisor's [classic CC index](http://colorconstancy.com/evaluation/datasets/index.html#colorcheckerhemrit)
  points toward illumination-estimation benchmarks; their role is auxiliary CC
  validation unless registered independent geometry is actually available. The
  index URL itself did not load through the web tool during this review.
