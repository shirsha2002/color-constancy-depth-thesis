# Sources and attribution

## Depth model

Lihe Yang et al. **Depth Anything V2** (2024).

- Official code: https://github.com/DepthAnything/Depth-Anything-V2
- Metric implementation: https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth
- Paper: https://arxiv.org/abs/2406.09414
- Upstream revision: a561b849ebae10a6f5ef49e26c83cbbcd36c71bf
- Checkpoint: https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small
- Checkpoint SHA-256: b782898d8a3e8be1f639de33837ed85e9b4b73e40f8f5e5cd99067588d722545

The upstream code remains under its original license. Downloaded code and weights
are excluded from this repository and can be obtained with the setup scripts.

## Reference RGB and depth

Silberman et al. **Indoor Segmentation and Support Inference from RGBD Images**,
ECCV 2012. Original NYU Depth V2 data and official train/test split.

- Dataset: https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html
- Per-sample provenance and preparation: [data/DATA_SOURCES.md](data/DATA_SOURCES.md)

Photographic panels in the research report and presentation are attributed NYU
research examples. The original dataset files and model weights are not included
in GitHub. Do not infer a dataset redistribution license from this project's MIT
license.

## Evaluation crop

The crop is the commonly used NYU Eigen crop (zero-based rows45:471,
columns41:601 at480x640). Our exact valid depth interval is0.1<depth<10meters,
and predictions are clipped to[0.1,10]meters with no per-image scale fitting.
This is a declared pilot protocol, not a claim of full benchmark replication.

Primary implementation references:

- https://github.com/isl-org/ZoeDepth/blob/main/zoedepth/utils/misc.py
- https://github.com/prs-eth/Marigold/blob/main/src/dataset/nyu_dataset.py

## Color correction

Gray-World uses the per-channel mean in linear RGB. White-Patch uses per-channel
maxima. Half-strength Gray-World raises its gains to power0.5. All gains preserve
the input mean luminance before clipping. These are implemented baseline methods,
not new algorithms. Synthetic perturbations are documented in scripts/science.py.

Color-constancy background: Finlayson and Trezzi, *Shades of Gray and Colour
Constancy*, CIC2004.
https://library.imaging.org/admin/apis/public/api/ist/website/downloadArticle/cic/12/1/art00008

## Follow-up paired-light data and geometry

- Vari/DCA author repository and public dataset links: https://github.com/zzt76/dca
- Associated paper: https://arxiv.org/abs/2112.08006
- Dataset acquisition, exact source revision, custom two-room split and explicit
  depth-decoder refinement: [Vari provenance](phase2/vari_research/README.md).
- DTU original data, cameras and structured-light reference surfaces:
  https://roboimagedata.compute.dtu.dk/?page_id=36
- DTU citation: Jensen et al., *Large Scale Multi-view Stereopsis Evaluation*,
  CVPR 2014; Aanæs et al., *Large-Scale Data for Multiple-View Stereopsis*, IJCV 2016.
- Native units, calibrated camera conventions, official/mirror member integrity
  checks and dataset use conditions: [DTU provenance](phase2/data/dtu/DATA_SOURCES.md).

Report figures include attributed rendered Vari examples and real DTU examples.
The small DTU point-cloud previews are derived algorithm outputs in millimeters,
not original reference clouds. Original raw images and reference surfaces are
not redistributed in this repository. Project licensing does not supersede
the authors' model or dataset rights.

## Integrated conditioning and radiometric controls

Feature-wise modulation follows established ideas such as
*FiLM: Visual Reasoning with a General Conditioning Layer*:
https://arxiv.org/abs/1709.07871

The internal chromatic adapter is an engineering prototype using a global
relative-channel cue, not verified physical illuminant recovery. Scalar gain,
diagonal gain and normalized-correlation stereo costs are known controls rather
than claimed novel algorithms. Equations and implementation references appear
in [the stereo design audit](phase2/research/stereo_design_audit.md) and the
[frozen research protocol](phase2/PROTOCOL.md).
