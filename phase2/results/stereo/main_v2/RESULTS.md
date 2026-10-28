# DTU real-light stereo: actual held-out results

One held-out scene (scene6), three reference views, six mismatched source-light conditions plus the matched diffuse control. All results below are descriptive; there are no population confidence intervals.

|Matching cost|Matched light MAE(mm)|Mismatched lights mean MAE(mm)|
|---|---:|---:|
|RGB SSD|13.189|54.214|
|Scalar gain|9.612|29.896|
|Diagonal gain|12.168|36.431|
|Gray ZNCC|22.182|37.967|
|Channel ZNCC|18.539|32.838|

Raw RGB matching changes from 13.189mm to 54.214mm under mismatched lighting (+311.1%).
The diagonal candidate averages 36.431mm versus 29.896mm for the scalar ablation and 32.838mm for established channelZNCC. It does not outperform the scalar ablation.

Independent float64 recomputation checked 210 per-case/method score rows. Scoring masks are identical across lighting and output support is identical across methods.

Geometry was checked using supplied projection decomposition, numerical roundtrips and RGB/STL support overlays. References are z-buffered original STL point samples, not inferred depth or interpolated dense ground truth.

Important scope limits: one held-out object; fixed low resolution and finite depth planes; no SGM/global smoothness; directional light rather than measured illuminant-color ground truth. Known radiometric matching costs are baselines, not a new color-constancy contribution.

The pointclouds_inference directory contains full finite outputs without GT masking. The pointclouds directory and comparison panels use common evaluation support and are labeled accordingly.

See phase2/research/stereo_design_audit.md and the copied protocol.json for sources, equations and exact settings.
