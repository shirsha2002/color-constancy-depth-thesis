# Local coupling: validation development before final test inspection

7 November 2026. The global conditional controls converge to nearly the same
validation score. Before inspecting any phase-three final test scores, this
second prototype adds a more specific local geometry/chroma coupling. The first
prototype and all its controls remain retained. The primary local candidate is
fixed as `spatial_geometry_eq`; no test-driven choice among candidates occurs.

## Coupled architecture

Each encoder token receives local linear-RGB, log-channel ratios and semantic
context. The system's own frozen depth prediction supplies normalized virtual
XYZ coordinates. A point-wise head predicts a two-dimensional relative chromatic
field. Neighbor affinities exp(-absolute log-depth difference/0.15) guide local
field smoothing. The field modulates bounded low-rank token residuals before
the shared frozen depth decoder. This forward graph receives a single RGB image;
measured depth, measured normals and test illumination labels are excluded.

The proposed geometric structure is a hypothesis, not measured physical 3D.
Its ingredients have prior art. In particular, estimated chromaticity still
mixes reflectance with illumination without absolute-illuminant supervision.

## Objective and matched controls

All variants retain common supervised depth, 0.1 paired log-depth agreement and
0.001 residual regularization. A local relative-chroma equivariance term with
weight0.2 matches the known augmentation's log(R/G),log(B/G) at each token.
The geometry-weighted local chroma variation penalty has weight0.01.

| Variant | Purpose |
|---|---|
| spatial_content | Generic semantic/geometry local field, explicit photometric input disabled; same objective |
| spatial_flat | Same active architecture and losses with constant depth/flat neighbor affinities |
| spatial_no_graph_loss | Full architecture without the geometry-weighted variation loss |
| spatial_no_equivariance | Full architecture without local relative-chroma equivariance |
| spatial_geometry_eq | Fixed primary candidate: full local coupling and both objectives |

All five instantiate the same parameter set and use the original 600-update
schedule, training scenes, original/augmented pairs, optimizer and validation
rule. Flat versus full has the same active architecture; content disables its
explicit color input, so total equal parameter count is not identical input
capacity. Every tested variant is reported.

The data roles and masks are unchanged from EXPERIMENT_V1.md. ReMID remains
external-only. Selection among the 200/400/600 checkpoints uses the same forty
NYU validation scenes. Development must not consult partial final-test CSVs.

No claim of innovation, state-of-the-art performance or CC-specific gain is
made before evaluation against the matched generic, brightness and flat-geometry
controls. A learned smoothing field is not calibrated illumination by definition.

## Validation-motivated matched-control extension

The no-equivariance ablation reached validation AbsRel0.15975, lower than the
generic local control (0.16492). Before inspecting any final test score, three
additional recipes are therefore locked in `learning/spatial_controls.py`:
content without equivariance, flat geometry without equivariance, and full
geometry with neither added loss. All use the same600updates and pair schedule.
These prevent a benefit from simply removing a regularizer being attributed to
explicit color handling or geometry. They are controls, not new test-selected
candidates. All recipes will be retained in the final tables.
