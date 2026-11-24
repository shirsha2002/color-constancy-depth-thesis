# NYU + DIODE camera-frame and scale experiment

Shirsha Mitra / 2023521460130 / 16 November 2026.

Nine new released-head PCCC models, 10000 epochs, official three folds. Old native and native-flat models reused; no gate or new architecture implemented.

| Geometry | Mean angular error | Median | Worst25% |
|---|---:|---:|---:|
| camera_real | 2.5471 deg | 1.7665 | 6.2195 |
| camera_unitdepth | 2.7192 deg | 1.7765 | 6.5476 |
| camera_flat | 2.8164 deg | 1.9717 | 6.7260 |
| native | 2.6135 deg | 1.7623 | 6.2850 |
| native_flat | 2.5061 deg | 1.6522 | 6.1271 |

## Predeclared paired contrasts

| Contrast | Role | Mean difference | Conditional95% interval |
|---|---|---:|---|
| camera_unitdepth minus native | primary | +0.1057 | [-0.1534, +0.3710] |
| camera_unitdepth minus camera_flat | primary | -0.0972 | [-0.3369, +0.1479] |
| camera_real minus native | secondary | -0.0664 | [-0.2386, +0.1051] |
| camera_unitdepth minus camera_real | secondary | +0.1720 | [-0.0734, +0.4240] |
| camera_flat minus native_flat | secondary | +0.3103 | [+0.1037, +0.5165] |

Negative differences favor the first condition. All contrasts are reported; intervals are conditional on these trained folds/captures, not independent rooms or repeated seeds. The two primary comparisons are not multiplicity adjusted. These are development data.

Per-source NYU/DIODE summaries and separately labelled test-selected references are in summary.csv. A coordinate normalization result is not physical calibration, proof of a new fusion algorithm, or a guarantee that geometry always helps.

Independent audit: PASS. New training time 5.758 GPU-hours. See audit.json for checkpoint verification and cross-backend inference differences.
