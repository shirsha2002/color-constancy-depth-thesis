# ReMID: complete real multi-illumination RGB-D release

Acquired 7 November 2026 from the public Drive folder linked in footnote 1 of
the primary [LumiDepth ECCV 2026 paper](https://link.springer.com/chapter/10.1007/978-3-032-37314-4_9#Fn1).
The publisher explicitly links
[this author release](https://drive.google.com/drive/folders/1-3txE3KiFiscACxs1Xch8UO1Ax82VBNg).

Five transport ZIPs contain 735 real 1280x720 RGB images in 50 captures:
284 indoor images and 451 tabletop images. All 885 extracted native files passed
archive CRC verification and have SHA-256 records in `acquisition_manifest.json`.
Raw archives, RGBs, depths and sensor metadata remain local.

The native `meta.json` files explicitly identify `depth_raw_mm.npy` as float32
millimeters, zero as invalid, and alignment via
`pyrealsense2.rs.align(rs.stream.color)`. The RGB variants of a capture share
the aligned 720x1280 reference. Intrinsics and distortion coefficients are
recorded in `manifest.json`. This study divides raw depth by 1,000 and uses a
single declared valid range, finite 0.1 < depth < 10 meters, for every method.
Native values above the range, including 65535-like returns, are preserved.

The author's README explains that `split_1` through `split_5` are separate upload
parts, not train/validation/test folds. Numeric scene IDs restart in different
parts, with visually different arrangements. A unique capture ID therefore
includes shard/category/scene. Conservative category/scene labels give 25 audit
groups for resampling, but neither these nor the 50 captures establish physical
room independence. No location identities are supplied.

All ReMID images are external evaluation only. No training, threshold choice or
checkpoint selection uses them. Frame names are ordinal lighting IDs rather
than calibrated illuminant RGB labels. Camera-processed RGB is distinct from
RAW color-checker-labeled CC data; this run cannot report illuminant angular
accuracy from ReMID alone.

No exact RGB duplicates or identical reference arrays across captures were
found in the completed source audit. This does not rule out similar scenes or
multiple views of one room. The selected valid-mask protocol and metric scores
are project-specific and must not be described as reproducing the complete
LumiDepth evaluation protocol.
