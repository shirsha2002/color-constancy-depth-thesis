# New scene-separated NYU Depth V2 subset

The 368 original RGB/reference pairs come directly from the original authors'
[labeled MATLAB file](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html).
The downloader reuses the verified one-megabyte byte-range reader and local
immutable block cache from the first experiment. Content-Range and received
length are checked; each exported PNG and float32 depth array has a SHA-256
identity in the new `manifest.json`.

The new selection is recorded in `selection.json` before model execution:
200 training scenes, 40 validation scenes and 128 evaluation scenes, one frame
per original scene, NumPy seed 20261107. Official train/test membership is
preserved. All 40 scenes from the first pilot are excluded across every new
partition. Scene and frame selection uses no model prediction or depth error.

RGBs are original camera-processed 640x480 images. Depth is the original authors'
projected and colorization-inpainted Kinect-derived `depths` field, in meters.
It is independently sourced reference geometry rather than a model prediction;
it is also not a dense direct sensor measurement at every filled pixel.

This is real RGB-D scene data. It does not provide physical repeated-illuminant
captures or calibrated illuminant ground truth. The controlled diagonal/scalar
changes used by this study are explicitly digital simulation, and are reported
separately from native real-image accuracy and captured ReMID lighting.

Raw data and large caches remain local. The source links, split IDs, content
hashes and acquisition code are versioned for reproduction.
