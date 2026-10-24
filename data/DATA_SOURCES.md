# Data provenance: NYU Depth V2 pilot

The experiment uses RGB and reference depth **directly from the original dataset authors' MATLAB file**, not predicted depth or a display heatmap.

- Official dataset description: https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html
- Original labeled MATLAB 7.3 / HDF5 file: https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
- Official 795-train / 654-test split indices: https://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat
- Requested citation: Silberman, N., Hoiem, D., Kohli, P., and Fergus, R. *Indoor Segmentation and Support Inference from RGBD Images*. ECCV, 2012.

## Encoding and preparation

The official `images` variable contains aligned uint8 RGB images. The official `depths` variable contains Kinect-derived depth **in meters**, with missing values filled by the dataset authors using the colorization method of Levin et al. These dense reference depths include interpolated values and are not purely direct measurements. The official `rawDepths` field would contain projected but unfilled measurements; it is not used in this initial pilot.

`scripts/download_data.py` reads only required byte ranges from the ~2.97 GB original file. Range offsets, sizes, and HTTP Content-Range headers are verified. It saves lossless RGB PNGs and unchanged float32 depth arrays in NPY format, with MATLAB's dimension order transposed into conventional height × width. No spatial resizing, depth scaling, cropping, or augmentation occurs during data export.

The downloader obtains original scene labels and the official split. Using NumPy PCG64 seed `20261023`, it samples **8 distinct training scenes for development and 32 distinct test scenes for evaluation**, one frame per scene. Development and evaluation scene sets are disjoint. Selection does not depend on depth model predictions, error measurements, or correction performance. The manifest records original one-based frame IDs, scene, split, per-file SHA-256, source, dimensions, and depth range.

The manifest proposes a fixed NYU evaluation crop, zero-based rows `[45:471]` and columns `[41:601]`, The actual experiment combines this crop with finite reference depth in `(0.1,10)` meters and clips predictions to `[0.1,10]` meters. The experiment must record its actual mask and resize protocol; this crop suggestion alone does not make the pilot identical to any published benchmark implementation.

## Scope and licensing

This is a small pilot; it does not reproduce the full 654-image benchmark. It is useful for paired, controlled color-cast comparisons on held-out scenes. Use model-frozen comparisons and identical valid pixels for every image variant.

The official webpage requests citation and does not present a clear general-purpose redistribution license. Do not assume that a third-party mirror's Apache or MIT metadata grants redistribution rights to the original images. Keep downloaded data and the byte-range cache out of GitHub; publish the downloader, source attribution, sample IDs, hashes, metrics, and generated research figures instead.

## Temporary benchmark pair

During acquisition, one genuine NYU pair was fetched from `val/official/00001.h5` in the FastDepth mirror https://huggingface.co/datasets/sayakpaul/nyu_depth_v2 (tar `data/val-000000.tar`). The resulting `nyu_val_00001.png/.npy` files were made available for runtime benchmarking only. They are **not part of the final experimental manifest**. The final manifest's `nyu_XXXXX` images all come directly from the original NYU `.mat` file.

Mirror provenance was checked against the FastDepth authors' repository: https://github.com/dwofk/fast-depth. Its loader preserves reference depth in meter units. The mirror was not needed for final experiment data.
