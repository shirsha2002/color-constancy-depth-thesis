# DTU multi-illumination geometry diagnostic: provenance and conventions

## Primary source

- Official dataset page: https://roboimagedata.compute.dtu.dk/?page_id=36
- Official SampleSet archive linked by that page: https://roboimagedata2.compute.dtu.dk/data/MVS/SampleSet.zip
- Official archive length checked through HTTP Content-Range: **6,905,656,531 bytes**.
- Dataset citation: Jensen, R., Dahl, A., Vogiatzis, G., Tola, E., and Aanæs, H. *Large Scale Multi-view Stereopsis Evaluation*. CVPR, 2014.
- Journal extension: Aanæs, H., Jensen, R., Vogiatzis, G., Tola, E., and Dahl, A. *Large-Scale Data for Multiple-View Stereopsis*. IJCV, 2016.
- Original README: `source/SampleSet/ReadMe.txt` (Windows-1252 text).

This dataset is described by its authors as **citeware**: cite the associated paper when using it. Downloaded native images, point clouds and caches should remain outside GitHub; distribute the downloader, provenance, source identifiers, and derived research evidence.

## Selected subset

The SampleSet contains complete native data for scenes 1 and 6. Its rectified folder has 49 viewpoints per scene, seven real lighting settings (`0` through `6`) and an additional `_max` composite. This pilot uses **views 25, 26 and 35**, preserving the original **one-based** view filenames, at **all seven real light settings** in **both scenes**: 42 native 1600 × 1200 RGB PNG files. The `_max` composite is excluded.

Original right/left/stereo calibration, all 64 supplied projection matrix text files, original scene 1/6 structured-light point clouds, observation masks, planes, and original MATLAB evaluation scripts are included. The reference point clouds are in `Points/stl`; the `Points/camp`, `Points/furu` and `Points/tola` reconstructions are competing MVS outputs and are not ground truth.

`manifest_pilot.json` provides the first scene 1 pair (views 25/26, lights 0/3), with matching projection matrices and the complete scene 1 structured-light reference. `manifest.json` is the complete requested 42-image subset after the full download completes. All manifest paths are **relative to the `phase2` directory**. Fields `view_id` and `light_id` preserve exact source IDs; `lighting_id` is an alias for compatibility.

## Exact geometry and camera conventions

The original README identifies light **3** as the most diffuse. It also states that the **color images come from the right camera**, whereas the **structured-light reference uses the left-camera coordinate frame**. The supplied `Calibration/cal18/pos_###.txt` projection matrices were provided to put reconstructions into the same reference frame as the structured-light point cloud.

Use each supplied 3 × 4 matrix **P** directly as `s [u,v,1]^T = P [X,Y,Z,1]^T`. For camera-Z reference depths, a decomposition `P = K [R | -R C]` gives `z_camera = (R (X-C))[2]`. Resolve the usual decomposition signs consistently so visible scene points have positive depth. Do not interpret the raw point cloud's world Z coordinate or an unnormalized homogeneous projection denominator as metric camera depth without checking this decomposition.

The original evaluation README labels error distances in **millimeters**, and the camera/point-cloud translation values use the same units. The native right-camera calibration stores `fc ≈ [2892.843725,2882.249450]`, `cc ≈ [824.425158,605.187152]`, with zero skew and 1600 × 1200 image dimensions. Use the supplied projection matrices for rectified images rather than silently assuming undistorted-image intrinsics equal raw calibration intrinsics.

The README does not explicitly specify a zero-based versus one-based raster origin for the supplied projection matrices. Keep matrices unchanged and validate a projected-reference overlay before evaluation; record any empirically justified convention explicitly. Resizing images requires the matching intrinsics/projection-coordinate transform, with the chosen pixel-center convention stated.

The native reference is a **point cloud**, not an official dense depth image. Scene 1 PLY has 2,880,879 vertices, binary little endian; each vertex is float32 `x,y,z,nx,ny,nz`, followed by uint8 `red,green,blue` (27 bytes/vertex). Its header uses unusual CR/CR/LF line endings, so parse the header robustly. Generate a projected reference with positive camera depth, front-most visibility/Z-buffer handling and an explicit valid-support mask. Empty projected pixels are unknown, not zero-depth measurements. Point splatting or interpolation changes reference support; document the choice and avoid filling across depth boundaries.

## Transport and integrity

The official server's slow transfer rate made selective access through a public mirror preferable. The mirror is https://huggingface.co/datasets/rhayat/DTU/blob/main/SampleSet.zip (archive length **6,924,030,144 bytes**). It is recompressed, so offsets and compressed sizes differ. Before use, **all 2,088 non-directory members** were checked against the authoritative official ZIP central directory: exact member name, uncompressed byte length and CRC32 agree for every file. Each selected member is decompressed and checked against that original CRC32/length; the output manifest records a SHA-256 digest for every extracted native file. This is ordinary ZIP integrity verification, not a claim that a full original-archive cryptographic hash was downloaded and checked.

Both archive indexes and downloaded ZIP64 directory tails are retained for review. `phase2/scripts/download_dtu.py` supports `--transport official` or `--transport verified-mirror` (default), reads exact HTTP byte ranges, verifies Content-Range and payload lengths, and never downloads the complete 6.9 GB archive. The 123 GB Rectified and 136 GB Cleaned archives are not downloaded.

## Interpretation limits

DTU's lighting variations mainly change **direction, shading, shadows and specular appearance**. They must not be described as verified chromatic-illumination changes or as a dataset on which global white balancing can remove all photometric differences. This is a complementary calibrated-geometry diagnostic for the monocular thesis, not a replacement of the main task by MVS.

Scenes 1 and 6 provide a small controlled demonstration, not a full DTU benchmark or enough independent scenes for broad statistical claims. Select parameters on scene 1 if needed and reserve scene 6 for the fixed-method check. Do not fit a per-image scale or depth range against scene 6 reference geometry. Original observation masks and planes describe the official 3D point-cloud evaluation support; a projected camera-depth pilot should state its own visibility/support protocol and must not be called the official DTU benchmark score.
