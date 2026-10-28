# Real-light calibrated stereo pilot

Run from the thesis project directory with its isolated interpreter:

```powershell
.\.venv\Scripts\python.exe phase2\stereo\run_experiment.py --scenes 1 --refs 25 --sources 26 --lights 3,0 --out calibration_pair_v2
.\.venv\Scripts\python.exe phase2\stereo\validate.py --pair-run calibration_pair_v2
.\.venv\Scripts\python.exe phase2\stereo\run_experiment.py --scenes 1,6 --out main_v2
.\.venv\Scripts\python.exe phase2\stereo\analyze.py --run main_v2
.\.venv\Scripts\python.exe phase2\stereo\finalize.py
```

Inputs are original, verified DTU SampleSet files acquired by `phase2/scripts` and recorded in `phase2/data/dtu/manifest.json`. The matching runner reads that original hierarchy without training a model. It uses the supplied right-camera projection matrices in the STL coordinate system.

`protocol.json` declares the fixed settings. Each output directory receives a complete copy and source-code hash before any matching. Cached output is reused only with the same protocol. Final results go to `phase2/results/stereo/main_v2/`: per-case arrays, metrics, descriptive summaries, camera/reference overlays and point clouds. Scene 1 is development; scene 6 is the single held-out scene. Earlier `main` and `calibration_pair` directories are diagnostic attempts with float32 photometric moments; they are not final results. Independent numerical checks caught variance cancellation in nearly saturated patches, so final costs use float64 moments.

Read `phase2/research/stereo_design_audit.md` before interpreting the results. These are known matching-cost models, not a new algorithm contribution. DTU real light changes are primarily directional; this study does not measure absolute illuminant-color recovery. Point-cloud previews and low-resolution depth scores do not reproduce the official full DTU reconstruction benchmark.

## Completed final pilot

The complete final run has 42 image conditions and 210 method score rows. `analyze.py` independently recalculates the four error scores from every saved array in float64 and checks common masks/support. `finalize.py` verifies source/protocol fingerprints, input RGB hashes, CSV/record/summary agreement, and writes `main_v2/report_handoff.json` for the combined research note. It does not tune or rerun inference.

On the held-out scene 6, the mean raw RGB SSD depth MAE rises from 13.189 mm under matched diffuse lighting to 54.214 mm under the six changed-light conditions. Changed-light scalar gain, diagonal gain, gray ZNCC and channel ZNCC average 29.896, 36.431, 37.967 and 32.838 mm respectively. These are means of per-view/light scores; scene 6 is one object, not 21 independent scenes. The color-channel candidate does not beat the scalar control or channel ZNCC in MAE. The study establishes lighting sensitivity and the usefulness of radiometric modeling in this pilot, but no additional CC-specific gain.

All matching and gain estimation use RGB/camera inputs only. STL, visibility and the supplied foreground plane define evaluation support. `pointclouds_inference` exports RGB SSD, scalar gain, diagonal gain and channel ZNCC on their full finite inference support, with no GT filtering. The `_preview30k.ply` files are deterministic small samples for inspection; they are not complete reconstructed meshes. `pointclouds` and comparison panels use GT-defined common evaluation support and must be labeled as such if shared.
