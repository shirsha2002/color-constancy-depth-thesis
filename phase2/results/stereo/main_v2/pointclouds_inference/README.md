# Small previews of actual stereo outputs

These binary little-endian PLY files contain deterministic samples of the
measured stereo reconstructions, not original DTU reference geometry. Each
preview contains 30,000 colored points in the supplied DTU world frame, with
coordinates in millimeters.

Files cover held-out scene 6, matched source light 3 and changed source light 0,
for RGB SSD, scalar gain, diagonal gain and channel ZNCC. Each full output fuses
the three reference-view predictions; 1 mm voxel duplicate removal uses only
predicted coordinates. The seed for preview sampling is 20261028.

No ground-truth foreground, visibility or accuracy mask filters these inference
exports. Background estimates and matching outliers therefore remain visible.
The previews are not complete meshes or official DTU benchmark reconstructions.
The metric tables use their separately declared, shared evaluation support.

`README.json` describes the full local exports. The full PLY files and numerical
depth arrays remain on the laptop; `phase2/stereo/analyze.py --run main_v2`
regenerates them from the saved matching outputs. A fresh checkout must first
acquire the data and run matching, as explained in
[the stereo instructions](../../../../stereo/README.md).

Source data: [DTU](https://roboimagedata.compute.dtu.dk/?page_id=36).
See [attribution](../../../../../THIRD_PARTY.md) and
[the measured results](../RESULTS.md).
