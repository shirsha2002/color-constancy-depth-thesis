"""Download a scene-disjoint NYU Depth V2 pilot using verified HTTP byte ranges.

The original NYU MATLAB file is ~2.97 GB. HDF5 random access avoids downloading
the entire file. RGB and reference depth come directly from the dataset authors.
The cached blocks are reusable and every range response is checked.

Usage: python scripts/download_data.py [--dev 8] [--eval 32]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import requests
from PIL import Image
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "nyu_pilot"
MAT_URL = "https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
SPLIT_URL = "https://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat"
HOMEPAGE = "https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html"
MAT_SIZE = 2_972_037_809
BLOCK_SIZE = 1_048_576
SEED = 20261023


def get(url: str, **kwargs) -> requests.Response:
    for attempt in range(5):
        try:
            response = requests.get(url, timeout=(20, 100), **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


class RemoteHDF5(io.RawIOBase):
    """Seekable read-only original .mat with on-disk range block cache."""

    def __init__(self):
        self.position = 0
        self.cache_dir = OUT / "_range_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory: dict[int, bytes] = {}
        super().__init__()

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.position = offset
        elif whence == io.SEEK_CUR:
            self.position += offset
        elif whence == io.SEEK_END:
            self.position = MAT_SIZE + offset
        else:
            raise ValueError(whence)
        return self.position

    def block(self, index):
        if index in self.memory:
            return self.memory[index]
        start = index * BLOCK_SIZE
        end = min(start + BLOCK_SIZE, MAT_SIZE) - 1
        path = self.cache_dir / f"{index:06d}.bin"
        expected = end - start + 1
        if path.exists() and path.stat().st_size == expected:
            payload = path.read_bytes()
        else:
            # A unique URL per range prevents intermediary caches from serving
            # a different request's cached partial response.
            response = get(
                f"{MAT_URL}?byte_range={start}-{end}",
                headers={"Range": f"bytes={start}-{end}"},
            )
            expected_header = f"bytes {start}-{end}/{MAT_SIZE}"
            if response.status_code != 206 or response.headers.get("Content-Range") != expected_header:
                raise RuntimeError(f"Unverified range response: {response.status_code}, {response.headers.get('Content-Range')}")
            payload = response.content
            if len(payload) != expected:
                raise RuntimeError("Truncated HTTP range")
            partial = path.with_suffix(f".{os.getpid()}.partial")
            partial.write_bytes(payload)
            partial.replace(path)
            print(f"Fetched official .mat block {index} ({expected:,} bytes)", flush=True)
        self.memory[index] = payload
        return payload

    def readinto(self, buffer):
        count = max(0, min(len(buffer), MAT_SIZE - self.position))
        view = memoryview(buffer)
        done = 0
        while done < count:
            index, offset = divmod(self.position, BLOCK_SIZE)
            payload = self.block(index)
            length = min(count - done, len(payload) - offset)
            view[done:done + length] = payload[offset:offset + length]
            self.position += length
            done += length
        return done


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_frames(indices, scenes, count, rng):
    """Choose scenes without replacement, then one image per selected scene."""
    by_scene = {}
    for matlab_id in np.ravel(indices).astype(int):
        by_scene.setdefault(scenes[matlab_id - 1], []).append(int(matlab_id))
    if count > len(by_scene):
        raise ValueError(f"Requested {count} distinct scenes; only {len(by_scene)} available")
    scene_names = sorted(by_scene)
    selected_scenes = rng.choice(scene_names, size=count, replace=False).tolist()
    return [(scene, int(rng.choice(by_scene[scene]))) for scene in sorted(selected_scenes)]


def export_pair(selection):
    split, scene, image_id = selection
    name = f"nyu_{image_id:05d}"
    rgb_path = OUT / "rgb" / f"{name}.png"
    depth_path = OUT / "depth" / f"{name}.npy"
    if not (rgb_path.exists() and depth_path.exists()):
        with h5py.File(RemoteHDF5(), "r") as dataset:
            rgb = dataset["images"][image_id - 1].transpose(2, 1, 0)
            depth = dataset["depths"][image_id - 1].T.astype(np.float32)
        if rgb.shape != (480, 640, 3) or depth.shape != (480, 640):
            raise ValueError(f"Unexpected frame shapes: {rgb.shape}, {depth.shape}")
        Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(rgb_path)
        np.save(depth_path, depth)
    print(f"Downloaded {split}: {name}, {scene}", flush=True)
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", type=int, default=8)
    parser.add_argument("--eval", type=int, default=32)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for folder in ("rgb", "depth"):
        (OUT / folder).mkdir(exist_ok=True)
    split_path = OUT / "splits.mat"
    if not split_path.exists():
        split_path.write_bytes(get(SPLIT_URL).content)
    splits = loadmat(split_path)
    reader = RemoteHDF5()
    records = []
    with h5py.File(reader, "r") as dataset:
        # Exact string scene labels from the original MATLAB reference objects.
        refs = dataset["scenes"][:].ravel()
        scenes = ["".join(chr(int(c)) for c in dataset[ref][:].ravel()) for ref in refs]
        (OUT / "official_scenes.json").write_text(json.dumps(scenes, indent=2), encoding="utf-8")
        train_scenes = {scenes[int(i) - 1] for i in splits["trainNdxs"].ravel()}
        test_scenes = {scenes[int(i) - 1] for i in splits["testNdxs"].ravel()}
        if train_scenes & test_scenes:
            raise RuntimeError("Official training and test scenes unexpectedly overlap")
        rng = np.random.default_rng(SEED)
        selections = {
            "development": select_frames(splits["trainNdxs"], scenes, args.dev, rng),
            "evaluation": select_frames(splits["testNdxs"], scenes, args.eval, rng),
        }
        print("Scene-disjoint selections:", json.dumps(selections), flush=True)
        tasks = [(split, scene, image_id) for split, selected in selections.items() for scene, image_id in selected]
        # Separate processes avoid h5py's global lock serializing network reads.
        # Cache writes are atomic and use per-process temporary names.
        with ProcessPoolExecutor(max_workers=6) as executor:
            list(executor.map(export_pair, tasks))
        for split, selected in selections.items():
            for scene, image_id in selected:
                name = f"nyu_{image_id:05d}"
                rgb_path = OUT / "rgb" / f"{name}.png"
                depth_path = OUT / "depth" / f"{name}.npy"
                if not (rgb_path.exists() and depth_path.exists()):
                    # MATLAB uses transposed dimensions as exposed through h5py.
                    rgb = dataset["images"][image_id - 1].transpose(2, 1, 0)
                    depth = dataset["depths"][image_id - 1].T.astype(np.float32)
                    if rgb.shape != (480, 640, 3) or depth.shape != (480, 640):
                        raise ValueError(f"Unexpected frame shapes: {rgb.shape}, {depth.shape}")
                    Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(rgb_path)
                    np.save(depth_path, depth)
                depth = np.load(depth_path)
                valid = np.isfinite(depth) & (depth > 0) & (depth <= 10)
                records.append({
                    "id": name,
                    "split": split,
                    "official_split": "train" if split == "development" else "test",
                    "original_matlab_id": image_id,
                    "scene": scene,
                    "rgb": rgb_path.relative_to(ROOT).as_posix(),
                    "depth": depth_path.relative_to(ROOT).as_posix(),
                    "source_url": MAT_URL,
                    "rgb_variable": f"images[:,:,:,{image_id}] (MATLAB one-based)",
                    "depth_variable": f"depths[:,:,{image_id}] (MATLAB one-based)",
                    "depth_units": "meters",
                    "reference_type": "official Kinect-derived, colorization-inpainted depths",
                    "width": 640,
                    "height": 480,
                    "valid_depth_fraction_0_to_10m": float(valid.mean()),
                    "depth_min_m": float(depth[valid].min()),
                    "depth_max_m": float(depth[valid].max()),
                    "rgb_sha256": sha256(rgb_path),
                    "depth_sha256": sha256(depth_path),
                })
                print(f"Ready {split}: {name}, {scene}", flush=True)
                # Partial manifest supports inspection during a long download.
                (OUT / "manifest.partial.json").write_text(json.dumps({"images": records}, indent=2), encoding="utf-8")
    manifest = {
        "dataset": "NYU Depth V2 labeled dataset (original authors' .mat)",
        "homepage": HOMEPAGE,
        "source_url": MAT_URL,
        "official_split_url": SPLIT_URL,
        "splits_mat_sha256": sha256(split_path),
        "selection_seed": SEED,
        "selection": "Random scenes without replacement from official split; one random frame per scene. Development selected before evaluation using one NumPy PCG64 generator.",
        "development_count": args.dev,
        "evaluation_count": args.eval,
        "scene_disjoint": True,
        "depth_units": "meters",
        "depth_reference": "depths field: Kinect-derived, projected and inpainted by original dataset authors; not model predictions",
        "evaluation_warning": "Small pilot subset, not the full 654-image benchmark. Inpainted depth is not a dense direct sensor measurement. No model is trained here.",
        "suggested_evaluation_crop": {"row_start": 45, "row_end_exclusive": 471, "column_start": 41, "column_end_exclusive": 601},
        "validity": "Finite reference depth >0 and <=10 meters within fixed crop. No depth scaling or predictions used in sample selection.",
        "images": records,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    import csv
    with (OUT / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"Complete: {len(records)} RGB/reference-depth pairs in {OUT}", flush=True)


if __name__ == "__main__":
    main()
