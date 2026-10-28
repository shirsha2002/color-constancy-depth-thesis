"""Selectively download authentic DTU MVS samples from the official ZIP.

No full archive is downloaded. ZIP64 metadata and exact member byte ranges are
used, with HTTP Content-Range checks, ZIP CRC32 validation, and SHA-256 hashes.
Run: python phase2/scripts/download_dtu.py --stage full
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
import os
from pathlib import Path
import re
import struct
import time
import zipfile
import zlib

import requests

PHASE2 = Path(__file__).resolve().parents[1]
OUT = PHASE2 / "data" / "dtu"
SOURCE = OUT / "source"
URL = "https://roboimagedata2.compute.dtu.dk/data/MVS/SampleSet.zip"
PAGE = "https://roboimagedata.compute.dtu.dk/?page_id=36"
ARCHIVE_SIZE = 6_905_656_531
MIRROR_URL = "https://huggingface.co/datasets/rhayat/DTU/resolve/main/SampleSet.zip"
MIRROR_SIZE = 6_924_030_144
ACTIVE_URL, ACTIVE_SIZE = URL, ARCHIVE_SIZE
CACHE_LABEL = "_ranges"
TAIL_NAME = "archive_tail512k.bin"
INDEX_NAME = "archive_index.json"
TRANSPORT = "official"
CHUNK_SIZE = 4 * 1024 * 1024
VIEWS = [25, 26, 35]
LIGHTS = list(range(7))
SCENES = [1, 6]


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def fetch_range(start, end):
    """Return a checked inclusive byte range, caching exact received bytes."""
    cache = OUT / CACHE_LABEL / f"{start:010d}_{end:010d}.bin"
    cache.parent.mkdir(parents=True, exist_ok=True)
    length = end - start + 1
    if cache.exists() and cache.stat().st_size == length:
        return cache.read_bytes()
    tail_path = OUT / TAIL_NAME
    if tail_path.exists() and start >= ACTIVE_SIZE - tail_path.stat().st_size:
        tail = tail_path.read_bytes()
        return tail[start - (ACTIVE_SIZE - len(tail)): end + 1 - (ACTIVE_SIZE - len(tail))]
    for attempt in range(5):
        try:
            response = requests.get(
                f"{ACTIVE_URL}?bytes={start}-{end}",
                headers={"Range": f"bytes={start}-{end}"},
                timeout=(25, 150),
            )
            response.raise_for_status()
            expected = f"bytes {start}-{end}/{ACTIVE_SIZE}"
            if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                raise RuntimeError(f"Unexpected range: {response.status_code}, {response.headers.get('Content-Range')}, expected {expected}")
            payload = response.content
            if len(payload) != length:
                raise RuntimeError(f"Truncated range: {len(payload)} != {length}")
            partial = cache.with_suffix(f".{os.getpid()}.partial")
            partial.write_bytes(payload)
            partial.replace(cache)
            print(f"Downloaded range {start}-{end} ({length / 1e6:.2f} MB)", flush=True)
            return payload
        except (requests.RequestException, RuntimeError):
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def load_index():
    path = OUT / INDEX_NAME
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    tail_start = ACTIVE_SIZE - 524288
    tail = fetch_range(tail_start, ACTIVE_SIZE - 1)
    (OUT / TAIL_NAME).write_bytes(tail)

    class TailReader(io.RawIOBase):
        def __init__(self): self.position = 0
        def readable(self): return True
        def seekable(self): return True
        def tell(self): return self.position
        def seek(self, offset, whence=0):
            self.position = offset if whence == 0 else self.position + offset if whence == 1 else ACTIVE_SIZE + offset
            return self.position
        def read(self, count=-1):
            if count < 0: count = ACTIVE_SIZE - self.position
            if self.position < tail_start: raise RuntimeError("Central directory exceeds saved tail")
            result = tail[self.position - tail_start:self.position - tail_start + count]
            self.position += len(result)
            return result

    with zipfile.ZipFile(TailReader()) as archive:
        entries = [dict(name=x.filename, compressed_size=x.compress_size, size=x.file_size,
                        offset=x.header_offset, compression=x.compress_type, crc32=x.CRC, is_dir=x.is_dir())
                   for x in archive.infolist()]
    path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    return entries


def destination(entry):
    # Retain the original archive hierarchy for primary-source inspection.
    candidate = (SOURCE / entry["name"]).resolve()
    if not candidate.is_relative_to(SOURCE.resolve()):
        raise RuntimeError("Unsafe ZIP member path")
    return candidate


def select_entries(index, stage):
    selected = []
    for entry in index:
        name = entry["name"]
        if entry["is_dir"]:
            continue
        include = name in {"SampleSet/ReadMe.txt", "SampleSet/CVPR2014bib.txt", "SampleSet/Datasets included.pdf"}
        include |= bool(re.fullmatch(r"SampleSet/MVS Data/Calibration/cal18/pos_\d{3}\.txt", name))
        include |= "/Calibration/cal18/Calib_Results_" in name and name.endswith((".m", ".mat"))
        include |= "/Matlab evaluation code/" in name and name.count("/") == 2 and name.endswith(".m")
        include |= any(name.endswith(f"/{suffix}{scene}{tail}") for scene in SCENES
                       for suffix, tail in [("ObsMask", "_10.mat"), ("Plane", ".mat")])
        if stage in ("pilot", "full"):
            include |= name.endswith("/Points/stl/stl001_total.ply")
            include |= bool(re.fullmatch(r"SampleSet/MVS Data/Rectified/scan1/rect_0(25|26)_[03]_r5000\.png", name))
        if stage == "full":
            include |= name.endswith("/Points/stl/stl006_total.ply")
            for scene in SCENES:
                for view in VIEWS:
                    for light in LIGHTS:
                        include |= name == f"SampleSet/MVS Data/Rectified/scan{scene}/rect_{view:03d}_{light}_r5000.png"
        if include:
            selected.append(entry)
    return selected


def member_range(entry):
    # Local headers contain variable name/extra fields. 1024 bytes are ample for
    # these official short member names; the header parser verifies the extent.
    return entry["offset"], min(ACTIVE_SIZE - 1, entry["offset"] + 30 + 1024 + entry["compressed_size"] - 1)


def download(entries, workers):
    missing = []
    for entry in entries:
        path = destination(entry)
        if path.exists() and path.stat().st_size == entry["size"] and zlib.crc32(path.read_bytes()) & 0xffffffff == entry["crc32"]:
            continue
        missing.append(entry)
    # Merge adjacent tiny metadata members to avoid dozens of network requests.
    spans = sorted(member_range(entry) for entry in missing)
    merged = []
    for start, end in spans:
        if merged and start <= merged[-1][1] + 4096 and end - merged[-1][0] < CHUNK_SIZE:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    chunks = []
    for start, end in merged:
        for offset in range(start, end + 1, CHUNK_SIZE):
            chunks.append((offset, min(end, offset + CHUNK_SIZE - 1)))
    print(f"Selected {len(entries)} members; {len(missing)} missing; network plan {sum(b-a+1 for a,b in chunks)/1e6:.2f} MB in {len(chunks)} ranges", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch_range, start, end): (start, end) for start, end in chunks}
        payloads = {futures[f]: f.result() for f in as_completed(futures)}
    for entry in missing:
        start, end = member_range(entry)
        relevant = sorted((a, b, payload) for (a, b), payload in payloads.items() if a <= end and b >= start)
        parts, cursor = [], start
        for a, b, payload in relevant:
            if b < cursor:
                continue
            if a > cursor:
                raise RuntimeError("Gap in downloaded ZIP member ranges")
            last = min(end, b)
            parts.append(payload[cursor-a:last-a+1])
            cursor = last + 1
            if cursor > end:
                break
        if cursor != end + 1:
            raise RuntimeError("Incomplete ZIP member ranges")
        blob = b"".join(parts)
        header = struct.unpack_from("<4s5H3I2H", blob, 0)
        if header[0] != b"PK\x03\x04":
            raise RuntimeError("Local ZIP header signature mismatch")
        name_length, extra_length = header[-2:]
        payload_offset = 30 + name_length + extra_length
        compressed = blob[payload_offset:payload_offset + entry["compressed_size"]]
        if len(compressed) != entry["compressed_size"]:
            raise RuntimeError("Local ZIP header is larger than reserved range")
        if entry["compression"] == 0:
            raw = compressed
        elif entry["compression"] == 8:
            raw = zlib.decompress(compressed, -15)
        else:
            raise RuntimeError(f"Unsupported ZIP compression {entry['compression']}")
        if len(raw) != entry["size"] or (zlib.crc32(raw) & 0xffffffff) != entry["crc32"]:
            raise RuntimeError(f"ZIP integrity check failed for {entry['name']}")
        path = destination(entry)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        print(f"Verified member {entry['name']}", flush=True)


def write_manifest(entries, stage):
    files, images = [], []
    for entry in entries:
        path = destination(entry)
        payload = path.read_bytes()
        record = dict(archive_member=entry["name"], path=path.relative_to(PHASE2).as_posix(),
                      bytes=len(payload), sha256=digest(payload), zip_crc32=f"{entry['crc32']:08x}",
                      archive_header_offset=entry["offset"], compressed_bytes=entry["compressed_size"])
        files.append(record)
        match = re.search(r"/Rectified/scan(\d+)/rect_(\d{3})_(\d)_r5000\.png$", entry["name"])
        if match:
            scene, view, light = map(int, match.groups())
            from PIL import Image
            with Image.open(path) as image:
                width, height = image.size
            images.append(dict(scene_id=scene, view_id=view, light_id=light, lighting_id=light,
                               width=width, height=height, image=record["path"], sha256=record["sha256"],
                               camera=f"data/dtu/source/SampleSet/MVS Data/Calibration/cal18/pos_{view:03d}.txt"))
    manifest = dict(dataset="DTU MVS 2014 official SampleSet selective subset", source_page=PAGE,
                    archive_url=URL, archive_bytes=ARCHIVE_SIZE, archive_index_sha256=digest((OUT/"archive_index.json").read_bytes()),
                    transport=TRANSPORT, transport_archive_url=ACTIVE_URL, transport_archive_bytes=ACTIVE_SIZE,
                    mirror_verification="All2088non-directory member names, uncompressed sizes and CRC32 values match the official ZIP central directory" if TRANSPORT == "verified-mirror" else None,
                    stage=stage, scenes=SCENES if stage == "full" else ([1] if stage == "pilot" else []),
                    selected_views=VIEWS if stage == "full" else ([25,26] if stage == "pilot" else []),
                    lighting_ids=LIGHTS if stage == "full" else ([0,3] if stage == "pilot" else []),
                    diffuse_lighting_id=3, image_count=len(images),
                    projection_convention="u~P[X,Y,Z,1]; supplied right-camera P maps the structured-light left-camera reference frame to rectified image pixels",
                    geometry_units="millimeters (official evaluation README labels distances 0-10 mm)",
                    geometry_reference="Official structured-light point clouds, not MVS baseline outputs or model-predicted depth",
                    native_dense_depth_provided=False,
                    images=images, files=files,
                    limitations=["Only scenes1and6, selected fixed views: pilot, not full benchmark",
                                 "Lighting variations include shading, shadow and specular changes; white balance alone cannot undo them",
                                 "Reference is a point cloud. Depth rasterization requires visibility/z-buffer handling and valid-coverage masks",
                                 "Projection matrices are supplied unchanged. Pixel-origin convention should be validated with projected reference geometry",
                                 "Do not use the _max composite images as a real eighth illumination condition"])
    path = OUT / f"manifest_{stage}.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if stage == "full":
        (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest ready: {path} ({len(images)} images, {len(files)} files)", flush=True)


def main():
    global ACTIVE_URL, ACTIVE_SIZE, CACHE_LABEL, TAIL_NAME, INDEX_NAME, TRANSPORT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["metadata", "pilot", "full"], default="full")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--transport", choices=["official", "verified-mirror"], default="verified-mirror")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    index = load_index()
    if args.transport == "verified-mirror":
        official = {entry["name"]: entry for entry in index if not entry["is_dir"]}
        ACTIVE_URL, ACTIVE_SIZE = MIRROR_URL, MIRROR_SIZE
        CACHE_LABEL, TAIL_NAME, INDEX_NAME = "_ranges_mirror", "mirror_archive_tail512k.bin", "mirror_archive_index.json"
        mirrored_index = load_index()
        mirrored = {entry["name"]: entry for entry in mirrored_index if not entry["is_dir"]}
        if set(official) != set(mirrored):
            raise RuntimeError("Mirror member set differs from authoritative original")
        for name, entry in official.items():
            if entry["size"] != mirrored[name]["size"] or entry["crc32"] != mirrored[name]["crc32"]:
                raise RuntimeError(f"Mirror does not match authoritative original: {name}")
        print(f"Verified mirror agrees with all {len(official)} original file names, sizes and CRC32 values", flush=True)
        index = mirrored_index
        TRANSPORT = "verified-mirror"
    entries = select_entries(index, args.stage)
    download(entries, args.workers)
    write_manifest(entries, args.stage)


if __name__ == "__main__":
    main()
