"""Check frozen mono inputs, or rerun the study in a strictly new result folder.

This wrapper preserves the exact numeric source used in the completed study.
It never resumes an existing feature/prediction/model cache. The old runner's
cache keys are incomplete, so a fresh directory is mandatory for a new run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "phase2/results/mono"
CHECKPOINT = ROOT / "checkpoints/depth_anything_v2_metric_hypersim_vits.pth"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def project_path(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    require(path.is_relative_to(ROOT), f"Manifest path leaves project: {relative}")
    return path


def verify() -> dict:
    snapshot = json.loads((FROZEN / "source_snapshot.json").read_text(encoding="utf-8"))
    config = json.loads((FROZEN / "config.json").read_text(encoding="utf-8"))
    audit = json.loads((FROZEN / "independent_audit.json").read_text(encoding="utf-8"))
    require(audit["audit"] == "PASS", "Saved independent audit did not pass")
    require(sha(FROZEN / "summary.json") == audit["audited_summary_sha256"],
            "Summary differs from the independently audited summary")
    for relative, expected in snapshot["files"].items():
        require(sha(project_path(relative)) == expected, f"Frozen source/input changed: {relative}")
    require(sha(CHECKPOINT) == config["base_checkpoint_sha256"], "Base checkpoint changed")
    manifest_path = ROOT / "phase2/data/vari/manifest.json"
    require(sha(manifest_path) == config["manifest_sha256"], "Manifest changed")
    # Reconstruct the original runner's configuration digest without importing it.
    without_hash = {key: value for key, value in config.items() if key != "configuration_sha256"}
    reconstructed = hashlib.sha256(json.dumps(without_hash, sort_keys=True).encode()
        + (ROOT / "phase2/mono/cc_model.py").read_bytes()).hexdigest()
    require(reconstructed == config["configuration_sha256"], "Configuration digest differs")

    upstream = ROOT / "external/Depth-Anything-V2"
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"],
                                       text=True).strip()
    require(revision == snapshot["upstream_git_revision"], "Upstream revision changed")
    for command in (["git", "-C", str(upstream), "diff", "--quiet"],
                    ["git", "-C", str(upstream), "diff", "--cached", "--quiet"]):
        require(subprocess.run(command, check=False).returncode == 0,
                "Tracked upstream files were modified")
    untracked = subprocess.check_output(["git", "-C", str(upstream), "ls-files",
                                        "--others", "--exclude-standard"], text=True).splitlines()
    require(not any(Path(name).suffix == ".py" for name in untracked),
            "Untracked Python source exists in upstream checkout")

    records = json.loads(manifest_path.read_text(encoding="utf-8"))["images"]
    unique_depths, raw_depths = {}, {}
    for record in records:
        require(sha(project_path(record["rgb"])) == record["rgb_sha256"],
                f"RGB content changed: {record['id']}")
        unique_depths[record["depth"]] = record["depth_sha256"]
        raw_name = (f"phase2/data/vari/raw/{record['scene']}/{record['camera']}/"
                    f"{record['scene']}_{record['camera']}_CGeometry_ZDepth{record['view']}.tif")
        raw_depths[raw_name] = record["raw_depth_sha256"]
    for relative, expected in {**unique_depths, **raw_depths}.items():
        require(sha(project_path(relative)) == expected, f"Depth content changed: {relative}")

    return {"guard": "PASS", "frozen_source_files": len(snapshot["files"]),
            "rgb_files_verified": len(records), "decoded_depth_files_verified": len(unique_depths),
            "raw_depth_files_verified": len(raw_depths), "upstream_git_revision": revision,
            "configuration_sha256": reconstructed,
            "cache_policy": "Existing cache directories are never reused for a new run",
            "note": "Input integrity guard; use audit_mono.py for numerical result verification"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-fresh", action="store_true",
                        help="After verification, execute the frozen numeric study in --out")
    parser.add_argument("--out", type=Path,
                        help="New directory below phase2/results, required with --run-fresh")
    args = parser.parse_args()
    require(args.run_fresh == (args.out is not None), "Use --run-fresh and --out together")
    destination = None
    if args.out is not None:
        destination = args.out.resolve()
        require(destination.is_relative_to((ROOT / "phase2/results").resolve()),
                "Fresh result directory must be below phase2/results")
        require(not destination.exists(), "Refusing existing output directory: stale cache risk")
    report = verify()
    print(json.dumps(report, indent=2), flush=True)
    if not args.run_fresh:
        return

    # Match the completed CPU experiment. The official image2tensor method can
    # auto-select CUDA/MPS even when model parameters remain on CPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import torch
    require(not torch.cuda.is_available() and not torch.backends.mps.is_available(),
            "This exact reproduction requires a CPU-only process")
    sys.path.insert(0, str(ROOT / "phase2/mono"))
    import run_study
    config = json.loads((FROZEN / "config.json").read_text(encoding="utf-8"))
    # No original source file is edited: override only the module's output root.
    # Every cache/model/prediction will therefore be computed from scratch.
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "reproduction_guard.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    run_study.OUT = destination
    sys.argv = [str(ROOT / "phase2/mono/run_study.py"), "--input-size", str(config["input_size"]),
                "--steps", str(config["steps"]), "--threads", str(config["threads"])]
    run_study.main()


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, FileNotFoundError, subprocess.CalledProcessError) as error:
        print(f"Reproduction guard failed: {error}", file=sys.stderr)
        raise SystemExit(2)
