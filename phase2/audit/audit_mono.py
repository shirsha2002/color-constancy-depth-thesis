"""Independent, read-only numerical audit of the saved Vari monocular study.

Reads native uint16 TIFF references, saved float32 predictions and model records.
Does not train, change predictions, or modify source/model/data files.
Outputs only audit evidence under phase2/audit and results/mono.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import cv2

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "phase2/results/mono"
AUDIT = Path(__file__).resolve().parent
METHODS = ["frozen", "gray_world_preprocess", "task_adapter", "achromatic_consistency", "cc_supervised", "cc_consistency"]
TRAINED = METHODS[2:]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measure(prediction, reference):
    p = np.asarray(prediction, dtype=np.float64)
    g = np.asarray(reference, dtype=np.float64)
    assert p.shape == g.shape
    assert np.isfinite(p).all()
    valid = np.logical_and.reduce((np.isfinite(g), g > 0.1, g < 9.99))
    assert np.count_nonzero(valid) >= 100
    estimated = np.minimum(10.0, np.maximum(0.1, p[valid]))
    actual = g[valid]
    errors = estimated - actual
    ratios = np.fmax(estimated / actual, actual / estimated)
    return {"abs_rel": float(np.average(np.fabs(errors) / actual)),
            "rmse_m": float(np.sqrt(np.average(np.square(errors)))),
            "delta1": float(np.count_nonzero(ratios < 1.25) / actual.size),
            "valid_pixels": int(actual.size)}


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(records, path):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def audit_data(manifest):
    records = manifest["images"]
    grouped = defaultdict(list)
    for record in records:
        grouped[record["view_group"]].append(record)
    split_images = Counter(r["split"] for r in records)
    split_groups = Counter(rs[0]["split"] for rs in grouped.values())
    assert dict(split_images) == {"training": 220, "validation": 44, "evaluation": 264}
    assert dict(split_groups) == {"training": 20, "validation": 4, "evaluation": 24}
    assert {str(r["scene"]) for r in records if r["split"] != "evaluation"} == {"212"}
    assert {str(r["scene"]) for r in records if r["split"] == "evaluation"} == {"485"}
    primary, native64, legacy = {}, {}, {}
    reference_notes = []
    for group, rs in sorted(grouped.items()):
        assert len(rs) == 11 and len({r["lighting"] for r in rs}) == 11
        assert len({r["split"] for r in rs}) == 1
        assert len({r["depth"] for r in rs}) == 1
        assert len({r["legacy_depth"] for r in rs}) == 1
        assert len({r["depth_sha256"] for r in rs}) == 1
        r = rs[0]
        # Resolve original source independently from native scene/camera/view IDs.
        raw = ROOT / "phase2/data/vari/raw" / str(r["scene"]) / r["camera"] / f"{r['scene']}_{r['camera']}_CGeometry_ZDepth{r['view']}.tif"
        # Independent TIFF reader from data preparation's tifffile backend.
        # Identical RGB channels make OpenCV's BGR ordering immaterial here.
        raw_array = cv2.imread(str(raw), cv2.IMREAD_UNCHANGED)
        assert raw_array.dtype == np.uint16 and raw_array.shape == (480, 640, 3)
        assert np.array_equal(raw_array[..., 0], raw_array[..., 1])
        assert np.array_equal(raw_array[..., 0], raw_array[..., 2])
        assert sha(raw) == r["raw_depth_sha256"]
        integers = raw_array[..., 0]
        decoded32 = (np.float32(1) - integers.astype(np.float32) / np.float32(65535)) * np.float32(10)
        stored32 = np.load(ROOT / r["depth"])
        assert stored32.dtype == np.float32 and np.array_equal(decoded32, stored32)
        assert sha(ROOT / r["depth"]) == r["depth_sha256"]
        original64 = (65535.0 - integers.astype(np.float64)) * (10.0 / 65535.0)
        with Image.open(raw) as image:
            eightbit = np.asarray(image.convert("L"), dtype=np.float32)
        decoded8 = (np.float32(1) - eightbit / np.float32(255)) * np.float32(10)
        legacy_stored = np.load(ROOT / r["legacy_depth"])
        assert np.array_equal(decoded8, legacy_stored)
        primary[group], native64[group], legacy[group] = stored32, original64, legacy_stored
        for item in rs:
            assert sha(ROOT / item["rgb"]) == item["rgb_sha256"]
        reference_notes.append({"view_group": group, "split": r["split"], "scene": str(r["scene"]),
                                "raw_tiff": raw.relative_to(ROOT).as_posix(),
                                "float32_vs_native64_max_rounding_m": float(np.max(np.abs(stored32.astype(np.float64) - original64))),
                                "primary_legacy_mean_abs_difference_m": float(np.mean(np.abs(stored32 - legacy_stored)))})
    return records, grouped, primary, native64, legacy, {
        "images_by_split": dict(split_images), "view_groups_by_split": dict(split_groups),
        "rooms_by_split": {"training": ["212"], "validation": ["212"], "evaluation": ["485"]},
        "distinct_rooms": 2, "evaluation_rooms": 1,
        "all_48_groups_have_11_unique_lights_and_one_identical_reference": True,
        "all_48_primary_npy_arrays_exactly_match_original_uint16_tiff_decode": True,
        "all_48_legacy_arrays_exactly_match_authors_pil_8bit_decode": True,
        "all_528_rgb_hashes_and_48_raw_depth_hashes_match_manifest": True,
        "references": reference_notes,
    }


def audit_models(summary, records, verify_validation):
    import torch
    config = summary["configuration"]
    checkpoint = ROOT / "checkpoints/depth_anything_v2_metric_hypersim_vits.pth"
    assert sha(checkpoint) == config["base_checkpoint_sha256"]
    snapshot = json.loads((OUT / "source_snapshot.json").read_text(encoding="utf-8"))
    source_hashes = {}
    for name, expected in snapshot["files"].items():
        actual = sha(ROOT / name)
        assert actual == expected, f"Source snapshot changed: {name}"
        source_hashes[name] = actual
    raw_config = dict(config)
    expected_config_hash = raw_config.pop("configuration_sha256")
    actual_config_hash = hashlib.sha256(json.dumps(raw_config, sort_keys=True).encode() + (ROOT / "phase2/mono/cc_model.py").read_bytes()).hexdigest()
    assert expected_config_hash == actual_config_hash
    expected_shapes = {"network.0.weight": (32, 3), "network.0.bias": (32,),
                       "network.2.weight": (3072, 32), "network.2.bias": (3072,)}
    state_records = []
    loaded = {}
    for variant in TRAINED:
        path = OUT / "models" / f"{variant}.pt"
        model = torch.load(path, map_location="cpu", weights_only=True)
        loaded[variant] = model
        assert model["variant"] == variant
        assert model["configuration_sha256"] == expected_config_hash
        state = model["state_dict"]
        assert set(state) == set(expected_shapes)
        assert all(tuple(state[k].shape) == shape for k, shape in expected_shapes.items())
        assert all(v.dtype == torch.float32 and torch.isfinite(v).all() for v in state.values())
        count = sum(v.numel() for v in state.values())
        assert count == model["trainable_parameters"] == 101504
        history = model["history"]
        assert [x["step"] for x in history] == list(range(1, 151))
        candidates = [x for x in history if "validation_abs_rel" in x]
        assert [x["step"] for x in candidates] == [50, 100, 150]
        best = min(candidates, key=lambda x: x["validation_abs_rel"])
        assert model["selected_step"] == best["step"]
        assert model["validation_abs_rel"] == best["validation_abs_rel"]
        if variant in ["task_adapter", "cc_supervised"]:
            assert all(x["depth_consistency"] == 0 and x["feature_consistency"] == 0 for x in history)
        metadata = json.loads((OUT / f"{variant}_training.json").read_text(encoding="utf-8"))
        assert metadata == {k: v for k, v in model.items() if k != "state_dict"}
        state_records.append({"variant": variant, "checkpoint_sha256": sha(path), "adapter_only_state_keys": sorted(state),
                              "adapter_parameters": count, "selected_step": model["selected_step"],
                              "validation_abs_rel": model["validation_abs_rel"], "selection_is_argmin_of_50_100_150_validation": True})
    assert len(list((OUT / "models").glob("*.pt"))) == 4
    # Reviewable source-level evidence: optimizer receives adapter parameters only;
    # frozen_model sets requires_grad=False for every base-model parameter.
    sys.path.insert(0, str(ROOT / "phase2/mono"))
    from cc_model import frozen_model, ColorFeatureAdapter, predict_from_features
    torch.set_num_interop_threads(1)
    base = frozen_model(8)
    assert all(not p.requires_grad for p in base.parameters())
    frozen_count = sum(p.numel() for p in base.parameters())
    deltas = []
    if verify_validation:
        validation_records = [r for r in records if r["split"] == "validation"]
        training_records = [r for r in records if r["split"] == "training"]
        train_descriptors = []
        for r in training_records:
            item = torch.load(OUT / "features" / f"{r['id']}__none.pt", map_location="cpu", weights_only=True)
            assert item["base_checkpoint_sha256"] == config["base_checkpoint_sha256"]
            assert item["manifest_sha256"] == config["manifest_sha256"]
            train_descriptors.append(item["descriptor"])
        checked_stats = {}
        for key in ["chromatic", "achromatic"]:
            values = np.array([d[key] for d in train_descriptors], dtype=np.float32)
            checked_stats[key] = {"mean": values.mean(axis=0).tolist(), "std": np.maximum(values.std(axis=0), .05).tolist()}
        for variant, saved in loaded.items():
            assert saved["descriptor_statistics"] == checked_stats
            adapter = ColorFeatureAdapter()
            adapter.load_state_dict(saved["state_dict"])
            adapter.eval()
            scores = []
            with torch.inference_mode():
                for r in validation_records:
                    item = torch.load(OUT / "features" / f"{r['id']}__none.pt", map_location="cpu", weights_only=True)
                    if variant == "task_adapter":
                        condition = torch.zeros((1, 3))
                    else:
                        key = "achromatic" if variant == "achromatic_consistency" else "chromatic"
                        stats = saved["descriptor_statistics"][key]
                        condition = (torch.tensor([item["descriptor"][key]]) - torch.tensor(stats["mean"])) / torch.tensor(stats["std"])
                    gt = np.load(ROOT / r["depth"])
                    prediction, _, _ = predict_from_features(base, item["features"], item["patch_hw"], adapter, condition, gt.shape)
                    scores.append(measure(prediction[0].numpy(), gt)["abs_rel"])
            actual = float(np.mean(scores))
            difference = abs(actual - saved["validation_abs_rel"])
            assert difference < 1e-7
            deltas.append({"variant": variant, "recomputed_44_image_validation_abs_rel": actual, "recorded": saved["validation_abs_rel"], "difference": difference})
            print(f"Validated selected checkpoint: {variant} AbsRel={actual:.9f}", flush=True)
    return {"base_checkpoint_sha256": config["base_checkpoint_sha256"], "base_checkpoint_hash_unchanged": True,
            "source_snapshot_all_files_unchanged": source_hashes, "configuration_fingerprint_matches": True,
            "base_model_trainable_parameters_at_reconstruction": 0, "base_model_parameters": frozen_count,
            "evidence_boundary": "Source freezes base parameters, optimizer receives only adapter.parameters; four saved checkpoints contain adapter tensors only. Audit reconstructs the frozen graph and rechecks selected validation checkpoints; it does not claim a historical in-memory weight trace was retained.",
            "four_adapter_checkpoints": state_records, "selected_validation_recomputations": deltas,
            "descriptor_statistics_verified_training_only": verify_validation}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-validation-reconstruction", action="store_true")
    args = parser.parse_args()
    start = time.perf_counter()
    AUDIT.mkdir(parents=True, exist_ok=True)
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    manifest_path = ROOT / "phase2/data/vari/manifest.json"
    assert sha(manifest_path) == summary["configuration"]["manifest_sha256"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest == summary["dataset"]
    records, grouped, gt, native64, legacy, data_audit = audit_data(manifest)
    evaluation = [r for r in records if r["split"] == "evaluation"]
    expected_rows = {(method, r["id"]) for method in METHODS for r in evaluation}
    original_rows = read_csv(OUT / "metrics.csv")
    assert len(original_rows) == len(expected_rows) == 1584
    indexed_rows = {(r["method"], r["id"]): r for r in original_rows}
    assert set(indexed_rows) == expected_rows
    original_instability = {(r["method"], r["view_group"]): r for r in read_csv(OUT / "instability.csv")}
    assert len(original_instability) == 6 * 24
    saved_summary = {r["method"]: r for r in summary["results"]}
    recomputed, native_rows, legacy_rows, inst_rows = [], [], [], []
    max_metric_difference = {k: 0.0 for k in ["abs_rel", "rmse_m", "delta1", "valid_pixels"]}
    max_instability_difference = 0.0
    aggregates = []
    for method in METHODS:
        predictions = {}
        method_rows, method_native, method_legacy, method_instability = [], [], [], []
        folder = OUT / "predictions" / method
        assert len(list(folder.glob("*.npy"))) == 264
        for record in evaluation:
            path = folder / f"{record['id']}.npy"
            prediction = np.load(path)
            assert prediction.dtype == np.float32 and prediction.shape == (480, 640)
            predictions[record["id"]] = prediction
            scores = measure(prediction, gt[record["view_group"]])
            saved = indexed_rows[(method, record["id"])]
            assert str(saved["scene"]) == str(record["scene"]) and saved["view_group"] == record["view_group"] and saved["lighting"] == record["lighting"]
            for key in max_metric_difference:
                difference = abs(scores[key] - float(saved[key]))
                max_metric_difference[key] = max(max_metric_difference[key], difference)
                assert difference < 1e-12
            common = {"method": method, "id": record["id"], "view_group": record["view_group"], "lighting": record["lighting"]}
            method_rows.append(dict(common, **scores))
            method_native.append(dict(common, **measure(prediction, native64[record["view_group"]])))
            method_legacy.append(dict(common, **measure(prediction, legacy[record["view_group"]])))
        for group, rs in sorted(grouped.items()):
            if rs[0]["split"] != "evaluation":
                continue
            reference = gt[group].astype(np.float64)
            valid = np.isfinite(reference) & (reference > .1) & (reference < 9.99)
            assert len({r["depth"] for r in rs}) == 1
            # Independent float64 summation; saved code averages float32 stacks.
            estimates = np.stack([np.clip(predictions[r["id"]], .1, 10).astype(np.float64)[valid] for r in rs])
            center = estimates.sum(axis=0) / len(rs)
            absolute_dispersion = np.abs(estimates - center[None, :])
            score = float(np.sum(absolute_dispersion / reference[valid][None, :]) / absolute_dispersion.size)
            difference = abs(score - float(original_instability[(method, group)]["lighting_instability"]))
            max_instability_difference = max(max_instability_difference, difference)
            assert difference < 1e-7
            row = {"method": method, "view_group": group, "scene": "485", "n_lights": len(rs), "normalized_mean_absolute_lighting_deviation": score,
                   "saved_lighting_instability": float(original_instability[(method, group)]["lighting_instability"]), "difference_float64_vs_saved": difference}
            method_instability.append(row)
        result = {"method": method, "n_images": 264, "n_view_groups": 24,
                  **{key: float(np.mean([r[key] for r in method_rows])) for key in ["abs_rel", "rmse_m", "delta1"]},
                  "lighting_instability_normalized_mad": float(np.mean([r["normalized_mean_absolute_lighting_deviation"] for r in method_instability])),
                  "legacy_8bit": {key: float(np.mean([r[key] for r in method_legacy])) for key in ["abs_rel", "rmse_m", "delta1"]},
                  "raw_16bit_float64_decode": {key: float(np.mean([r[key] for r in method_native])) for key in ["abs_rel", "rmse_m", "delta1"]}}
        for key in ["abs_rel", "rmse_m", "delta1"]:
            assert abs(result[key] - saved_summary[method][key]) < 1e-12
        assert abs(result["lighting_instability_normalized_mad"] - saved_summary[method]["lighting_instability"]) < 1e-7
        aggregates.append(result)
        recomputed.extend(method_rows); native_rows.extend(method_native); legacy_rows.extend(method_legacy); inst_rows.extend(method_instability)
        print(f"Audited {method}:264predictions; AbsRel={result['abs_rel']:.9f}; legacy={result['legacy_8bit']['abs_rel']:.9f}", flush=True)
    model_audit = audit_models(summary, records, not args.skip_validation_reconstruction)
    ranking16 = sorted(aggregates, key=lambda r: r["abs_rel"])
    ranking8 = sorted(aggregates, key=lambda r: r["legacy_8bit"]["abs_rel"])
    lookup = {r["method"]: r for r in aggregates}
    assert lookup["task_adapter"]["abs_rel"] < lookup["cc_consistency"]["abs_rel"] < lookup["frozen"]["abs_rel"]
    assert lookup["task_adapter"]["legacy_8bit"]["abs_rel"] < lookup["cc_consistency"]["legacy_8bit"]["abs_rel"] < lookup["frozen"]["legacy_8bit"]["abs_rel"]
    report = {"audit": "PASS", "audit_code_sha256": sha(Path(__file__)), "audited_summary_sha256": sha(OUT / "summary.json"),
              "evaluation_protocol": {"frame": "full640x480", "valid_ground_truth": "finite and strictly0.1<depth<9.99meters", "prediction_clip_m": [.1, 10], "aggregation": "arithmeticmeanper-imagemetrics", "fitted_scale_shift": False},
              "dataset": data_audit, "numerical_checks": {"saved_float32_predictions": 1584, "metrics_exactly_match_rows": True,
                  "maximum_metric_difference": max_metric_difference, "lighting_group_checks": 144,
                  "lighting_instability_definition": "mean over11lights and validpixels of abs(prediction-lightmean)/fixedGT, then arithmeticmeanover24groups; dimensionlessMAD, notMSE",
                  "maximum_instability_difference_independent_float64": max_instability_difference},
              "results": aggregates, "models_and_source": model_audit,
              "decoder_sensitivity": {"all_six_methods_rescored_on_legacy_8bit": True,
                  "primary_absrel_ranking": [r["method"] for r in ranking16], "legacy_absrel_ranking": [r["method"] for r in ranking8],
                  "critical_frozen_task_cc_order_preserved": True},
              "interpretation": {"cc_specific_gain_proven": False,
                  "reason": "CCconsistencyimprovesoverfrozenbasebutordinarytaskadapterhaslowerAbsRel,lowerRMSEandlowerlightingMAD; domainadaptationcontrolisrequired. CCconsistencyhasdifferentdelta1tradeoff.",
                  "cc_vs_task_absrel_increase": lookup["cc_consistency"]["abs_rel"] - lookup["task_adapter"]["abs_rel"],
                  "cc_vs_task_absrel_relative_increase_pct": 100 * (lookup["cc_consistency"]["abs_rel"] / lookup["task_adapter"]["abs_rel"] - 1),
                  "cc_vs_task_lighting_mad_ratio": lookup["cc_consistency"]["lighting_instability_normalized_mad"] / lookup["task_adapter"]["lighting_instability_normalized_mad"],
                  "scope": "Oneheldoutroom,24correlatedviewgroups×11renderedlights; not264independentscenes orofficialVaribenchmark"},
              "wall_seconds": time.perf_counter() - start}
    write_csv(recomputed, AUDIT / "mono_metrics_recomputed.csv")
    write_csv(legacy_rows, AUDIT / "mono_metrics_legacy_8bit.csv")
    write_csv(native_rows, AUDIT / "mono_metrics_native16_float64.csv")
    write_csv(inst_rows, AUDIT / "mono_instability_recomputed.csv")
    content = json.dumps(report, indent=2)
    (AUDIT / "mono_independent_audit.json").write_text(content, encoding="utf-8")
    (OUT / "independent_audit.json").write_text(content, encoding="utf-8")
    print("PASS:1584predictions,144lightinggroups,48nativeGTs,4adaptercheckpoints; nooriginalrun/data/modelmodified.", flush=True)


if __name__ == "__main__":
    main()
