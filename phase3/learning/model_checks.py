"""Meaningful identity, trainability and inference-input checks for the adapter."""
from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import GeometryChromaticAdapter, MODES


def main():
    torch.manual_seed(20261107)
    torch.set_num_threads(2)
    batch, height, width, channels = 2, 8, 10, 384
    features = [(torch.randn(batch, height * width, channels),
                 torch.randn(batch, channels)) for _ in range(4)]
    rgb = torch.rand(batch, 3, 112, 140).clamp(.02, .95)
    # This is a synthetic positive stand-in for a frozen prediction, not GT.
    predicted_depth = .5 + torch.rand(batch, 112, 140) * 4
    model = GeometryChromaticAdapter()
    count = sum(p.numel() for p in model.parameters())
    checks = {}
    for mode in MODES:
        outputs, aux = model(features, rgb, predicted_depth, (height, width), mode)
        assert all(torch.equal(out[0], source[0]) for out, source in zip(outputs, features))
        assert all(out[1] is source[1] for out, source in zip(outputs, features))
        assert aux["illumination"].shape == (batch, 2)
        assert aux["regularization"].ndim == 0
        assert aux["condition"].abs().max() <= .75
        assert torch.isfinite(aux["illumination"]).all()
        checks[mode] = {"initial_identity": True, "class_tokens_preserved": True}

    # A scalar downstream proxy supplies gradients without loading a depth model.
    weights = [torch.randn_like(f[0]) for f in features]
    for mode in MODES:
        current = GeometryChromaticAdapter()
        optimizer = torch.optim.AdamW(current.parameters(), lr=.002)
        norms = {}
        for step in range(4):
            outputs, aux = current(features, rgb, predicted_depth, (height, width), mode)
            loss = sum((out[0] * weight).mean() for out, weight in zip(outputs, weights))
            if mode in ("chromatic", "geometry_chromatic"):
                loss = loss + .01 * (aux["illumination"] - .2).square().mean()
            optimizer.zero_grad(); loss.backward()
            for name, parameter in current.named_parameters():
                if parameter.grad is not None:
                    assert torch.isfinite(parameter.grad).all(), name
                    norms[name] = max(norms.get(name, 0.), float(parameter.grad.norm()))
            optimizer.step()
        assert norms.get("task_gamma", 0.) > 0 and norms.get("task_beta", 0.) > 0
        if mode != "ordinary":
            assert any(norms.get(f"up.{i}.weight", 0.) > 0 for i in range(4))
            assert any(norms.get(f"down.{i}.weight", 0.) > 0 for i in range(4))
        if mode == "geometry_chromatic":
            assert norms.get("geometry_encoder.0.weight", 0.) > 0
        checks[mode]["constant_adapter_gradients"] = True
        checks[mode]["residual_gradients"] = mode != "ordinary"
        if mode == "geometry_chromatic": checks[mode]["geometry_branch_gradients"] = True

    context1 = model._geometry_context(rgb, predicted_depth, (height, width))
    context2 = model._geometry_context(rgb, predicted_depth * 3, (height, width))
    assert torch.allclose(context1, context2, atol=1e-6, rtol=1e-5)
    signature = inspect.signature(GeometryChromaticAdapter.forward)
    assert tuple(signature.parameters) == ("self", "features", "rgb_linear", "base_depth", "patch_hw", "mode")
    assert all(not tensor.requires_grad for feature in features for tensor in feature)
    print(json.dumps({"adapter_checks": "PASS", "trainable_parameters": count,
                      "modes": checks, "pseudo_geometry_scale_invariance": True,
                      "reference_depth_or_intrinsics_argument": False,
                      "note": "Synthetic module selfcheck; not an experiment result or novelty proof"}, indent=2))


if __name__ == "__main__": main()
