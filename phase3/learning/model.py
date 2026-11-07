"""Small task-internal illumination-conditioned depth adapter.

This is a research hypothesis using known feature-conditioning/low-rank ideas,
not a claim of established architectural novelty or measured improvement.
Inference receives RGB and a frozen model's predicted depth, never reference
depth. The geometry branch uses normalized virtual rays rather than calibrated
world coordinates; it must not be presented as a measured 3D point cloud.
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


MODES = ("ordinary", "achromatic", "content", "chromatic", "flat_geometry_chromatic", "geometry_chromatic")


class GeometryChromaticAdapter(nn.Module):
    """Shared task adaptation plus a bounded, cue-dependent low-rank residual.

    `aux['illumination']` is a Bx2 log-ratio estimate. Only the chromatic modes
    interpret its components as log(R/G), log(B/G). A differentiable global-color
    proxy supplies an initial anchor; the learned semantic/geometric residual
    can change it. Without absolute illuminant supervision this is not a
    calibrated illuminant estimate. The correction uses the separately bounded
    `aux['condition']`, not an unconstrained correction image.
    """

    def __init__(self, channels: int = 384, layers: int = 4, rank: int = 8,
                 hidden: int = 64, strength: float = 0.1,
                 constant_strength: float = 0.25):
        super().__init__()
        if min(channels, layers, rank, hidden) < 1:
            raise ValueError("Adapter dimensions must be positive")
        self.channels, self.layers, self.rank = channels, layers, rank
        self.strength, self.constant_strength = strength, constant_strength
        self.task_gamma = nn.Parameter(torch.zeros(layers, channels))
        self.task_beta = nn.Parameter(torch.zeros(layers, channels))
        # 384 semantic channels + 12 photometric statistics at the defaults.
        self.context_encoder = nn.Sequential(
            nn.Linear(channels + 12, hidden), nn.GELU())
        self.illumination_head = nn.Linear(hidden, 2)
        self.geometry_encoder = nn.Sequential(
            nn.Linear(6, 32), nn.GELU(), nn.Linear(32, hidden), nn.GELU())
        self.geometry_projection = nn.Linear(hidden, hidden, bias=False)
        self.reliability_gate = nn.Linear(hidden, 1)
        self.condition_projection = nn.Linear(2, rank, bias=False)
        self.down = nn.ModuleList(nn.Linear(channels, rank, bias=False)
                                  for _ in range(layers))
        self.up = nn.ModuleList(nn.Linear(rank, channels, bias=False)
                                for _ in range(layers))
        for layer in self.up:
            nn.init.zeros_(layer.weight)
        # Nonzero cue initialization avoids a dead product of two zero paths,
        # especially for the semantic-only control. Output remains identity
        # because every final residual projection is zero-initialized.
        nn.init.normal_(self.illumination_head.weight, std=0.02)
        nn.init.zeros_(self.illumination_head.bias)
        nn.init.zeros_(self.reliability_gate.weight)
        nn.init.constant_(self.reliability_gate.bias, -1.0)
        self.register_buffer("luma_weights", torch.tensor([.2126, .7152, .0722]))

    def _photo(self, rgb: Tensor, achromatic: bool = False):
        # A small, deterministic pooling grid keeps repeated CPU training cheap.
        small = F.adaptive_avg_pool2d(rgb, (min(32, rgb.shape[-2]),
                                         min(32, rgb.shape[-1])))
        flat = small.flatten(2)
        means = flat.mean(-1).clamp_min(1e-5)
        stds = flat.std(-1, unbiased=False).clamp_min(1e-5)
        luma = (flat * self.luma_weights[None, :, None]).sum(1)
        quantiles = torch.quantile(luma, luma.new_tensor([.1, .5, .9]),
                                  dim=-1).transpose(0, 1).clamp_min(1e-5)
        saturated = (flat >= .98).to(flat.dtype).mean(-1)
        log_luma = luma.mean(-1).clamp_min(1e-5).log()
        if achromatic:
            # The explicit photo descriptor supplies brightness information only.
            log_means = log_luma[:, None].expand(-1, 3)
            log_stds = luma.std(-1, unbiased=False).clamp_min(1e-5).log()[:, None].expand(-1, 3)
            saturated = (luma >= .98).to(flat.dtype).mean(-1)[:, None].expand(-1, 3)
        else:
            log_means, log_stds = means.log(), stds.log()
        photo = torch.cat((log_means, log_stds, quantiles.log(), saturated), -1)
        log_means_original = means.log()
        ratios = torch.stack((log_means_original[:, 0] - log_means_original[:, 1],
                              log_means_original[:, 2] - log_means_original[:, 1]), -1)
        return photo, ratios, log_luma

    def _geometry_context(self, rgb: Tensor, predicted_depth: Tensor,
                          patch_hw: tuple[int, int]):
        height, width = patch_hw
        color = F.adaptive_avg_pool2d(rgb, patch_hw).flatten(2).transpose(1, 2)
        z = F.interpolate(predicted_depth[:, None], size=patch_hw,
                          mode="bilinear", align_corners=False)[:, 0]
        median = z.flatten(1).median(1).values.clamp_min(1e-3)
        z = (z / median[:, None, None]).clamp(.05, 20.)
        yy, xx = torch.meshgrid(
            torch.linspace(-1., 1., height, device=z.device, dtype=z.dtype),
            torch.linspace(-1., 1., width, device=z.device, dtype=z.dtype),
            indexing="ij")
        xyz = torch.stack((xx[None] * z, yy[None] * z, z), -1).flatten(1, 2)
        points = torch.cat((color, xyz), -1)
        encoded = self.geometry_encoder(points)
        # Fixed-size PointNet-like pooling, not measured camera/world geometry.
        pooled = .5 * (encoded.mean(1) + encoded.max(1).values)
        return self.geometry_projection(pooled)

    def forward(self, features: Sequence[tuple[Tensor, Tensor]], rgb_linear: Tensor,
                base_depth: Tensor, patch_hw: tuple[int, int], mode: str):
        if mode not in MODES:
            raise ValueError(f"Unknown adapter mode: {mode}")
        if len(features) != self.layers:
            raise ValueError("Unexpected number of feature layers")
        height, width = (int(v) for v in patch_hw)
        if min(height, width) < 1:
            raise ValueError("Patch dimensions must be positive")
        first = features[0][0]
        if first.ndim != 3:
            raise ValueError("Feature tokens must have shape B,N,C")
        batch, tokens, channels = first.shape
        if channels != self.channels or tokens != height * width:
            raise ValueError("Feature dimensions disagree with adapter/patch grid")
        for token, cls in features:
            if token.shape != first.shape or cls.shape != (batch, channels):
                raise ValueError("All feature and class-token shapes must agree")
        if rgb_linear.ndim != 4 or rgb_linear.shape[:2] != (batch, 3):
            raise ValueError("RGB must have shape B,3,H,W")
        if base_depth.ndim != 3 or base_depth.shape[0] != batch:
            raise ValueError("Frozen predicted depth must have shape B,H,W")
        rgb = rgb_linear.detach().to(device=first.device, dtype=first.dtype).clamp(0., 1.)
        predicted_depth = base_depth.detach().to(device=first.device, dtype=first.dtype)
        if not torch.isfinite(rgb).all() or not torch.isfinite(predicted_depth).all():
            raise ValueError("RGB/predicted depth contains a non-finite value")
        if (predicted_depth <= 0).any():
            raise ValueError("Frozen predicted depth must be strictly positive")
        semantic = torch.stack([
            F.layer_norm(token, (channels,)).mean(1) for token, _ in features
        ]).mean(0)
        photo, ratios, log_luma = self._photo(rgb, achromatic=mode == "achromatic")
        if mode in ("ordinary", "content"):
            photo = torch.zeros_like(photo)
        context = self.context_encoder(torch.cat((semantic, photo), -1))
        if mode in ("flat_geometry_chromatic", "geometry_chromatic"):
            geometry_depth = (predicted_depth if mode == "geometry_chromatic"
                              else torch.ones_like(predicted_depth))
            context = context + self._geometry_context(rgb, geometry_depth, (height, width))
        if mode == "ordinary":
            illumination = first.new_zeros(batch, 2)
        elif mode == "achromatic":
            illumination = log_luma[:, None].expand(-1, 2)
        elif mode == "content":
            illumination = self.illumination_head(context)
        else:
            illumination = ratios + self.illumination_head(context)
        condition = .75 * torch.tanh(illumination)
        modulation = torch.tanh(self.condition_projection(condition))[:, None]
        gate = torch.sigmoid(self.reliability_gate(context))[:, None]
        gamma = self.constant_strength * torch.tanh(self.task_gamma)
        beta = self.constant_strength * torch.tanh(self.task_beta)
        adapted, penalties = [], [gamma.square().mean(), beta.square().mean()]
        for index, (token, cls) in enumerate(features):
            scale = token.detach().std(dim=(1, 2), unbiased=False, keepdim=True).clamp_min(.1)
            normalized = F.layer_norm(token, (channels,))
            residual = torch.tanh(self.up[index](self.down[index](normalized) * modulation))
            new = token * (1 + gamma[index][None, None]) + beta[index][None, None] * scale
            new = new + self.strength * gate * residual * scale
            adapted.append((new, cls))
            penalties.append(residual.square().mean())
        auxiliary = {
            "illumination": illumination,
            "condition": condition,
            "regularization": torch.stack(penalties).mean(),
            "gate": gate[:, 0, 0],
            "geometry_source": ("predicted_frozen_depth_normalized_virtual_rays"
                                if mode == "geometry_chromatic" else
                                "constant_depth_virtual_rays" if mode == "flat_geometry_chromatic" else "disabled"),
        }
        return adapted, auxiliary
