"""Candidate RGB/geometry fusion for illuminant estimation; no quality oracle."""
import torch
from torch import nn
from torch.nn import functional as F

MODES = ('rgb', 'ordinary', 'mask', 'learned')


class OriginalPCCC(nn.Module):
    """Pinned original network under the same worker/splits/loss as controls."""
    def __init__(self):
        super().__init__()
        import importlib.util
        from pathlib import Path
        path = Path(__file__).resolve().parents[1] / 'phase5_pccc/model.py'
        spec = importlib.util.spec_from_file_location('fusion_original_pccc', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        author, _, _ = module.load()
        self.net = module.make(author, 'released', 'cpu')

    def forward(self, cloud):
        points, _ = self.net(cloud)
        return dict(illuminant=points.sum(-1), gate=torch.ones_like(cloud[:, :1]))

    def capacity(self):
        n = sum(p.numel() for p in self.parameters())
        return dict(allocated=n, active=n)


def make_model(mode):
    return OriginalPCCC() if mode == 'pccc' else ReliabilityFusion(mode)


def reliability_cues(xyz, neighbors=8):
    """Input-only mask, valid fraction and local log-depth disagreement.

    Ray-nearest neighbors require XYZ with camera Z, but not ordered points.
    Depth discontinuities can also produce disagreement: these are cues, not
    measured reliability labels. Invalid/subthreshold coordinates are excluded.
    """
    valid = xyz[:, 2:3].abs() > 1e-8
    safe_z = torch.where(valid, xyz[:, 2:3], torch.ones_like(xyz[:, 2:3]))
    rays = (xyz[:, :2] / safe_z).transpose(1, 2)
    distance = torch.cdist(rays, rays, compute_mode='donot_use_mm_for_euclid_dist')
    count = xyz.shape[-1]
    blocked = ~valid.squeeze(1)
    distance = distance.masked_fill(blocked[:, None, :], float('inf'))
    distance = distance.masked_fill(torch.eye(count, device=xyz.device, dtype=torch.bool)[None], float('inf'))
    k = min(neighbors, count - 1)
    if k < 1:
        raise ValueError('Need at least two points')
    cutoff = distance.topk(k, dim=-1, largest=False).values[:, :, -1:]
    log_z = safe_z.abs().log().squeeze(1)
    # Include all neighbors tied at the kth distance. Index-based tie breaking
    # on a regular ray grid would make the answer depend on point ordering.
    support = torch.isfinite(distance) & (distance <= cutoff) & valid.squeeze(1)[:, :, None]
    delta = (log_z[:, None, :] - log_z[:, :, None]).abs()
    delta = torch.where(support, delta, torch.zeros_like(delta))
    denom = support.sum(-1).clamp_min(1)
    mean = delta.sum(-1) / denom
    rms = (delta.square().sum(-1) / denom).sqrt()
    fraction = valid.float().mean(-1, keepdim=True).expand_as(valid)
    return torch.cat([valid.float(), fraction, mean[:, None], rms[:, None]], dim=1), valid


class PointBranch(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv1d(3, width, 1), nn.ReLU(),
                                    nn.Conv1d(width, width, 1), nn.ReLU())

    def forward(self, x):
        return self.layers(x)


class ReliabilityFusion(nn.Module):
    """PointNet-style branches, scalar per-point geometry gate, common head.

    All modes allocate identical modules; report active and allocated capacity
    separately. No branch feature depends on another point before gating.
    """
    def __init__(self, mode='learned', width=64):
        super().__init__()
        if mode not in MODES:
            raise ValueError(mode)
        self.mode = mode
        self.color = PointBranch(width)
        self.geometry = PointBranch(width)
        self.gate = nn.Sequential(nn.Conv1d(4, 16, 1), nn.ReLU(), nn.Conv1d(16, 1, 1))
        self.head = nn.Sequential(nn.Conv1d(width * 3, 128, 1), nn.ReLU(), nn.Conv1d(128, 3, 1))

    def forward(self, cloud):
        if cloud.ndim != 3 or cloud.shape[1] != 6 or not torch.isfinite(cloud).all():
            raise ValueError('Expected finite B x 6 x N XYZRGB')
        xyz, rgb = cloud[:, :3], cloud[:, 3:]
        mask = xyz[:, 2:3].abs() > 1e-8
        clean_xyz = torch.where(mask, xyz, torch.zeros_like(xyz))
        color = self.color(rgb)
        if self.mode == 'rgb':
            weight = torch.zeros_like(xyz[:, :1])
            fused = color
        else:
            geometry = self.geometry(clean_xyz)
            if self.mode == 'ordinary':
                weight = torch.ones_like(xyz[:, :1])
            elif self.mode == 'mask':
                weight = mask.to(xyz.dtype)
            else:
                cues, _ = reliability_cues(xyz.detach())
                weight = mask.to(xyz.dtype) * torch.sigmoid(self.gate(cues))
            fused = color + weight * geometry
        context = torch.cat([fused.mean(-1, keepdim=True), fused.amax(-1, keepdim=True)], dim=1)
        local = F.softplus(self.head(torch.cat([fused, context.expand(-1, -1, fused.shape[-1])], dim=1))) + 1e-6
        return dict(illuminant=local.mean(-1), gate=weight, local_illuminant=local)

    def capacity(self):
        allocated = sum(p.numel() for p in self.parameters())
        inactive = sum(p.numel() for p in self.gate.parameters()) if self.mode != 'learned' else 0
        if self.mode == 'rgb':
            inactive += sum(p.numel() for p in self.geometry.parameters())
        return dict(allocated=allocated, active=allocated-inactive)


def angular_loss(prediction, label):
    cosine = (F.normalize(prediction, dim=-1) * F.normalize(label, dim=-1)).sum(-1)
    return torch.rad2deg(torch.acos(cosine.clamp(-.999999, .999999))).mean()


def objective(clean, damaged, label, consistency=0.1):
    supervised = (angular_loss(clean['illuminant'], label) + angular_loss(damaged['illuminant'], label)) / 2
    agreement = (F.normalize(clean['illuminant'], dim=-1) - F.normalize(damaged['illuminant'], dim=-1)).square().sum(-1).mean()
    return supervised + consistency * agreement
