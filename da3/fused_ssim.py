from __future__ import annotations

import torch
import torch.nn.functional as F


def _avg_pool(image: torch.Tensor, padding: int) -> torch.Tensor:
    return F.avg_pool2d(image, kernel_size=3, stride=1, padding=padding)


def fused_ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    padding: str = "same",
    data_range: float = 1.0,
) -> torch.Tensor:
    if x.shape != y.shape:
        raise ValueError(f"SSIM expects identical shapes, got {x.shape} and {y.shape}")
    if x.ndim != 4:
        raise ValueError(f"SSIM expects BCHW tensors, got ndim={x.ndim}")

    pad = 0 if padding == "valid" else 1
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    mu_x = _avg_pool(x, pad)
    mu_y = _avg_pool(y, pad)
    sigma_x = _avg_pool(x * x, pad) - mu_x * mu_x
    sigma_y = _avg_pool(y * y, pad) - mu_y * mu_y
    sigma_xy = _avg_pool(x * y, pad) - mu_x * mu_y

    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    ssim_map = numerator / torch.clamp(denominator, min=1e-8)
    return ssim_map.mean()
