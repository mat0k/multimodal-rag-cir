import torch


def _normalize(features: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(features, p=2, dim=-1)


def _slerp(start: torch.Tensor, end: torch.Tensor, alpha: float) -> torch.Tensor:
    # Spherical interpolation is numerically stable for multimodal embedding fusion.
    start_n = _normalize(start)
    end_n = _normalize(end)
    dot = torch.clamp(torch.sum(start_n * end_n, dim=-1, keepdim=True), -0.9995, 0.9995)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)

    near_linear = sin_omega.abs() < 1e-6
    blended = (torch.sin((1.0 - alpha) * omega) / sin_omega) * start_n + (torch.sin(alpha * omega) / sin_omega) * end_n
    linear = (1.0 - alpha) * start_n + alpha * end_n
    return torch.where(near_linear, linear, blended)


def fusion(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    fusion_type: str = "sum",
    alpha: float = 0.5,
) -> torch.Tensor:
    if image_features.shape != text_features.shape:
        raise ValueError(
            f"Fusion expects same shape tensors. Got image={image_features.shape}, text={text_features.shape}."
        )

    if fusion_type == "sum":
        fused = image_features + text_features
    elif fusion_type == "mean":
        fused = (image_features + text_features) * 0.5
    elif fusion_type == "slerp":
        fused = _slerp(image_features, text_features, alpha=alpha)
    else:
        raise ValueError(f"Unsupported fusion_type: {fusion_type}")

    return _normalize(fused)
