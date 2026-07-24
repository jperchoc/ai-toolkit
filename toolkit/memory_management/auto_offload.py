"""Automatic layer-offload sizing.

Picks a `layer_offloading_*_percent` from the actual VRAM budget and the module's
own weight footprint, so users don't have to trial-and-error the value to avoid
OOM. The estimate is deliberately conservative (biased toward offloading a bit
more) and is only ever a starting point — it is logged and can be overridden, and
the training loop can raise it further on OOM.
"""
import math

import torch


def _module_weight_bytes(module: torch.nn.Module) -> int:
    """Total bytes of a module's weights, counting quantized buffers too."""
    seen = set()
    total = 0
    for t in list(module.parameters()) + list(module.buffers()):
        if t is None or id(t) in seen:
            continue
        seen.add(id(t))
        total += t.numel() * t.element_size()
    return total


def compute_offload_percent(
    module: torch.nn.Module,
    device: torch.device,
    reserved_gb: float = 4.0,
    safety: float = 0.9,
    granularity: float = 0.05,
    label: str = "transformer",
) -> float:
    """Fraction of `module`'s layers to offload so its resident weights fit the
    VRAM budget. 0.0 = keep everything on GPU, 1.0 = stream everything from CPU.

    budget = total_vram * safety - reserved_gb   (reserved covers activations,
    the trained adapter's grads/optimizer state, VAE/TE, and CUDA context).
    """
    if not torch.cuda.is_available() or device is None or torch.device(device).type != "cuda":
        # nothing to offload against; keep the safe default (full offload)
        return 1.0

    props = torch.cuda.get_device_properties(device)
    total = props.total_memory
    try:
        free, _ = torch.cuda.mem_get_info(device)
    except Exception:
        free = total

    weight_bytes = _module_weight_bytes(module)
    budget = total * safety - reserved_gb * (1024 ** 3)

    if weight_bytes <= 0:
        percent = 1.0
    elif budget <= 0:
        percent = 1.0
    elif weight_bytes <= budget:
        percent = 0.0
    else:
        percent = 1.0 - (budget / weight_bytes)

    # bias conservative: round UP to the granularity, then clamp
    percent = math.ceil(percent / granularity) * granularity
    percent = min(1.0, max(0.0, percent))

    gb = 1024 ** 3
    print(
        f"[auto-offload] {label}: vram total={total/gb:.1f}GB free={free/gb:.1f}GB, "
        f"weights={weight_bytes/gb:.1f}GB, reserved={reserved_gb:.1f}GB "
        f"-> offloading {percent*100:.0f}%"
    )
    return percent
