import contextlib

import torch

PRECISION_MAP = {
    None: torch.float32,
    32: torch.float32,
    16: torch.float16,
    torch.float32: torch.float32,
    torch.float16: torch.float16,
    torch.bfloat16: torch.bfloat16,
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float": torch.float32,
    "half": torch.float16,
    "32": torch.float32,
    "16": torch.float16,
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def get_autocast_context(
    device: str | torch.device, precision: str | int | torch.dtype
) -> contextlib.AbstractContextManager:
    # fucking autocast only wants pure strings like 'cpu' or 'cuda'
    # we need to convert the model device to that
    device_type_for_autocast = str(device).split(":")[0]

    # autocast doesn't work with CPU and stuff different from bfloat16
    autocast_manager = (
        contextlib.nullcontext()
        if device_type_for_autocast in ["cpu", "mps"]
        and PRECISION_MAP[precision] != torch.bfloat16
        else (
            torch.autocast(
                device_type=device_type_for_autocast,
                dtype=PRECISION_MAP[precision],
            )
        )
    )
    return autocast_manager
