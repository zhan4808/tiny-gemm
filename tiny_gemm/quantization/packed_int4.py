import torch


def quantize_per_tensor_int4(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-tensor INT4 quantization (signed, -8..7)."""
    max_abs = x.abs().max()
    scale = max_abs / 7.0 if max_abs > 0 else torch.tensor(1.0, device=x.device)
    q = torch.clamp(torch.round(x / scale), -8, 7).to(torch.int8)
    return q, scale


def dequantize_int4(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.to(torch.float32) * scale


def pack_int4_signed(q: torch.Tensor, axis: int = -1) -> torch.Tensor:
    """Pack signed int4 values (-8..7) into uint8, 2 per byte along axis."""
    if q.dtype != torch.int8:
        raise ValueError("q must be int8 with values in [-8, 7]")
    axis = axis % q.ndim
    if q.shape[axis] % 2 != 0:
        pad_shape = list(q.shape)
        pad_shape[axis] = 1
        q = torch.cat([q, torch.zeros(pad_shape, device=q.device, dtype=q.dtype)], dim=axis)
    q_u = (q & 0x0F).to(torch.uint8)
    q_lo = q_u.index_select(axis, torch.arange(0, q_u.shape[axis], 2, device=q.device))
    q_hi = q_u.index_select(axis, torch.arange(1, q_u.shape[axis], 2, device=q.device))
    packed = q_lo | (q_hi << 4)
    return packed.contiguous()


def unpack_int4_signed(
    packed: torch.Tensor, original_shape: torch.Size, axis: int = -1
) -> torch.Tensor:
    """Unpack uint8 into signed int4 values (-8..7) along axis."""
    if packed.dtype != torch.uint8:
        raise ValueError("packed must be uint8")
    axis = axis % packed.ndim
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    q = torch.stack([lo, hi], dim=axis + 1)
    new_shape = list(packed.shape)
    new_shape[axis] = new_shape[axis] * 2
    q = q.reshape(new_shape).to(torch.int8)
    q = torch.where(q >= 8, q - 16, q)
    return q.reshape(original_shape)
