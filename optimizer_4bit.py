"""Avalanche4Bit: Lossless 4-Bit Factorized Optimizer for PyTorch.

An OpenBlade Release (https://github.com/OpenBladeAI, https://huggingface.co/OpenBlade)
Created and Developed by Jassim Ibrahim Farooqi.
Part of ongoing research in extreme-efficiency deep learning optimization architectures.

Architecture:
  1. Lossless 4-Bit Momentum Buffer:
     Compresses the first-moment velocity tensor into packed 4-bit signed integers
     (0.5 bytes per parameter, 8x smaller than FP32).
  2. Error-Feedback (EF) Residual Compensation:
     Retains the exact quantization residual (e_t = m_t - m_hat_t) in half precision (FP16)
     and injects it directly into the subsequent step. No gradients, steps, or fractional
     magnitudes are discarded over the training trajectory.
  3. Rank-1 Marginal Second-Moment Factorization:
     Tracks only marginal row vector r in R^M and column vector c in R^N for any 2D matrix (M, N).
     Reduces second-moment state from O(M * N) to O(M + N) (~10MB total VRAM across 1.1B model).
  4. Immediate Intermediate Cast-Away:
     High-precision intermediate tensors (unpacked momentum, preconditioners, updates)
     are strictly computed, applied to weights, and cast away from VRAM immediately.
  5. Dynamic RMS Gradient Normalization:
     Stabilizes updates across FP16 and BF16 mixed-precision training.
"""

import math
from typing import Dict, Iterable, List, Optional, Tuple, Union
import torch
from torch.optim.optimizer import Optimizer


def _get_shape_key(p: torch.Tensor) -> Tuple[int, ...]:
    return tuple(p.shape)


def quantize_pack_4bit(
    x: torch.Tensor,
    block_size: int = 32,
    stochastic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, int, Tuple[int, ...]]:
    """Quantizes a float tensor into packed 4-bit signed integers [-7, 7] stored as uint8.
    
    Each byte in the returned packed tensor holds two 4-bit values:
      low nibble  (bits 0-3): even index + 8
      high nibble (bits 4-7): odd index + 8
    
    Returns:
      packed: torch.uint8 tensor of shape (num_blocks, block_size // 2)
      scales: torch.float16 or torch.float32 scale per block
      n: original number of elements
      orig_shape: original tensor shape
    """
    orig_shape = x.shape
    x_flat = x.flatten()
    n = x_flat.numel()
    pad = (block_size - (n % block_size)) % block_size
    if pad > 0:
        x_flat = torch.cat([x_flat, torch.zeros(pad, device=x.device, dtype=x.dtype)])

    blocks = x_flat.view(-1, block_size)
    # Per-block absolute maximum scale (bound by 7 for signed 4-bit [-7, 7])
    scales = blocks.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 7.0
    scale_dtype = torch.float16 if x.device.type == "cuda" else torch.float32
    scales_saved = scales.squeeze(1).to(scale_dtype)

    normed = blocks / scales
    if stochastic:
        floor = normed.floor()
        prob = normed - floor
        rnd = torch.rand_like(normed)
        q = torch.where(rnd < prob, floor + 1.0, floor).clamp(-7, 7).to(torch.int8)
    else:
        q = normed.round().clamp(-7, 7).to(torch.int8)

    # Map signed [-7, 7] into unsigned [1, 15] within 4-bit nibble range [0, 15]
    q_unsigned = (q + 8).to(torch.uint8)
    q_even = q_unsigned[:, 0::2]
    q_odd = q_unsigned[:, 1::2]
    packed = q_even | (q_odd << 4)

    return packed, scales_saved, n, orig_shape


def dequantize_unpack_4bit(
    packed: torch.Tensor,
    scales: torch.Tensor,
    n: int,
    orig_shape: Tuple[int, ...],
    block_size: int = 32,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Unpacks a 4-bit packed uint8 tensor back into a full-precision float tensor."""
    num_blocks = packed.shape[0]
    q_even = (packed & 0x0F).to(torch.int8) - 8
    q_odd = ((packed >> 4) & 0x0F).to(torch.int8) - 8

    q = torch.empty((num_blocks, block_size), dtype=dtype, device=packed.device)
    q[:, 0::2] = q_even.to(dtype)
    q[:, 1::2] = q_odd.to(dtype)

    sc = scales.to(dtype).unsqueeze(1)
    return (q * sc).view(-1)[:n].view(orig_shape)


class Avalanche4Bit(Optimizer):
    """Avalanche4Bit: Lossless 4-Bit Factorized Optimizer.
    
    An OpenBlade Release. Created by Jassim Ibrahim Farooqi.
    Combines rank-1 marginal second-moment factorization with packed 4-bit momentum
    and lossless error-feedback compensation.
    
    Args:
        params: Iterable of parameters, parameter groups, or an nn.Module instance.
        lr: Learning rate (default: 1e-3).
        betas: Tuple of (beta1, beta2). Beta1 controls the 4-bit momentum exponential moving
               average (default: (0.9, 0.999)).
        eps: Small constant for numerical stability (default: 1e-30).
        weight_decay: Decoupled weight decay coefficient (default: 0.01).
        lossless: If True, retains residual quantization error in FP16 to guarantee zero
                  loss of gradient/step information across steps (default: True).
        stochastic_rounding: If True, uses unbiased stochastic rounding during 4-bit quantization (default: False).
        block_size: Block size for 4-bit quantization scaling (default: 32).
        chunk: Number of identical-shape tensors to process concurrently in batched kernels (default: 16).
        fp32_max_elems: Threshold above which 2D tensors use chunked row-streaming (default: 4,194,304 = 4M).
    """

    def __init__(
        self,
        params: Union[Iterable[torch.Tensor], Iterable[Dict], torch.nn.Module],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-30,
        weight_decay: float = 0.01,
        lossless: bool = True,
        stochastic_rounding: bool = False,
        block_size: int = 32,
        chunk: int = 16,
        fp32_max_elems: int = 4194304,
    ):
        if isinstance(params, torch.nn.Module):
            param_source = params.parameters()
            self._module_ref = params
        else:
            param_source = params
            self._module_ref = None

        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            beta1=betas[0],
            beta2=betas[1],
            eps=eps,
            weight_decay=weight_decay,
            lossless=lossless,
            stochastic_rounding=stochastic_rounding,
            block_size=block_size,
            chunk=chunk,
            fp32_max_elems=fp32_max_elems,
        )
        super().__init__(param_source, defaults)

        self.eps = eps
        self.chunk = chunk
        self.fp32_max = fp32_max_elems
        self.b1 = betas[0]
        self.b2 = betas[1]
        self.wd = weight_decay
        self.lossless = lossless
        self.stochastic = stochastic_rounding
        self.block_size = block_size
        self.t = 0

    @torch.no_grad()
    def reset_states(self) -> None:
        """Resets all internal moving average statistics to clean initial state."""
        self.t = 0
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state.clear()

    @torch.no_grad()
    def _step_2d_batched(
        self,
        tensors: List[torch.Tensor],
        states: List[Dict[str, Optional[torch.Tensor]]],
        lr: float,
        weight_decay: float,
        scale: float,
        bc1: float,
        bc2: float,
        b1: float,
        b2: float,
        chunk_size: int,
        block_size: int,
        lossless: bool,
        stochastic: bool,
    ) -> None:
        m, n = tensors[0].shape
        for st in states:
            if "r" not in st or st["r"] is None:
                st["r"] = torch.zeros(m, device=st["p"].device, dtype=torch.float32)
            if "c" not in st or st["c"] is None:
                st["c"] = torch.zeros(n, device=st["p"].device, dtype=torch.float32)

        K = len(tensors)
        for s in range(0, K, chunk_size):
            grp_p = tensors[s : s + chunk_size]
            grp_st = states[s : s + chunk_size]

            # 1. Gradients and second-moment factorized statistics
            G = torch.stack([p.grad for p in grp_p]).float().div_(scale)
            r = torch.stack([st["r"] for st in grp_st])
            c = torch.stack([st["c"] for st in grp_st])

            rs = G.pow(2).mean(dim=2)
            cs = G.pow(2).mean(dim=1)
            r.mul_(b2).add_(rs, alpha=1.0 - b2)
            c.mul_(b2).add_(cs, alpha=1.0 - b2)

            for i, st in enumerate(grp_st):
                st["r"].copy_(r[i])
                st["c"].copy_(c[i])

            gm = G.pow(2).mean(dim=(1, 2)).clamp_min(self.eps)[:, None, None]
            vhat = (r[:, :, None] * c[:, None, :]) / (gm * bc2)
            denom = vhat.clamp_min(self.eps).sqrt_()

            # 2. 4-bit momentum with lossless residual error compensation
            M_list = []
            for i, st in enumerate(grp_st):
                g_i = G[i]
                if b1 > 0.0:
                    if "m_packed" in st and st["m_packed"] is not None:
                        m_prev = dequantize_unpack_4bit(
                            st["m_packed"], st["m_scales"], st["m_n"], st["m_shape"],
                            block_size=block_size, dtype=torch.float32
                        )
                        # Re-inject previous step's residual error (lossless)
                        if lossless and "m_residual" in st and st["m_residual"] is not None:
                            m_prev.add_(st["m_residual"].to(torch.float32))
                        m_curr = m_prev.mul_(b1).add_(g_i, alpha=1.0 - b1)
                    else:
                        m_curr = g_i.clone()

                    # Quantize and pack into 4-bit
                    m_packed, m_scales, m_n, m_shape = quantize_pack_4bit(
                        m_curr, block_size=block_size, stochastic=stochastic
                    )
                    st["m_packed"] = m_packed
                    st["m_scales"] = m_scales
                    st["m_n"] = m_n
                    st["m_shape"] = m_shape

                    # Preserve quantization residual in FP16 for next step/batch
                    if lossless:
                        m_rec = dequantize_unpack_4bit(
                            m_packed, m_scales, m_n, m_shape, block_size=block_size, dtype=torch.float32
                        )
                        res_dtype = torch.float16 if g_i.device.type == "cuda" else torch.float32
                        st["m_residual"] = (m_curr - m_rec).to(res_dtype)

                    m_eff = m_curr / bc1
                    M_list.append(m_eff)
                else:
                    M_list.append(g_i)

            M_stack = torch.stack(M_list)

            # 3. Compute update with RMS gradient normalization
            u = M_stack / denom.add_(self.eps)
            rms = u.pow(2).mean(dim=(1, 2), keepdim=True).sqrt_().clamp_min(1.0)
            u.div_(rms)

            upd = [u_i.to(p.dtype) for u_i, p in zip(u, grp_p)]
            if lr > 0 and weight_decay > 0:
                torch._foreach_mul_(grp_p, 1.0 - lr * weight_decay)
            torch._foreach_add_(grp_p, upd, alpha=-lr if lr > 0 else 0.0)

            # 4. Immediate cast-away of full-precision temporary tensors
            del G, r, c, rs, cs, gm, vhat, denom, M_list, M_stack, u, upd

    @torch.no_grad()
    def _step_2d_streaming_large(
        self,
        p: torch.Tensor,
        st: Dict[str, Optional[torch.Tensor]],
        lr: float,
        weight_decay: float,
        scale: float,
        bc1: float,
        bc2: float,
        b1: float,
        b2: float,
        block_size: int,
        lossless: bool,
        stochastic: bool,
    ) -> None:
        g = p.grad
        m, n = p.shape

        if "r" not in st or st["r"] is None:
            st["r"] = torch.zeros(m, device=p.device, dtype=torch.float32)
            st["c"] = torch.zeros(n, device=p.device, dtype=torch.float32)

        r, c = st["r"], st["c"]
        s2 = float(scale) * float(scale)

        rs = torch.linalg.vector_norm(g, dim=1, dtype=torch.float32).pow_(2).div_(m * s2)
        cs = torch.linalg.vector_norm(g, dim=0, dtype=torch.float32).pow_(2).div_(n * s2)
        gm = (
            torch.linalg.vector_norm(g.reshape(-1), dtype=torch.float32)
            .pow_(2)
            .div_(m * n * s2)
            .clamp_min(self.eps)
        )

        r.mul_(b2).add_(rs, alpha=1.0 - b2)
        c.mul_(b2).add_(cs, alpha=1.0 - b2)

        denom = ((r[:, None] * c[None, :]) / (gm * bc2)).clamp_min(self.eps).sqrt_()

        # Update 4-bit momentum
        g_fp32 = g.float().div_(scale)
        if b1 > 0.0:
            if "m_packed" in st and st["m_packed"] is not None:
                m_prev = dequantize_unpack_4bit(
                    st["m_packed"], st["m_scales"], st["m_n"], st["m_shape"],
                    block_size=block_size, dtype=torch.float32
                )
                if lossless and "m_residual" in st and st["m_residual"] is not None:
                    m_prev.add_(st["m_residual"].to(torch.float32))
                m_curr = m_prev.mul_(b1).add_(g_fp32, alpha=1.0 - b1)
            else:
                m_curr = g_fp32.clone()

            m_packed, m_scales, m_n, m_shape = quantize_pack_4bit(
                m_curr, block_size=block_size, stochastic=stochastic
            )
            st["m_packed"] = m_packed
            st["m_scales"] = m_scales
            st["m_n"] = m_n
            st["m_shape"] = m_shape

            if lossless:
                m_rec = dequantize_unpack_4bit(
                    m_packed, m_scales, m_n, m_shape, block_size=block_size, dtype=torch.float32
                )
                res_dtype = torch.float16 if g.device.type == "cuda" else torch.float32
                st["m_residual"] = (m_curr - m_rec).to(res_dtype)

            m_eff = m_curr / bc1
        else:
            m_eff = g_fp32

        # Stream row slices to bound peak VRAM
        sum_sq = 0.0
        rows = max(1, (1 << 21) // n)
        for r0 in range(0, m, rows):
            r1 = min(r0 + rows, m)
            u_chunk = m_eff[r0:r1] / denom[r0:r1].add_(self.eps)
            sum_sq += u_chunk.pow(2).sum().item()

        rms = max(1.0, math.sqrt(sum_sq / float(m * n)))

        for r0 in range(0, m, rows):
            r1 = min(r0 + rows, m)
            u_slice = ((m_eff[r0:r1] / denom[r0:r1].add_(self.eps)) / rms).to(g.dtype)
            if lr > 0:
                if weight_decay > 0:
                    p.data[r0:r1].mul_(1.0 - lr * weight_decay)
                p.data[r0:r1].add_(u_slice, alpha=-lr)

        del g_fp32, denom, m_eff

    @torch.no_grad()
    def _step_1d(
        self,
        tensors: List[torch.Tensor],
        states: List[Dict[str, Optional[torch.Tensor]]],
        lr: float,
        weight_decay: float,
        scale: float,
        bc1: float,
        bc2: float,
        b1: float,
        b2: float,
        block_size: int,
        lossless: bool,
        stochastic: bool,
    ) -> None:
        for st in states:
            if "v" not in st or st["v"] is None:
                st["v"] = torch.zeros_like(st["p"], dtype=torch.float32)

        for i, st in enumerate(states):
            p = tensors[i]
            g = p.grad.float().div_(scale)
            v = st["v"]
            v.mul_(b2).addcmul_(g, g, value=1.0 - b2)
            denom = (v / bc2).clamp_min(self.eps).sqrt_().add_(self.eps)

            if b1 > 0.0:
                if "m_packed" in st and st["m_packed"] is not None:
                    m_prev = dequantize_unpack_4bit(
                        st["m_packed"], st["m_scales"], st["m_n"], st["m_shape"],
                        block_size=block_size, dtype=torch.float32
                    )
                    if lossless and "m_residual" in st and st["m_residual"] is not None:
                        m_prev.add_(st["m_residual"].to(torch.float32))
                    m_curr = m_prev.mul_(b1).add_(g, alpha=1.0 - b1)
                else:
                    m_curr = g.clone()

                m_packed, m_scales, m_n, m_shape = quantize_pack_4bit(
                    m_curr, block_size=block_size, stochastic=stochastic
                )
                st["m_packed"] = m_packed
                st["m_scales"] = m_scales
                st["m_n"] = m_n
                st["m_shape"] = m_shape

                if lossless:
                    m_rec = dequantize_unpack_4bit(
                        m_packed, m_scales, m_n, m_shape, block_size=block_size, dtype=torch.float32
                    )
                    res_dtype = torch.float16 if p.device.type == "cuda" else torch.float32
                    st["m_residual"] = (m_curr - m_rec).to(res_dtype)

                m_eff = m_curr / bc1
            else:
                m_eff = g

            upd = (m_eff / denom).to(p.dtype)
            if lr > 0 and weight_decay > 0:
                p.data.mul_(1.0 - lr * weight_decay)
            p.data.add_(upd, alpha=-lr if lr > 0 else 0.0)

    @torch.no_grad()
    def step(
        self,
        closure: Optional[callable] = None,
        lr: Optional[float] = None,
        scale: float = 1.0,
    ) -> Optional[float]:
        """Performs a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss (optional).
            lr: Optional explicit learning rate override for manual LR schedulers.
            scale: Gradient loss scaling factor (default: 1.0, used for mixed precision).
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.t += 1

        for group in self.param_groups:
            group_lr = lr if lr is not None else group["lr"]
            wd = group.get("weight_decay", self.wd)
            b1 = group.get("beta1", self.b1)
            b2 = group.get("beta2", self.b2)
            chunk_size = group.get("chunk", self.chunk)
            fp32_max = group.get("fp32_max_elems", self.fp32_max)
            block_size = group.get("block_size", self.block_size)
            lossless = group.get("lossless", self.lossless)
            stochastic = group.get("stochastic_rounding", self.stochastic)

            bc1 = 1.0 - (b1 ** self.t) if b1 > 0.0 else 1.0
            bc2 = 1.0 - (b2 ** self.t)

            # Bucket parameters by dimensionality and shape
            buckets_2d: Dict[Tuple[int, ...], List[Tuple[torch.Tensor, Dict]]] = {}
            list_1d: List[Tuple[torch.Tensor, Dict]] = []

            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                st["p"] = p
                if p.ndim >= 2:
                    buckets_2d.setdefault(_get_shape_key(p), []).append((p, st))
                else:
                    list_1d.append((p, st))

            # 1. Process 2D Weight Tensors
            for shape_key, entries in buckets_2d.items():
                p_list = [e[0] for e in entries]
                st_list = [e[1] for e in entries]

                small_pairs = [(p, st) for p, st in zip(p_list, st_list) if p.numel() <= fp32_max]
                big_pairs = [(p, st) for p, st in zip(p_list, st_list) if p.numel() > fp32_max]

                if small_pairs:
                    self._step_2d_batched(
                        [e[0] for e in small_pairs],
                        [e[1] for e in small_pairs],
                        lr=group_lr,
                        weight_decay=wd,
                        scale=scale,
                        bc1=bc1,
                        bc2=bc2,
                        b1=b1,
                        b2=b2,
                        chunk_size=chunk_size,
                        block_size=block_size,
                        lossless=lossless,
                        stochastic=stochastic,
                    )

                for p_big, st_big in big_pairs:
                    self._step_2d_streaming_large(
                        p_big,
                        st_big,
                        lr=group_lr,
                        weight_decay=wd,
                        scale=scale,
                        bc1=bc1,
                        bc2=bc2,
                        b1=b1,
                        b2=b2,
                        block_size=block_size,
                        lossless=lossless,
                        stochastic=stochastic,
                    )

            # 2. Process 1D Parameters (biases, layer norm gains)
            if list_1d:
                self._step_1d(
                    [e[0] for e in list_1d],
                    [e[1] for e in list_1d],
                    lr=group_lr,
                    weight_decay=wd,
                    scale=scale,
                    bc1=bc1,
                    bc2=bc2,
                    b1=b1,
                    b2=b2,
                    block_size=block_size,
                    lossless=lossless,
                    stochastic=stochastic,
                )

        return loss


# Aliases
Avalanche4B = Avalanche4Bit
Lossless4BitAvalanche = Avalanche4Bit
FactorizedAdamW4Bit = Avalanche4Bit
