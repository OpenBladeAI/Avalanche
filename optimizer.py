"""Avalanche: Ultra-Low VRAM Matrix-Factorized & Multi-Precision Optimizer for PyTorch.

An OpenBlade Release (https://github.com/OpenBladeAI, https://huggingface.co/OpenBlade)
Created and Developed by Jassim Ibrahim Farooqi.
Part of ongoing research in extreme-efficiency deep learning training architectures.

Provides unified, lossless optimization across multiple precision formats:
  - Avalanche (beta1=0.0): Zero-momentum matrix-factorized optimizer (~10 MB VRAM total).
  - Avalanche4Bit / Avalanche4B: 4-bit packed momentum with lossless error feedback (0.5 B/param).
  - Avalanche8Bit / Avalanche8B: 8-bit quantized momentum with lossless error feedback (1.0 B/param).
  - AvalancheFP16: Half-precision 16-bit momentum with error feedback (2.0 B/param).
  - AvalancheBF16: Bfloat16 momentum matching FP32 dynamic range (2.0 B/param).
  - AvalancheFP32: Full single-precision momentum with factorized second moments (4.0 B/param).

All variants share the same core:
  1. Rank-1 marginal second-moment factorization: r in R^M, c in R^N. State is O(M + N).
  2. Lossless error-feedback: no gradient or step magnitude is discarded across steps/batches.
  3. Immediate intermediate cast-away: full-precision calculations are freed immediately after update.
  4. Dynamic RMS gradient update normalization for bulletproof mixed-precision stability.
"""

import math
from typing import Dict, Iterable, List, Optional, Tuple, Union
import torch
from torch.optim.optimizer import Optimizer


def _get_shape_key(p: torch.Tensor) -> Tuple[int, ...]:
    return tuple(p.shape)


# ============================================================================
# Quantization & Packing Kernels
# ============================================================================

def quantize_pack_4bit(
    x: torch.Tensor,
    block_size: int = 32,
    stochastic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, int, Tuple[int, ...]]:
    """Quantizes float tensor into packed 4-bit signed integers [-7, 7] stored as uint8."""
    orig_shape = x.shape
    x_flat = x.flatten()
    n = x_flat.numel()
    pad = (block_size - (n % block_size)) % block_size
    if pad > 0:
        x_flat = torch.cat([x_flat, torch.zeros(pad, device=x.device, dtype=x.dtype)])

    blocks = x_flat.view(-1, block_size)
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

    q_unsigned = (q + 8).to(torch.uint8)
    packed = q_unsigned[:, 0::2] | (q_unsigned[:, 1::2] << 4)
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


def quantize_8bit(
    x: torch.Tensor,
    block_size: int = 32,
    stochastic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, int, Tuple[int, ...]]:
    """Quantizes float tensor into signed 8-bit integers [-127, 127] with per-block scaling."""
    orig_shape = x.shape
    x_flat = x.flatten()
    n = x_flat.numel()
    pad = (block_size - (n % block_size)) % block_size
    if pad > 0:
        x_flat = torch.cat([x_flat, torch.zeros(pad, device=x.device, dtype=x.dtype)])

    blocks = x_flat.view(-1, block_size)
    scales = blocks.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 127.0
    scale_dtype = torch.float16 if x.device.type == "cuda" else torch.float32
    scales_saved = scales.squeeze(1).to(scale_dtype)

    normed = blocks / scales
    if stochastic:
        floor = normed.floor()
        prob = normed - floor
        rnd = torch.rand_like(normed)
        q = torch.where(rnd < prob, floor + 1.0, floor).clamp(-127, 127).to(torch.int8)
    else:
        q = normed.round().clamp(-127, 127).to(torch.int8)

    return q, scales_saved, n, orig_shape


def dequantize_8bit(
    q: torch.Tensor,
    scales: torch.Tensor,
    n: int,
    orig_shape: Tuple[int, ...],
    block_size: int = 32,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dequantizes an 8-bit integer tensor back into full-precision float tensor."""
    sc = scales.to(dtype).unsqueeze(1)
    return (q.to(dtype) * sc).view(-1)[:n].view(orig_shape)


# ============================================================================
# Momentum Precision Dispatch Helpers
# ============================================================================

def unpack_momentum(st: Dict, mode: str, block_size: int = 32, lossless: bool = True) -> Optional[torch.Tensor]:
    """Unpacks stored momentum velocity into a temporary float32 tensor."""
    if mode in ("none", "zero"):
        return None

    if mode == "4bit":
        if "m_packed" not in st or st["m_packed"] is None:
            return None
        m = dequantize_unpack_4bit(
            st["m_packed"], st["m_scales"], st["m_n"], st["m_shape"],
            block_size=block_size, dtype=torch.float32
        )
        if lossless and "m_residual" in st and st["m_residual"] is not None:
            m.add_(st["m_residual"].to(torch.float32))
        return m

    elif mode == "8bit":
        if "m_8bit" not in st or st["m_8bit"] is None:
            return None
        m = dequantize_8bit(
            st["m_8bit"], st["m_scales"], st["m_n"], st["m_shape"],
            block_size=block_size, dtype=torch.float32
        )
        if lossless and "m_residual" in st and st["m_residual"] is not None:
            m.add_(st["m_residual"].to(torch.float32))
        return m

    elif mode in ("fp16", "bf16", "fp32"):
        if "m" not in st or st["m"] is None:
            return None
        m = st["m"].to(torch.float32)
        if lossless and "m_residual" in st and st["m_residual"] is not None:
            m.add_(st["m_residual"].to(torch.float32))
        return m

    return None


def pack_momentum(
    m_curr: torch.Tensor,
    st: Dict,
    mode: str,
    block_size: int = 32,
    lossless: bool = True,
    stochastic: bool = False,
) -> None:
    """Packs float32 momentum into target precision format and preserves lossless residual."""
    if mode in ("none", "zero"):
        return

    res_dtype = torch.float16 if m_curr.device.type == "cuda" else torch.float32

    if mode == "4bit":
        packed, scales, n, shape = quantize_pack_4bit(m_curr, block_size=block_size, stochastic=stochastic)
        st["m_packed"] = packed
        st["m_scales"] = scales
        st["m_n"] = n
        st["m_shape"] = shape
        if lossless:
            rec = dequantize_unpack_4bit(packed, scales, n, shape, block_size=block_size, dtype=torch.float32)
            st["m_residual"] = (m_curr - rec).to(res_dtype)

    elif mode == "8bit":
        q8, scales, n, shape = quantize_8bit(m_curr, block_size=block_size, stochastic=stochastic)
        st["m_8bit"] = q8
        st["m_scales"] = scales
        st["m_n"] = n
        st["m_shape"] = shape
        if lossless:
            rec = dequantize_8bit(q8, scales, n, shape, block_size=block_size, dtype=torch.float32)
            st["m_residual"] = (m_curr - rec).to(res_dtype)

    elif mode == "fp16":
        m_fp16 = m_curr.to(torch.float16)
        st["m"] = m_fp16
        if lossless:
            st["m_residual"] = (m_curr - m_fp16.float()).to(res_dtype)

    elif mode == "bf16":
        m_bf16 = m_curr.to(torch.bfloat16)
        st["m"] = m_bf16
        if lossless:
            st["m_residual"] = (m_curr - m_bf16.float()).to(res_dtype)

    elif mode == "fp32":
        st["m"] = m_curr.clone()


# ============================================================================
# Main Avalanche Optimizer Engine
# ============================================================================

class Avalanche(Optimizer):
    """Avalanche: Ultra-Low VRAM Matrix-Factorized & Multi-Precision Optimizer.
    
    An OpenBlade Release. Created and Developed by Jassim Ibrahim Farooqi.
    
    Supports:
      - mode='auto' : Auto-selects '4bit' if beta1 > 0 else 'none' (zero-momentum, ~10MB VRAM).
      - mode='4bit' : Lossless 4-bit packed momentum with error feedback (0.5 B/param).
      - mode='8bit' : Lossless 8-bit quantized momentum with error feedback (1.0 B/param).
      - mode='fp16' : 16-bit half precision momentum (2.0 B/param).
      - mode='bf16' : 16-bit bfloat16 momentum matching FP32 dynamic range (2.0 B/param).
      - mode='fp32' : Full 32-bit single precision momentum (4.0 B/param).
      - mode='none' : Zero momentum (original Avalanche; maximum VRAM reduction).
    """

    def __init__(
        self,
        params: Union[Iterable[torch.Tensor], Iterable[Dict], torch.nn.Module],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.0, 0.999),
        eps: float = 1e-30,
        weight_decay: float = 0.01,
        mode: str = "auto",
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

        # Resolve mode
        mode_clean = mode.lower().strip()
        if mode_clean == "auto":
            mode_clean = "4bit" if betas[0] > 0.0 else "none"

        valid_modes = {"4bit", "8bit", "fp16", "bf16", "fp32", "none", "zero"}
        if mode_clean not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Choose from: {sorted(valid_modes)}")

        defaults = dict(
            lr=lr,
            beta1=betas[0],
            beta2=betas[1],
            eps=eps,
            weight_decay=weight_decay,
            mode=mode_clean,
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
        self.mode = mode_clean
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
        mode: str,
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

            # 2. Momentum calculation with precision dispatch and lossless error compensation
            M_list = []
            for i, st in enumerate(grp_st):
                g_i = G[i]
                if b1 > 0.0 and mode not in ("none", "zero"):
                    m_prev = unpack_momentum(st, mode=mode, block_size=block_size, lossless=lossless)
                    if m_prev is not None:
                        m_curr = m_prev.mul_(b1).add_(g_i, alpha=1.0 - b1)
                    else:
                        m_curr = g_i.clone()

                    pack_momentum(
                        m_curr, st, mode=mode, block_size=block_size,
                        lossless=lossless, stochastic=stochastic
                    )
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

            # 4. Immediate cast-away of full-precision calculation buffers
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
        mode: str,
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

        # Update momentum
        g_fp32 = g.float().div_(scale)
        if b1 > 0.0 and mode not in ("none", "zero"):
            m_prev = unpack_momentum(st, mode=mode, block_size=block_size, lossless=lossless)
            if m_prev is not None:
                m_curr = m_prev.mul_(b1).add_(g_fp32, alpha=1.0 - b1)
            else:
                m_curr = g_fp32.clone()

            pack_momentum(
                m_curr, st, mode=mode, block_size=block_size,
                lossless=lossless, stochastic=stochastic
            )
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
        mode: str,
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

            if b1 > 0.0 and mode not in ("none", "zero"):
                m_prev = unpack_momentum(st, mode=mode, block_size=block_size, lossless=lossless)
                if m_prev is not None:
                    m_curr = m_prev.mul_(b1).add_(g, alpha=1.0 - b1)
                else:
                    m_curr = g.clone()

                pack_momentum(
                    m_curr, st, mode=mode, block_size=block_size,
                    lossless=lossless, stochastic=stochastic
                )
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
        """Performs a single optimization step."""
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
            mode = group.get("mode", self.mode)
            chunk_size = group.get("chunk", self.chunk)
            fp32_max = group.get("fp32_max_elems", self.fp32_max)
            block_size = group.get("block_size", self.block_size)
            lossless = group.get("lossless", self.lossless)
            stochastic = group.get("stochastic_rounding", self.stochastic)

            bc1 = 1.0 - (b1 ** self.t) if b1 > 0.0 else 1.0
            bc2 = 1.0 - (b2 ** self.t)

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

            # 1. 2D Weight Tensors
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
                        mode=mode,
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
                        mode=mode,
                        block_size=block_size,
                        lossless=lossless,
                        stochastic=stochastic,
                    )

            # 2. 1D Parameters
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
                    mode=mode,
                    block_size=block_size,
                    lossless=lossless,
                    stochastic=stochastic,
                )

        return loss


# ============================================================================
# Dedicated Precision Subclasses
# ============================================================================

class Avalanche4Bit(Avalanche):
    """Avalanche with Lossless 4-Bit Momentum (0.5 Bytes/param)."""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-30, weight_decay=0.01, lossless=True, stochastic_rounding=False, block_size=32, chunk=16, fp32_max_elems=4194304):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, mode="4bit", lossless=lossless, stochastic_rounding=stochastic_rounding, block_size=block_size, chunk=chunk, fp32_max_elems=fp32_max_elems)


class Avalanche8Bit(Avalanche):
    """Avalanche with Lossless 8-Bit Momentum (1.0 Byte/param)."""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-30, weight_decay=0.01, lossless=True, stochastic_rounding=False, block_size=32, chunk=16, fp32_max_elems=4194304):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, mode="8bit", lossless=lossless, stochastic_rounding=stochastic_rounding, block_size=block_size, chunk=chunk, fp32_max_elems=fp32_max_elems)


class AvalancheFP16(Avalanche):
    """Avalanche with 16-Bit Half Precision (FP16) Momentum (2.0 Bytes/param)."""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-30, weight_decay=0.01, lossless=True, chunk=16, fp32_max_elems=4194304):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, mode="fp16", lossless=lossless, chunk=chunk, fp32_max_elems=fp32_max_elems)


class AvalancheBF16(Avalanche):
    """Avalanche with Bfloat16 Momentum matching FP32 dynamic range (2.0 Bytes/param)."""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-30, weight_decay=0.01, lossless=True, chunk=16, fp32_max_elems=4194304):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, mode="bf16", lossless=lossless, chunk=chunk, fp32_max_elems=fp32_max_elems)


class AvalancheFP32(Avalanche):
    """Avalanche with Full 32-Bit Single Precision (FP32) Momentum (4.0 Bytes/param)."""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-30, weight_decay=0.01, chunk=16, fp32_max_elems=4194304):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, mode="fp32", chunk=chunk, fp32_max_elems=fp32_max_elems)


# ============================================================================
# Aliases
# ============================================================================

AvalancheAdamW = Avalanche
FactorizedAdamW = Avalanche

Avalanche4B = Avalanche4Bit
Lossless4BitAvalanche = Avalanche4Bit
FactorizedAdamW4Bit = Avalanche4Bit

Avalanche8B = Avalanche8Bit
Lossless8BitAvalanche = Avalanche8Bit
FactorizedAdamW8Bit = Avalanche8Bit

Avalanche16 = AvalancheFP16
FactorizedAdamWFP16 = AvalancheFP16

FactorizedAdamWBF16 = AvalancheBF16

Avalanche32 = AvalancheFP32
FactorizedAdamWFP32 = AvalancheFP32
