"""Avalanche: Ultra-Low VRAM Matrix-Factorized & Multi-Precision Optimizer for PyTorch.

An OpenBlade Release (https://github.com/OpenBladeAI, https://huggingface.co/OpenBlade)
Created and Developed by Jassim Ibrahim Farooqi.
Part of ongoing research and optimizations in extreme-efficiency training architectures.

Variants:
  - Avalanche (mode="auto" / "none"): Zero-momentum matrix-factorized optimizer (~10MB VRAM).
  - Avalanche4Bit / Avalanche4B: 4-bit packed momentum with lossless error feedback (0.5 B/param).
  - Avalanche8Bit / Avalanche8B: 8-bit quantized momentum with lossless error feedback (1.0 B/param).
  - AvalancheFP16 / Avalanche16: 16-bit half precision momentum (2.0 B/param).
  - AvalancheBF16: 16-bit bfloat16 momentum matching FP32 dynamic range (2.0 B/param).
  - AvalancheFP32 / Avalanche32: Full 32-bit single precision momentum (4.0 B/param).
"""

from .optimizer import (
    Avalanche,
    AvalancheAdamW,
    FactorizedAdamW,
    Avalanche4Bit,
    Avalanche4B,
    Lossless4BitAvalanche,
    FactorizedAdamW4Bit,
    Avalanche8Bit,
    Avalanche8B,
    Lossless8BitAvalanche,
    FactorizedAdamW8Bit,
    AvalancheFP16,
    Avalanche16,
    FactorizedAdamWFP16,
    AvalancheBF16,
    FactorizedAdamWBF16,
    AvalancheFP32,
    Avalanche32,
    FactorizedAdamWFP32,
    quantize_pack_4bit,
    dequantize_unpack_4bit,
    quantize_8bit,
    dequantize_8bit,
)

__version__ = "1.2.0"
__author__ = "Jassim Ibrahim Farooqi"
__organization__ = "OpenBlade"
__license__ = "MIT"

__all__ = [
    "Avalanche",
    "AvalancheAdamW",
    "FactorizedAdamW",
    "Avalanche4Bit",
    "Avalanche4B",
    "Lossless4BitAvalanche",
    "FactorizedAdamW4Bit",
    "Avalanche8Bit",
    "Avalanche8B",
    "Lossless8BitAvalanche",
    "FactorizedAdamW8Bit",
    "AvalancheFP16",
    "Avalanche16",
    "FactorizedAdamWFP16",
    "AvalancheBF16",
    "FactorizedAdamWBF16",
    "AvalancheFP32",
    "Avalanche32",
    "FactorizedAdamWFP32",
    "quantize_pack_4bit",
    "dequantize_unpack_4bit",
    "quantize_8bit",
    "dequantize_8bit",
]
