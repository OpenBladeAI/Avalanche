"""Avalanche: Ultra-Low VRAM Matrix-Factorized Optimizer for PyTorch.

An OpenBlade Release (https://github.com/OpenBladeAI, https://huggingface.co/OpenBlade)
Created and Developed by Jassim Ibrahim Farooqi.
Part of ongoing research and optimizations in extreme-efficiency training architectures.

Reduces optimizer state VRAM consumption by up to 99.99% compared to standard FP32
AdamW (~10MB total VRAM footprint across entire model training regardless of model size).
"""

from .optimizer import Avalanche, AvalancheAdamW, FactorizedAdamW

__version__ = "1.0.0"
__author__ = "Jassim Ibrahim Farooqi"
__organization__ = "OpenBlade"
__license__ = "MIT"

__all__ = ["Avalanche", "AvalancheAdamW", "FactorizedAdamW"]
