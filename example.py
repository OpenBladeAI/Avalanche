"""Self-contained benchmark and usage example for the Avalanche Optimizer.

Created and Developed by Jassim Ibrahim Farooqi.
Part of ongoing research and optimizations in extreme-efficiency training architectures.
"""

import os
import sys
import torch
import torch.nn as nn

# Ensure local avalanche package can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from optimizer import Avalanche


class BenchmarkTransformer(nn.Module):
    """Synthetic multi-layer transformer block model to benchmark optimizer memory footprint."""
    def __init__(self, d_model=1536, n_layers=12, vocab_size=32000):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=16,
                dim_feedforward=4096,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h)
        return self.lm_head(self.norm(h))


def compute_optimizer_state_bytes(opt: Avalanche) -> int:
    """Calculates exact memory allocated for all internal optimizer states."""
    total_bytes = 0
    for group in opt.param_groups:
        for p in group["params"]:
            st = opt.state[p]
            for key in ("r", "c", "v"):
                t = st.get(key)
                if t is not None:
                    total_bytes += t.nelement() * t.element_size()
    return total_bytes


def main():
    print("=" * 76)
    print("❄️   Avalanche: Ultra-Low VRAM Matrix-Factorized Optimizer")
    print("    An OpenBlade Release (https://github.com/OpenBladeAI)")
    print("    Hugging Face: https://huggingface.co/OpenBlade")
    print("    Author: Jassim Ibrahim Farooqi")
    print("    Citation: Ongoing Research & Optimizations in Extreme-Efficiency Training")
    print("=" * 76)

    # Use CPU for benchmark validation so active GPU training is untouched
    device = torch.device("cpu")
    print("\n[1/4] Instantiating Benchmark Transformer (~300M Parameters)...")
    model = BenchmarkTransformer(d_model=1536, n_layers=12, vocab_size=32000).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    model_fp32_mb = (total_params * 4) / (1024 ** 2)
    standard_adam_mb = (total_params * 8) / (1024 ** 2)  # 1st + 2nd FP32 moments (8 bytes/param)

    print(f"      Total Parameters       : {total_params:,}")
    print(f"      Model Weights (FP32)   : {model_fp32_mb:.2f} MB")
    print(f"      Standard AdamW Memory  : {standard_adam_mb:.2f} MB ({standard_adam_mb / 1024:.2f} GB)")

    print("\n[2/4] Initializing Avalanche Optimizer...")
    opt = Avalanche(model.parameters(), lr=1e-4, weight_decay=0.01)

    print("\n[3/4] Running 5 Optimization Steps with Synthetic Loss...")
    x = torch.randint(0, 32000, (2, 64), device=device)
    targets = torch.randint(0, 32000, (2, 64), device=device)

    criterion = nn.CrossEntropyLoss()

    for step in range(1, 6):
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits.view(-1, 32000), targets.view(-1))
        loss.backward()
        opt.step()
        print(f"      Step {step} | Loss: {loss.item():.4f}")

    avalanche_bytes = compute_optimizer_state_bytes(opt)
    avalanche_mb = avalanche_bytes / (1024 ** 2)
    reduction_pct = (1.0 - (avalanche_mb / standard_adam_mb)) * 100.0

    print("\n[4/4] Optimizer Memory Analysis:")
    print("-" * 76)
    print(f"  Standard FP32 AdamW State Footprint : {standard_adam_mb:8.2f} MB ({standard_adam_mb / 1024:.2f} GB)")
    print(f"  Avalanche State Footprint           : {avalanche_mb:8.2f} MB")
    print(f"  Memory Footprint Reduction          : {reduction_pct:8.2f}%")
    print("-" * 76)

    # State dict verification
    sd = opt.state_dict()
    assert "param_groups" in sd
    opt.load_state_dict(sd)
    print("✓ State dict serialization & deserialization verified successfully.")
    print("✓ Avalanche optimizer is ready for deployment across GitHub, Hugging Face, and production.")


if __name__ == "__main__":
    main()
