from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="avalanche-opt",
    version="1.2.0",
    author="Jassim Ibrahim Farooqi",
    maintainer="OpenBlade",
    description="Ultra-Low VRAM Matrix-Factorized & Multi-Precision (4-bit, 8-bit, FP16, BF16, FP32) Lossless Optimizer for PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/OpenBladeAI/Avalanche",
    project_urls={
        "Organization": "https://github.com/OpenBladeAI",
        "HuggingFace": "https://huggingface.co/OpenBlade",
        "Source": "https://github.com/OpenBladeAI/Avalanche",
    },
    packages=["avalanche"],
    package_dir={"avalanche": "."},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.13.0",
    ],
)
