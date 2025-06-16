"""
Setup script for RivalAI package.
"""

from setuptools import setup, find_packages

setup(
    name="rival_ai",
    version="0.1.0",
    description="A chess engine using graph neural networks",
    author="RivalAI Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.12.0",
        "tensorboard>=2.12.0",
        "tqdm>=4.65.0",
        "pytest>=7.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
            "flake8>=6.0.0",
        ],
    },
) 