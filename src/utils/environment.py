from __future__ import annotations

import hashlib
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch


def _version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _source_tree_sha256() -> str:
    source_root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(source_root.rglob("*.py"), key=lambda item: item.as_posix()):
        digest.update(path.relative_to(source_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def environment_metadata() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "source_tree_sha256": _source_tree_sha256(),
        "packages": {
            name: _version(name)
            for name in (
                "torch",
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "pyyaml",
                "tqdm",
                "einops",
                "matplotlib",
            )
        },
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }
