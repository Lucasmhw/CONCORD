import torch

from concord.utils.environment import environment_metadata
from concord.utils.seed import set_seed


def test_deterministic_seed_and_source_hash_are_recorded() -> None:
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        set_seed(7, deterministic=True)
        metadata = environment_metadata()
        assert metadata["deterministic_algorithms"] is True
        assert len(metadata["source_tree_sha256"]) == 64
    finally:
        torch.use_deterministic_algorithms(previous)
