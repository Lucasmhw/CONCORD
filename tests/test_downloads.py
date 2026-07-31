import hashlib

import pytest

from scripts.download_benchmarks import _verify_sha256


def test_checksum_accepts_an_audited_byte_variant(tmp_path) -> None:
    path = tmp_path / "data.csv"
    path.write_bytes(b"a,b\r\n1,2\r\n")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()

    _verify_sha256(path, ("unused", actual), "test data")


def test_checksum_rejects_an_unknown_variant(tmp_path) -> None:
    path = tmp_path / "data.csv"
    path.write_bytes(b"changed")

    with pytest.raises(ValueError, match="Checksum mismatch"):
        _verify_sha256(path, "expected", "test data")
