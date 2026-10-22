# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""L0 tests for resolving Hugging Face downloads from ``COSMOS_HF_LOCAL_SNAPSHOTS``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cosmos_framework.configs.base.defaults.reasoner import download_tokenizer_files
from cosmos_framework.utils import checkpoint_db
from cosmos_framework.utils.checkpoint_db import (
    LOCAL_SNAPSHOTS_ENV,
    CheckpointDirHf,
    CheckpointFileHf,
)

_REPO = "nvidia/Cosmos3-Nano"


@pytest.fixture
def snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A local copy of ``_REPO`` named by the environment; the Hub is unreachable."""
    (tmp_path / "sound_tokenizer").mkdir()
    (tmp_path / "sound_tokenizer" / "config.json").write_text("{}")
    (tmp_path / "tokenizer.json").write_text("{}")
    monkeypatch.setenv(LOCAL_SNAPSHOTS_ENV, json.dumps({_REPO: str(tmp_path)}))

    def _no_hub(cmd_args: list[str]) -> str:
        raise AssertionError(f"fetched from the Hub: {cmd_args}")

    monkeypatch.setattr(checkpoint_db, "_hf_download", _no_hub)
    return tmp_path


@pytest.mark.L0
@pytest.mark.CPU
def test_mapped_directory_resolves_locally_whatever_the_revision(snapshot: Path) -> None:
    assert CheckpointDirHf(repository=_REPO, revision="main").download() == str(snapshot)


@pytest.mark.L0
@pytest.mark.CPU
def test_mapped_subdirectory_resolves_inside_the_snapshot(snapshot: Path) -> None:
    checkpoint = CheckpointDirHf(repository=_REPO, revision="main", subdirectory="sound_tokenizer")
    assert checkpoint.download() == str(snapshot / "sound_tokenizer")


@pytest.mark.L0
@pytest.mark.CPU
def test_mapped_file_resolves_inside_the_snapshot(snapshot: Path) -> None:
    checkpoint = CheckpointFileHf(repository=_REPO, revision="main", filename="tokenizer.json")
    assert checkpoint.download() == str(snapshot / "tokenizer.json")


@pytest.mark.L0
@pytest.mark.CPU
def test_mapped_snapshot_without_the_file_fails_loudly(snapshot: Path) -> None:
    checkpoint = CheckpointFileHf(repository=_REPO, revision="main", filename="Wan2.2_VAE.pth")
    with pytest.raises(FileNotFoundError, match="Wan2.2_VAE.pth"):
        checkpoint.download()


@pytest.mark.L0
@pytest.mark.CPU
def test_mapped_snapshot_without_the_subdirectory_fails_loudly(snapshot: Path) -> None:
    checkpoint = CheckpointDirHf(repository=_REPO, revision="main", subdirectory="vision_encoder")
    with pytest.raises(FileNotFoundError, match="vision_encoder"):
        checkpoint.download()


@pytest.mark.L0
@pytest.mark.CPU
def test_mapping_to_a_missing_directory_fails_loudly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(LOCAL_SNAPSHOTS_ENV, json.dumps({_REPO: str(tmp_path / "absent")}))
    with pytest.raises(FileNotFoundError, match="absent"):
        CheckpointDirHf(repository=_REPO, revision="main").download()


@pytest.mark.L0
@pytest.mark.CPU
def test_unmapped_repository_still_downloads(snapshot: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(checkpoint_db, "_hf_download", lambda cmd_args: calls.append(cmd_args) or str(snapshot))
    CheckpointFileHf(repository="Wan-AI/Wan2.2-TI2V-5B", revision="main", filename="tokenizer.json").download()
    assert calls and calls[0][0] == "Wan-AI/Wan2.2-TI2V-5B"


@pytest.mark.L0
@pytest.mark.CPU
def test_hf_tokenizer_files_resolve_to_the_mapped_directory(snapshot: Path) -> None:
    assert download_tokenizer_files(_REPO, "hf") == str(snapshot)


@pytest.mark.L0
@pytest.mark.CPU
def test_hf_tokenizer_files_name_the_repository_when_unmapped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(LOCAL_SNAPSHOTS_ENV, raising=False)
    assert download_tokenizer_files(_REPO, "hf") == _REPO


@pytest.mark.L0
@pytest.mark.CPU
def test_processor_node_builds_from_the_mapped_directory(snapshot: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The node a Cosmos3-Nano model config's 'vlm_config.tokenizer' instantiates.
    from cosmos_framework.data.generator import processors

    monkeypatch.setattr(processors, "build_processor", lambda path, **kwargs: path)
    assert processors.build_processor_lazy(repository=_REPO, revision="main") == str(snapshot)
