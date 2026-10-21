# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import subprocess
import sys
from typing import Any

import pytest
import torch
from torch.utils.data import Dataset

from cosmos_framework.data.generator.action.datasets.action_sft_wrappers import (
    ActionIterableShuffleDataset,
    get_action_external_sft_dataset,
)


class RecordingDataset(Dataset):
    """Stands in for a downstream package's dataset: two episodes of one window each."""

    def __init__(self, num_episodes: int, action_dim: int) -> None:
        self.num_episodes = num_episodes
        self.action_dim = action_dim

    def __len__(self) -> int:
        return self.num_episodes

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {
            "ai_caption": "Fold the towel.",
            "video": torch.zeros(3, 33, 64, 64),  # [C,T,H,W]
            "action": torch.full((32, self.action_dim), float(idx)),  # [T,D]
            "mode": "wam",
            "domain_id": torch.tensor(12),
        }

    def get_shuffle_blocks(self) -> list[tuple[int, int]]:
        return [(idx, 1) for idx in range(self.num_episodes)]


_TRANSFORM_KWARGS = {
    "tokenizer_config": None,
    "max_action_dim": 32,
    "append_viewpoint_info": False,
    "append_duration_fps_timestamps": False,
    "append_resolution_info": False,
}


@pytest.mark.L0
def test_external_dataset_class_is_located_and_transformed() -> None:
    dataset = get_action_external_sft_dataset(
        dataset_class=f"{__name__}.RecordingDataset",
        dataset_kwargs={"num_episodes": 2, "action_dim": 20},
        resolution="256",
        transform_kwargs=_TRANSFORM_KWARGS,
    )

    assert len(dataset) == 2
    sample = dataset[1]
    assert sample["action"].shape == (32, 32)  # [T,D_max]
    assert torch.equal(sample["action"][:, :20], torch.ones(32, 20))
    assert "sequence_plan" in sample


@pytest.mark.L0
def test_external_dataset_streams_episodes_when_shuffled() -> None:
    dataset = get_action_external_sft_dataset(
        dataset_class=f"{__name__}.RecordingDataset",
        dataset_kwargs={"num_episodes": 2, "action_dim": 20},
        resolution="256",
        transform_kwargs=_TRANSFORM_KWARGS,
        iterable_shuffle=True,
    )

    assert isinstance(dataset, ActionIterableShuffleDataset)
    stream = iter(dataset)
    firsts = {next(stream)["action"][0, 0].item() for _ in range(2)}
    assert firsts == {0.0, 1.0}


@pytest.mark.L0
def test_external_dataset_class_must_exist() -> None:
    with pytest.raises(ImportError):
        get_action_external_sft_dataset(
            dataset_class=f"{__name__}.MissingDataset",
            dataset_kwargs={},
            resolution="256",
        )


@pytest.mark.L0
def test_wrappers_import_without_the_concrete_datasets_dependencies() -> None:
    # A downstream package wraps its own dataset without LeRobot or pandas installed.
    code = (
        "import sys\n"
        "sys.modules['lerobot'] = None\n"
        "sys.modules['pandas'] = None\n"
        "import cosmos_framework.data.generator.action.datasets.action_sft_wrappers\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
