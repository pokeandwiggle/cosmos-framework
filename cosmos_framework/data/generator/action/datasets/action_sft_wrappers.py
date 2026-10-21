# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Dataset-agnostic action SFT wrappers: a raw action dataset -> ``ActionTransformPipeline``.

Kept free of the concrete datasets so that a downstream package can wrap its own
dataset without their LeRobot and pandas dependencies.
"""

from __future__ import annotations

from typing import Any

from torch.utils.data import Dataset, IterableDataset, get_worker_info

from cosmos_framework.data.generator.action.utils.transforms import ActionTransformPipeline
from cosmos_framework.utils.lazy_config.registry import locate


class ActionSFTDataset(Dataset):
    """Wraps a map-style action dataset and applies ``ActionTransformPipeline`` per sample."""

    def __init__(self, dataset: Dataset, transform: ActionTransformPipeline, resolution: str | int | None):
        super().__init__()
        self._dataset = dataset
        self._transform = transform
        self._resolution = resolution

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._transform(self._dataset[idx], self._resolution)

    def get_shuffle_blocks(self):
        """Delegate to the inner DROIDLeRobotDataset (per-episode/segment flat-index blocks)."""
        return self._dataset.get_shuffle_blocks()


class ActionIterableShuffleDataset(IterableDataset):
    """Streaming view of a map-style ``ActionSFTDataset``.

    Each ``(rank, worker)`` is assigned a DISJOINT subset of episodes (sharded over
    ``shard_world_size * num_workers``), shuffles its episode ORDER, and streams the
    windows WITHIN each episode sequentially -> within-rank batch diversity (the N
    workers of a rank stream N different episodes) AND cross-rank diversity, while
    keeping reads sequential (I/O locality + COW; no RandomSampler random-access OOM).
    Re-shuffles each epoch and streams indefinitely (the trainer stops at ``max_iter``).

    ``shard_world_size`` / ``shard_rank`` are set by ``RankPartitionedDataLoader``.
    """

    def __init__(self, dataset: "ActionSFTDataset", seed: int = 42):
        super().__init__()
        self._dataset = dataset
        self._seed = int(seed)
        self.shard_world_size = 1
        self.shard_rank = 0

    def __len__(self) -> int:  # informational only; iteration is infinite
        return len(self._dataset)

    def __iter__(self):
        import torch

        blocks = self._dataset.get_shuffle_blocks()
        wi = get_worker_info()
        wid = wi.id if wi is not None else 0
        nw = wi.num_workers if wi is not None else 1
        global_shard = int(self.shard_rank) * nw + wid
        total_shards = max(1, int(self.shard_world_size) * nw)
        epoch = 0
        while True:
            g = torch.Generator()
            g.manual_seed(self._seed + epoch)  # same permutation across all (rank,worker) -> disjoint shard
            order = torch.randperm(len(blocks), generator=g).tolist()
            for b in order[global_shard::total_shards]:
                start, length = blocks[b]
                for idx in range(start, start + length):
                    yield self._dataset[idx]
            epoch += 1


def get_action_external_sft_dataset(
    *,
    dataset_class: str,
    dataset_kwargs: dict[str, Any],
    resolution: str | int,
    transform_kwargs: dict[str, Any] | None = None,
    iterable_shuffle: bool = False,
    episode_shuffle_seed: int = 42,
) -> Dataset:
    """Build an action SFT dataset around a dataset class named by import path.

    Lets a downstream package train on its own recordings without adding a dataset
    to this repository: its class only has to return the raw sample that
    ``DROIDLeRobotDataset.__getitem__`` returns (``video``/``action``/``ai_caption``/
    ``viewpoint``/``mode``/``domain_id``/...), and ``get_shuffle_blocks()`` when
    ``iterable_shuffle`` is set.

    Args:
        dataset_class: Import path of the map-style dataset class, e.g.
            ``"my_package.recordings.RecordingDataset"``.
        dataset_kwargs: Keyword arguments for ``dataset_class``.
        resolution: Resolution bucket passed to ``ActionTransformPipeline``.
        transform_kwargs: Keyword arguments for ``ActionTransformPipeline``.
        iterable_shuffle: Stream through ``ActionIterableShuffleDataset``.
        episode_shuffle_seed: Seed of the episode-order shuffle.
    """
    dataset: Dataset = locate(dataset_class)(**dataset_kwargs)
    sft = ActionSFTDataset(dataset, ActionTransformPipeline(**(transform_kwargs or {})), resolution)
    if iterable_shuffle:
        return ActionIterableShuffleDataset(sft, seed=episode_shuffle_seed)
    return sft
