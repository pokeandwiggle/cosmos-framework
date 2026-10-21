# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import pytest
import torch

from cosmos_framework.inference.action import build_action_batch
from cosmos_framework.inference.args import ModelMode


def _build(action_prefix: torch.Tensor | None, model_mode: ModelMode = ModelMode.WAM) -> dict:
    return build_action_batch(
        video=torch.zeros(3, 1, 64, 64, dtype=torch.uint8),
        action=torch.zeros(32, 64),
        raw_action_dim=20,
        prompt="Fold the towel.",
        view_point="concat_view",
        domain_name="robomind-franka-dual",
        model_mode=model_mode,
        action_chunk_size=32,
        fps=15,
        resolution="256",
        input_video_key="video",
        device="cpu",
        action_prefix=action_prefix,
    )


@pytest.mark.L0
def test_action_prefix_becomes_clean_leading_rows() -> None:
    prefix = torch.randn(15, 20)  # [d,raw_action_dim]

    batch = _build(prefix)

    action = batch["action"][0][0]  # [T,D_max]
    torch.testing.assert_close(action[:15, :20], prefix)
    assert not action[15:].any() and not action[:, 20:].any()
    assert batch["sequence_plan"][0].condition_frame_indexes_action == list(range(15))


@pytest.mark.L0
def test_no_action_prefix_keeps_every_action_noisy() -> None:
    assert _build(None)["sequence_plan"][0].condition_frame_indexes_action == []


@pytest.mark.L0
def test_action_prefix_is_refused_outside_wam_and_beyond_the_chunk() -> None:
    with pytest.raises(ValueError, match="model_mode=wam"):
        _build(torch.zeros(4, 20), ModelMode.INVERSE_DYNAMICS)
    with pytest.raises(ValueError, match="raw_action_dim"):
        _build(torch.zeros(4, 10))
    with pytest.raises(ValueError, match="rtc_delay=33"):
        _build(torch.zeros(33, 20))


@pytest.mark.L0
def test_additional_view_description_reaches_the_caption() -> None:
    description = "The scene camera is on top; the left and right wrist cameras are below."

    batch = build_action_batch(
        video=torch.zeros(3, 1, 64, 64, dtype=torch.uint8),
        action=torch.zeros(32, 64),
        raw_action_dim=20,
        prompt="Fold the towel.",
        view_point="concat_view",
        domain_name="robomind-franka-dual",
        model_mode=ModelMode.WAM,
        action_chunk_size=32,
        fps=15,
        resolution="256",
        input_video_key="video",
        device="cpu",
        additional_view_description=description,
    )

    assert description in batch["ai_caption"][0]
    assert description not in _build(None)["ai_caption"][0]
