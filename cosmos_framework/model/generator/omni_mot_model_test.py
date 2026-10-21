# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def test_reasoner_only_setup_skips_vision_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    from cosmos_framework.model.generator import omni_mot_model

    vlm_tokenizer = SimpleNamespace(eos_token_id=42)
    vlm_processor = SimpleNamespace(tokenizer=vlm_tokenizer)
    vlm_config = SimpleNamespace(tokenizer="vlm-tokenizer-config")
    vision_config = SimpleNamespace(temporal_compression_factor=4)
    config = SimpleNamespace(
        load_vision_tokenizer=False,
        lidar_tokenizer=None,
        sound_gen=False,
        tokenizer=vision_config,
        vlm_config=vlm_config,
    )
    instantiated = []

    def _instantiate(candidate):
        instantiated.append(candidate)
        return vlm_processor

    monkeypatch.setattr(omni_mot_model, "lazy_instantiate", _instantiate)
    monkeypatch.setattr(omni_mot_model, "add_special_tokens", lambda tokenizer: (tokenizer, {}))

    model = SimpleNamespace(config=config)
    omni_mot_model.OmniMoTModel.set_up_tokenizers(model)

    assert instantiated == [vlm_config.tokenizer]
    assert model.tokenizer_vision_gen is None
    assert model.tokenizer_sound_gen is None


def test_default_setup_loads_vision_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    from cosmos_framework.model.generator import omni_mot_model

    vlm_tokenizer = SimpleNamespace(eos_token_id=42)
    vlm_processor = SimpleNamespace(tokenizer=vlm_tokenizer)
    vision_tokenizer = SimpleNamespace(latent_ch=48, reset_dtype=Mock())
    vlm_config = SimpleNamespace(tokenizer="vlm-tokenizer-config")
    vision_config = SimpleNamespace(temporal_compression_factor=4)
    config = SimpleNamespace(
        load_vision_tokenizer=True,
        lidar_tokenizer=None,
        sound_gen=False,
        state_ch=48,
        tokenizer=vision_config,
        vlm_config=vlm_config,
    )

    def _instantiate(candidate):
        if candidate == vlm_config.tokenizer:
            return vlm_processor
        assert candidate is vision_config
        return vision_tokenizer

    monkeypatch.setattr(omni_mot_model, "lazy_instantiate", _instantiate)
    monkeypatch.setattr(omni_mot_model, "add_special_tokens", lambda tokenizer: (tokenizer, {}))

    model = SimpleNamespace(config=config)
    omni_mot_model.OmniMoTModel.set_up_tokenizers(model)

    assert model.tokenizer_vision_gen is vision_tokenizer
    vision_tokenizer.reset_dtype.assert_called_once_with()


@pytest.mark.L0
def test_sampling_returns_the_rtc_action_prefix_unchanged() -> None:
    import torch

    from cosmos_framework.model.generator.diffusion.samplers.unipc import UniPCSampler
    from cosmos_framework.model.generator.omni_mot_model import keep_condition_values

    torch.manual_seed(0)
    chunk, action_dim, delay = 32, 20, 15
    plan = torch.randn(chunk, action_dim)  # [T,D] committed plan; rows < delay are the prefix
    mask = torch.zeros(chunk, action_dim)  # [T,D]
    mask[:delay] = 1.0
    target = torch.randn(chunk, action_dim)  # [T,D]
    noise = (mask * plan + (1.0 - mask) * torch.randn(chunk, action_dim)).reshape(-1)  # [T*D]

    def velocity_fn(latents: list[torch.Tensor], timestep: torch.Tensor) -> list[torch.Tensor]:
        # Like OmniMoTModel._get_velocity: zero velocity on conditioning entries.
        sigma = max(timestep.item() / 1000.0, 1e-3)
        return [(x - target.reshape(-1)) / sigma * (1.0 - mask.reshape(-1)) for x in latents]

    sampler = UniPCSampler(tensor_kwargs={"device": "cpu", "dtype": torch.float32})
    latents = sampler(velocity_fn, [noise], num_steps=2, shift=5.0, seed=[0])
    actions = keep_condition_values(latents, [plan.reshape(-1)], [mask.reshape(-1)])[0].reshape(chunk, action_dim)

    assert torch.equal(actions[:delay], plan[:delay])
    assert not torch.equal(actions[delay:], plan[delay:])
