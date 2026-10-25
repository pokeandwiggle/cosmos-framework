# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
from pathlib import Path

import pytest
import safetensors.torch
import torch
from torch import nn

from cosmos_framework.configs.base.defaults.quantization import QuantizationConfig
from cosmos_framework.utils.generator.quantization import (
    FP8_ACTIVATION_AMAX_FLOOR,
    _quantize_linear,
    _torchao_config,
    apply_modelopt_fp8_checkpoint_inplace,
    is_modelopt_fp8_checkpoint,
    quantize_per_layer_to_device,
)


class TinyLinearModel(nn.Module):
    selected: nn.Linear
    unselected: nn.Linear

    def __init__(self, device: torch.device | str = "cuda") -> None:
        super().__init__()
        self.selected = nn.Linear(16, 16, bias=False, device=device, dtype=torch.bfloat16)
        self.unselected = nn.Linear(16, 16, bias=False, device=device, dtype=torch.bfloat16)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # inputs: [B,D], returns: [B,D]
        return self.selected(inputs) + self.unselected(inputs)  # [B,D]


def _write_modelopt_fp8_checkpoint(
    checkpoint_path: Path,
    quantized_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: torch.Tensor,
    additional_module: str | None = None,
    additional_module_has_scales: bool = True,
) -> dict[str, str]:
    """Write a minimal ModelOpt-style FP8 export with weights and scales in separate shards."""
    checkpoint_path.mkdir()
    (checkpoint_path / "hf_quant_config.json").write_text(
        json.dumps({"quant_method": "modelopt", "quant_algo": "FP8"}),
        encoding="utf-8",
    )
    weight_shard_name = "model-00001-of-00002.safetensors"
    scale_shard_name = "model-00002-of-00002.safetensors"
    module_names = ["selected"]
    if additional_module is not None:
        module_names.append(additional_module)
    scale_module_names = module_names if additional_module_has_scales else ["selected"]
    weight_tensors = {f"{module_name}.weight": quantized_weight.clone() for module_name in module_names}
    scale_tensors = {
        key: value
        for module_name in scale_module_names
        for key, value in (
            (f"{module_name}.input_scale", input_scale.clone()),
            (f"{module_name}.weight_scale", weight_scale.clone()),
        )
    }
    safetensors.torch.save_file(weight_tensors, checkpoint_path / weight_shard_name)
    safetensors.torch.save_file(scale_tensors, checkpoint_path / scale_shard_name)
    weight_map = {
        **{f"{module_name}.weight": weight_shard_name for module_name in module_names},
        **{
            f"{module_name}.{scale_name}": scale_shard_name
            for module_name in scale_module_names
            for scale_name in ("input_scale", "weight_scale")
        },
    }
    (checkpoint_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}),
        encoding="utf-8",
    )
    return weight_map


def _identity_key_mapper(source_key: str, shard_path: str) -> str:
    del shard_path
    return source_key


def _ignore_extra_key_mapper(source_key: str, shard_path: str) -> str | None:
    del shard_path
    return None if source_key.startswith("ignored.") else source_key


def test_is_modelopt_fp8_checkpoint_without_config(tmp_path: Path) -> None:
    assert not is_modelopt_fp8_checkpoint(tmp_path)


def test_is_modelopt_fp8_checkpoint_rejects_malformed_config(tmp_path: Path) -> None:
    (tmp_path / "hf_quant_config.json").write_text("{", encoding="utf-8")

    with pytest.raises(ValueError, match="Unable to read a valid checkpoint quantization config"):
        is_modelopt_fp8_checkpoint(tmp_path)


@pytest.mark.parametrize(
    "quant_config",
    [
        pytest.param([], id="non-object"),
        pytest.param({"quant_method": "other", "quant_algo": "FP8"}, id="unsupported-method"),
        pytest.param({"quant_method": "modelopt", "quant_algo": "OTHER"}, id="unsupported-algorithm"),
    ],
)
def test_is_modelopt_fp8_checkpoint_rejects_unsupported_config(tmp_path: Path, quant_config: object) -> None:
    (tmp_path / "hf_quant_config.json").write_text(json.dumps(quant_config), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported checkpoint quantization configuration"):
        is_modelopt_fp8_checkpoint(tmp_path)


def test_apply_modelopt_fp8_checkpoint_preserves_exported_tensors(tmp_path: Path) -> None:
    model = TinyLinearModel(device="cpu").eval()
    original_selected = model.selected
    quantized_weight = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).reshape(16, 16)
    weight_scale = torch.tensor(0.125, dtype=torch.float32)
    input_scale = torch.tensor(0.25, dtype=torch.float32)
    weight_map = _write_modelopt_fp8_checkpoint(tmp_path / "checkpoint", quantized_weight, weight_scale, input_scale)

    converted = apply_modelopt_fp8_checkpoint_inplace(
        model,
        tmp_path / "checkpoint",
        key_mapper=_identity_key_mapper,
        weight_map=weight_map,
    )

    assert is_modelopt_fp8_checkpoint(tmp_path / "checkpoint")
    assert converted == ["selected"]
    assert isinstance(model.selected, nn.Linear)
    assert model.selected is not original_selected
    assert type(original_selected) is nn.Linear
    assert type(model.selected.weight).__name__ == "PrototypeFloat8Tensor"
    assert torch.equal(model.selected.weight.qdata.view(torch.uint8), quantized_weight.view(torch.uint8))
    assert torch.equal(model.selected.weight.scale, weight_scale.reshape(1, 1))
    assert torch.equal(model.selected.weight.act_quant_scale, input_scale.reshape(1, 1))
    assert type(model.unselected.weight) is nn.Parameter


def test_apply_modelopt_fp8_checkpoint_skips_ignored_keys(tmp_path: Path) -> None:
    model = TinyLinearModel(device="cpu").eval()
    quantized_weight = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).reshape(16, 16)
    weight_scale = torch.tensor(0.125, dtype=torch.float32)
    input_scale = torch.tensor(0.25, dtype=torch.float32)
    weight_map = _write_modelopt_fp8_checkpoint(
        tmp_path / "checkpoint",
        quantized_weight,
        weight_scale,
        input_scale,
        additional_module="ignored",
        additional_module_has_scales=False,
    )

    converted = apply_modelopt_fp8_checkpoint_inplace(
        model,
        tmp_path / "checkpoint",
        key_mapper=_ignore_extra_key_mapper,
        weight_map=weight_map,
    )

    assert converted == ["selected"]


def test_apply_modelopt_fp8_checkpoint_skips_absent_target(tmp_path: Path) -> None:
    model = TinyLinearModel(device="cpu").eval()
    quantized_weight = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).reshape(16, 16)
    weight_scale = torch.tensor(0.125, dtype=torch.float32)
    input_scale = torch.tensor(0.25, dtype=torch.float32)
    weight_map = _write_modelopt_fp8_checkpoint(
        tmp_path / "checkpoint",
        quantized_weight,
        weight_scale,
        input_scale,
        additional_module="missing",
        additional_module_has_scales=False,
    )

    converted = apply_modelopt_fp8_checkpoint_inplace(
        model,
        tmp_path / "checkpoint",
        key_mapper=_identity_key_mapper,
        weight_map=weight_map,
    )

    assert converted == ["selected"]


@pytest.mark.gpus(1)
def test_apply_modelopt_fp8_checkpoint_uses_torchao_linear_dispatch(tmp_path: Path) -> None:
    if torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("requires an Ada or newer GPU")
    pytest.importorskip("torchao")

    model = TinyLinearModel().eval()
    original_weight = model.selected.weight.detach().cpu()
    weight_scale = original_weight.abs().amax().float() / torch.finfo(torch.float8_e4m3fn).max
    quantized_weight = (original_weight / weight_scale).to(torch.float8_e4m3fn)
    input_scale = torch.tensor(0.025, dtype=torch.float32)
    weight_map = _write_modelopt_fp8_checkpoint(tmp_path / "checkpoint", quantized_weight, weight_scale, input_scale)
    apply_modelopt_fp8_checkpoint_inplace(
        model,
        tmp_path / "checkpoint",
        key_mapper=_identity_key_mapper,
        weight_map=weight_map,
    )

    inputs = torch.randn((6, 16), device="cuda", dtype=torch.bfloat16)
    output = model.selected(inputs)
    reasoning_inputs = torch.randn((2, 3, 16), device="cuda", dtype=torch.bfloat16)
    reasoning_output = model.selected(reasoning_inputs)
    output_after_reasoning = model.selected(inputs)
    compiled_model = torch.compile(model.selected, dynamic=True)
    compiled_output = compiled_model(inputs)
    compiled_reasoning_output = compiled_model(reasoning_inputs)
    empty_output = model.selected(inputs[:0])

    assert output.shape == (6, 16)
    assert torch.isfinite(output).all()
    assert reasoning_output.shape == (2, 3, 16)
    assert torch.isfinite(reasoning_output).all()
    assert output_after_reasoning.shape == (6, 16)
    assert torch.isfinite(output_after_reasoning).all()
    assert compiled_output.shape == (6, 16)
    assert torch.isfinite(compiled_output).all()
    assert compiled_reasoning_output.shape == (2, 3, 16)
    assert torch.isfinite(compiled_reasoning_output).all()
    assert empty_output.shape == (0, 16)
    assert model.selected.weight.act_quant_scale.shape == (1, 1)


def test_quantize_per_layer_to_device_never_holds_two_bf16_linears_on_device() -> None:
    # "meta" stands in for the GPU so the loader logic runs on CPU; the torchao
    # FP8 kernels themselves need CUDA (sm_89+) and are checked on a GPU only.
    model = nn.Sequential(
        *(
            nn.Sequential(nn.Linear(8, 8, dtype=torch.bfloat16), nn.BatchNorm1d(8, dtype=torch.bfloat16))
            for _ in range(3)
        ),
        nn.Linear(8, 8, dtype=torch.bfloat16),
    )
    config = QuantizationConfig(method="fp8", exclude_regex=[r"^3$"])
    linears = [module for module in model.modules() if isinstance(module, nn.Linear)]
    quantized: list[nn.Linear] = []

    def fake_quantize(linear: nn.Linear, quantization_config: QuantizationConfig) -> None:
        on_device_bf16 = [
            module for module in linears if module.weight.is_meta and module.weight.dtype == torch.bfloat16
        ]
        assert on_device_bf16 == [linear]
        assert quantization_config is config
        linear.weight = nn.Parameter(linear.weight.to(torch.float8_e4m3fn), requires_grad=False)
        quantized.append(linear)

    matched = quantize_per_layer_to_device(model, config, "meta", quantize_linear=fake_quantize)

    assert matched == ["0.0", "1.0", "2.0"]
    assert quantized == linears[:3]
    assert all(tensor.is_meta for tensor in [*model.parameters(), *model.buffers()])
    assert model[3].weight.dtype == torch.bfloat16


def test_quantize_per_layer_to_device_requires_a_method() -> None:
    with pytest.raises(ValueError, match="method"):
        quantize_per_layer_to_device(nn.Linear(4, 4), QuantizationConfig(), "meta")


@pytest.mark.parametrize("granularity", ["per_row", "per_tensor"])
def test_fp8_floors_the_activation_amax(granularity: str) -> None:
    config = _torchao_config(QuantizationConfig(method="fp8", fp8_granularity=granularity))
    assert config.activation_value_lb == FP8_ACTIVATION_AMAX_FLOOR > 0


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
    reason="torchao's FP8 kernels need CUDA sm_89+",
)
def test_fp8_keeps_an_all_zero_activation_row_finite() -> None:
    linear = nn.Linear(64, 64, bias=False, device="cuda", dtype=torch.bfloat16)
    _quantize_linear(linear, QuantizationConfig(method="fp8"))
    inputs = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)  # [B,D]
    inputs[1] = 0

    outputs = linear(inputs)  # [B,D]

    assert torch.isfinite(outputs).all()
    assert (outputs[1] == 0).all()
