# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for Qwen4-Exp MTP-specific MoE dimensions."""

from __future__ import annotations

from tests.qwen4_vendored import QWEN4_COMPAT, import_into


def _load_vendored():
    package = import_into(globals(), "mlx_vlm.models.qwen4_exp")
    global TextConfig, language
    TextConfig = package.TextConfig
    language = package.language


PATCH_ISOLATION = (QWEN4_COMPAT,)
PATCH_ISOLATION_SETUP = _load_vendored

# Bound by _load_vendored() before the first test runs.
TextConfig = None
language = None


def _config(
    *,
    mtp_num_experts: int | None = 512,
    mtp_num_experts_per_tok: int | None = 4,
) -> TextConfig:
    return TextConfig(
        model_type="qwen4_exp_text",
        hidden_size=8,
        num_hidden_layers=2,
        num_attention_heads=1,
        linear_num_value_heads=1,
        linear_num_key_heads=1,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=2,
        num_experts=384,
        num_experts_per_tok=10,
        shared_expert_intermediate_size=8,
        moe_intermediate_size=8,
        rms_norm_eps=1e-6,
        vocab_size=16,
        num_key_value_heads=1,
        max_position_embeddings=16,
        heads_per_ngram=1,
        mtp_num_experts=mtp_num_experts,
        mtp_num_experts_per_tok=mtp_num_experts_per_tok,
    )


def test_mtp_module_uses_its_own_expert_counts(monkeypatch):
    captured = {}

    class FakeDecoderLayer:
        def __init__(self, config, *, layer_idx):
            captured["num_experts"] = config.num_experts
            captured["num_experts_per_tok"] = config.num_experts_per_tok
            captured["layer_idx"] = layer_idx

    class FakeGatedResidual:
        def __init__(self, config, *, use_combine):
            captured["residual_num_experts"] = config.num_experts
            captured["use_combine"] = use_combine

    monkeypatch.setattr(language, "Qwen4ExpDecoderLayer", FakeDecoderLayer)
    monkeypatch.setattr(language, "Qwen4ExpGatedResidual", FakeGatedResidual)

    language.Qwen4ExpMTPModule(_config())

    assert captured == {
        "num_experts": 512,
        "num_experts_per_tok": 4,
        "layer_idx": 0,
        "residual_num_experts": 512,
        "use_combine": False,
    }


def test_mtp_module_defaults_to_model_expert_counts(monkeypatch):
    captured = {}

    class FakeDecoderLayer:
        def __init__(self, config, *, layer_idx):
            captured["num_experts"] = config.num_experts
            captured["num_experts_per_tok"] = config.num_experts_per_tok

    class FakeGatedResidual:
        def __init__(self, config, *, use_combine):
            captured["residual_num_experts"] = config.num_experts

    monkeypatch.setattr(language, "Qwen4ExpDecoderLayer", FakeDecoderLayer)
    monkeypatch.setattr(language, "Qwen4ExpGatedResidual", FakeGatedResidual)

    language.Qwen4ExpMTPModule(
        _config(mtp_num_experts=None, mtp_num_experts_per_tok=None)
    )

    assert captured == {
        "num_experts": 384,
        "num_experts_per_tok": 10,
        "residual_num_experts": 384,
    }


def test_mtp_module_falls_back_per_missing_override(monkeypatch):
    captured = {}

    class FakeDecoderLayer:
        def __init__(self, config, *, layer_idx):
            captured["num_experts"] = config.num_experts
            captured["num_experts_per_tok"] = config.num_experts_per_tok

    class FakeGatedResidual:
        def __init__(self, config, *, use_combine):
            pass

    monkeypatch.setattr(language, "Qwen4ExpDecoderLayer", FakeDecoderLayer)
    monkeypatch.setattr(language, "Qwen4ExpGatedResidual", FakeGatedResidual)

    language.Qwen4ExpMTPModule(_config(mtp_num_experts=None))

    assert captured == {
        "num_experts": 384,
        "num_experts_per_tok": 4,
    }
