"""Gemma 4 (google/gemma-4-E2B) in plain PyTorch — no `transformers` import anywhere.

Ported line by line from transformers v5.16.1:
    transformers/models/gemma4/modeling_gemma4.py
    transformers/models/gemma4/configuration_gemma4.py

Design rule: **mirror the reference module tree exactly** so that
`model.load_state_dict(ckpt)` works on the raw HuggingFace safetensors
without any key renaming. Every class/parameter name below matches the
reference implementation on purpose; do not "clean up" the names.

Covers
  * text tower: RMSNorm, QK-norm attention, per-layer-type RoPE
    (default / proportional), sliding-window + full attention, KV sharing,
    double-wide MLP on KV-shared layers, Per-Layer Embeddings (PLE)
  * vision tower: patch embedder with 2-D learned positions, bidirectional
    encoder with 2-D RoPE, spatial pooling to `max_soft_tokens`
  * audio tower: SSCP conv subsampling, chunked local attention with
    relative position bias, LightConv, conformer-ish block
  * multimodal stitching: soft-token scatter into the text embedding stream
  * head + loss: `tanh` logit soft-capping, shifted cross entropy

Not covered (deliberately, see README): KV cache / generation loop.
The forward+loss path is exact and verified against the reference.
"""

from __future__ import annotations

import json
import math
import os
from collections import UserDict
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn

try:  # imported as a package (normal case)
    from .kv_cache import KVCache
except ImportError:  # `python modeling_gemma4.py` from inside the folder
    from kv_cache import KVCache

__all__ = [
    "Gemma4Config",
    "Gemma4TextConfig",
    "Gemma4VisionConfig",
    "Gemma4AudioConfig",
    "Gemma4ForConditionalGeneration",
    "load_gemma4_from_safetensors",
]


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _act(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    table = {
        "gelu_pytorch_tanh": lambda x: F.gelu(x, approximate="tanh"),
        "gelu": F.gelu,
        "gelu_new": lambda x: F.gelu(x, approximate="tanh"),
        "silu": F.silu,
        "swish": F.silu,
        "relu": F.relu,
    }
    if name not in table:
        raise KeyError(f"unsupported activation {name!r}")
    return table[name]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim: int = 1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def _min_dtype(dtype: torch.dtype) -> float:
    return torch.finfo(dtype).min


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
@dataclass
class Gemma4TextConfig:
    vocab_size: int = 262_144
    hidden_size: int = 2304
    intermediate_size: int = 9216
    num_hidden_layers: int = 30
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    head_dim: int = 256
    global_head_dim: int = 512
    hidden_activation: str = "gelu_pytorch_tanh"
    max_position_embeddings: int = 131_072
    rms_norm_eps: float = 1e-6
    pad_token_id: int = 0
    eos_token_id: int = 1
    bos_token_id: int = 2
    tie_word_embeddings: bool = True
    attention_bias: bool = False
    attention_dropout: float = 0.0
    sliding_window: int = 512
    layer_types: list[str] | None = None
    final_logit_softcapping: float | None = None
    use_bidirectional_attention: str | None = None
    vocab_size_per_layer_input: int = 262_144
    hidden_size_per_layer_input: int = 256
    attention_k_eq_v: bool = False
    num_kv_shared_layers: int = 0
    num_global_key_value_heads: int | None = None
    enable_moe_block: bool = False
    use_double_wide_mlp: bool = False
    num_experts: int | None = None
    top_k_experts: int | None = None
    moe_intermediate_size: int | None = None
    rope_parameters: dict | None = None

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention" if bool((i + 1) % 6) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        if self.layer_types and self.layer_types[-1] != "full_attention":
            self.layer_types[-1] = "full_attention"
        if self.rope_parameters is None:
            self.rope_parameters = {
                "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
                "full_attention": {
                    "rope_type": "proportional",
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 1_000_000.0,
                },
            }

    # --- per-layer-type views (reference: config.per_layer_config[layer_type]) ---
    def layer_head_dim(self, layer_idx: int) -> int:
        return self.global_head_dim if self.layer_types[layer_idx] == "full_attention" else self.head_dim

    def layer_num_kv_heads(self, layer_idx: int) -> int:
        if self.layer_types[layer_idx] == "full_attention" and self.attention_k_eq_v:
            if self.num_global_key_value_heads is not None:
                return self.num_global_key_value_heads
        return self.num_key_value_heads

    def type_head_dim(self, layer_type: str) -> int:
        return self.global_head_dim if layer_type == "full_attention" else self.head_dim

    @property
    def first_kv_shared_layer_idx(self) -> int:
        return self.num_hidden_layers - self.num_kv_shared_layers


@dataclass
class Gemma4VisionConfig:
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 16
    num_attention_heads: int = 12
    num_key_value_heads: int = 12
    head_dim: int = 64
    hidden_activation: str = "gelu_pytorch_tanh"
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    attention_bias: bool = False
    patch_size: int = 16
    pooling_kernel_size: int = 3
    position_embedding_size: int = 10 * 1024
    max_position_embeddings: int = 131_072
    rope_parameters: dict | None = None
    use_clipped_linears: bool = False
    standardize: bool = False
    initializer_range: float = 0.02

    def __post_init__(self):
        if self.rope_parameters is None:
            self.rope_parameters = {"rope_type": "default", "rope_theta": 100.0}


@dataclass
class Gemma4AudioConfig:
    hidden_size: int = 1024
    num_hidden_layers: int = 12
    num_attention_heads: int = 8
    hidden_act: str = "silu"
    subsampling_conv_channels: list[int] = field(default_factory=lambda: [128, 32])
    conv_kernel_size: int = 5
    residual_weight: float = 0.5
    attention_chunk_size: int = 12
    attention_context_left: int = 13
    attention_context_right: int = 0
    attention_logit_cap: float = 50.0
    attention_invalid_logits_value: float = -1.0e9
    use_clipped_linears: bool = True
    rms_norm_eps: float = 1e-6
    gradient_clipping: float = 1e10
    output_proj_dims: int = 1536


@dataclass
class Gemma4Config:
    text_config: Gemma4TextConfig
    vision_config: Gemma4VisionConfig | None = None
    audio_config: Gemma4AudioConfig | None = None
    image_token_id: int = 258_880
    video_token_id: int = 258_884
    audio_token_id: int = 258_881
    boi_token_id: int = 255_999
    eoi_token_id: int = 258_882
    boa_token_id: int = 256_000
    eoa_token_index: int = 258_883
    tie_word_embeddings: bool = True
    # top-level "dtype" string from config.json ("bfloat16" ...).
    # `from_pretrained` uses it as the default load dtype — see load_gemma4_from_safetensors(dtype="auto").
    dtype: str | None = None

    @classmethod
    def from_dict(cls, raw: dict) -> "Gemma4Config":
        def sub(klass, d):
            if d is None:
                return None
            keep = {f.name for f in klass.__dataclass_fields__.values()}
            return klass(**{k: v for k, v in d.items() if k in keep})

        return cls(
            text_config=sub(Gemma4TextConfig, raw.get("text_config")) or Gemma4TextConfig(),
            vision_config=sub(Gemma4VisionConfig, raw.get("vision_config")),
            audio_config=sub(Gemma4AudioConfig, raw.get("audio_config")),
            image_token_id=raw.get("image_token_id", 258_880),
            video_token_id=raw.get("video_token_id", 258_884),
            audio_token_id=raw.get("audio_token_id", 258_881),
            boi_token_id=raw.get("boi_token_id", 255_999),
            eoi_token_id=raw.get("eoi_token_id", 258_882),
            boa_token_id=raw.get("boa_token_id", 256_000),
            eoa_token_index=raw.get("eoa_token_index", 258_883),
            tie_word_embeddings=raw.get("tie_word_embeddings", True),
            dtype=raw.get("dtype", None),
        )

    @classmethod
    def from_json(cls, path: str) -> "Gemma4Config":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------
class Gemma4RMSNorm(nn.Module):
    """Reference keeps the norm math in float32 and casts back at the end."""

    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, hidden_states):
        mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        return hidden_states * torch.pow(mean_squared, -0.5)

    def forward(self, hidden_states):
        normed_output = self._norm(hidden_states.float())
        if self.with_scale:
            normed_output = normed_output * self.weight.float()
        return normed_output.type_as(hidden_states)


class Gemma4ClippableLinear(nn.Module):
    """`nn.Linear` + loadable clamp bounds (kept as buffers: they are in the ckpt)."""

    def __init__(self, config, in_features: int, out_features: int):
        super().__init__()
        self.use_clipped_linears = getattr(config, "use_clipped_linears", False)
        self.linear = nn.Linear(in_features, out_features, bias=False)
        if self.use_clipped_linears:
            self.input_min = nn.Buffer(torch.tensor(-float("inf")))
            self.input_max = nn.Buffer(torch.tensor(float("inf")))
            self.output_min = nn.Buffer(torch.tensor(-float("inf")))
            self.output_max = nn.Buffer(torch.tensor(float("inf")))

    def forward(self, hidden_states):
        if self.use_clipped_linears:
            hidden_states = torch.clamp(hidden_states, self.input_min, self.input_max)
        hidden_states = self.linear(hidden_states)
        if self.use_clipped_linears:
            hidden_states = torch.clamp(hidden_states, self.output_min, self.output_max)
        return hidden_states


class Gemma4TextScaledWordEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, padding_idx, embed_scale: float = 1.0):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.scalar_embed_scale = embed_scale
        self.embed_scale = nn.Buffer(torch.tensor(embed_scale), persistent=False)

    def forward(self, input_ids):
        return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)


# ---------------------------------------------------------------------------
# text tower
# ---------------------------------------------------------------------------
class Gemma4TextRotaryEmbedding(nn.Module):
    """One inv_freq tensor per layer type (default / proportional RoPE).

    NOTE: inv_freq is deliberately a plain attribute, **not** a registered buffer.
    `nn.Module._apply` (i.e. `model.to(dtype)`) casts every floating buffer, and
    bf16-casting inv_freq destroys the angles permanently: bf16(0.1) = 0.10009765625,
    and upcasting back to float32 cannot recover it. The reference keeps inv_freq in
    float32 and upcasts inside `forward`; we mirror that by never letting it be cast
    (measured effect when it does get cast: 1.26e-2 logits drift vs the reference,
    starting at the very first q@k matmul). Not a buffer also means it is not in
    `state_dict`, which is correct -- it is derived from config, not from the ckpt.
    """

    def __init__(self, config: Gemma4TextConfig):
        super().__init__()
        self.config = config
        self.layer_types = sorted(set(config.layer_types))
        self._inv_freqs: dict[str, torch.Tensor] = {}
        for layer_type in self.layer_types:
            self._inv_freqs[layer_type] = self._build_inv_freq(layer_type)

    def _build_inv_freq(self, layer_type: str) -> torch.Tensor:
        params = self.config.rope_parameters[layer_type]
        head_dim = self.config.type_head_dim(layer_type)
        base = params["rope_theta"]
        if params["rope_type"] == "default":
            return 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
        if params["rope_type"] != "proportional":
            raise NotImplementedError(f"rope_type={params['rope_type']} not ported")
        # proportional: rotated angles first, then zero-frequency (identity) angles
        rope_angles = int(params.get("partial_rotary_factor", 1.0) * head_dim // 2)
        inv_freq_rotated = 1.0 / (
            base ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.float) / head_dim)
        )
        nope_angles = head_dim // 2 - rope_angles
        if nope_angles > 0:
            inv_freq = torch.cat([inv_freq_rotated, torch.zeros(nope_angles)], dim=0)
        else:
            inv_freq = inv_freq_rotated
        return inv_freq / params.get("factor", 1.0)

    @torch.no_grad()
    def forward(self, x, position_ids, layer_type: str):
        inv_freq = self._inv_freqs[layer_type].to(device=x.device, dtype=torch.float)
        inv_freq_expanded = inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)


class Gemma4TextAttention(nn.Module):
    def __init__(self, config: Gemma4TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.is_sliding = self.layer_type == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_sliding else None

        self.head_dim = config.layer_head_dim(layer_idx)
        self.num_kv_heads = config.layer_num_kv_heads(layer_idx)
        self.use_alternative_attention = config.attention_k_eq_v and not self.is_sliding
        self.num_key_value_groups = config.num_attention_heads // self.num_kv_heads
        self.scaling = 1.0  # QK-norm makes 1/sqrt(d) redundant
        self.attention_dropout = config.attention_dropout
        self.is_causal = config.use_bidirectional_attention != "all"

        first_shared = config.first_kv_shared_layer_idx
        self.is_kv_shared_layer = layer_idx >= first_shared >= 0
        prev_layers = config.layer_types[:first_shared]
        self.store_full_length_kv = (
            not self.is_kv_shared_layer
            and layer_idx == len(prev_layers) - 1 - prev_layers[::-1].index(config.layer_types[layer_idx])
        )

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.q_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # shared layers carry no k/v projections at run time (the ckpt still has them)
        if not self.is_kv_shared_layer:
            self.k_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.v_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps, with_scale=False)
            self.k_proj = nn.Linear(
                config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias
            )
            self.v_proj = (
                nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
                if not self.use_alternative_attention
                else None
            )

        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    def forward(self, hidden_states, position_embeddings, attention_mask, shared_kv_states,
                past_key_values: KVCache | None = None):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        cos, sin = position_embeddings

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape))
        query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2).transpose(1, 2)

        if self.is_kv_shared_layer:
            # No k_proj/v_proj here: reuse the k/v produced earlier by the layer that
            # owns this layer type.  We read `shared_kv_states` rather than the cache
            # because a sliding layer's cache may not hold the full history it needs
            # (see kv_cache.py, `crop`).
            key_states, value_states = shared_kv_states[self.layer_type]
            key_states = key_states.to(query_states.device)
            value_states = value_states.to(query_states.device)
        else:
            key_states = self.k_proj(hidden_states).view(hidden_shape)
            value_states = self.v_proj(hidden_states).view(hidden_shape) if self.v_proj is not None else key_states
            key_states = self.k_norm(key_states)
            key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2).transpose(1, 2)
            value_states = self.v_norm(value_states).transpose(1, 2)

            if past_key_values is not None:
                # Append the new positions; get back the whole history, which may be
                # longer than `hidden_states` (prefill) or longer than 1 (decode).
                key_states, value_states = past_key_values.update(
                    key_states, value_states, self.layer_idx
                )

        if self.store_full_length_kv:
            # Written *after* the cache update on purpose: shared layers downstream
            # need the full history, not just this chunk.
            shared_kv_states[self.layer_type] = (key_states, value_states)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
        return self.o_proj(attn_output)


class Gemma4TextMLP(nn.Module):
    def __init__(self, config: Gemma4TextConfig, layer_idx: int):
        super().__init__()
        first_shared = config.first_kv_shared_layer_idx
        is_kv_shared_layer = layer_idx >= first_shared > 0
        use_double_wide_mlp = config.use_double_wide_mlp and is_kv_shared_layer
        self.intermediate_size = config.intermediate_size * (2 if use_double_wide_mlp else 1)
        self.gate_proj = nn.Linear(config.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = _act(config.hidden_activation)

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Gemma4TextExperts(nn.Module):
    """MoE experts (unused by E2B, kept so that larger Gemma 4 checkpoints load)."""

    def __init__(self, config: Gemma4TextConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = _act(config.hidden_activation)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        return final_hidden_states


class Gemma4TextRouter(nn.Module):
    def __init__(self, config: Gemma4TextConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.scalar_root_size = self.hidden_size**-0.5
        self.norm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps, with_scale=False)
        self.proj = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.scale = nn.Parameter(torch.ones(self.hidden_size))
        self.per_expert_scale = nn.Parameter(torch.ones(config.num_experts))

    def forward(self, hidden_states):
        hidden_states = self.norm(hidden_states) * self.scale * self.scalar_root_size
        expert_scores = self.proj(hidden_states)
        router_probabilities = nn.functional.softmax(expert_scores, dim=-1, dtype=torch.float32)
        top_k_weights, top_k_index = torch.topk(router_probabilities, k=self.config.top_k_experts, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        top_k_weights = top_k_weights * self.per_expert_scale[top_k_index]
        return router_probabilities, top_k_weights, top_k_index


class Gemma4TextDecoderLayer(nn.Module):
    def __init__(self, config: Gemma4TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.self_attn = Gemma4TextAttention(config, layer_idx)
        self.mlp = Gemma4TextMLP(config, layer_idx)
        self.input_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layer_scalar = nn.Buffer(torch.ones(1))

        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        if self.hidden_size_per_layer_input:
            self.act_fn = _act(config.hidden_activation)
            self.per_layer_input_gate = nn.Linear(config.hidden_size, self.hidden_size_per_layer_input, bias=False)
            self.per_layer_projection = nn.Linear(self.hidden_size_per_layer_input, config.hidden_size, bias=False)
            self.post_per_layer_input_norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.enable_moe_block = config.enable_moe_block
        if self.enable_moe_block:
            self.router = Gemma4TextRouter(config)
            self.experts = Gemma4TextExperts(config)
            self.post_feedforward_layernorm_1 = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_feedforward_layernorm_2 = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.pre_feedforward_layernorm_2 = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, per_layer_input, shared_kv_states, position_embeddings,
                attention_mask, past_key_values: KVCache | None = None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            shared_kv_states=shared_kv_states,
            past_key_values=past_key_values,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        if self.enable_moe_block:
            hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states)
            hidden_states_flat = residual.reshape(-1, residual.shape[-1])
            _, top_k_weights, top_k_index = self.router(hidden_states_flat)
            hidden_states_2 = self.pre_feedforward_layernorm_2(hidden_states_flat)
            hidden_states_2 = self.experts(hidden_states_2, top_k_index, top_k_weights)
            hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2.reshape(residual.shape))
            hidden_states = hidden_states_1 + hidden_states_2

        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        if self.hidden_size_per_layer_input:
            residual = hidden_states
            hidden_states = self.act_fn(self.per_layer_input_gate(hidden_states))
            hidden_states = hidden_states * per_layer_input
            hidden_states = self.per_layer_projection(hidden_states)
            hidden_states = self.post_per_layer_input_norm(hidden_states)
            hidden_states = residual + hidden_states

        return hidden_states * self.layer_scalar


# ---------------------------------------------------------------------------
# masks
# ---------------------------------------------------------------------------
def build_causal_mask(batch_size, q_len, kv_len, dtype, device, sliding_window=None,
                      attention_mask=None, bidirectional_blocks=None, is_causal=True):
    """Additive 0/-min mask, shape (B, 1, q_len, kv_len).

    bidirectional_blocks: (B, kv_len) int tensor; equal values attend bidirectionally.
    During decode `q_len < kv_len`: the queries sit at the *end* of the cached
    history, so their absolute positions are offset by `kv_len - q_len`.  Prefill
    has `q_len == kv_len` and the offset is 0, so this is a no-op there.
    """
    q_offset = kv_len - q_len
    q_idx = torch.arange(q_len, device=device)[:, None] + q_offset
    kv_idx = torch.arange(kv_len, device=device)[None, :]
    allowed = torch.ones((q_len, kv_len), dtype=torch.bool, device=device) if not is_causal else kv_idx <= q_idx
    if is_causal and bidirectional_blocks is not None:
        same_block = bidirectional_blocks[:, kv_idx] == bidirectional_blocks[:, q_idx]  # (B,q,kv)
        same_block = same_block & (bidirectional_blocks[:, kv_idx] >= 0)
        allowed = allowed | same_block
    if sliding_window is not None:
        allowed = allowed & (kv_idx > q_idx - sliding_window)
    mask = torch.zeros((batch_size, 1, q_len, kv_len), dtype=dtype, device=device)
    mask = mask.masked_fill(~allowed, _min_dtype(dtype))
    if attention_mask is not None:
        padding = attention_mask[:, None, None, :].to(torch.bool)
        mask = mask.masked_fill(~padding, _min_dtype(dtype))
    return mask


def build_bidirectional_allowed(batch_size, q_len, kv_len, device, valid=None, sliding_window=None):
    """Boolean (B, 1, q, kv) 'allowed' mask (True = attends)."""
    allowed = torch.ones((q_len, kv_len), dtype=torch.bool, device=device)
    if sliding_window is not None:
        left, right = sliding_window
        q_idx = torch.arange(q_len, device=device)[:, None]
        kv_idx = torch.arange(kv_len, device=device)[None, :]
        dist = q_idx - kv_idx
        allowed = ((dist >= 0) & (dist < left)) | ((dist < 0) & (-dist < right))
    allowed = allowed[None, None].expand(batch_size, 1, q_len, kv_len).clone()
    if valid is not None:
        allowed = allowed & valid[:, None, None, :].to(device).to(torch.bool)
    return allowed


def build_bidirectional_mask(batch_size, q_len, kv_len, dtype, device, valid=None, sliding_window=None):
    allowed = build_bidirectional_allowed(batch_size, q_len, kv_len, device, valid, sliding_window)
    return torch.zeros(allowed.shape, dtype=dtype, device=device).masked_fill(~allowed, _min_dtype(dtype))


def block_sequence_ids_for_mask(mm_token_type_ids: torch.Tensor, device) -> torch.Tensor:
    mm = mm_token_type_ids.to(device)
    is_vision = (mm == 1) | (mm == 2)
    is_prev_vision = torch.roll(is_vision, shifts=1, dims=-1)
    is_prev_vision[..., 0] = False
    new_vision_starts = is_vision & ~is_prev_vision
    vision_group_ids = torch.cumsum(new_vision_starts.int(), dim=1) - 1
    return torch.where(is_vision, vision_group_ids, -1)


# ---------------------------------------------------------------------------
# text model
# ---------------------------------------------------------------------------
@dataclass
class Gemma4TextModelOutput:
    last_hidden_state: torch.Tensor
    past_key_values: KVCache | None = None


class Gemma4TextModel(nn.Module):
    def __init__(self, config: Gemma4TextConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = Gemma4TextScaledWordEmbedding(
            config.vocab_size, config.hidden_size, self.padding_idx, embed_scale=config.hidden_size**0.5
        )
        self.layers = nn.ModuleList([Gemma4TextDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma4TextRotaryEmbedding(config)
        self.unique_layer_types = set(config.layer_types)

        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        if self.hidden_size_per_layer_input:
            self.embed_tokens_per_layer = Gemma4TextScaledWordEmbedding(
                config.vocab_size_per_layer_input,
                config.num_hidden_layers * config.hidden_size_per_layer_input,
                self.padding_idx,
                embed_scale=config.hidden_size_per_layer_input**0.5,
            )
            self.per_layer_input_scale = 2.0**-0.5
            self.per_layer_model_projection = nn.Linear(
                config.hidden_size, config.num_hidden_layers * config.hidden_size_per_layer_input, bias=False
            )
            self.per_layer_model_projection_scale = config.hidden_size**-0.5
            self.per_layer_projection_norm = Gemma4RMSNorm(config.hidden_size_per_layer_input, eps=config.rms_norm_eps)

    # --- Per-Layer Embeddings (PLE) ---
    def get_per_layer_inputs(self, input_ids: torch.Tensor) -> torch.Tensor:
        if not self.hidden_size_per_layer_input:
            raise RuntimeError("config has no per-layer embeddings")
        return self.embed_tokens_per_layer(input_ids).reshape(
            *input_ids.shape, self.config.num_hidden_layers, self.hidden_size_per_layer_input
        )

    def project_per_layer_inputs(self, inputs_embeds, per_layer_inputs=None):
        per_layer_projection = self.per_layer_model_projection(inputs_embeds) * self.per_layer_model_projection_scale
        per_layer_projection = per_layer_projection.reshape(
            *inputs_embeds.shape[:-1], self.config.num_hidden_layers, self.hidden_size_per_layer_input
        )
        per_layer_projection = self.per_layer_projection_norm(per_layer_projection)
        if per_layer_inputs is None:
            return per_layer_projection
        return (per_layer_projection + per_layer_inputs) * self.per_layer_input_scale

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, per_layer_inputs=None,
                position_ids=None, mm_token_type_ids=None, past_key_values: KVCache | None = None,
                use_cache: bool = False):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("specify exactly one of input_ids / inputs_embeds")
        if input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self.hidden_size_per_layer_input:
            if per_layer_inputs is None:
                if input_ids is None:
                    raise ValueError(
                        "per_layer_inputs must be passed when calling with inputs_embeds only "
                        "(reversing the embedding table is not implemented)"
                    )
                per_layer_inputs = self.get_per_layer_inputs(input_ids)
            per_layer_inputs = self.project_per_layer_inputs(inputs_embeds, per_layer_inputs)

        batch_size, seq_len = inputs_embeds.shape[:2]
        device, dtype = inputs_embeds.device, inputs_embeds.dtype

        if use_cache and past_key_values is None:
            past_key_values = KVCache(len(self.layers))
        past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
        kv_len = past_seen + seq_len

        if attention_mask is not None and attention_mask.shape[-1] != kv_len:
            raise ValueError(
                f"attention_mask covers {attention_mask.shape[-1]} positions but this call needs "
                f"{kv_len} ({past_seen} cached + {seq_len} new). During decode the mask must span the "
                f"whole history, not just the new token."
            )

        if position_ids is None:
            # Absolute positions: on decode the new token is *not* at index 0.
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0) + past_seen

        block_ids = None
        if self.config.use_bidirectional_attention == "vision" and mm_token_type_ids is not None:
            block_ids = block_sequence_ids_for_mask(mm_token_type_ids, device)

        mask_kwargs = dict(batch_size=batch_size, q_len=seq_len, kv_len=kv_len, dtype=dtype,
                           device=device, attention_mask=attention_mask)
        masks = {
            "full_attention": build_causal_mask(**mask_kwargs),
            "sliding_attention": build_causal_mask(**mask_kwargs, sliding_window=self.config.sliding_window),
        }
        if block_ids is not None:
            # global layers: causal only; local layers: AND(sliding, OR(causal, blockwise))
            masks["full_attention"] = build_causal_mask(**mask_kwargs)
            masks["sliding_attention"] = build_causal_mask(
                **mask_kwargs, sliding_window=self.config.sliding_window, bidirectional_blocks=block_ids
            )

        position_embeddings = {
            layer_type: self.rotary_emb(inputs_embeds, position_ids, layer_type)
            for layer_type in self.unique_layer_types
        }

        shared_kv_states = UserDict()
        hidden_states = inputs_embeds
        for i, decoder_layer in enumerate(self.layers):
            per_layer_input = per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None
            hidden_states = decoder_layer(
                hidden_states,
                per_layer_input,
                shared_kv_states=shared_kv_states,
                position_embeddings=position_embeddings[self.config.layer_types[i]],
                attention_mask=masks[self.config.layer_types[i]],
                past_key_values=past_key_values,
            )
        return Gemma4TextModelOutput(
            last_hidden_state=self.norm(hidden_states),
            past_key_values=past_key_values,
        )


# ---------------------------------------------------------------------------
# vision tower
# ---------------------------------------------------------------------------
class Gemma4VisionPatchEmbedder(nn.Module):
    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.position_embedding_size = config.position_embedding_size
        self.input_proj = nn.Linear(3 * self.patch_size**2, config.hidden_size, bias=False)
        self.position_embedding_table = nn.Parameter(torch.ones(2, self.position_embedding_size, config.hidden_size))

    def _position_embeddings(self, pixel_position_ids, padding_positions):
        clamped = pixel_position_ids.clamp(min=0)
        x_emb = F.embedding(clamped[..., 0], self.position_embedding_table[0])
        y_emb = F.embedding(clamped[..., 1], self.position_embedding_table[1])
        position_embeddings = x_emb + y_emb
        return torch.where(padding_positions.unsqueeze(-1), 0.0, position_embeddings)

    def forward(self, pixel_values, pixel_position_ids, padding_positions):
        pixel_values = 2 * (pixel_values - 0.5)
        if (target_dtype := self.input_proj.weight.dtype).is_floating_point:
            pixel_values = pixel_values.to(target_dtype)
        return self.input_proj(pixel_values) + self._position_embeddings(pixel_position_ids, padding_positions)


class Gemma4VisionRotaryEmbedding(nn.Module):
    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        base = config.rope_parameters["rope_theta"]
        dim = config.head_dim or config.hidden_size // config.num_attention_heads
        spatial_dim = dim // 2
        # plain attribute, not a buffer -- see Gemma4TextRotaryEmbedding for why
        self._inv_freq = 1.0 / (base ** (torch.arange(0, spatial_dim, 2, dtype=torch.float) / spatial_dim))

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq = self._inv_freq.to(device=x.device, dtype=torch.float)
        inv_freq_expanded = inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1)
        all_cos, all_sin = [], []
        for i in range(2):
            dim_position_ids = position_ids[:, :, i]
            freqs = (inv_freq_expanded @ dim_position_ids[:, None, :].float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            all_cos.append(emb.cos())
            all_sin.append(emb.sin())
        return torch.cat(all_cos, dim=-1).to(x.dtype), torch.cat(all_sin, dim=-1).to(x.dtype)


def apply_multidimensional_rope(x, cos, sin, position_ids, unsqueeze_dim=2):
    ndim = position_ids.shape[-1]
    num_input_channels = x.shape[-1]
    n = 2 * (num_input_channels // (2 * ndim))
    x_parts = torch.split(x, [n] * ndim, dim=-1)
    cos_parts = torch.split(cos, [n] * ndim, dim=-1)
    sin_parts = torch.split(sin, [n] * ndim, dim=-1)
    return torch.cat(
        [apply_rotary_pos_emb(x_parts[k], cos_parts[k], sin_parts[k], unsqueeze_dim) for k in range(ndim)], dim=-1
    )


class Gemma4VisionAttention(nn.Module):
    def __init__(self, config: Gemma4VisionConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.head_dim = config.head_dim or config.hidden_size // config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = 1.0
        self.attention_dropout = config.attention_dropout
        self.q_proj = Gemma4ClippableLinear(config, config.hidden_size, config.num_attention_heads * self.head_dim)
        self.k_proj = Gemma4ClippableLinear(config, config.hidden_size, config.num_key_value_heads * self.head_dim)
        self.v_proj = Gemma4ClippableLinear(config, config.hidden_size, config.num_key_value_heads * self.head_dim)
        self.o_proj = Gemma4ClippableLinear(config, config.num_attention_heads * self.head_dim, config.hidden_size)
        self.q_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps, with_scale=False)

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, position_ids=None):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        cos, sin = position_embeddings

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape))
        query_states = apply_multidimensional_rope(query_states, cos, sin, position_ids).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape))
        key_states = apply_multidimensional_rope(key_states, cos, sin, position_ids).transpose(1, 2)
        value_states = self.v_norm(self.v_proj(hidden_states).view(hidden_shape)).transpose(1, 2)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
        return self.o_proj(attn_output)


class Gemma4VisionMLP(nn.Module):
    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.gate_proj = Gemma4ClippableLinear(config, config.hidden_size, config.intermediate_size)
        self.up_proj = Gemma4ClippableLinear(config, config.hidden_size, config.intermediate_size)
        self.down_proj = Gemma4ClippableLinear(config, config.intermediate_size, config.hidden_size)
        self.act_fn = _act(config.hidden_activation)

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Gemma4VisionEncoderLayer(nn.Module):
    def __init__(self, config: Gemma4VisionConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Gemma4VisionAttention(config, layer_idx)
        self.mlp = Gemma4VisionMLP(config)
        self.input_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, position_ids=None):
        residual = hidden_states
        hidden_states = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings, attention_mask, position_ids
        )
        hidden_states = residual + self.post_attention_layernorm(hidden_states)

        residual = hidden_states
        hidden_states = self.mlp(self.pre_feedforward_layernorm(hidden_states))
        return residual + self.post_feedforward_layernorm(hidden_states)


class Gemma4VisionEncoder(nn.Module):
    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.config = config
        self.rotary_emb = Gemma4VisionRotaryEmbedding(config)
        self.layers = nn.ModuleList([Gemma4VisionEncoderLayer(config, i) for i in range(config.num_hidden_layers)])

    def forward(self, inputs_embeds, attention_mask, pixel_position_ids):
        batch_size, seq_len = inputs_embeds.shape[:2]
        mask = None
        if attention_mask is not None:
            mask = build_bidirectional_mask(
                batch_size, seq_len, seq_len, inputs_embeds.dtype, inputs_embeds.device, valid=attention_mask
            )
        position_embeddings = self.rotary_emb(inputs_embeds, pixel_position_ids)
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings, mask, pixel_position_ids)
        return hidden_states


class Gemma4VisionPooler(nn.Module):
    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.root_hidden_size = config.hidden_size**0.5

    def _avg_pool_by_positions(self, hidden_states, pixel_position_ids, length):
        input_seq_len = hidden_states.shape[1]
        k = int((input_seq_len // length) ** 0.5)
        k_squared = k**2
        if k_squared * length != input_seq_len:
            raise ValueError(f"cannot pool {hidden_states.shape} to {length}")
        clamped = pixel_position_ids.clamp(min=0)
        max_x = clamped[..., 0].max(dim=-1, keepdim=True)[0] + 1
        kernel_idxs = torch.div(clamped, k, rounding_mode="floor")
        kernel_idxs = kernel_idxs[..., 0] + (max_x // k) * kernel_idxs[..., 1]
        weights = F.one_hot(kernel_idxs.long(), length).float() / k_squared
        output = weights.transpose(1, 2) @ hidden_states.float()
        mask = torch.logical_not((weights == 0).all(dim=1))
        return output.to(hidden_states.dtype), mask

    def forward(self, hidden_states, pixel_position_ids, padding_positions, output_length):
        if output_length > hidden_states.shape[1]:
            raise ValueError("more soft tokens requested than there are patches")
        hidden_states = hidden_states.masked_fill(padding_positions.unsqueeze(-1), 0.0)
        if hidden_states.shape[1] != output_length:
            hidden_states, padding_positions = self._avg_pool_by_positions(
                hidden_states, pixel_position_ids, output_length
            )
        return hidden_states.float() * self.root_hidden_size, padding_positions


class Gemma4VisionModel(nn.Module):
    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.config = config
        self.patch_embedder = Gemma4VisionPatchEmbedder(config)
        self.encoder = Gemma4VisionEncoder(config)
        self.pooler = Gemma4VisionPooler(config)
        if config.standardize:
            self.std_bias = nn.Buffer(torch.empty(config.hidden_size))
            self.std_scale = nn.Buffer(torch.empty(config.hidden_size))

    def forward(self, pixel_values, pixel_position_ids):
        output_length = pixel_values.shape[-2] // (self.config.pooling_kernel_size**2)
        padding_positions = (pixel_position_ids == -1).all(dim=-1)
        inputs_embeds = self.patch_embedder(pixel_values, pixel_position_ids, padding_positions)
        hidden_states = self.encoder(inputs_embeds, ~padding_positions, pixel_position_ids)

        hidden_states, pooler_mask = self.pooler(
            hidden_states, pixel_position_ids, padding_positions, output_length
        )
        hidden_states = hidden_states[pooler_mask]
        if self.config.standardize:
            hidden_states = (hidden_states - self.std_bias.float()) * self.std_scale.float()
        return hidden_states.to(inputs_embeds.dtype)


# ---------------------------------------------------------------------------
# audio tower
# ---------------------------------------------------------------------------
class Gemma4AudioRelPositionalEncoding(nn.Module):
    def __init__(self, config: Gemma4AudioConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.context_size = (
            config.attention_chunk_size + config.attention_context_left - 1 + config.attention_context_right
        )
        num_timescales = self.hidden_size // 2
        log_increment = math.log(10000.0 / 1.0) / max(num_timescales - 1, 1)
        inv_timescales = 1.0 * torch.exp(torch.arange(num_timescales) * -log_increment)
        self.inv_timescales = nn.Buffer(inv_timescales.unsqueeze(0).unsqueeze(0), persistent=False)

    @torch.no_grad()
    def forward(self, hidden_states):
        position_ids = torch.arange(self.context_size // 2, -1, -1, device=hidden_states.device)[..., None]
        scaled_time = position_ids * self.inv_timescales.to(hidden_states.device)
        pos_embed = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=-1)
        return pos_embed.to(dtype=hidden_states.dtype)


class Gemma4AudioAttention(nn.Module):
    """Chunked local attention with relative position bias."""

    def __init__(self, config: Gemma4AudioConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads
        self.q_scale = (self.head_dim**-0.5) / math.log(2)
        self.k_scale = math.log(1 + math.e) / math.log(2)
        self.chunk_size = config.attention_chunk_size
        self.max_past_horizon = config.attention_context_left - 1
        self.max_future_horizon = config.attention_context_right
        self.context_size = self.chunk_size + self.max_past_horizon + self.max_future_horizon

        self.q_proj = Gemma4ClippableLinear(config, config.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = Gemma4ClippableLinear(config, config.hidden_size, self.num_heads * self.head_dim)
        self.v_proj = Gemma4ClippableLinear(config, config.hidden_size, self.num_heads * self.head_dim)
        self.post = Gemma4ClippableLinear(config, config.hidden_size, config.hidden_size)
        self.relative_k_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.per_dim_scale = nn.Parameter(torch.zeros(self.head_dim))
        self.softcap = nn.Buffer(torch.tensor(config.attention_logit_cap), persistent=False)

    def _convert_to_block(self, hidden_states):
        batch_size, seq_len, num_heads, head_dim = hidden_states.shape
        num_blocks = (seq_len + self.chunk_size - 1) // self.chunk_size
        pad = num_blocks * self.chunk_size - seq_len
        hidden_states = F.pad(hidden_states, (0, 0, 0, 0, 0, pad))
        return hidden_states.reshape(batch_size, num_blocks, self.chunk_size, num_heads, head_dim).contiguous()

    def _extract_block_context(self, hidden_states):
        batch_size, seq_len, num_heads, head_dim = hidden_states.shape
        hidden_states = F.pad(
            hidden_states, (0, 0, 0, 0, self.max_past_horizon, self.max_future_horizon + self.chunk_size - 1)
        )
        hidden_states = hidden_states.unfold(1, self.context_size, self.chunk_size)
        return torch.movedim(hidden_states, -1, 2).contiguous()

    def _rel_shift(self, x):
        batch_size, num_heads, num_blocks, block_size, position_length = x.shape
        context_size = self.context_size
        x = F.pad(x, (0, context_size + 1 - position_length))
        x = x.view(batch_size, num_heads, num_blocks, block_size * (context_size + 1))
        x = x[..., : block_size * context_size]
        return x.view(batch_size, num_heads, num_blocks, block_size, context_size)

    def forward(self, hidden_states, position_embeddings, attention_mask=None):
        batch_size, seq_length, _ = hidden_states.shape
        hidden_shape = (batch_size, seq_length, self.num_heads, self.head_dim)

        query_states = self.q_proj(hidden_states).float().view(hidden_shape)
        key_states = self.k_proj(hidden_states).float().view(hidden_shape)
        value_states = self.v_proj(hidden_states).float().view(hidden_shape)

        query_states = query_states * self.q_scale * F.softplus(self.per_dim_scale)
        key_states = key_states * self.k_scale

        query_states = self._convert_to_block(query_states)
        key_states = self._extract_block_context(key_states)
        value_states = self._extract_block_context(value_states)
        num_blocks = query_states.shape[1]

        relative_key_states = self.relative_k_proj(position_embeddings).view(-1, self.num_heads, self.head_dim)
        relative_key_states = relative_key_states.to(dtype=query_states.dtype)

        queries = query_states.permute(0, 3, 1, 2, 4)
        matrix_ac = queries @ key_states.permute(0, 3, 1, 4, 2)
        queries_flat = queries.reshape(batch_size, self.num_heads, -1, self.head_dim)
        matrix_bd = queries_flat @ relative_key_states.permute(1, 2, 0)
        matrix_bd = matrix_bd.reshape(batch_size, self.num_heads, num_blocks, self.chunk_size, -1)
        matrix_bd = self._rel_shift(matrix_bd)

        attn_weights = matrix_ac + matrix_bd
        attn_weights = torch.tanh(attn_weights / self.softcap) * self.softcap
        if attention_mask is not None:
            attn_weights = attn_weights.masked_fill(
                attention_mask.logical_not(), self.config.attention_invalid_logits_value
            )
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value_states.dtype)
        attn_output = attn_weights @ value_states.permute(0, 3, 1, 2, 4)
        attn_output = attn_output.permute(0, 2, 3, 1, 4).reshape(batch_size, num_blocks * self.chunk_size, -1)
        attn_output = attn_output[:, :seq_length].contiguous()
        return self.post(attn_output.to(hidden_states.dtype)), None


class Gemma4AudioSubSampleConvProjectionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, norm_eps):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=1, bias=False)
        self.norm = nn.LayerNorm(out_channels, eps=norm_eps, elementwise_affine=True, bias=False)
        self.act = nn.ReLU()

    def forward(self, hidden_states, mask=None):
        if mask is not None:
            mask = mask.to(device=hidden_states.device)
            hidden_states = hidden_states * mask[:, None, :, None]
        hidden_states = self.conv(hidden_states.to(self.conv.weight.dtype))
        hidden_states = self.act(self.norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous())
        if mask is not None:
            mask = mask[:, ::2]
        return hidden_states, mask


class Gemma4AudioSubSampleConvProjection(nn.Module):
    def __init__(self, config: Gemma4AudioConfig):
        super().__init__()
        self.layer0 = Gemma4AudioSubSampleConvProjectionLayer(1, config.subsampling_conv_channels[0], config.rms_norm_eps)
        self.layer1 = Gemma4AudioSubSampleConvProjectionLayer(
            config.subsampling_conv_channels[0], config.subsampling_conv_channels[1], config.rms_norm_eps
        )
        proj_input_dim = (config.subsampling_conv_channels[0] // 4) * config.subsampling_conv_channels[1]
        self.input_proj_linear = nn.Linear(proj_input_dim, config.hidden_size, bias=False)

    def forward(self, input_features, input_features_mask=None):
        hidden_states = input_features.unsqueeze(1)
        hidden_states, mask = self.layer0(hidden_states, input_features_mask)
        hidden_states, mask = self.layer1(hidden_states, mask)
        batch_size, _, seq_len, _ = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 3, 1).contiguous().reshape(batch_size, seq_len, -1)
        return self.input_proj_linear(hidden_states), mask


class Gemma4AudioFeedForward(nn.Module):
    def __init__(self, config: Gemma4AudioConfig):
        super().__init__()
        self.ffw_layer_1 = Gemma4ClippableLinear(config, config.hidden_size, config.hidden_size * 4)
        self.ffw_layer_2 = Gemma4ClippableLinear(config, config.hidden_size * 4, config.hidden_size)
        self.pre_layer_norm = Gemma4RMSNorm(config.hidden_size)
        self.post_layer_norm = Gemma4RMSNorm(config.hidden_size)
        self.act_fn = _act(config.hidden_act)
        self.gradient_clipping = config.gradient_clipping
        self.post_layer_scale = config.residual_weight

    def forward(self, hidden_states):
        clip = min(self.gradient_clipping, torch.finfo(hidden_states.dtype).max)
        residual = hidden_states
        hidden_states = self.pre_layer_norm(torch.clamp(hidden_states, -clip, clip))
        hidden_states = self.ffw_layer_2(self.act_fn(self.ffw_layer_1(hidden_states)))
        hidden_states = self.post_layer_norm(torch.clamp(hidden_states, -clip, clip))
        return hidden_states * self.post_layer_scale + residual


class Gemma4AudioCausalConv1d(nn.Conv1d):
    @property
    def left_pad(self):
        effective_kernel_size = (self.kernel_size[0] - 1) * self.dilation[0] + 1
        return effective_kernel_size - self.stride[0]

    def forward(self, x):
        x = nn.functional.pad(x, (self.left_pad, 0))
        return super().forward(x)


class Gemma4AudioLightConv1d(nn.Module):
    def __init__(self, config: Gemma4AudioConfig):
        super().__init__()
        self.linear_start = Gemma4ClippableLinear(config, config.hidden_size, config.hidden_size * 2)
        self.linear_end = Gemma4ClippableLinear(config, config.hidden_size, config.hidden_size)
        self.depthwise_conv1d = Gemma4AudioCausalConv1d(
            in_channels=config.hidden_size,
            out_channels=config.hidden_size,
            kernel_size=config.conv_kernel_size,
            groups=config.hidden_size,
            bias=False,
        )
        self.pre_layer_norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps, with_scale=True)
        self.conv_norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps, with_scale=True)
        self.act_fn = _act(config.hidden_act)
        self.gradient_clipping = config.gradient_clipping

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = nn.functional.glu(self.linear_start(self.pre_layer_norm(hidden_states)), dim=-1)
        hidden_states = self.depthwise_conv1d(hidden_states.transpose(1, 2)).transpose(1, 2)
        clip = min(self.gradient_clipping, torch.finfo(hidden_states.dtype).max)
        hidden_states = self.conv_norm(torch.clamp(hidden_states, -clip, clip))
        hidden_states = self.linear_end(self.act_fn(hidden_states))
        return hidden_states + residual


class Gemma4AudioLayer(nn.Module):
    def __init__(self, config: Gemma4AudioConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.feed_forward1 = Gemma4AudioFeedForward(config)
        self.feed_forward2 = Gemma4AudioFeedForward(config)
        self.self_attn = Gemma4AudioAttention(config, layer_idx)
        self.lconv1d = Gemma4AudioLightConv1d(config)
        self.norm_pre_attn = Gemma4RMSNorm(config.hidden_size)
        self.norm_post_attn = Gemma4RMSNorm(config.hidden_size)
        self.norm_out = Gemma4RMSNorm(config.hidden_size)
        self.gradient_clipping = config.gradient_clipping

    def forward(self, hidden_states, attention_mask, position_embeddings):
        clip = min(self.gradient_clipping, torch.finfo(self.norm_pre_attn.weight.dtype).max)
        hidden_states = self.feed_forward1(hidden_states)
        residual = hidden_states
        hidden_states = self.norm_pre_attn(torch.clamp(hidden_states, -clip, clip))
        hidden_states, _ = self.self_attn(hidden_states, position_embeddings, attention_mask)
        hidden_states = self.norm_post_attn(torch.clamp(hidden_states, -clip, clip)) + residual
        hidden_states = self.lconv1d(hidden_states)
        hidden_states = self.feed_forward2(hidden_states)
        return self.norm_out(torch.clamp(hidden_states, -clip, clip))


@dataclass
class Gemma4AudioModelOutput:
    last_hidden_state: torch.Tensor
    attention_mask: torch.Tensor


class Gemma4AudioModel(nn.Module):
    def __init__(self, config: Gemma4AudioConfig):
        super().__init__()
        self.config = config
        self.subsample_conv_projection = Gemma4AudioSubSampleConvProjection(config)
        self.rel_pos_enc = Gemma4AudioRelPositionalEncoding(config)
        self.layers = nn.ModuleList([Gemma4AudioLayer(config, i) for i in range(config.num_hidden_layers)])
        self.output_proj = nn.Linear(config.hidden_size, config.output_proj_dims, bias=True)

    def _convert_4d_mask_to_blocked_5d(self, mask_4d):
        batch_size, _, seq_len, _ = mask_4d.shape
        device = mask_4d.device
        chunk_size = self.config.attention_chunk_size
        max_past = self.config.attention_context_left - 1
        max_future = self.config.attention_context_right
        num_blocks = (seq_len + chunk_size - 1) // chunk_size
        padded_seq_len = num_blocks * chunk_size
        pad_amount = padded_seq_len - seq_len

        mask_4d = F.pad(mask_4d, (0, pad_amount, 0, pad_amount), value=False)
        mask_5d = mask_4d.reshape(batch_size, 1, num_blocks, chunk_size, padded_seq_len)
        mask_5d = F.pad(mask_5d, (max_past, max_future), value=False)
        block_starts = torch.arange(num_blocks, device=device) * chunk_size
        offsets = torch.arange(chunk_size + max_past + max_future, device=device)
        kv_indices = block_starts[:, None] + offsets[None, :]
        kv_indices = kv_indices[None, None, :, None, :].expand(batch_size, 1, -1, chunk_size, -1)
        return mask_5d.gather(-1, kv_indices)

    def forward(self, input_features, attention_mask=None):
        hidden_states, output_mask = self.subsample_conv_projection(input_features, attention_mask)
        position_embeddings = self.rel_pos_enc(hidden_states)

        # bidirectional AND sliding-window, then reshaped into the blocked layout the
        # chunked attention expects: (B, 1, num_blocks, chunk, context_size)
        allowed = build_bidirectional_allowed(
            hidden_states.shape[0],
            hidden_states.shape[1],
            hidden_states.shape[1],
            hidden_states.device,
            valid=output_mask.to(hidden_states.device) if output_mask is not None else None,
            sliding_window=(self.config.attention_context_left - 1, self.config.attention_context_right),
        )
        blocked_mask = self._convert_4d_mask_to_blocked_5d(allowed)

        for layer in self.layers:
            hidden_states = layer(hidden_states, blocked_mask, position_embeddings)
        return Gemma4AudioModelOutput(last_hidden_state=self.output_proj(hidden_states), attention_mask=output_mask)


# ---------------------------------------------------------------------------
# multimodal stitch
# ---------------------------------------------------------------------------
class Gemma4MultimodalEmbedder(nn.Module):
    def __init__(self, multimodal_config, text_config: Gemma4TextConfig):
        super().__init__()
        self.multimodal_hidden_size = getattr(multimodal_config, "output_proj_dims", multimodal_config.hidden_size)
        self.embedding_projection = nn.Linear(self.multimodal_hidden_size, text_config.hidden_size, bias=False)
        self.embedding_pre_projection_norm = Gemma4RMSNorm(
            self.multimodal_hidden_size, eps=multimodal_config.rms_norm_eps, with_scale=False
        )

    def forward(self, inputs_embeds):
        return self.embedding_projection(self.embedding_pre_projection_norm(inputs_embeds))


@dataclass
class Gemma4CausalLMOutput:
    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    image_hidden_states: torch.Tensor | None = None
    audio_hidden_states: torch.Tensor | None = None
    hidden_states: torch.Tensor | None = None
    past_key_values: KVCache | None = None


class Gemma4Model(nn.Module):
    def __init__(self, config: Gemma4Config):
        super().__init__()
        self.config = config
        self.vision_tower = Gemma4VisionModel(config.vision_config) if config.vision_config else None
        self.language_model = Gemma4TextModel(config.text_config)
        self.audio_tower = Gemma4AudioModel(config.audio_config) if config.audio_config else None
        self.embed_vision = (
            Gemma4MultimodalEmbedder(config.vision_config, config.text_config) if config.vision_config else None
        )
        self.embed_audio = (
            Gemma4MultimodalEmbedder(config.audio_config, config.text_config) if config.audio_config else None
        )

    # --- vision ---
    def get_image_features(self, pixel_values, image_position_ids):
        hidden_states = self.vision_tower(pixel_values=pixel_values, pixel_position_ids=image_position_ids)
        pooler_output = self.embed_vision(hidden_states)
        output_length = pixel_values.shape[-2] // (self.config.vision_config.pooling_kernel_size**2)
        k_squared = int((image_position_ids.shape[1] // output_length) ** 0.5) ** 2
        non_pad_mask = (image_position_ids != -1).all(dim=-1)
        split_sizes = (non_pad_mask.sum(dim=-1) // k_squared).tolist()
        return torch.split(pooler_output, split_sizes)

    # --- audio ---
    def get_audio_features(self, input_features, input_features_mask):
        out = self.audio_tower(input_features, input_features_mask)
        out.pooler_output = self.embed_audio(out.last_hidden_state)
        return out

    def get_placeholder_mask(self, input_ids=None, inputs_embeds=None):
        if input_ids is not None:
            image_mask = input_ids == self.config.image_token_id
            video_mask = input_ids == self.config.video_token_id
            audio_mask = input_ids == self.config.audio_token_id
        else:
            def eq_embed(token_id):
                emb = self.language_model.embed_tokens(
                    torch.full((), token_id, dtype=torch.long, device=inputs_embeds.device)
                )
                return (inputs_embeds == emb).all(-1)

            image_mask = eq_embed(self.config.image_token_id)
            video_mask = eq_embed(self.config.video_token_id)
            audio_mask = eq_embed(self.config.audio_token_id)
        return image_mask, video_mask, audio_mask

    def forward(self, input_ids=None, pixel_values=None, pixel_values_videos=None, input_features=None,
                attention_mask=None, input_features_mask=None, position_ids=None, mm_token_type_ids=None,
                inputs_embeds=None, image_position_ids=None, video_position_ids=None, per_layer_inputs=None,
                past_key_values: KVCache | None = None, use_cache: bool = False):
        image_mask, video_mask, audio_mask = self.get_placeholder_mask(input_ids, inputs_embeds)
        multimodal_mask = image_mask | video_mask | audio_mask

        llm_input_ids = None
        if inputs_embeds is None:
            llm_input_ids = torch.where(multimodal_mask, self.config.text_config.pad_token_id, input_ids)
            inputs_embeds = self.language_model.embed_tokens(llm_input_ids)

        if per_layer_inputs is None and self.config.text_config.hidden_size_per_layer_input:
            if llm_input_ids is None:
                raise ValueError(
                    "per_layer_inputs must be provided when calling with inputs_embeds only "
                    "(the reference reverses the embedding table, which is O(B*S*V))"
                )
            # raw token-identity part only: the context-aware projection happens inside
            # Gemma4TextModel.forward, using the post-scatter embeddings (as in the reference)
            per_layer_inputs = self.language_model.get_per_layer_inputs(llm_input_ids)

        image_features = None
        if pixel_values is not None:
            if self.vision_tower is None:
                raise ValueError("pixel_values given but config has no vision_config")
            feats = self.get_image_features(pixel_values, image_position_ids)
            image_features = torch.cat(feats, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            n_slots = image_mask.sum()
            image_mask_expanded = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            if inputs_embeds[image_mask_expanded].numel() != image_features.numel():
                raise ValueError(
                    f"image features ({image_features.shape[0]}) and image placeholders ({n_slots}) mismatch"
                )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask_expanded, image_features)

        audio_features = None
        if input_features is not None and input_features_mask is not None:
            if self.audio_tower is None:
                raise ValueError("input_features given but config has no audio_config")
            audio_out = self.get_audio_features(input_features, input_features_mask)
            audio_features = audio_out.pooler_output[audio_out.attention_mask.to(audio_out.pooler_output.device)]
            audio_mask_expanded = audio_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            if inputs_embeds[audio_mask_expanded].numel() != audio_features.numel():
                raise ValueError(
                    f"audio features ({audio_features.numel()}) and audio placeholders "
                    f"({audio_mask.sum() * inputs_embeds.shape[-1]}) mismatch"
                )
            inputs_embeds = inputs_embeds.masked_scatter(
                audio_mask_expanded, audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
            )

        if position_ids is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = (
                torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0) + past_seen
            )

        use_bidir_ids = None
        if self.config.text_config.use_bidirectional_attention == "vision" and mm_token_type_ids is not None:
            use_bidir_ids = mm_token_type_ids

        outputs = self.language_model(
            input_ids=None,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            per_layer_inputs=per_layer_inputs,
            position_ids=position_ids,
            mm_token_type_ids=use_bidir_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        return Gemma4CausalLMOutput(
            logits=None,
            image_hidden_states=image_features,
            audio_hidden_states=audio_features,
            hidden_states=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
        )


# ---------------------------------------------------------------------------
# top level model: head + loss
# ---------------------------------------------------------------------------
class Gemma4ForConditionalGeneration(nn.Module):
    """Mirrors `Gemma4ForConditionalGeneration` state-dict layout 1:1."""

    def __init__(self, config: Gemma4Config):
        super().__init__()
        self.config = config
        self.model = Gemma4Model(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

    # convenience aliases used by the demo / by HF-style call sites
    @property
    def language_model(self):
        return self.model.language_model

    def forward(self, input_ids=None, pixel_values=None, pixel_values_videos=None, input_features=None,
                attention_mask=None, input_features_mask=None, position_ids=None, image_position_ids=None,
                video_position_ids=None, mm_token_type_ids=None, inputs_embeds=None, labels=None,
                logits_to_keep: int = 0, per_layer_inputs=None,
                past_key_values: KVCache | None = None, use_cache: bool = False):
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            input_features=input_features,
            attention_mask=attention_mask,
            input_features_mask=input_features_mask,
            position_ids=position_ids,
            mm_token_type_ids=mm_token_type_ids,
            inputs_embeds=inputs_embeds,
            image_position_ids=image_position_ids,
            video_position_ids=video_position_ids,
            per_layer_inputs=per_layer_inputs,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        hidden_states = outputs.hidden_states
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        softcap = self.config.text_config.final_logit_softcapping
        if softcap is not None:
            logits = torch.tanh(logits / softcap) * softcap

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.config.text_config.vocab_size)

        return Gemma4CausalLMOutput(
            loss=loss,
            logits=logits,
            image_hidden_states=outputs.image_hidden_states,
            audio_hidden_states=outputs.audio_hidden_states,
            hidden_states=hidden_states,
            past_key_values=outputs.past_key_values,
        )

    @staticmethod
    def loss_function(logits, labels, vocab_size, ignore_index: int = -100):
        """Same as transformers.loss.loss_utils.ForCausalLMLoss (no num_items_in_batch)."""
        logits = logits.float()
        labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()
        return nn.functional.cross_entropy(
            logits.view(-1, vocab_size), shift_labels.view(-1).to(logits.device), ignore_index=ignore_index
        )


# ---------------------------------------------------------------------------
# checkpoint loading (safetensors, no transformers)
# ---------------------------------------------------------------------------
def _iter_safetensors_files(model_path: str):
    if os.path.isfile(model_path):
        return [model_path]
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            shard_files = sorted(set(json.load(f)["weight_map"].values()))
        return [os.path.join(model_path, name) for name in shard_files]
    single = os.path.join(model_path, "model.safetensors")
    if os.path.exists(single):
        return [single]
    raise FileNotFoundError(f"no safetensors found under {model_path}")


_DTYPE_MAP = {
    "float32": torch.float32,
    "float": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "half": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def _unused_kv_keys(config: Gemma4Config):
    """KV-shared decoder layers still ship k/v weights in the ckpt; they are unused."""
    text = config.text_config
    first_shared = text.first_kv_shared_layer_idx
    keys = []
    for i in range(text.num_hidden_layers):
        if i >= first_shared >= 0:
            keys.extend(
                f"model.language_model.layers.{i}.self_attn.{n}" for n in ("k_proj.weight", "v_proj.weight", "k_norm.weight")
            )
    return set(keys)


@torch.no_grad()
def load_gemma4_from_safetensors(model_path: str, dtype: torch.dtype | str | None = None,
                                 device: str | torch.device = "cpu", verbose: bool = True):
    """Build the model from `<model_path>/config.json` and load raw HF weights.

    Handles: single/shards, tied `lm_head`, unused KV-sharing weights, dtype cast.
    Returns (model, report) where report holds missing/unexpected key lists.

    `dtype`:
      * `None`            → keep float32 (default here; safest for parity work)
      * `"auto"`          → use `config.json`'s top-level `"dtype"`, which is exactly
                            what `from_pretrained()` does by default (usually bf16)
      * a `torch.dtype`   → that dtype

    Gotcha: `from_pretrained(ckpt)` on a bf16 checkpoint gives you a bf16 model. If you
    compare against it while your port runs in float32 you will see diffs of ~1e-2 that
    are pure precision, not bugs. Either pass `dtype="auto"` here or force the reference
    to float32 (`from_pretrained(..., dtype=torch.float32)`).
    """
    from safetensors import safe_open

    if os.path.isdir(model_path):
        config_path = os.path.join(model_path, "config.json")
    else:
        config_path = os.path.join(os.path.dirname(model_path), "config.json")
    config = Gemma4Config.from_json(config_path)

    if dtype == "auto":
        dtype = _DTYPE_MAP.get(config.dtype or "", None)
        if dtype is None:
            raise ValueError(f"config.json has dtype={config.dtype!r}, cannot map to a torch dtype; pass dtype explicitly")
    elif dtype is not None and not isinstance(dtype, torch.dtype):
        raise TypeError(f"dtype must be None, 'auto' or a torch.dtype, got {dtype!r}")

    model = Gemma4ForConditionalGeneration(config)
    if dtype is not None:
        model = model.to(dtype)

    state_dict = {}
    for shard in _iter_safetensors_files(model_path):
        with safe_open(shard, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)

    # tied embeddings: ckpt usually omits lm_head
    if "lm_head.weight" not in state_dict and config.text_config.tie_word_embeddings:
        state_dict["lm_head.weight"] = state_dict["model.language_model.embed_tokens.weight"]

    ignored = _unused_kv_keys(config)
    dropped = sorted(k for k in state_dict if k in ignored)
    for k in dropped:
        del state_dict[k]

    if dtype is not None:
        state_dict = {k: v.to(dtype) for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    report = {"missing": list(missing), "unexpected": list(unexpected), "dropped": dropped}
    report["dtype"] = str(next(model.parameters()).dtype)
    if verbose:
        print(f"[load] tensors={len(state_dict)} dropped_unused={len(dropped)} "
              f"missing={len(missing)} unexpected={len(unexpected)} dtype={report['dtype']}")
        if dtype is None and config.dtype not in (None, "float32"):
            print(f"  note: config.json says dtype={config.dtype!r}; from_pretrained() would load in that "
                  f"precision. You are running float32 — pass dtype='auto' for an apples-to-apples diff.")
        for name in report["missing"][:10]:
            print("  missing:", name)
        for name in report["unexpected"][:10]:
            print("  unexpected:", name)

    model = model.to(device)
    model.eval()
    return model, report
