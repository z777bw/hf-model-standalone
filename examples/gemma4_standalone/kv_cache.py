"""Minimal KV cache.

A deliberately small, readable cache: one tensor pair per layer, appended with
`torch.cat`.  It exists so the *idea* is legible and hackable, not so it is fast.

Layout
    key_cache[i]   : (batch, num_kv_heads, seq_len, head_dim)
    value_cache[i] : (batch, num_kv_heads, seq_len, head_dim)

`update()` concatenates along dim=-2 (the sequence axis) and returns the full
k/v, so attention always sees "everything so far" while only computing queries
for the new token(s).

Two facts that surprise people
------------------------------
1. **Only non-shared layers write.**  Gemma 4 lets the last `num_kv_shared_layers`
   layers reuse the k/v of an earlier layer of the same type; those layers have
   no `k_proj` / `v_proj` at all, so there is nothing to append.  They read from
   `shared_kv_states` in `modeling_gemma4.Gemma4TextAttention` instead.

2. **The cache is not trimmed by the sliding window.**  What restricts a sliding
   layer to the last `w` tokens is the attention *mask*, so the stored tensor
   grows to the full sequence length.  Trimming would save memory and is the
   obvious first optimization -- see `crop()` and the note at the bottom.

Cost model, honestly
--------------------
`torch.cat` copies the whole history every step, so a single decode step is
O(seq_len) in copying alone.  That is fine for a 100-token demo and wrong for
production.  Every real implementation preallocates a buffer:

    key_cache[i] = torch.empty(batch, heads, max_len, head_dim)   # once
    key_cache[i][:, :, pos] = k_new                               # in place

Doing it that way is the first exercise once the readable version makes sense.
"""

from __future__ import annotations

import torch


class KVCache:
    """Per-layer key/value store for incremental decoding.

    Example
    -------
    >>> cache = KVCache(num_layers=26)
    >>> out = model(input_ids=prompt, past_key_values=cache, use_cache=True)
    >>> for _ in range(max_new_tokens):
    ...     out = model(input_ids=next_tok, past_key_values=cache, use_cache=True)
    """

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.key_cache: list[torch.Tensor | None] = [None] * num_layers
        self.value_cache: list[torch.Tensor | None] = [None] * num_layers

    # -- introspection ------------------------------------------------------
    def __len__(self) -> int:
        """Number of layers (not sequence length)."""
        return self.num_layers

    def __repr__(self) -> str:
        return f"KVCache(layers={self.num_layers}, seq_len={self.get_seq_length()})"

    def get_seq_length(self, layer_idx: int | None = None) -> int:
        """How many tokens are already cached.

        With `layer_idx=None` we return the first non-empty layer: KV-shared
        layers never write, so layer 0 is usually the safe choice, but scanning
        is cheap and works even if the model shares from layer 0.
        """
        if layer_idx is not None:
            k = self.key_cache[layer_idx]
            return 0 if k is None else k.shape[-2]
        for k in self.key_cache:
            if k is not None:
                return k.shape[-2]
        return 0

    def get(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Cached (k, v) for one layer, or None if the layer has not run yet."""
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    # -- the one method that matters ----------------------------------------
    def update(self, key_states: torch.Tensor, value_states: torch.Tensor,
               layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Append the new k/v and return the whole history for that layer.

        `key_states` / `value_states` are (batch, heads, new_len, head_dim).
        On the first call (`new_len == prompt_len`) this is the prefill; every
        later call appends a single position.
        """
        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat(
                (self.key_cache[layer_idx], key_states), dim=-2
            )
            self.value_cache[layer_idx] = torch.cat(
                (self.value_cache[layer_idx], value_states), dim=-2
            )
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    # -- housekeeping -------------------------------------------------------
    def reset(self) -> None:
        """Drop everything (start a new conversation)."""
        self.key_cache = [None] * self.num_layers
        self.value_cache = [None] * self.num_layers

    def crop(self, max_length: int) -> None:
        """Keep only the last `max_length` positions of every layer.

        This is the knob for sliding-window memory savings.  Two caveats before
        you wire it up:

        * every masked layer type must use the same trim length, otherwise
          shapes disagree when layers of different types are mixed;
        * `shared_kv_states` (used by KV-shared layers) must keep the *full*
          length, because a shared layer of type `full_attention` may need
          positions that a sliding layer already dropped.  In transformers this
          is exactly why shared layers read a separate dict instead of the cache.

        So crop() is safe only if no layer type needs a longer view than
        `max_length`.  For Gemma 4 with `sliding_window=1024` you would need
        `max_length >=` the longest full-attention context, which defeats the
        purpose -- which is why the reference simply keeps the full cache.
        """
        if max_length < 0:
            raise ValueError("max_length must be >= 0")
        for i in range(self.num_layers):
            k = self.key_cache[i]
            if k is not None and k.shape[-2] > max_length:
                self.key_cache[i] = k[..., -max_length:, :]
                self.value_cache[i] = self.value_cache[i][..., -max_length:, :]

    def reorder_cache(self, beam_idx: torch.Tensor) -> None:
        """Reorder the batch axis, for beam search.

        `beam_idx` is (batch,) long: `beam_idx[b]` says which old batch entry
        the new row `b` should come from.
        """
        for i in range(self.num_layers):
            k = self.key_cache[i]
            if k is not None:
                self.key_cache[i] = k.index_select(0, beam_idx.to(k.device))
                self.value_cache[i] = self.value_cache[i].index_select(0, beam_idx.to(k.device))


# ---------------------------------------------------------------------------
# What a production cache looks like (read this after the code above makes sense)
# ---------------------------------------------------------------------------
# transformers ships several, and the differences are instructive:
#
#   DynamicCache        append-only, like this file but with better bookkeeping
#   StaticCache         preallocated to max_len; no cat, no realloc -> CUDA graphs
#   SlidingWindowCache  StaticCache that keeps a rolling window; needs the mask
#                       and the cache to agree on absolute positions, which is
#                       why `cache_position` exists as a separate input
#   QuantizedCache      stores k/v in int8/4bit; trades accuracy for memory
#
# The one idea worth stealing from all of them: positions are *absolute*
# (derived from `cache_position`), storage is *relative* (an index or a slice).
# Mixing the two up is the classic off-by-window bug.
