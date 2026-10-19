"""Minimal autoregressive generation loop.

Two phases, which is the whole idea:

    prefill   the prompt goes through the model in ONE forward pass; every layer
              writes its k/v into the cache.  Cost O(prompt_len^2).

    decode    one token per forward pass.  Only that token's query is computed;
              keys/values come from the cache.  Cost O(seq_len) per step.

`generate()` is those two glued together plus sampling.  The pieces are kept
separate on purpose so you can replace one without touching the others -- for
example swap in beam search, or a speculative-decoding draft model.

What this baseline deliberately does NOT do
-------------------------------------------
* no stopping criteria beyond `max_new_tokens` / eos
* no `generate`-style config resolution (GenerationConfig, logits processors)
* no CUDA graphs, no static cache, no batched-in-flight tricks
* finished sequences keep being forwarded (masked to pad) instead of being
  dropped from the batch

Each of those is a reasonable next exercise; see the notes at the bottom.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:  # imported as a package (normal case)
    from .kv_cache import KVCache
except ImportError:  # `python generation.py` from inside the folder
    from kv_cache import KVCache


# ---------------------------------------------------------------------------
# phase 1: prefill
# ---------------------------------------------------------------------------
@torch.no_grad()
def prefill(model, input_ids, attention_mask=None, **model_kwargs):
    """Run the prompt through the model once and fill a fresh cache.

    Returns `(last_logits, cache, attention_mask)` where `last_logits` is the
    distribution for the **next** token, shape (batch, vocab).
    """
    if input_ids.dtype != torch.long:
        input_ids = input_ids.long()

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        **model_kwargs,
    )
    return out.logits[:, -1, :], out.past_key_values, attention_mask


# ---------------------------------------------------------------------------
# phase 2: one decode step
# ---------------------------------------------------------------------------
@torch.no_grad()
def decode_step(model, next_token, cache: KVCache, attention_mask):
    """Feed a single token per sequence and return its next-token logits.

    `next_token` is (batch, 1).  Multimodal inputs are NOT passed here: images
    and audio only exist in the prompt, and re-running the vision/audio towers
    every step would waste most of the time budget.
    """
    attention_mask = torch.cat(
        [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
    )
    out = model(
        input_ids=next_token,
        attention_mask=attention_mask,
        past_key_values=cache,
        use_cache=True,
    )
    return out.logits[:, -1, :], out.past_key_values, attention_mask


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------
def sample(logits, do_sample: bool = False, temperature: float = 1.0,
           top_k: int | None = None, top_p: float | None = None,
           generator: torch.Generator | None = None) -> torch.Tensor:
    """(batch, vocab) -> (batch, 1) token ids."""
    if not do_sample:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits.float() / max(temperature, 1e-5)

    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, order = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        # `cum - probs` is the mass *before* this token: keep the first token
        # that crosses top_p, drop everything after it.
        drop = (probs.cumsum(dim=-1) - probs) > top_p
        sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
        logits = sorted_logits.scatter(1, order, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)


def _as_id_list(x) -> list[int]:
    if x is None:
        return []
    if isinstance(x, int):
        return [x]
    return list(x)


def _is_eos(tokens: torch.Tensor, eos_ids: list[int]) -> torch.Tensor:
    """(batch, 1) -> (batch,) bool."""
    hit = torch.zeros_like(tokens[:, 0], dtype=torch.bool)
    for e in eos_ids:
        hit |= tokens[:, 0] == e
    return hit


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate(model, input_ids, attention_mask=None, max_new_tokens: int = 32,
             do_sample: bool = False, temperature: float = 1.0,
             top_k: int | None = None, top_p: float | None = None,
             eos_token_id=None, pad_token_id: int | None = None,
             on_token=None, generator: torch.Generator | None = None,
             **model_kwargs) -> torch.Tensor:
    """Greedy / sampling generation.  Returns prompt + new tokens.

    Parameters
    ----------
    model
        `Gemma4ForConditionalGeneration` (or anything with the same forward
        signature: `past_key_values`, `use_cache`, `out.logits`, `out.past_key_values`).
    input_ids
        (batch, prompt_len) long tensor.
    eos_token_id
        int or list of ints.  Gemma 4 ends a turn with `<end_of_turn>` and stops
        the sequence with `<eos>`, so pass both.
    on_token
        optional `callable(step, token_ids)`, for streaming output in a notebook.
    model_kwargs
        forwarded to the **prefill** call only (`pixel_values`, `image_position_ids`,
        `input_features`, ...).  Never forwarded to decode steps.

    Notes
    -----
    Finished rows are padded with `pad_token_id` and keep being forwarded.  That
    wastes compute but keeps shapes static; dropping them means reindexing the
    cache batch axis (`KVCache.reorder_cache` is a starting point).
    """
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    eos_ids = _as_id_list(eos_token_id)
    if pad_token_id is None:
        pad_token_id = eos_ids[0] if eos_ids else 0

    logits, cache, attention_mask = prefill(
        model, input_ids, attention_mask, **model_kwargs
    )

    generated = input_ids
    finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

    for step in range(max_new_tokens):
        next_token = sample(
            logits,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generator=generator,
        ).to(input_ids.device)

        if eos_ids:
            finished |= _is_eos(next_token, eos_ids)
        # rows that already ended emit pad from now on
        next_token = next_token.masked_fill(finished[:, None], pad_token_id)

        generated = torch.cat([generated, next_token], dim=-1)

        if on_token is not None:
            on_token(step, next_token)
        if eos_ids and bool(finished.all()):
            break

        logits, cache, attention_mask = decode_step(model, next_token, cache, attention_mask)

    return generated


# ---------------------------------------------------------------------------
# next exercises, roughly in order of payoff
# ---------------------------------------------------------------------------
# 1. **Trim the cache.**  A sliding layer only ever attends to the last `w`
#    positions, so it could store just `w` of them.  The trap: KV-shared layers
#    of type `full_attention` still need the long history, which is why the
#    reference feeds them from `shared_kv_states` instead of the cache
#    (see `kv_cache.py::KVCache.crop`).
#
# 2. **Static cache + preallocated buffer.**  Replaces `torch.cat` with an
#    in-place write at `cache_position`, which removes the per-step copy and
#    makes CUDA graphs possible.
#
# 3. **Batching with real early exit.**  Sort by prompt length, drop finished
#    rows from the batch, reindex the cache with `reorder_cache`.
#
# 4. **Speculative decoding.**  A tiny draft model proposes k tokens, the big
#    model verifies them in one forward pass.  The verify step is exactly
#    `prefill` with a multi-token chunk, so this file already has the pieces.
