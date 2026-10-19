"""Numerical parity: our standalone Gemma 4 vs. transformers' own implementation.

Strategy (no 10 GB download, no GPU):
  1. build a *tiny* Gemma4Config with the exact same architecture knobs as
     google/gemma-4-E2B (per-layer head_dim, KV sharing, double-wide MLP, PLE,
     logit soft-capping, 2-D vision RoPE, chunked audio attention ...)
  2. instantiate the reference `Gemma4ForConditionalGeneration`, random-init it
  3. export its state_dict to a real `model.safetensors` + `config.json`
  4. load that checkpoint with `gemma4_standalone.load_gemma4_from_safetensors`
  5. compare logits / loss / pixel_values / input_features / token ids
  6. decode the same sequence incrementally through the KV cache, and compare
     greedy generation against the reference `generate()`

The text+vision suite runs against the reference's **eager** attention path
(bit-exact there). The audio suite runs against **sdpa**: the reference's eager
path feeds an additive 0/-inf mask into `Gemma4AudioAttention`, which does
`masked_fill(mask.logical_not(), -1e9)` and therefore overwrites every *allowed*
logit with -1e9 -- the encoder degenerates to uniform attention over the local
window. sdpa receives a boolean mask and behaves as intended, so that is what we
match. See references/pitfalls.md.

Run:
    python examples/gemma4_standalone/test_parity.py     # from the repo root
"""

import copy
import json
import os
import sys
import tempfile

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from transformers import (  # noqa: E402
    AutoTokenizer,
    Gemma4AudioConfig as HFAudioConfig,
    Gemma4AudioFeatureExtractor,
    Gemma4Config as HFConfig,
    Gemma4ForConditionalGeneration as HFModel,
    Gemma4ImageProcessorPil,
    Gemma4TextConfig as HFTextConfig,
    Gemma4VisionConfig as HFVisionConfig,
)

from gemma4_standalone import generation as mine_gen  # noqa: E402
from gemma4_standalone import modeling_gemma4 as mine  # noqa: E402
from gemma4_standalone import preprocessing as mine_pre  # noqa: E402

CKPT_DIR = os.environ.get("GEMMA4_CKPT") or os.path.join(
    os.path.dirname(ROOT), "ckpt"
)
RESULTS = []


def record(name, ok, detail=""):
    ok = bool(ok)
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def diff(a, b):
    if isinstance(a, torch.Tensor):
        a = a.detach().float().cpu()
    if isinstance(b, torch.Tensor):
        b = b.detach().float().cpu()
    return (torch.as_tensor(a) - torch.as_tensor(b)).abs().max().item()


def check_tensor(name, a, b, tol=2e-4):
    d = diff(a, b)
    record(name, d <= tol, f"max_abs_diff={d:.3e} tol={tol:.1e}")


# ---------------------------------------------------------------------------
# tiny reference model (same knobs as E2B, ~0.4M params)
# ---------------------------------------------------------------------------
TEXT_LAYER_TYPES = [
    "sliding_attention", "sliding_attention", "sliding_attention",
    "full_attention", "sliding_attention", "full_attention",
]


def build_reference(attn_impl: str):
    text = HFTextConfig(
        vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=6,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, global_head_dim=16,
        hidden_size_per_layer_input=8, vocab_size_per_layer_input=128,
        sliding_window=8, final_logit_softcapping=30.0, num_kv_shared_layers=2,
        use_double_wide_mlp=True, pad_token_id=0, bos_token_id=2, eos_token_id=1,
        layer_types=TEXT_LAYER_TYPES,
    )
    vision = HFVisionConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, head_dim=8,
        patch_size=16, pooling_kernel_size=3, position_embedding_size=128,
        use_clipped_linears=True,
    )
    audio = HFAudioConfig(
        hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
        # channels[0] must stay 128: the SSCP flattens (channels[1], mel_bins // 4)
        subsampling_conv_channels=[128, 4], output_proj_dims=32,
    )
    cfg = HFConfig(
        text_config=text, vision_config=vision, audio_config=audio,
        image_token_id=50, audio_token_id=51, video_token_id=52,
        boi_token_id=53, eoi_token_id=54, boa_token_id=55, eoa_token_index=56,
    )
    cfg._attn_implementation = attn_impl
    torch.manual_seed(0)
    model = HFModel(cfg)
    model.eval()
    return model, cfg


def export_checkpoint(model, cfg, out_dir, dtype_name=None):
    """Write config.json + model.safetensors the way the hub does.

    `dtype_name` emulates checkpoints that ship a top-level `"dtype"` (the real
    gemma-4-E2B one says bfloat16, and `from_pretrained` then loads in bf16).
    """
    d = cfg.to_dict()
    d["dtype"] = dtype_name
    # transformers 5.16 pops `global_head_dim` into `per_layer_config`, so it never
    # reaches to_dict(); the real gemma-4-E2B config.json *does* ship it. Re-add it so
    # both sides resolve the same per-layer head dims.
    d["text_config"]["global_head_dim"] = cfg.text_config.per_layer_config["full_attention"].head_dim
    d["text_config"]["num_global_key_value_heads"] = None
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)

    sd = {k: v.contiguous().clone() for k, v in model.state_dict().items()}
    if dtype_name is not None:
        tdt = getattr(torch, dtype_name)
        sd = {k: v.to(tdt) for k, v in sd.items()}
    sd.pop("lm_head.weight", None)  # tied to embed_tokens: the hub omits it

    # The real checkpoint still stores k/v weights for KV-shared layers (unused).
    # Emulate that so the loader's "drop unused keys" path is exercised.
    first_shared = cfg.text_config.num_hidden_layers - cfg.text_config.num_kv_shared_layers
    for layer_idx in range(first_shared, cfg.text_config.num_hidden_layers):
        ltype = cfg.text_config.layer_types[layer_idx]
        src = max(i for i in range(first_shared) if cfg.text_config.layer_types[i] == ltype)
        for name in ("k_proj", "v_proj", "k_norm"):
            sd[f"model.language_model.layers.{layer_idx}.self_attn.{name}.weight"] = sd[
                f"model.language_model.layers.{src}.self_attn.{name}.weight"
            ].clone()

    from safetensors.torch import save_file
    save_file(sd, os.path.join(out_dir, "model.safetensors"))


# ---------------------------------------------------------------------------
# preprocessing checks (run once)
# ---------------------------------------------------------------------------
def test_tokenizer():
    if not os.path.exists(os.path.join(CKPT_DIR, "tokenizer.json")):
        print(f"  [skip] no real tokenizer.json under {CKPT_DIR}")
        print("         run: python scripts/fetch_minimal.py google/gemma-4-E2B --out ./ckpt")
        return
    ref = AutoTokenizer.from_pretrained(CKPT_DIR)
    ours = mine_pre.Gemma4Tokenizer(
        os.path.join(CKPT_DIR, "tokenizer.json"),
        json.load(open(os.path.join(CKPT_DIR, "tokenizer_config.json"), encoding="utf-8")),
    )
    samples = [
        "Hello world",
        "Gemma 4 是一个多模态模型，支持图像、音频和文本输入。",
        "def f(x):\n    return x ** 2  # math\n",
        "emoji 🚀🎯 mixed 中文 English 123",
        "  leading and trailing spaces  ",
        "a" * 200,
    ]
    ok = True
    for s in samples:
        a = ref(s, add_special_tokens=True)["input_ids"]
        b = ours.encode(s, add_special_tokens=True)
        if list(a) != list(b):
            ok = False
            print("  mismatch:", s[:40], a[:10], b[:10])
    record("tokenizer ids (CJK/emoji/code/long)", ok)

    ids = ref("decode me", add_special_tokens=True)["input_ids"]
    record("tokenizer decode(skip_special)",
           ref.decode(ids, skip_special_tokens=True) == ours.decode(ids, skip_special_tokens=True))

    batch = ["short", "a much longer sentence here"]
    a = ref(batch, padding=True, return_tensors="pt")
    b = ours.batch_encode(batch, padding=True)
    record("tokenizer batch + left padding",
           torch.equal(a["input_ids"], b["input_ids"]) and torch.equal(a["attention_mask"], b["attention_mask"]))
    record("special token ids",
           (ours.image_token_id, ours.audio_token_id, ours.video_token_id)
           == (ref.image_token_id, ref.audio_token_id, ref.convert_tokens_to_ids("<|video|>")),
           f"image={ours.image_token_id} audio={ours.audio_token_id} video={ours.video_token_id}")


def test_image_processor():
    rng = np.random.default_rng(0)
    imgs = [
        Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)),
        Image.fromarray(rng.integers(0, 256, (100, 37, 3), dtype=np.uint8)),  # extreme aspect ratio
    ]
    ref = Gemma4ImageProcessorPil(max_soft_tokens=70, patch_size=16, pooling_kernel_size=3)
    a = ref(images=imgs, return_tensors="pt")
    ours = mine_pre.Gemma4ImageProcessor(patch_size=16, max_soft_tokens=70, pooling_kernel_size=3)
    b = ours.preprocess(imgs)

    check_tensor("image pixel_values", a["pixel_values"], b["pixel_values"], tol=1e-6)
    record("image image_position_ids (exact)", torch.equal(a["image_position_ids"], b["image_position_ids"]))
    ref_soft = a["num_soft_tokens_per_image"]
    ref_soft = ref_soft.tolist() if hasattr(ref_soft, "tolist") else list(ref_soft)
    record("image num_soft_tokens", ref_soft == b["num_soft_tokens_per_image"],
           f"{b['num_soft_tokens_per_image']}")


def test_audio_features():
    sr = 16_000
    wav = (0.4 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr)).astype(np.float64)
    a = Gemma4AudioFeatureExtractor()([wav], return_tensors="pt", sampling_rate=sr)
    b = mine_pre.Gemma4FeatureExtractor()([wav])
    check_tensor("audio log-mel features", a["input_features"], b["input_features"], tol=5e-4)
    record("audio features mask", torch.equal(a["input_features_mask"].bool(), b["input_features_mask"].bool()),
           f"{tuple(b['input_features'].shape)}")


# ---------------------------------------------------------------------------
# forward parity
# ---------------------------------------------------------------------------
def test_forward(hfmodel, cfg, ckpt_parent, pixel_values, image_position_ids, n_image_tokens,
                 audio_feats, include_audio: bool, tag: str):
    ours = mine.load_gemma4_from_safetensors(os.path.join(ckpt_parent, "model"), verbose=False)[0]
    ours.eval()

    ids = torch.tensor([[0, 2, 5, 6, 7, 8, 9], [0, 0, 2, 11, 12, 13, 14]])
    mask = torch.tensor([[0, 1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1, 1]])
    labels = ids.clone()
    labels[mask == 0] = -100
    # the first real token is predicted from a fully-masked (undefined) position in the
    # sdpa path, so ignore that boundary label -- standard padding-loss hygiene
    for b, first in enumerate(mask.float().argmax(dim=1).tolist()):
        labels[b, first] = -100
    with torch.no_grad():
        ra = hfmodel(input_ids=ids, attention_mask=mask, labels=labels)
        rb = ours(input_ids=ids, attention_mask=mask, labels=labels)
    # fully-masked query rows (left padding) have no defined output: only compare real tokens
    d_valid = diff(ra.logits[mask.bool()], rb.logits[mask.bool()])
    record(f"{tag}text logits (left-padded, valid positions)", d_valid <= 2e-4, f"max_abs_diff={d_valid:.3e}")
    check_tensor(f"{tag}text loss", ra.loss, rb.loss, tol=1e-5)

    with torch.no_grad():
        ra = hfmodel(input_ids=ids[:, 2:], labels=labels[:, 2:])
        rb = ours(input_ids=ids[:, 2:], labels=labels[:, 2:])
    check_tensor(f"{tag}text logits (no padding)", ra.logits, rb.logits)

    img = cfg.image_token_id
    seq = [2, 20, 21, cfg.boi_token_id] + [img] * n_image_tokens + [cfg.eoi_token_id, 22, 23, 24]
    ids_im = torch.tensor([seq])
    labels_im = torch.tensor([[0] + seq[1:]])
    labels_im[0, : 4 + n_image_tokens] = -100
    with torch.no_grad():
        ra = hfmodel(input_ids=ids_im, pixel_values=pixel_values,
                     image_position_ids=image_position_ids, labels=labels_im)
        rb = ours(input_ids=ids_im, pixel_values=pixel_values,
                  image_position_ids=image_position_ids, labels=labels_im)
    check_tensor(f"{tag}image_hidden_states", ra.image_hidden_states, rb.image_hidden_states)
    check_tensor(f"{tag}image+text logits", ra.logits, rb.logits)
    check_tensor(f"{tag}image+text loss", ra.loss, rb.loss, tol=1e-5)

    with torch.no_grad():
        a = hfmodel(input_ids=ids, attention_mask=mask, logits_to_keep=1).logits
        b = ours(input_ids=ids, attention_mask=mask, logits_to_keep=1).logits
    check_tensor(f"{tag}logits_to_keep=1", a, b)

    if not include_audio:
        return

    feats, feats_mask = audio_feats["input_features"], audio_feats["input_features_mask"]
    with torch.no_grad():
        n_audio_tokens = int(hfmodel.model.audio_tower(feats, feats_mask).attention_mask[0].bool().sum())
    seq = [2, 30, cfg.boa_token_id] + [cfg.audio_token_id] * n_audio_tokens + [cfg.eoa_token_index, 31, 32]
    ids_au = torch.tensor([seq])
    labels_au = torch.tensor([[0] + seq[1:]])
    labels_au[0, : 3 + n_audio_tokens] = -100
    with torch.no_grad():
        ra = hfmodel(input_ids=ids_au, input_features=feats, input_features_mask=feats_mask, labels=labels_au)
        rb = ours(input_ids=ids_au, input_features=feats, input_features_mask=feats_mask, labels=labels_au)
    check_tensor(f"{tag}audio_hidden_states", ra.audio_hidden_states, rb.audio_hidden_states, tol=1e-4)
    check_tensor(f"{tag}audio+text logits", ra.logits, rb.logits, tol=1e-3)
    check_tensor(f"{tag}audio+text loss", ra.loss, rb.loss, tol=1e-4)

    seq = ([2, 40, cfg.boi_token_id] + [img] * n_image_tokens + [cfg.eoi_token_id, 41, cfg.boa_token_id]
           + [cfg.audio_token_id] * n_audio_tokens + [cfg.eoa_token_index, 42, 43])
    ids_all = torch.tensor([seq])
    labels_all = torch.tensor([[0] + seq[1:]])
    labels_all[0, : len(seq) - 3] = -100
    with torch.no_grad():
        ra = hfmodel(input_ids=ids_all, pixel_values=pixel_values, image_position_ids=image_position_ids,
                     input_features=feats, input_features_mask=feats_mask, labels=labels_all)
        rb = ours(input_ids=ids_all, pixel_values=pixel_values, image_position_ids=image_position_ids,
                  input_features=feats, input_features_mask=feats_mask, labels=labels_all)
    check_tensor(f"{tag}image+audio+text logits", ra.logits, rb.logits, tol=1e-3)
    check_tensor(f"{tag}image+audio+text loss", ra.loss, rb.loss, tol=1e-4)


def test_weight_loading(ckpt_parent):
    model, report = mine.load_gemma4_from_safetensors(os.path.join(ckpt_parent, "model"), verbose=False)
    record("load: no unexpected keys", len(report["unexpected"]) == 0, f"{report['unexpected'][:3]}")
    record("load: no missing keys", len(report["missing"]) == 0, f"{report['missing'][:3]}")
    record("load: unused KV-shared k/v dropped", len(report["dropped"]) == 6, f"dropped={len(report['dropped'])}")
    record("load: lm_head tied to embed_tokens",
           torch.equal(model.lm_head.weight, model.model.language_model.embed_tokens.weight))


def test_dtype_auto(hfmodel, cfg, ckpt_parent):
    """A bf16 checkpoint + `from_pretrained()` = a bf16 reference.

    Comparing that against a float32 port gives ~1e-2 diffs that are pure precision.
    `dtype="auto"` must reproduce from_pretrained's default so the diff is apples-to-apples.

    Note we build the reference with `from_pretrained(..., dtype="auto")` rather than
    `deepcopy(model).to(bf16)`: the latter also casts the reference's own `inv_freq`
    buffers to bf16 (plain `.to()` casts every buffer), while from_pretrained only casts
    parameters. Only the from_pretrained route is what users actually hit.
    """
    out = os.path.join(ckpt_parent, "bf16")
    os.makedirs(out, exist_ok=True)
    export_checkpoint(hfmodel, cfg, out, dtype_name="bfloat16")

    ours, report = mine.load_gemma4_from_safetensors(out, dtype="auto", verbose=False)
    record("load auto: config.dtype honoured", report["dtype"] == "torch.bfloat16", report["dtype"])

    # RoPE angles must be built from full-precision inv_freq even in a bf16 model:
    # if it were a registered buffer, `.to(bf16)` would round it (bf16(0.1)=0.10009765625)
    # and cos/sin drift by a bf16 ulp -> ~1e-2 logits.
    inv = ours.model.language_model.rotary_emb._inv_freqs["sliding_attention"]
    record("load auto: inv_freq stays fp32", inv.dtype == torch.float32, str(inv.dtype))

    ref = HFModel.from_pretrained(
        out, attn_implementation="eager", dtype="auto"
    ).eval()
    ids = torch.tensor([[2, 5, 6, 7, 8, 9]])
    with torch.no_grad():
        a = ref(input_ids=ids).logits.float()
        b = ours(input_ids=ids).logits.float()
    d = float((a - b).abs().max())
    record("load auto: bf16 logits bit-exact", d == 0.0, f"max_abs_diff={d:.3e}")


# ---------------------------------------------------------------------------
# incremental decoding / generation
# ---------------------------------------------------------------------------
def test_kv_cache(hfmodel, cfg, ckpt_parent, tag=""):
    """Decode one token at a time and compare against a single full-sequence pass.

    The sequence is 12 tokens, longer than the tiny `sliding_window` (8), so the
    sliding masks really do have to clip the history mid-decode.  A wrong
    q/kv offset or a broken shared-KV hand-off shows up here and nowhere else.
    """
    model, _ = mine.load_gemma4_from_safetensors(os.path.join(ckpt_parent, "model"), verbose=False)
    model.eval()

    ids = torch.tensor([[2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]])
    with torch.no_grad():
        full = hfmodel(input_ids=ids).logits
        # Same implementation, cache on vs off: isolates "does the cache change the
        # math?" from "does decoding change the reduction order?" -- and from
        # "eager vs sdpa kernel" (the port is always hand-written eager, so comparing
        # against an sdpa reference here would measure the wrong thing).
        uncached = model(input_ids=ids).logits
        cached_full = model(input_ids=ids, use_cache=True).logits
    d_prefill = float((uncached - cached_full).abs().max())
    record(f"{tag}kv cache: cached prefill bit-exact", d_prefill == 0.0,
           f"max_abs_diff={d_prefill:.3e}")

    prefill_len = 5
    with torch.no_grad():
        out = model(input_ids=ids[:, :prefill_len], use_cache=True)
        cache = out.past_key_values
        chunks = [out.logits[:, -1:]]
        mask = torch.ones((1, prefill_len), dtype=torch.long)
        for t in range(prefill_len, ids.shape[1]):
            mask = torch.cat([mask, mask.new_ones((1, 1))], dim=-1)
            out = model(
                input_ids=ids[:, t:t + 1], attention_mask=mask,
                past_key_values=cache, use_cache=True,
            )
            chunks.append(out.logits[:, -1:])

    inc = torch.cat(chunks, dim=1)                    # (1, T - prefill_len + 1, V)
    ref = full[:, prefill_len - 1:]                   # predictions for tokens prefill..T-1
    diff = float((ref - inc).abs().max())
    # Mathematically identical, but not bit-identical: the softmax normaliser sums
    # 6 terms in one pass and 12 in the other, and matmul sees a different batch
    # shape, so the reduction order changes. ~1e-6 in fp32 is arithmetic noise.
    # A real off-by-one in the q/kv offset or a broken shared-KV hand-off lands
    # at 1e-2 or worse, so the two are easy to tell apart.
    record(f"{tag}kv cache: incremental == full forward", diff < 1e-5, f"max_abs_diff={diff:.3e}")

    record(f"{tag}kv cache: seq length", cache.get_seq_length() == ids.shape[1],
           f"{cache.get_seq_length()} vs {ids.shape[1]}")

    layers = model.model.language_model.layers
    shared = [i for i, layer in enumerate(layers) if layer.self_attn.is_kv_shared_layer]
    record(f"{tag}kv cache: shared layers store nothing",
           all(cache.key_cache[i] is None for i in shared), f"shared={shared}")
    record(f"{tag}kv cache: non-shared layers filled",
           all(cache.key_cache[i] is not None for i in range(len(layers)) if i not in shared))


def test_generation(hfmodel, cfg, ckpt_parent, tag=""):
    """Greedy decoding must yield the same ids as transformers' `generate`."""
    model, _ = mine.load_gemma4_from_safetensors(os.path.join(ckpt_parent, "model"), verbose=False)
    model.eval()

    prompt = torch.tensor([[2, 5, 6, 7]])
    n_new = 6
    # an eos id that essentially never wins, so both sides run the full 6 steps
    eos = cfg.text_config.vocab_size - 1

    ours = mine_gen.generate(
        model, prompt, max_new_tokens=n_new, do_sample=False,
        eos_token_id=eos, pad_token_id=0,
    )
    with torch.no_grad():
        ref = hfmodel.generate(
            prompt, max_new_tokens=n_new, do_sample=False,
            eos_token_id=eos, pad_token_id=0, use_cache=True,
        )
    record(f"{tag}generate: greedy ids match transformers", torch.equal(ours, ref),
           f"ours={ours.tolist()[0][-n_new:]} ref={ref.tolist()[0][-n_new:]}")


# ---------------------------------------------------------------------------
def main():
    print(f"== preprocessing (tokenizer checkpoint: {CKPT_DIR}) ==")
    test_tokenizer()
    test_image_processor()
    test_audio_features()

    imgs = [Image.fromarray(np.random.default_rng(1).integers(0, 256, (64, 64, 3), dtype=np.uint8))]
    ref_img = Gemma4ImageProcessorPil(max_soft_tokens=70, patch_size=16, pooling_kernel_size=3)
    img_in = ref_img(images=imgs, return_tensors="pt")
    n_image_tokens = int(ref_img(images=imgs, return_tensors="pt")["num_soft_tokens_per_image"][0])
    audio_in = Gemma4AudioFeatureExtractor()(
        [(0.4 * np.sin(2 * np.pi * 440 * np.arange(16_000) / 16_000)).astype(np.float64)],
        return_tensors="pt",
    )

    for impl, include_audio, tag in (("eager", False, ""), ("sdpa", True, "sdpa ")):
        print(f"\n== tiny reference model, attn_implementation={impl} ==")
        d = tempfile.mkdtemp(prefix=f"gemma4_{impl}_")
        os.makedirs(os.path.join(d, "model"), exist_ok=True)
        hfmodel, cfg = build_reference(impl)
        print(f"   params: {sum(p.numel() for p in hfmodel.parameters())/1e6:.2f}M")
        export_checkpoint(hfmodel, cfg, os.path.join(d, "model"))
        test_forward(hfmodel, cfg, d, img_in["pixel_values"], img_in["image_position_ids"],
                     n_image_tokens, audio_in, include_audio, tag)
        test_kv_cache(hfmodel, cfg, d, tag)
        if impl == "eager":
            test_weight_loading(d)
            test_dtype_auto(hfmodel, cfg, d)
            test_generation(hfmodel, cfg, d)

    ok = all(r[1] for r in RESULTS)
    print("\n" + "=" * 72)
    print(f"RESULT: {sum(r[1] for r in RESULTS)}/{len(RESULTS)} checks passed")
    for name, passed, _ in RESULTS:
        if not passed:
            print("  FAILED:", name)
    print("=" * 72)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
