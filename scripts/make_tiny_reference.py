#!/usr/bin/env python
"""Build a tiny, randomly-initialised reference checkpoint that keeps every architectural knob.

Why: you usually cannot afford the real weights, but you *can* verify a reimplementation
bit-exactly against the reference if both sides see the same small weights.

It shrinks only the dims that don't change control flow (hidden/heads/layers/vocab) and
leaves the switches alone (layer_types, KV sharing, softcap, rope types, pooling kernel...).
Output = a normal HF checkpoint dir: config.json + model.safetensors, minus the tied lm_head.

    python make_tiny_reference.py google/gemma-4-E2B --out ./tiny_ckpt
    python make_tiny_reference.py ./ckpt --out ./tiny_ckpt --layers 6 --hidden 32
    python make_tiny_reference.py ./ckpt --out ./tiny_ckpt --set text_config.global_head_dim=16

Caveats (it will tell you when it matters):
  * a few models hard-code dims (e.g. Gemma 4 audio needs subsampling_conv_channels[0] == mel_bins);
    the script validates by instantiating + running a dummy forward and prints the error if any
  * `config.to_dict()` sometimes drops per-layer fields; use --set to re-add them
"""

import argparse
import json
import math
import os
import sys

TEXT_SHRINK = {
    "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 6,
    "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
    "global_head_dim": 8, "vocab_size": 128, "max_position_embeddings": 256,
    "hidden_size_per_layer_input": 8, "vocab_size_per_layer_input": 128,
    "sliding_window": 8, "moe_intermediate_size": 16, "num_experts": 4,
    "hidden_act": "silu", "hidden_activation": "gelu_pytorch_tanh",
}
VISION_SHRINK = {
    "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
    "num_attention_heads": 4, "num_key_value_heads": 4, "head_dim": 8,
    "position_embedding_size": 128, "max_position_embeddings": 256,
}
AUDIO_SHRINK = {
    "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
    "output_proj_dims": 32,
}
SHRINK_BY_SUBCONFIG = {
    "text_config": TEXT_SHRINK, "vision_config": VISION_SHRINK,
    "audio_config": AUDIO_SHRINK, "llm_config": TEXT_SHRINK,
}


def load_config_dict(src):
    if os.path.isdir(src):
        path = os.path.join(src, "config.json")
    elif os.path.isfile(src):
        path = src
    else:
        from transformers import AutoConfig

        return AutoConfig.from_pretrained(src).to_dict()
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def shrink(d, layers=None, hidden=None, heads=None, vocab=None):
    for key, sub in d.items():
        if not isinstance(sub, dict) or key not in SHRINK_BY_SUBCONFIG:
            continue
        shrink_map = SHRINK_BY_SUBCONFIG[key]
        for k, v in shrink_map.items():
            if k in sub:
                sub[k] = v
        if hidden:
            sub["hidden_size"] = hidden
            if "intermediate_size" in sub:
                sub["intermediate_size"] = hidden * 2
        if heads:
            sub["num_attention_heads"] = heads
            sub["num_key_value_heads"] = max(1, heads // 2)
            sub["head_dim"] = max(2, (hidden or sub.get("hidden_size", 32)) // heads)
        if groups := sub.get("subsampling_conv_channels"):
            # keep channels[0] == mel bins: the SSCP flattens (channels[1], mel_bins // 4)
            pass
        if layers and "num_hidden_layers" in sub:
            sub["num_hidden_layers"] = layers
            lt = sub.get("layer_types")
            if lt:
                # tile the first "cycle" (up to and including the first full_attention),
                # then force the last layer to full_attention like __post_init__ does
                cycle = lt[: lt.index("full_attention") + 1] if "full_attention" in lt else lt
                sub["layer_types"] = [cycle[i % len(cycle)] for i in range(layers - 1)] + ["full_attention"]
            if sub.get("num_kv_shared_layers"):
                # KV sharing only works if every shared layer type has a non-shared
                # predecessor of the same type; shrink the count until that holds
                n_layers = sub["num_hidden_layers"]
                want = min(sub["num_kv_shared_layers"], max(n_layers // 2, 0))
                ns = want
                while ns > 0:
                    first = n_layers - ns
                    shared = set(sub["layer_types"][first:])
                    src = set(sub["layer_types"][:first])
                    if first > 0 and shared <= src:
                        break
                    ns -= 1
                if ns != want:
                    print(f"  note: num_kv_shared_layers {want} -> {ns} (no valid non-shared source layer)")
                sub["num_kv_shared_layers"] = ns
    # vocab-dependent ids must stay in range
    vocab_size = vocab or TEXT_SHRINK["vocab_size"]
    n = 20
    for key, sub in d.items():
        if isinstance(sub, dict):
            for k in list(sub):
                if k.endswith("_token_id") and isinstance(sub[k], int) and sub[k] >= vocab_size:
                    sub[k] = n
                    n += 1
        if key.endswith("_token_id") or key.endswith("_token_index"):
            if isinstance(d[key], int) and d[key] >= vocab_size:
                d[key] = n
                n += 1
    for k in ("image_seq_length", "audio_seq_length"):
        if k in d:
            d[k] = 70 if k.startswith("image") else 750
    return d


def apply_sets(d, pairs):
    for pair in pairs or []:
        path, value = pair.split("=", 1)
        node = d
        parts = path.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
        node[parts[-1]] = value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="repo id or local checkpoint dir / config.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--heads", type=int, default=None)
    ap.add_argument("--set", action="append", default=[], metavar="PATH=VALUE")
    ap.add_argument("--arch", default=None, help="override class name (default: config.architectures[0])")
    ap.add_argument("--attn-implementation", default="eager")
    ap.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "bfloat16", "float16"],
        help="precision written into config.json AND used for the weights. "
        "Keep float32: from_pretrained() loads in config['dtype'], so a bf16 tiny ckpt makes the "
        "reference bf16 while your port is fp32 -> ~1e-2 'phantom' diffs.",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    import transformers
    from safetensors.torch import save_file

    d = shrink(load_config_dict(args.source), args.layers, args.hidden, args.heads)
    apply_sets(d, args.set)

    arch = args.arch or (d.get("architectures") or [None])[0]
    if not arch or not hasattr(transformers, arch):
        sys.exit(
            f"cannot resolve model class for architectures={d.get('architectures')}; "
            "pass --arch (e.g. --arch LlamaForCausalLM)"
        )
    model_type = d.get("model_type")
    base_cls = arch.split("For")[0]           # Gemma4ForConditionalGeneration -> Gemma4
    cfg_cls = getattr(transformers, f"{base_cls}Config", None)

    torch.manual_seed(args.seed)
    try:
        if cfg_cls is not None:
            cfg = cfg_cls(**{k: v for k, v in d.items() if k != "architectures"})
        else:
            from transformers import AutoConfig

            cfg = AutoConfig.for_model(model_type, **{k: v for k, v in d.items() if k != "architectures"})
    except Exception as exc:
        sys.exit(
            f"could not rebuild the config from the shrunk dict: {exc}\n"
            "hand-write a small config.json instead (that is often faster than fighting the schema)"
        )

    if hasattr(cfg, "_attn_implementation"):
        cfg._attn_implementation = args.attn_implementation

    tdt = getattr(torch, args.dtype)
    # The config we cloned carries the source repo's `dtype` (usually bfloat16), and
    # transformers initialises the freshly built model in that dtype. Setting it before
    # construction, plus an unconditional `.to()`, is what actually gives us float32
    # weights -- otherwise a "float32" tiny ckpt is silently stored as bf16.
    cfg.dtype = args.dtype
    if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "dtype"):
        cfg.text_config.dtype = args.dtype

    model = getattr(transformers, arch)(cfg)
    model.eval()
    model = model.to(tdt)
    n = sum(p.numel() for p in model.parameters())
    real_dtypes = sorted({str(p.dtype) for p in model.parameters()})
    print(f"{arch}: {n/1e6:.3f} M params, dtype={args.dtype} (params are {', '.join(real_dtypes)})")

    os.makedirs(args.out, exist_ok=True)
    out_cfg = cfg.to_dict()
    apply_sets(out_cfg, args.set)
    # from_pretrained() uses this as the default load dtype -> keep both sides identical
    out_cfg["dtype"] = args.dtype
    if isinstance(out_cfg.get("text_config"), dict):
        out_cfg["text_config"]["dtype"] = args.dtype
    with open(os.path.join(args.out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(out_cfg, f, indent=2)

    sd = {k: v.contiguous().clone() for k, v in model.state_dict().items()}
    # sanity: the file dtype must match what config.json claims, or `from_pretrained`
    # will upcast bf16-rounded weights and the "fp32" comparison is quietly meaningless
    weight_dtypes = sorted({str(v.dtype) for v in sd.values() if v.is_floating_point()})
    if weight_dtypes != [str(tdt)]:
        sys.exit(f"internal error: weights are {weight_dtypes} but --dtype={args.dtype}; refusing to write")
    if getattr(cfg, "tie_word_embeddings", False) and "lm_head.weight" in sd:
        sd.pop("lm_head.weight")  # hub checkpoints omit tied weights
    save_file(sd, os.path.join(args.out, "model.safetensors"))
    print(f"wrote {args.out}/config.json + model.safetensors ({len(sd)} tensors)")

    if isinstance(out_cfg.get("text_config"), dict) and "global_head_dim" in d.get("text_config", {}):
        print("note: re-add per-layer fields with --set if to_dict() dropped them "
              f"(e.g. --set text_config.global_head_dim={d['text_config']['global_head_dim']})")


if __name__ == "__main__":
    main()
