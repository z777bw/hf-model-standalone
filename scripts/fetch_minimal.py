#!/usr/bin/env python
"""Download only the *small* files of a checkpoint: configs, tokenizer, processor.

Weight files are never touched.  Everything the preprocessing port needs lives here.

    python fetch_minimal.py google/gemma-4-E2B --out ./ckpt
    HF_ENDPOINT=https://hf-mirror.com python fetch_minimal.py google/gemma-4-E2B --out ./ckpt
    python fetch_minimal.py google/gemma-4-E2B --out ./ckpt --endpoint https://hf-mirror.com --no-proxy
    python fetch_minimal.py google/gemma-4-E2B --out ./ckpt --max-mb 80 --dry-run

Prints a "what to port" map at the end: which file drives which piece of preprocessing.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hf_net import FetchError, add_net_args, http_get  # noqa: E402  (same-dir helper)

DEFAULT_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
WEIGHT_SUFFIXES = (
    ".safetensors", ".bin", ".pt", ".pth", ".gguf", ".onnx", ".h5", ".msgpack",
    ".ot", ".ckpt", ".tar", ".zip",
)

ROLE = {
    "config.json": "model architecture (THE source of truth)",
    "generation_config.json": "sampling defaults / eos pairs",
    "processor_config.json": "image processor + video processor + audio feature extractor defaults",
    "preprocessor_config.json": "image processor defaults (older layout)",
    "tokenizer_config.json": "special tokens, padding_side, chat_template",
    "tokenizer.json": "BPE merges/vocab + normalizer + post_processor (feeds the `tokenizers` lib)",
    "tokenizer.model": "SentencePiece proto (only if the repo is slow-tokenizer based)",
    "special_tokens_map.json": "extra special tokens",
    "chat_template.jinja": "chat formatting (often the only place it lives)",
    "vocab.json": "BPE vocab (legacy split layout)",
    "merges.txt": "BPE merges (legacy split layout)",
}


def repo_files(repo, endpoint, use_proxy=True):
    url = f"{endpoint}/api/models/{repo}/tree/main?recursive=true"
    try:
        data = json.loads(http_get(url, use_proxy=use_proxy, endpoint=endpoint))
        return [(d["path"], d.get("size", 0)) for d in data if d.get("type") == "file"]
    except FetchError as e:
        if "HTTP 4" not in e.reason:  # network trouble: don't mask it with a second request
            raise
        data = json.loads(http_get(f"{endpoint}/api/models/{repo}", use_proxy=use_proxy, endpoint=endpoint))
        return [(s["rfilename"], 0) for s in data.get("siblings", [])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("--out", required=True)
    add_net_args(ap, DEFAULT_ENDPOINT)
    ap.add_argument("--max-mb", type=float, default=64.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--with-readme", action="store_true")
    args = ap.parse_args()

    endpoint = args.endpoint.rstrip("/")
    os.makedirs(args.out, exist_ok=True)
    try:
        files = repo_files(args.repo, endpoint, use_proxy=not args.no_proxy)
    except FetchError as e:
        sys.exit(f"fetch failed.\n  {e}")

    picked, skipped = [], []
    for path, size in files:
        name = os.path.basename(path)
        if path.lower().endswith(WEIGHT_SUFFIXES):
            skipped.append((path, size, "weights"))
            continue
        if name == "README.md" and not args.with_readme:
            continue
        if size and size > args.max_mb * 1e6:
            skipped.append((path, size, f">{args.max_mb} MB"))
            continue
        picked.append((path, size))

    print(f"{args.repo}: {len(files)} files, downloading {len(picked)}\n")
    for path, size in picked:
        dest = os.path.join(args.out, path)
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        if args.dry_run:
            print(f"  [dry-run] {path:44s} {size/1e6:7.2f} MB")
            continue
        url = f"{endpoint}/{args.repo}/resolve/main/{path}"
        try:
            data = http_get(url, use_proxy=not args.no_proxy, endpoint=endpoint)
        except FetchError as e:
            sys.exit(f"download failed for {path}.\n  {e}")
        with open(dest, "wb") as f:
            f.write(data)
        print(f"  saved {path:44s} {len(data)/1e6:7.2f} MB")

    if skipped:
        print("\nskipped:")
        for path, size, why in skipped[:12]:
            print(f"  {path:44s} {size/1e6:8.2f} MB  ({why})")

    print("\nwhat to port:")
    for path, _ in picked:
        name = os.path.basename(path)
        if name in ROLE:
            print(f"  {name:26s} -> {ROLE[name]}")
        elif name.endswith(".py"):
            print(f"  {name:26s} -> remote code (trust_remote_code): read it before porting")


if __name__ == "__main__":
    main()
