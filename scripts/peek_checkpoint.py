#!/usr/bin/env python
"""List every parameter of a (remote or local) safetensors checkpoint WITHOUT downloading it.

safetensors layout:  [u64 LE header_len][header_len bytes of JSON][tensor bytes]
so two HTTP Range requests give you all names / shapes / dtypes / byte sizes.

    python peek_checkpoint.py google/gemma-4-E2B
    python peek_checkpoint.py google/gemma-4-E2B --list          # every key
    python peek_checkpoint.py /data/models/gemma-4-E2B            # local dir (offline)
    HF_ENDPOINT=https://hf-mirror.com python peek_checkpoint.py ...
    python peek_checkpoint.py google/gemma-4-E2B --endpoint https://hf-mirror.com --no-proxy

Aggregated view collapses layer indices (`layers.12.` -> `layers.N.`) so you can see
the module tree and where the megabytes are.
"""

import argparse
import json
import os
import struct
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hf_net import FetchError, add_net_args, http_get  # noqa: E402  (same-dir helper)

DEFAULT_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")


def repo_files(repo, endpoint, use_proxy=True):
    """[(path, size)] from the tree API, fallback to siblings for old endpoints."""
    url = f"{endpoint}/api/models/{repo}/tree/main?recursive=true"
    try:
        data = json.loads(http_get(url, use_proxy=use_proxy, endpoint=endpoint))
        return [(d["path"], d.get("size", 0)) for d in data if d.get("type") == "file"]
    except FetchError as e:
        if "HTTP 4" not in e.reason:  # network trouble: don't mask it with a second request
            raise
        data = json.loads(http_get(f"{endpoint}/api/models/{repo}", use_proxy=use_proxy, endpoint=endpoint))
        return [(s["rfilename"], 0) for s in data.get("siblings", [])]


def header_from_url(url, timeout=60, use_proxy=True, endpoint=None):
    raw = http_get(url, {"Range": "bytes=0-7"}, timeout, use_proxy=use_proxy, endpoint=endpoint)
    (n,) = struct.unpack("<Q", raw[:8])
    payload = http_get(url, {"Range": f"bytes=8-{8 + n - 1}"}, timeout, use_proxy=use_proxy, endpoint=endpoint)
    head = json.loads(payload.decode("utf-8"))
    head.pop("__metadata__", None)
    return head


def header_from_file(path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        head = json.loads(f.read(n).decode("utf-8"))
    head.pop("__metadata__", None)
    return head


def collect(source, endpoint, use_proxy=True):
    """dict of tensor_name -> {shape, dtype, bytes}"""
    out = {}
    if os.path.exists(source):
        if os.path.isfile(source):
            files = [source]
        else:
            index = os.path.join(source, "model.safetensors.index.json")
            if os.path.exists(index):
                with open(index, "r", encoding="utf-8") as f:
                    files = [os.path.join(source, s) for s in sorted(set(json.load(f)["weight_map"].values()))]
            else:
                files = sorted(
                    os.path.join(source, f) for f in os.listdir(source) if f.endswith(".safetensors")
                )
        for p in files:
            out.update(header_from_file(p))
        return out

    files = [p for p, _ in repo_files(source, endpoint, use_proxy)]
    index_name = next((p for p in files if p.endswith(".safetensors.index.json")), None)
    if index_name:
        index = json.loads(
            http_get(f"{endpoint}/{source}/resolve/main/{index_name}", use_proxy=use_proxy, endpoint=endpoint)
        )
        shards = sorted(set(index["weight_map"].values()))
    else:
        shards = [p for p in files if p.endswith(".safetensors")]
    if not shards:
        sys.exit(f"no safetensors found in {source}")
    for s in shards:
        out.update(header_from_url(f"{endpoint}/{source}/resolve/main/{s}", use_proxy=use_proxy, endpoint=endpoint))
    return out


def collapse(name):
    return ".".join("N" if part.isdigit() else part for part in name.split("."))


def human_count(n: float) -> str:
    """1.23 M / 45.6 K style, so tiny checkpoints don't print as 0.000 B."""
    for scale, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(n) >= scale:
            return f"{n / scale:.3f} {suffix}"
    return str(int(n))


def human_bytes(n: float) -> str:
    for scale, suffix in ((1e9, "GB"), (1e6, "MB"), (1e3, "KB")):
        if abs(n) >= scale:
            return f"{n / scale:.1f} {suffix}"
    return f"{int(n)} B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="repo id or local path")
    add_net_args(ap, DEFAULT_ENDPOINT)
    ap.add_argument("--list", action="store_true", help="print every key instead of the aggregated view")
    ap.add_argument("--all", action="store_true", help="also list the 0-dim scalar buffers (default: rolled up)")
    ap.add_argument("--sort", default="bytes", choices=["bytes", "name"])
    args = ap.parse_args()

    try:
        tensors = collect(args.source, args.endpoint.rstrip("/"), use_proxy=not args.no_proxy)
    except FetchError as e:
        sys.exit(f"fetch failed.\n  {e}")
    total = sum(v["data_offsets"][1] - v["data_offsets"][0] for v in tensors.values())
    n_params = sum(
        __import__("math").prod(v["shape"]) if v["shape"] else 1 for v in tensors.values()
    )
    size_str = f"{total/1e9:.2f} GB" if total >= 1e9 else human_bytes(total)
    print(f"{args.source}: {len(tensors)} tensors, {human_count(n_params)} params, {size_str}\n")

    if args.list:
        for k in sorted(tensors):
            v = tensors[k]
            print(f"{k:80s} {str(v['shape']):26s} {v['dtype']}")
        return

    groups = defaultdict(lambda: {"count": 0, "bytes": 0, "params": 0, "shape": None, "dtype": None})
    scalars = defaultdict(lambda: {"count": 0, "bytes": 0})
    for k, v in tensors.items():
        n = 1
        for d in v["shape"]:
            n *= d
        if not v["shape"]:
            # 0-dim clamp bounds (input_min / output_max ...): dozens of them, all 0.0 MB.
            # Printing each one drowns the module tree, so roll them up per module.
            s = scalars[collapse(k.rsplit(".", 1)[0])]
            s["count"] += 1
            s["bytes"] += v["data_offsets"][1] - v["data_offsets"][0]
            continue
        g = groups[collapse(k)]
        g["count"] += 1
        g["bytes"] += v["data_offsets"][1] - v["data_offsets"][0]
        g["params"] += n
        g["shape"] = v["shape"]
        g["dtype"] = v["dtype"]

    keys = sorted(groups, key=(lambda k: -groups[k]["bytes"]) if args.sort == "bytes" else (lambda k: k))
    for k in keys:
        g = groups[k]
        print(f"{k:74s} x{g['count']:<3d} {str(g['shape']):24s} {g['dtype']:8s} "
              f"{human_count(g['params']).rjust(10)} {human_bytes(g['bytes']).rjust(10)}")

    if scalars:
        n_sc = sum(s["count"] for s in scalars.values())
        print(f"\n+ {n_sc} scalar buffers (0-dim clamp bounds such as input_min/output_max) "
              f"across {len(scalars)} modules, rolled up"
              + ("" if args.all else " -- pass --all to list them"))
        if args.all:
            for k in sorted(scalars):
                print(f"  {k:72s} x{scalars[k]['count']}")


if __name__ == "__main__":
    main()
