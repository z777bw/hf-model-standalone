"""Load the real google/gemma-4-E2B (10.2 GB bf16) with the standalone code.

Needs ~11 GB of disk and, for a 5.1B model, a CUDA GPU (bf16, weights + activations
for a short sequence fits in ~12 GB).  Run this on the server, not on a laptop.

    python examples/run_gemma4_e2b.py --model-dir /data/models/gemma-4-E2B

What it does
  1. loads config + tokenizer + processor config from the checkpoint dir (no transformers)
  2. loads raw safetensors weights into the standalone module tree
  3. runs one image+text forward with labels -> loss
  4. (optional) --compare-transformers checks the same batch against transformers
"""

import argparse
import io
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gemma4_standalone import Gemma4Processor, load_gemma4_from_safetensors


def make_demo_image(size=(448, 448)):
    """Replace with a real image; kept dependency-free on purpose."""
    from PIL import Image

    img = Image.new("RGB", size)
    px = img.load()
    for y in range(size[1]):
        for x in range(size[0]):
            px[x, y] = (x % 256, y % 256, (x + y) % 256)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="local dir with config.json/model.safetensors")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-soft-tokens", type=int, default=280)
    ap.add_argument("--compare-transformers", action="store_true")
    ap.add_argument("--prompt", default=None)
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    print(f"loading from {args.model_dir} as {args.dtype} on {args.device} ...")
    model, report = load_gemma4_from_safetensors(args.model_dir, dtype=dtype, device=args.device)
    print("  missing   :", report["missing"][:5], "..." if len(report["missing"]) > 5 else "")
    print("  unexpected:", report["unexpected"][:5])
    print("  dropped   :", len(report["dropped"]), "unused KV-shared tensors")

    proc = Gemma4Processor.from_pretrained_dir(args.model_dir)
    if args.max_soft_tokens != 280:
        proc.image_processor.max_soft_tokens = args.max_soft_tokens

    image = make_demo_image()
    n_soft = proc.image_processor.preprocess([image])["num_soft_tokens_per_image"][0]
    text = args.prompt or "<|image|>Describe the image in one sentence."
    batch = proc(text=text, images=[image])
    # proc already expanded <|image|> into <boi> + n_soft soft tokens + <eoi>
    print("  input_ids:", tuple(batch["input_ids"].shape),
          "pixel_values:", tuple(batch["pixel_values"].shape),
          "n_soft:", n_soft)

    dev = args.device
    batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
    with torch.no_grad():
        out = model(**batch, labels=batch["input_ids"])

    print("  logits:", tuple(out.logits.shape), "loss:", float(out.loss))
    print("  image soft tokens:", tuple(out.image_hidden_states.shape))

    if args.compare_transformers:
        from transformers import AutoProcessor, Gemma4ForConditionalGeneration

        ref = Gemma4ForConditionalGeneration.from_pretrained(
            args.model_dir, dtype=dtype, attn_implementation="eager"
        ).to(dev).eval()
        with torch.no_grad():
            ref_out = ref(**batch)
        d = (ref_out.logits.float() - out.logits.float()).abs().max().item()
        print(f"  dtype: ours={next(model.parameters()).dtype} ref={next(ref.parameters()).dtype}")
        print(f"  max |logits_ours - logits_transformers| = {d:.3e}")
        if args.dtype != "float32":
            print("  (bf16/fp16 下 1e-2 量级属正常；要逐位相同请加 --dtype float32)")


if __name__ == "__main__":
    main()
