# gemma4_standalone — google/gemma-4-E2B 去 transformers 实现

从 `transformers` v5.16.1 的 `models/gemma4/` 逐行移植，**运行时不 import transformers**。
只用 `torch / numpy / pillow / tokenizers / safetensors`。

```
examples/gemma4_standalone/
├── modeling_gemma4.py   ← 模型全部代码（单文件）：config 树 + 所有 nn.Module + forward/loss + 权重加载
├── preprocessing.py     ← 输入预处理：tokenizer / 图像处理 / 音频 log-mel / processor 拼装
├── kv_cache.py          ← 增量解码：每层一对 (k, v)，append + crop + reorder
├── generation.py        ← prefill / decode_step / sample / generate
├── __init__.py
└── test_parity.py       ← 与 transformers 的数值对齐测试（tiny 随机权重，CPU，秒级）

examples/run_gemma4_e2b.py  ← 真实 10GB 权重的前向 + loss 示例（在服务器上跑）
```

`modeling_gemma4.py` 刻意**完全镜像**参考实现的模块树与参数名，所以
`state_dict` 不需要任何 key 重命名就能吃 HuggingFace 原始 safetensors。

---

## 覆盖范围

| 模块 | 内容 |
| --- | --- |
| 文本塔 | RMSNorm(fp32 内部)、QK-norm、**逐层 head_dim**、两种 RoPE（`default` / `proportional`+`partial_rotary_factor`）、sliding/full 混合注意力、**KV 共享**（`num_kv_shared_layers`）、共享层的**双倍宽 MLP**、**Per-Layer Embeddings (PLE)**、可选 MoE |
| 视觉塔 | patch embedder（`Linear` 直接吃 patch，2D 可学习位置表）、双向编码器 + 2D 多维 RoPE、按位置的空间池化到 `max_soft_tokens` |
| 音频塔 | SSCP 两次 stride-2 卷积下采样、chunked local attention（相对位置偏置 + per-dim scale + rel_shift + logit softcap）、LightConv1d、conformer-ish block |
| 融合 | `<|image|>` / `<|audio|>` 占位符被 soft token `masked_scatter` 替换；PLE 在替换前用 pad embedding 计算 |
| 头/损失 | `tanh` logit soft-capping(30) + 移位交叉熵（`-100` 忽略） |
| 权重 | 单文件/分片 safetensors、tied `lm_head`、丢弃 KV 共享层里多余的 k/v、dtype 转换 |
| 增量解码 | `KVCache`（append 式）+ `prefill` / `decode_step` / `generate`；支持 greedy / temperature / top-k / top-p |

未覆盖：beam search、静态预分配 cache、KV cache 量化、多卡与 CUDA graph。
`use_cache` 默认关闭，纯前向路径不受影响。

---

## 用法

```python
import torch
from gemma4_standalone import Gemma4Processor, load_gemma4_from_safetensors

model, report = load_gemma4_from_safetensors("/path/to/gemma-4-E2B", dtype="auto")  # 跟随 config.dtype(bf16)
print(report)            # missing / unexpected / dropped / dtype

proc = Gemma4Processor.from_pretrained_dir("/path/to/gemma-4-E2B")
batch = proc(text="<|image|>Describe this image.", images=open("cat.png", "rb").read())

out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
            image_position_ids=batch["image_position_ids"],
            labels=batch["input_ids"])
print(out.logits.shape, out.loss.item())
```

纯文本 / 纯音频同理，`input_features` + `input_features_mask` 走音频塔。

### 增量解码

```python
from gemma4_standalone import generate

new_ids = generate(
    model, batch["input_ids"], max_new_tokens=32, do_sample=False,
    eos_token_id=[proc.tokenizer.eos_token_id],
    # 多模态输入只喂给 prefill；decode 阶段不会再跑视觉塔
    pixel_values=batch["pixel_values"],
    image_position_ids=batch["image_position_ids"],
)
print(proc.tokenizer.decode(new_ids[0, batch["input_ids"].shape[1]:]))
```

`generate` 就是 `prefill`（一整段 prompt 一次前向，填满 cache）+ 循环 `decode_step`
（每次一个 token，只算它的 query）。`kv_cache.py` 是一个"每层一对 tensor、沿 seq 维
`cat`"的最小实现 —— 故意写得能一眼看完，方便你换成 static cache / 量化 cache / beam search。

---

## 验证（`tests/test_parity.py`）

没有 10GB 权重也能验证：造一个 **架构开关和 E2B 完全一致** 的 tiny 模型
（per-layer head_dim 8/16、KV 共享 2 层、双倍宽 MLP、PLE、softcap、2D 视觉 RoPE、
chunked 音频注意力），随机初始化，导出成真的 `model.safetensors`，再由本实现加载比对。

```
$ python examples/gemma4_standalone/test_parity.py     # 仓库根目录下执行
...
RESULT: 46/46 checks passed
```

其中 tokenizer 那几项需要真实的 `tokenizer.json`（约 31 MB，不含权重）：

```bash
python scripts/fetch_minimal.py google/gemma-4-E2B --out ./ckpt
# 或指向任意已下载的目录
GEMMA4_CKPT=/data/models/gemma-4-E2B python examples/gemma4_standalone/test_parity.py
```

没有的话这几项会打印 `[skip]` 并从计数里剔除，其余 42 项照常跑。

实测结果：

| 检查 | max abs diff |
| --- | --- |
| tokenizer ids（含中文/emoji/代码）、batch 左 padding、decode | 完全一致 |
| `pixel_values` / `image_position_ids` / `num_soft_tokens` | 5.96e-8 / 完全一致 / 一致 |
| 音频 log-mel + mask | 2.49e-4 / 完全一致 |
| 纯文本 logits + loss（eager 参考） | **0.0**（逐位相同） |
| 图像 + 文本 logits + loss | **0.0** |
| 音频 + 文本 / 图像+音频+文本（sdpa 参考） | ≤ 5.6e-7 |
| 权重加载：missing/unexpected/dropped | 0 / 0 / 6（丢弃 KV 共享层的冗余 k/v） |
| `dtype="auto"` 读 bf16 checkpoint vs `from_pretrained` 的 bf16 参考 | **0.0** |
| `use_cache=True` vs `False` 的整段前向（隔离 cache 本身） | **0.0** |
| 增量 decode 拼接 vs 一次全序列 forward（12 token > sliding window 8） | 9.09e-7 / 8.35e-7 |
| greedy 生成的 token ids vs `model.generate()` | 完全一致 |

**为什么"增量 vs 全量"不是 0？** 两者数学上等价，但 softmax 分母的求和项数不同
（prefill 时一次算 6 项，decode 时算 12 项），matmul 的 batch shape 也不同，归约顺序变了。
fp32 下 ~1e-6 就是算术噪声；**1e-2 才是 bug**（位置偏移错、共享层取错层）。
把"`use_cache` 开/关的整段前向"单独测成 0.0，就是为了把这两件事区分开。

三个必须知道的参考实现细节（测试里已处理，详见 [`references/pitfalls.md`](../../references/pitfalls.md)）：

1. `config.to_dict()` 会把 `global_head_dim` 吞进 `per_layer_config` 而丢失 —— 官方
   checkpoint 的 `config.json` 是带这个字段的，导出 tiny 配置时要手动补回去。
2. `Gemma4AudioAttention` 里写的是 `masked_fill(mask.logical_not(), -1e9)`，即**期望 bool 掩码**；
   eager 路径给的是 0/-inf 的 float 掩码，于是窗口内所有 logit 被覆盖成 -1e9，注意力退化为
   窗口内均匀平均。sdpa 路径给 bool 掩码才是正常语义 —— 音频对齐要用 sdpa。
3. **`inv_freq` 不能注册成 buffer**：`model.to(bf16)` 会把所有浮点 buffer 一起降精度，
   `bf16(0.1) = 0.10009765625`，`.float()` 也救不回来，RoPE 角度整体偏 ~1e-3 rad。
   参考实现让 `inv_freq` 一直是 fp32、forward 里再 upcast；本实现用普通属性存 fp32 张量。

### 精度对齐：先统一 dtype，再谈 diff

`config.json` 顶层的 `"dtype": "bfloat16"` 会被 `from_pretrained()` 当成**默认加载精度**。
所以「我的实现 fp32 vs 参考 bf16」会看到 ~1e-2 的差异（逐层放大，第一层就有），
这是精度不是 bug。两种做法任选：

```python
# A. 参考也压成 fp32（推荐做逐位比对）
ref = Gemma4ForConditionalGeneration.from_pretrained(p, dtype=torch.float32)
ours, _ = load_gemma4_from_safetensors(p, dtype=torch.float32)

# B. 让本实现跟随 checkpoint 精度，复现 from_pretrained 的默认行为
ours, _ = load_gemma4_from_safetensors(p, dtype="auto")   # 读 config.dtype
```

`dtype=None`（默认）就是 fp32；`verbose=True` 时若 `config.dtype` 不是 float32 会打印提示。
统一精度后，本实现与参考在 fp32 下是**逐位相同**（logits diff 0.0），在 bf16 下也是 0.0
（前提是 RoPE 的 `inv_freq` 保持 fp32，见上面第 3 条）。

本案例的开发过程**没有下载过权重**：本地只有 `config.json / processor_config.json /
tokenizer_config.json / generation_config.json / tokenizer.json`（约 31 MB），
模块命名与形状是通过 HTTP Range 读 safetensors header（258 KB）得到的。
真实 10 GB 权重的加载与对拍交给 `examples/run_gemma4_e2b.py`，在有 GPU 的机器上跑。
