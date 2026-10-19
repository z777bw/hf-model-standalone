---
name: hf-model-standalone
description: 把一个 HuggingFace 模型从 transformers 里"抽"成不依赖 transformers 的纯 PyTorch 实现——含 tokenizer/图像/音频预处理、权重加载、forward 与 loss、KV cache 增量解码与生成循环，并用 tiny 随机权重做逐层数值对齐。当用户说"脱离 transformers 跑某个模型/自己实现某模型的 forward/loss/推理"、"把模型代码抽出来、去依赖、单文件化"、"不用 transformers 加载 hf 权重"、"实现 KV cache / 增量解码 / 生成循环"、"复现某模型的预处理"时使用。
---

# 从 transformers 抽出一个可独立运行的模型

目标：给定 `org/model`，产出一份 `torch / numpy / pillow / tokenizers / safetensors` 就能跑的实现，
forward、loss、权重加载、增量解码全部自洽，并且**能和 transformers 的输出对上**。

---

## 0. 先回答"该关注哪里"

四个位置，按信息密度排序。别从 `modeling_*.py` 第一行开始读。

| # | 看什么 | 在哪 | 为什么关键 |
|---|---|---|---|
| 1 | `config.json` | 权重仓库根目录 | 架构的**唯一真相**。所有开关（层类型、head_dim、rope 参数、共享层、softcap、池化核）都在这里 |
| 2 | 权重 key 树 | `model.safetensors` 的 **header**（不用下载） | 决定你的模块**命名**。照抄 key 就免费获得 `load_state_dict` 兼容 |
| 3 | `modeling_*.py` 的 forward 链 | `site-packages/transformers/models/<model_type>/` | 顺序、残差、norm 位置、mask 语义、dtype 处理 |
| 4 | `image_processing_*.py` / `feature_extraction_*.py` / `tokenizer.json` | 同上 + 仓库 | 预处理差 1e-3，模型输出就可能差 1e-1；而且它常**不按常理出牌**（aspect-ratio resize、直接 patchify、不做 normalize） |

以及两个"隐形的第 5 处"：`configuration_utils` 的 `__post_init__`（派生态，如
`layer_types` 默认 5:1、`per_layer_config` 的 head_dim 覆盖）和
`masking_utils` 的 `cache_position` / mask 语义（0 允许 / min 屏蔽、sliding 是 `k > q - w`）。

> 经验：**模块树 / 参数名 100% 镜像参考实现**。不要"顺便美化"成 `self.layers[i].attn.qkv`，
> 否则你得手写一套 key 映射，而 key 映射是最容易错、最难调的一层。

---

## 1. 标准工作流

```bash
# ① 只拉配置和预处理器（几 MB～几十 MB），绝不拉权重
python scripts/fetch_minimal.py google/gemma-4-E2B --out ./ckpt

# ② 通过 HTTP Range 读 safetensors header，拿到完整参数名/形状/dtype/字节数
python scripts/peek_checkpoint.py google/gemma-4-E2B            # 聚合视图（建议先看这个）
python scripts/peek_checkpoint.py google/gemma-4-E2B --list      # 全量 key
```

网络先自查（`502 Bad Gateway` 基本都是本地代理，不是网站挂了）：

```bash
# 内地网络 + 本地代理的常见可用组合
python scripts/fetch_minimal.py google/gemma-4-E2B --out ./ckpt \
       --endpoint https://hf-mirror.com --no-proxy
# 或者
export HF_ENDPOINT=https://hf-mirror.com
```

两个脚本都接受**本地路径**（含 `config.json` / `*.safetensors` 的目录），断网也能干活。
（`scripts/` 下三个脚本共用 `hf_net.py`，拷走时要连同它一起拷。）

```bash
# ③ 读参考实现：先列结构，再读关键函数
grep -n "^class \|^def " .../models/<arch>/modeling_<arch>.py
grep -n "^from\|^import" .../models/<arch>/modeling_<arch>.py   # 依赖面 = 你要自己实现的算子清单
```

④ **写实现**（`modeling_<arch>.py` 单文件 + `preprocessing.py`），逐条对照：

- config → dataclass（只保留会用到的键，其余走 `**kwargs`），把 `__post_init__` 里的派生态写出来
- 每个 class 的 `__init__` 里的 submodule 名**逐字**照抄（含 `nn.Buffer`！很多 buffer 是 persist 的）
- forward 的算子顺序、残差位置、norm 的 dtype、mask 的形状与取值，逐行对

⑤ **数值对齐**（见第 3 节）。先对齐 embedding → 逐层 hidden → 最后才看 logits。

⑥ **增量解码**（见第 5 节）。forward 对齐之后再接 cache，顺序反了会同时面对两类 bug。

⑦ 真实权重验证（在服务器上）：`report["missing"] == []`、`unexpected == []` 才算通。

---

## 2. 抽取策略：两条路，选一条

**A. 重写（推荐给"要读懂/要改"的场景）**
参考实现只当规范读。优点是干净、可控、单文件；缺点是要逐行对照。
多模态模型基本只能走这条（依赖面太大：masking_utils、cache_utils、rope_utils、
generation、integrations…）。

**B. 搬运 + 打桩（推荐给"只是想去掉依赖"的场景）**
把 `modeling_<arch>.py` 复制出来，重写 `from ...x import y`：
- 能一行顶掉的（`ACT2FN`、`PreTrainedModel`、`ModelOutput`）→ 写进 `_compat.py`
- 复杂子系统（`masking_utils`、`cache_utils`、`modeling_rope_utils`）→ 按需**裁剪**成 50~200 行，
  只保留本架构实际走到的分支
判别标准：`grep -n "^from \.\." modeling_<arch>.py` 数一下有多少个不同的子系统。
>3 个就建议走 A。

---

## 3. 没有权重怎么验证？造 tiny 参考模型

这是本 skill 的核心手法，**不需要 GPU、不需要下载权重**：

```bash
python scripts/make_tiny_reference.py google/gemma-4-E2B --out ./tiny_ckpt
# 读 config.json → 缩小维度（保留所有架构开关）→ 实例化参考模型 → 随机初始化
# → 存成真的 config.json + model.safetensors（tied 权重按 hub 习惯剔除）
```

然后写一个 `test_parity.py`：

1. 用你的加载器读 `./tiny_ckpt`，`missing/unexpected` 必须为空
2. 同一批输入喂两边，比较：
   **embedding → 每层 hidden → final norm → logits → loss**（用 forward hook 抓中间量）
3. 预处理单独比：`pixel_values` / `input_features` / `token ids` 必须逐位或 1e-6 内相等
4. 容差：fp32 文本塔应当 **0.0（逐位相同）**；多模态/音频 1e-6 量级；
   若只能做到 1e-3，几乎一定是某个 mask/rope/pooling 错了，别急着放宽容差

调试顺序（从便宜到贵）：
`embed → PLE/位置编码 → 单层 attn → 单层 mlp → 多了几层后开始漂`。
一旦某层输入相同、输出不同，问题就在这一层内部，hook 该层子模块逐个比。

更详细的对齐清单见 `references/parity.md`。

---

## 4. 权重加载（safetensors）

```python
from safetensors import safe_open
for shard in shards:                       # 单文件或 index.json 的 weight_map
    with safe_open(shard, framework="pt") as f:
        state_dict.update({k: f.get_tensor(k) for k in f.keys()})
```

必处理的 5 件事：

1. **tied embeddings**：`tie_word_embeddings=True` 时 hub 上**没有** `lm_head.weight`，
   要从 `embed_tokens.weight` 补
2. **冗余参数**：官方权重里常留着模型运行时用不到的权重（KV 共享层的 `k_proj/v_proj/k_norm`），
   加载时按 `_keys_to_ignore_on_load_unexpected` 的等价规则丢掉
3. **dtype**：`config.json` 的 `dtype` 是 bf16；先 `model.to(dtype)` 再 `load_state_dict`，
   或加载后统一 cast（注意 `inf` 边界 buffer 不能变 NaN）。**同时记住 `from_pretrained()`
   默认就按这个 `dtype` 加载**——比对时两边精度必须一致，否则你会看到 1e-2 的假差异
4. **buffer 也是参数**：`layer_scalar`、`input_min/max`、`output_min/max`、`embed_scale`
   （persist 的才在 ckpt 里）。`nn.Buffer(persistent=False)` 的不在，别去找
5. **分片 + 报错**：`strict=False` 然后打印 missing/unexpected，**不要**用 `strict=True` 硬扛

---

## 5. KV cache 与生成循环

`use_cache=False` 保持默认（纯前向，最容易和训练/对拍对齐）。要推理时再打开。
三块分开，便于单独替换：

| 文件 | 职责 |
|---|---|
| `kv_cache.py` | 每层一对 `(k, v)`；`update()` 沿 seq 维 append，返回完整历史；`crop()` / `reorder_cache()` 留作改造点 |
| `modeling_*.py` | attention 里 `past_key_values.update(...)`，**只有非共享层写** |
| `generation.py` | `prefill` → 循环 `decode_step` → `sample`，拼成 `generate` |

### 四个必须处理的点

**① mask 要做位置偏移。** decode 时 `q_len=1`、`kv_len=past+1`，query 的绝对位置是
`kv_len - q_len`，不是 0。忘了这一步，sliding mask 会永远放行（或全屏蔽）——
**短序列完全看不出来**。

```python
q_offset = kv_len - q_len
q_idx = torch.arange(q_len)[:, None] + q_offset
kv_idx = torch.arange(kv_len)[None, :]
```

**② KV 共享层不写 cache。** 它们没有 `k_proj/v_proj`，从 `shared_kv_states` 读。
写入方是"该 `layer_type` 的最后一个非共享层"（`store_full_length_kv`），而且必须在
`cache.update()` **之后**写，否则共享层拿到的是半截历史。

**③ position_ids 与 attention_mask 都要跟着 cache 走。**
`position_ids = arange(seq_len) + past_seen`；`attention_mask` 必须覆盖**全长**
（`past + new`），只传新 token 会 shape 报错或在广播里静默串味。

**④ decode 不要重跑视觉/音频塔。** 它们只在 prefill 出现。每步重算会吃掉全部时间预算，
而且 soft token 已经进 cache 了，重算也不会改变结果。

### 怎么验证

| 检查 | 期望 | 说明 |
| --- | --- | --- |
| 同一实现 `use_cache=True` vs `False` 的整段前向 | **逐位相同（0.0）** | cache 读写不该改变数值 |
| 增量 decode 拼接 vs 一次全序列 forward | **1e-6 量级，不是 0** | 见下 |
| greedy 生成的 token ids vs 参考实现 | 完全一致 | 最贴近用户感受的一条 |

第二项**不是 0 才是正常**：softmax 分母的求和项数不同（6 项 vs 12 项），matmul 的
batch shape 也不同，归约顺序变了。把第一项单独测，就是为了区分"cache 写错了"和
"归约顺序不同"——**1e-2 才是 bug**（位置偏移错、共享层取错层）。

---

## 6. 常见坑（移植时高频出现）

| 坑 | 症状 | 处理 |
|---|---|---|
| **参考按 `config.dtype` 加载 bf16** | 第一层就有 1e-2 差异，逐层放大，fp32 下本该 0.0 | 参考显式传 `dtype=torch.float32`，或自己传 `dtype="auto"`；先看 `next(ref.parameters()).dtype` |
| **`inv_freq` 注册成 buffer** | fp32 下 0.0，bf16 下第一层就差 1e-2；`cos` 逐位对、`sin` 差 1 ulp | `.to(bf16)` 会舍入所有浮点 buffer（`bf16(0.1)=0.10009765625`）。改成普通属性存 fp32，forward 里再 upcast |
| `config.to_dict()` 丢掉派生字段 | tiny 配置里 full-attention 层 head_dim 变成默认值 | 导出 tiny 配置后手动补回（`--set text_config.<field>=...`） |
| `per_layer_config` 让 `config.head_dim` 抛异常 | `AmbiguousGlobalPerLayerAttributeError` | 用 `config.per_layer_config[i].head_dim`，或自己在 config 里写 `layer_head_dim(i)` |
| eager 下音频 mask 语义反了 | 音频输出差 1e-1 且怎么调都不收敛 | 某些实现的 `masked_fill(mask.logical_not(), ...)` 期望 bool mask，eager 给 float → 退化为均匀注意力。**对齐时换 `attn_implementation` 试** |
| 同一份 embedding 被投影两次 | 每层 hidden 差 1e-2，logits 差 5e-2 | 投影只在内层做一次，外层只传 raw embedding |
| 左 padding 的 loss | loss 差 3e-2，logits 只在有效位对上 | 第一个真实 token 由"全屏蔽行"预测，label 要一起忽略 |
| 位置表 buffer 初始化成 `ones` | 不报错，但输出不对 | 先查 `__init__` 的初值，别随手 `torch.empty` |
| sliding window 边界 | 只在长序列时错 | 语义是 `kv_idx > q_idx - w`（含自身，窗口宽 w） |
| **decode 时 mask 没做位置偏移** | 短序列正常，超过窗口就全乱 | q 的绝对位置是 `kv_len - q_len`，不是 0 |
| **共享层读到半截 k/v** | 只有解码长度超过 prefill 时才错 | `store_full_length_kv` 必须在 `cache.update()` 之后写 |
| 视觉 patch 顺序 | 图看起来对但差 1e-2 | patch 内通道放最后一维，与 `Linear(C*P*P, hidden)` 对应 |

完整清单：`references/pitfalls.md`。

---

## 7. 交付物组织

```
<model>_standalone/
├── modeling_<arch>.py    # 单文件：config + 所有 nn.Module + forward/loss + 权重加载
├── preprocessing.py      # tokenizer / image / audio / processor 拼装
├── kv_cache.py           # 增量解码：最小 K/V 存储（可替换为 static / quantized）
├── generation.py         # prefill / decode_step / generate
├── README.md             # 覆盖范围 + 未覆盖范围 + 实测 diff 表
├── test_parity.py        # tiny 参考模型对齐 + cache 一致性
└── run_<arch>.py         # 真实权重的前向 / loss / 生成 demo
```

模型代码坚持一个文件：多模态模型动辄 3000 行，拆成 20 个模块后"和参考实现对拍"会变成
在 20 个文件间跳转，成本远大于单文件的阅读成本。`kv_cache.py` 与 `generation.py` 单独成文
是例外——它们是**要被替换**的部分，和"读懂模型"是两件事。

逐项打勾的执行清单见 `references/checklist.md`（配置 → 模块树 → 预处理 → 权重 → 对齐 → cache → 交付）。
