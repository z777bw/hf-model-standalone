---
name: hf-model-standalone
description: 把一个 HuggingFace 模型从 transformers 里"抽"成不依赖 transformers 的纯 PyTorch 实现——含 tokenizer/图像/音频预处理、权重加载、forward 与 loss，并用 tiny 随机权重做逐层数值对齐。当用户说"脱离 transformers 跑某个模型/自己实现某模型的 forward/loss/推理"、"把模型代码抽出来、去依赖、单文件化"、"不用 transformers 加载 hf 权重"、"复现某模型的预处理"时使用。
---

# 从 transformers 抽出一个可独立运行的模型

目标：给定 `org/model`，产出一份 `torch / numpy / pillow / tokenizers / safetensors` 就能跑的实现，
forward、loss、权重加载全部自洽，并且**能和 transformers 的输出逐位对上**。

---

## 0. 先回答"该关注哪里"

四个位置，按信息密度排序。别从 `modeling_*.py` 第一行开始读。

| # | 看什么 | 在哪 | 为什么关键 |
|---|---|---|---|
| 1 | `config.json` | 权重仓库根目录 | 架构的**唯一真相**。所有开关（层类型、head_dim、rope 参数、共享层、softcap、池化核）都在这里 |
| 2 | 权重 key 树 | `model.safetensors` 的 **header**（不用下载） | 决定你的模块**命名**。照抄 key 就免费获得 `load_state_dict` 兼容 |
| 3 | `modeling_*.py` 的 forward 链 | `site-packages/transformers/models/<arch>/` | 顺序、残差、norm 位置、mask 语义、dtype 处理 |
| 4 | `image_processing_*.py` / `feature_extraction_*.py` / `tokenizer.json` | 同上 + 仓库 | 预处理差 1e-3，模型输出就可能差 1e-1；而且它常**不按常理出牌**（aspect-ratio resize、直接 patchify、不做 normalize） |

以及两个"隐形的第 5 处"：`configuration_utils` 的 `__post_init__`（派生态，如
`layer_types` 默认 5:1、`per_layer_config` 的 head_dim 覆盖）和
`masking_utils` 的 mask 语义（0 允许 / min 屏蔽、sliding 是 `k > q - w`）。

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

⑥ 真实权重验证（在服务器上）：`report["missing"] == []`、`unexpected == []` 才算通。

---

## 2. 抽取策略：两条路，选一条

**A. 重写（推荐给"要读懂/要改"的场景）**
参考实现只当规范读。优点是干净、可控、单文件；缺点是要逐行对照。
Gemma 4 / 多模态模型基本只能走这条（依赖面太大：masking_utils、cache_utils、rope_utils、
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

## 5. 常见坑（Gemma 4 实测，全部踩过）

| 坑 | 症状 | 处理 |
|---|---|---|
| **参考按 `config.dtype` 加载 bf16** | 第一层就有 1e-2 差异，逐层放大，fp32 下本该 0.0 | 参考显式传 `dtype=torch.float32`，或自己传 `dtype="auto"`；先看 `next(ref.parameters()).dtype` |
| **`inv_freq` 注册成 buffer** | fp32 下 0.0，bf16 下第一层就差 1e-2；`cos` 逐位对、`sin` 差 1 ulp | `.to(bf16)` 会舍入所有浮点 buffer（`bf16(0.1)=0.10009765625`）。改成普通属性存 fp32，forward 里再 upcast |
| `config.to_dict()` 丢掉 `global_head_dim` | tiny 配置里 full-attention 层 head_dim 变成默认 512 | 导出 tiny 配置后手动补 `text_config.global_head_dim` |
| `per_layer_config` 让 `config.head_dim` 抛异常 | `AmbiguousGlobalPerLayerAttributeError` | 用 `config.per_layer_config[i].head_dim`，或自己在 config 里写 `layer_head_dim(i)` |
| eager 下音频 mask 语义反了 | 音频输出差 1e-1 且怎么调都不收敛 | 参考实现的 `Gemma4AudioAttention` 期望 bool mask，eager 给 float → 窗口内注意力退化为均匀。**对齐要用 sdpa** |
| PLE 被投影两次 | 每层 hidden 差 1e-2，logits 差 5e-2 | 投影只在 text model 内部做一次，外层只传 raw token embedding |
| 左 padding 的 loss | loss 差 3e-2，logits 只在有效位对上 | 第一个真实 token 由"全屏蔽行"预测，label 要一起忽略 |
| 位置表 buffer 初始化成 `ones` | 不报错，但输出不对 | `position_embedding_table` 初值是 1 不是 0/正态 |
| sliding window 边界 | 只在长序列时错 | 语义是 `kv_idx > q_idx - w`（含自身，窗口宽 w） |
| 视觉 patch 顺序 | 图看起来对但差 1e-2 | reshape 顺序 `(C,ph,P,pw,P)->(ph,pw,P,P,C)`，别用 unfold 的默认序 |

完整清单：`references/pitfalls.md`。

---

## 6. 交付物组织

```
<model>_standalone/
├── modeling_<arch>.py    # 单文件：config + 所有 nn.Module + forward/loss + 权重加载
├── preprocessing.py      # tokenizer / image / audio / processor 拼装
├── README.md             # 覆盖范围 + 未覆盖范围 + 实测 diff 表
└── tests/test_parity.py  # tiny 参考模型对齐
```

模型代码坚持一个文件：多模态模型动辄 3000 行，拆成 20 个模块后"和参考实现对拍"会变成
在 20 个文件间跳转，成本远大于单文件的阅读成本。

逐项打勾的执行清单见 `references/checklist.md`（配置 → 模块树 → 预处理 → 权重 → 对齐 → 交付）。

---

## 7. 参考案例

`examples/gemma4_standalone/`（google/gemma-4-E2B，5.1B，文本 + 视觉 + 音频）

- `modeling_gemma4.py`：1575 行单文件实现，运行时不 import transformers
- `preprocessing.py`：tokenizer / 图像切 patch / 音频 log-mel / processor 拼装
- `test_parity.py`：35 项检查全过，文本与视觉在 fp32 下**逐位相同（0.0）**

它包含三个参考实现自身的坑（`global_head_dim` 被 `per_layer_config` 吞掉、eager 音频 mask
类型、`inv_freq` 被 bf16 舍入），以及"dtype 不一致导致 1e-2 假差异"的完整排查过程，
都记录在 `references/pitfalls.md`。改造新模型时，先照着这份案例的目录结构摆。
