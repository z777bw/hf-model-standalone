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

## 2. 抽取方式：最大限度复用 transformers 的实现

原则：**以 transformers 的对应代码为蓝本，能搬就搬，只做"去依赖 + 去分支"的删减，不要凭理解重写。**

- 搬的是**源码**，不是运行时依赖：代码进你的文件，产物依然不 import transformers。
- 搬进来的部分保留原来的算子顺序、dtype、mask 取值、norm 位置 —— 这些细节最容易被"我觉得应该这样"改错。
- 只有三类东西需要自己写：config dataclass、权重加载器、把各部件串起来的接口（forward 的胶水）。

### 2.1 先列依赖面：决定搬哪些代码

```bash
grep -n "^from \.\|^import" modeling_<arch>.py   # 依赖面 = 要搬的源文件清单
grep -n "^class \|^def " modeling_<arch>.py      # 结构总览：类层次 + forward 链
```

| 依赖 | 搬法 |
|---|---|
| 工具件（用到的激活函数、`ModelOutput`、日志）| 搬用到的那个函数/表项；`ModelOutput` 用 `dataclass` 顶掉 |
| 计算子系统（mask、rope、cache、attention、generation）| 把对应文件里的相关函数整段搬进来，删掉本架构走不到的分支，通常剩 50~200 行 |
| 预处理（image processor / feature extractor / tokenizer）| 搬 `image_processing_*.py` / `feature_extraction_*.py` 的实现，剥掉 `ProcessorMixin` 外壳；tokenizer 用 `tokenizers` 直接加载仓库里的 `tokenizer.json` |

按"最小闭包"搬：缺哪个函数补哪个，只搬 forward 真正用到的部分，但搬到的部分逐行保留原样。

### 2.2 定点读：只读要搬的那几段

- **从 `__init__` 对结构**：照着权重文件的索引（`peek_checkpoint.py --list`）逐个核对 submodule
  名，对不上就是漏了层或写错了名。
- **从 `forward` 追调用**：只沿 `forward` 实际走到的路径往下，`generate()`、训练脚本、
  `prepare_inputs_for_generation` 不看。
- **遇到 `if` 按当前 config 定值**：只搬这个配置走的分支，另一条在注释里写一句"若有 X 则……"，
  方便以后改。

### 2.3 落笔规则

1. **保留命名**：类名 / 属性名 / 参数名逐字一致，这是加载官方权重的前提。
2. **保留实现细节**：算子顺序、dtype、mask 取值、池化边界照搬，不顺手"优化"。想改进，等数字对上之后再单独改。
3. **只删外壳，不动计算**：backend 分派收成一条自己看得懂的 attention 路径；`Cache` 换成最小的
   `KVCache`；`**kwargs` 展开成具名参数。
4. **标注出处**：文件头写清搬自哪个 transformers 版本、哪个文件，后面改动才好回溯。

### 2.4 收尾自检

```bash
grep -rn "import transformers\|from transformers" modeling_<arch>.py preprocessing.py   # 必须为空
```

再抽查两处：随便挑 forward 里的一个张量，能不能说出它的 shape 和来源？随便挑一个搬来的函数，
能不能指出参考实现里的对应位置？

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

## 6. 出现数值差异时怎么查

每个模型的具体问题都不一样，所以这里不列具体案例，只给一套定位方法：**每一步都先确认两边
的输入相同，再把范围缩小一半。** 顺序不要跳。

### ① 先排除假差异

在怀疑自己的实现之前，确认两边算的是同一件事：

- **dtype**：`from_pretrained()` 默认按参考 config 里的精度加载，先打 `next(ref.parameters()).dtype`
- **attention 实现**：eager / sdpa / flash 的归约顺序不同，1e-7 量级是正常的
- **模式**：`model.eval()` + `torch.no_grad()`，别让 dropout 参与
- **输入**：同一份 `input_ids` / `attention_mask`；`position_ids` 是参考自己算的还是你传的
- **设备**：CPU 与 GPU 的 kernel 不同，先在同一设备上比完再谈别的

### ② 从最前面开始，逐段二分

固定顺序：**预处理 → embedding → 第 1 层 → 前几层 → 全部层 → logits → loss**。
找到第一个"输入相同、输出不同"的模块，问题就在它内部；它之后的所有差异都是这一处的后果。

用 forward hook 抓中间量，两边注册同样的 hook：

```python
buf = {}
def hook(name):
    def fn(module, inp, out):
        buf[name] = (out[0] if isinstance(out, tuple) else out).detach()
    return fn
```

### ③ 一层内部还看不出，就继续往下钻

- 整层输出有差异 → hook 该层的子模块（attn / mlp / norm），继续二分
- 子模块也一致 → 差异在没被 hook 的中间量：q/k/v、rope 的 cos/sin、mask 本身、池化后的 token
- 还不够细 → 拦截算子（`torch.matmul`、`F.softmax`），给每次输出打个 checksum，
  **第一个对不上的算子**就是出事的地方

### ④ 按差异的形态判断方向

| 现象 | 大概率原因 |
|---|---|
| 1e-7 量级、只出现在 logits | 不同 attention backend 的归约顺序，正常 |
| fp32 下本该 0.0，却稳定在 1e-3 | 语义写错了（mask 取值、rope 拼法、投影次数），别用容差掩盖 |
| 第一层就差、之后一路放大 | 精度问题：浮点 buffer 被 `.to()` 舍入、参考按 bf16 加载、初始化值不同 |
| 前几层 0.0，某一层突然起跳 | 该层独有的逻辑：异构 head_dim、共享 KV 取错层、逐层缩放系数 |
| 只在长序列错 | 窗口 / 分块注意力的边界（窗口宽度、是否含自身） |
| 只在有 padding 时错 | mask 方向、有限负值；padded 行的输出本来就未定义 |
| 只在带 cache 解码时错 | query 位置偏移、cache 更新顺序、mask 没覆盖全长 |
| 只在某一种模态错 | 该模态的预处理或池化，不是语言塔的问题 |
| 怎么调都差 1e-1，且不随层数收敛 | 先怀疑参考实现自己在当前 backend 下的分支，换 `attn_implementation` 再试 |

### ⑤ 收尾

- 容差写进交付文档：哪一项、差多少、为什么。
- 每个修好的差异都补一条测试，防止以后改架构时复发。

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
