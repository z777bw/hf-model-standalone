# hf-model-standalone 🧩

> **把 HuggingFace 模型从 `transformers` 里抽出来，变成你能读懂、能改的一个文件。**

🧠 一份代码读懂一个架构 · 🔌 运行时不依赖 `transformers` · ✅ 官方权重照样能加载 · 🔬 对得上官方实现

<p>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.9%2B-blue.svg">
  <img alt="PyTorch" src="https://img.shields.io/badge/pytorch-2.0%2B-orange.svg">
  <img alt="Agent Skill" src="https://img.shields.io/badge/agent-skill-8a2be2.svg">
</p>

---

## 💡 为什么做这个

### 😤 1. `transformers` 用起来很方便，读起来、改起来很难

`transformers` 要同时支持 100+ 架构、3 种 attention backend、4 种 cache 策略、量化、编译、
张量并行。所以它的代码必然是分层的、带大量分支的。

这对**使用者**无害 —— `from_pretrained` 一行就能跑，这恰恰是它最成功的地方。
但如果你想**读懂**某一层到底做了什么，或者想把 attention 换个写法、把位置编码换成别的，
就会立刻撞上一堵墙：mask 是 `masking_utils.py` 生成的，KV 从 `cache_utils.py` 里来，
`rope_theta` 在 `modeling_rope_utils.py` 里变成角度，`layer_types` 在 `configuration_utils.py`
的 `__post_init__` 里算出来，而真正的 attention 计算在 `integrations/sdpa_attention.py`
—— **根本不在你正在读的那个文件里**。

于是你熟悉的三个场景会反复出现：

- 🌀 一个 `forward` 里 6 个 `if`，你永远不确定自己的配置走的是哪条分支
- 💥 改一个维度，shape 错误栈 40 层，一多半在框架内部
- 🕵️ debug 时无法区分"模型写错了"还是"框架分支没走到"

这不是 `transformers` 写得差，这是"通用性"必须付的代价。
只是这个代价，不该由**想学一个模型**的人来付。

### 🚀 2. 最前沿的开源模型，总是先出现在 HuggingFace 上

新架构（多模态、混合线性注意力、稀疏 MoE、每层不同的配置……）发布时，官方权重和参考实现
几乎总是最先落在 HuggingFace 上。问题在于"能用"和"能读"是两件事：一个新模型刚进
`transformers` 时，往往同时带着每层不同的配置、新的缓存规则、新的位置编码、自定义算子
—— 正好是最难读的那部分。

所以这个 skill 把"从零读一个新架构"变成了一套固定的操作：

1. 📄 只下载几 KB 的配置文件（`config.json`），看清模型由哪些层组成、每层用什么参数
2. 📋 花几秒钟读一下权重文件的开头，看里面记着哪些参数、各自是什么形状
   （只有几 MB 的索引，不下载权重）
3. 🏗️ 照着这份清单，用普通 PyTorch 把模型写进一个 `.py` 文件。
   参数名和官方保持一致，官方权重就能直接加载进去
4. 🔬 造一个缩小版的模型（结构一点不改，只把维度改小），把自己的输出和官方代码的输出
   逐层比一遍 —— 数字对得上，就说明写对了

按这套流程，任何新模型都能在很短时间里变成一份 ~1500 行、能从头读到尾的代码，
而不是在几千行框架代码里迷路。🧭

想改架构、加一种 attention、换一种位置编码？直接在单文件上动手，改完还能加载官方权重继续
预训练 / 微调 / 上 LoRA —— 这才是"读懂一个模型"之后该有的自由。

## 🧱 三条硬约束，缺一不可

**① 一个模型 = 一个文件。**
配置、所有网络层、forward、loss、权重加载，全都放在一个 `modeling_<arch>.py` 里。
拆成 20 个模块之后，"和官方实现逐行比对"会变成在 20 个文件之间跳转，成本远大于单文件的阅读成本。

**② 参数名和官方完全一致：读得懂，也训得动。**
单文件不等于"重写一套命名"。每一层的组织方式、每个参数的名字和形状，都和官方代码一致，
所以你不需要维护任何名字映射表 —— `model.load_state_dict(hf_safetensors)` 直接就能用，
官方 tokenizer、官方配置也能原样复用，抽出来的模型可以立刻接上下游训练与微调
（含 LoRA / 继续预训练）。想改架构？改完 shape，加载器会明确告诉你哪几个参数对不上，
而不是静默出错。而名字映射恰好是整个移植过程中最容易错、最难调的一环，直接省掉它。

**③ 用一个缩小版的模型自测：不下 10 GB，也能知道实现对不对。**
不需要 GPU，也不需要下载完整权重。把每个维度缩到 32、结构和开关全部保留
（每层不同的 head_dim、两种位置编码、KV 共享、softcap、池化核……），用官方代码生成一个
随机初始化的模型，存成真正的 `config.json + model.safetensors`，再用自己的实现加载，
逐层对比中间结果和最终输出。改一行代码，几十秒内就知道对不对。

而且这份验证是可迁移的：在小模型上逐位对上之后，同一套实现和加载器换到真正的预训练权重
就能直接用，`report["missing"] == [] and report["unexpected"] == []` 就是通关凭据。
✅ 先在小模型上把"实现对不对"钉死，再用大权重做下游任务，两件事不会互相干扰。

## 📦 安装

### 方式一：装成 agent skill（推荐）

这个仓库本身就是一个 skill —— `SKILL.md` 是入口，`references/` 是细则，`scripts/` 可直接调用。
克隆到 skills 目录即可：

```bash
# WorkBuddy
git clone https://github.com/z777bw/hf-model-standalone ~/.workbuddy/skills/hf-model-standalone

# Claude Code
git clone https://github.com/z777bw/hf-model-standalone ~/.claude/skills/hf-model-standalone
```

装好后不用手动记流程：说一句"把 Qwen3 从 transformers 里抽出来"之类的话，
agent 会自动加载 `SKILL.md` 并按它的工作流走。

### 方式二：只当工具箱用

```bash
git clone https://github.com/z777bw/hf-model-standalone
cd hf-model-standalone

# 运行时依赖：不含 transformers
pip install torch numpy pillow tokenizers safetensors

# 只有"造缩小版模型 / 和官方对比"时需要，它只当标准答案
pip install transformers
```

## 🎯 使用

### 对着 agent 说一句话

```
把 google/gemma-4-E2B 从 transformers 里抽出来，单文件实现，不要依赖 transformers，
用缩小版的随机权重模型和官方实现逐层比对，最后给我一份能加载官方权重的报告。
```

agent 会按固定顺序推进，你也可以按这个顺序自己手动做：

1. 📥 只拉 `config.json` 和预处理器（几 MB～几十 MB），**绝不下载权重**
2. 🔍 花几秒钟读一下权重文件的索引（只读文件开头，不下载权重），拿到每个参数的名字 / 形状 / 精度
3. 🏗️ 照官方的参数名写 `modeling_<arch>.py`（单文件）+ `preprocessing.py`
4. 🧪 造一个缩小版模型，逐层对比中间结果和最终输出
5. ⚡ 接 KV cache 与生成循环（`kv_cache.py` / `generation.py`）
6. 🚀 在服务器上用真实权重验证 `missing == [] / unexpected == []`

### 直接跑脚本

```bash
# ① 只拉配置和预处理器，不碰权重
python scripts/fetch_minimal.py google/gemma-4-E2B --out ./ckpt

# ② 只看权重文件的索引（几秒）：每个参数的名字 / 形状 / 精度，不下载 10 GB
python scripts/peek_checkpoint.py google/gemma-4-E2B

# ③ 造一个 1 MB 的缩小版模型（结构开关全保留）
python scripts/make_tiny_reference.py ./ckpt --out ./tiny_ckpt
```

> 🌐 内地网络：`502 Bad Gateway` 基本都是本地代理拒绝隧道，不是网站挂了。
> 加 `--endpoint https://hf-mirror.com --no-proxy`，或 `export HF_ENDPOINT=https://hf-mirror.com`。
> 三个脚本都接受**本地路径**，断网也能核对参数名、造缩小版模型。

### 交付产物目录

跑完之后，你的实现应该是这样一个目录（`<out>` 由你指定）：

```text
<out>/<arch>_standalone/
├── modeling_<arch>.py   # 单文件：配置 + 全部网络层 + forward + loss + 权重加载
├── preprocessing.py     # tokenizer / image processor / audio feature extractor / processor 拼装
├── kv_cache.py          # 最小 K/V 存储：每层一对 (k, v)，沿 seq 维 append（第一个改造点）
├── generation.py        # prefill → decode_step → generate（greedy / temperature / top-k / top-p）
├── test_parity.py       # 缩小版模型逐层对齐 + cache 开关一致性 + 增量解码检查
├── run_<arch>.py        # 真实权重的前向 / loss / 生成 demo
└── README.md            # 覆盖范围 + 未覆盖范围 + 数值 diff 记录
```

这个仓库自己的目录：

```text
hf-model-standalone/
├── SKILL.md                  # agent 入口：工作流、抽取路线、权重加载、KV cache、坑表
├── references/
│   ├── pitfalls.md           # A~G 七组坑位，★ 标注实测踩过的
│   ├── parity.md             # 数值对齐手册（含算子级定位法）
│   └── checklist.md          # 逐项打勾清单（配置 → 参数名 → forward → 预处理 → 权重 → cache）
├── scripts/
│   ├── hf_net.py             # 网络层：重试 / 绕代理 / 镜像 / 可读报错
│   ├── fetch_minimal.py      # 只拉配置和小文件
│   ├── peek_checkpoint.py    # 只读权重文件的索引，不下载权重
│   └── make_tiny_reference.py# 缩维造缩小版模型
└── examples/
    └── gemma4_standalone/    # 完整案例：单文件实现 + 预处理 + KV cache + 生成 + 对齐测试 + demo
```

> `scripts/` 下四个文件互相依赖（共用 `hf_net.py`），单独拷走时要一起拷。

## 📄 License

MIT。案例代码由 `transformers`（Apache-2.0）逐行移植而来，`modeling_gemma4.py` 头部注明了来源版本。
