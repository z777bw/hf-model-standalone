# 数值对齐手册

## 原则

1. **先预处理，后模型。** `pixel_values` / `input_features` / `input_ids` 不逐位相等，
   后面的比对没有意义（误差来源不可分离）。
2. **先中间量，后 logits。** logits 是最后一步，用它定位不了问题。用 forward hook 抓每层输出，
   找到"输入相同、输出不同"的那一层。
3. **fp32 起步，而且两边都要 fp32。** fp32 下文本塔应该做到 **0.0（逐位相同）**。做不到就说明
   语义有偏差，不要用 bf16 的宽容差掩盖。**注意 `from_pretrained()` 默认按 `config.json` 的
   顶层 `"dtype"` 加载**（gemma-4-E2B 是 bfloat16）——"我 fp32 vs 参考 bf16"会得到 1e-2 量级、
   且第一层就有的差异，非常像 bug。做逐位比对时显式给参考传 `dtype=torch.float32`，或者让
   自己的实现传 `dtype="auto"` 跟随 checkpoint。bf16 只在最后做一次 smoke test。
4. **容差留档。** 交付时把"哪一项、差多少、为什么"写成表（见 README 的 diff 表），
   比一句"已验证一致"有用得多。

## tiny 参考模型的构造

用 `scripts/make_tiny_reference.py`。要求：

- **架构开关全保留**：`layer_types`、`num_kv_shared_layers`、`use_double_wide_mlp`、
  `hidden_size_per_layer_input`、`final_logit_softcapping`、`rope_parameters` 的两种类型、
  `pooling_kernel_size`、音频 `attention_chunk_size/context_left`
- **只缩连续维度**：hidden / heads / layers / vocab / intermediate
- **缩完必须能跑**：脚本会实例化并跑一遍，报 shape 错就说明有硬编码的比例关系
  （例：Gemma 4 音频要求 `subsampling_conv_channels[0] == mel_bins`，因为 SSCP 展平的是
  `(channels[1], mel_bins//4)`）
- **左侧层数要够**：要测到 KV 共享，层数必须 > `num_kv_shared_layers`，且共享段里每种
  `layer_type` 都要有一个非共享的"上游层"，否则 `shared_kv_states` 取不到 key 直接 KeyError

## 比对序列

| 顺序 | 比对对象 | 期望 |
| --- | --- | --- |
| 1 | `embed_tokens(ids)` | 完全相等（缩放系数别写错） |
| 2 | 位置编码 `cos/sin`、PLE 投影 | 完全相等 |
| 3 | 单层 `attn` 输出 | 完全相等 |
| 4 | 单层 `mlp` 输出 | 完全相等 |
| 5 | 单层整体输出 | 完全相等 |
| 6 | 全部层 → final norm | 完全相等（fp32） |
| 7 | logits / loss | 完全相等（fp32）/ ≤1e-6（bf16、sdpa、多模态） |

hook 写法（两边同样注册）：

```python
buf = {}
def hook(name):
    def fn(module, inp, out):
        if hasattr(out, "last_hidden_state"):
            out = out.last_hidden_state
        elif isinstance(out, tuple):
            out = out[0]
        buf[name] = out.detach()
    return fn
```

## 算子级定位（逐层也不够细的时候）

hook 到"整层 / 整个子模块"还看不出问题时，直接拦截算子、给每个输出打个 checksum：

```python
_orig = torch.matmul
log = []
def patched(a, b, *args, **kw):
    out = _orig(a, b, *args, **kw)
    log.append((tuple(a.shape), tuple(b.shape), str(a.dtype), float(out.float().sum())))
    return out
torch.matmul = patched      # 同样可以包 F.softmax
```

跑一遍参考、跑一遍自己，逐条比对 → **第一个不一致的算子**就是出事的地方。
比 hook 更快定位：能直接区分"输入就不同"还是"算子本身不同"。
（实测靠它发现第一个 `q@kᵀ` 就不同，从而锁定到 RoPE 而不是 attention。）

## 采样输入要覆盖的分支

- 纯文本、**无 padding**（最干净的基线）
- 纯文本、**左 padding**（测 mask；padded 行是未定义输出，只比有效位）
- 短序列（< sliding_window）和**长于 sliding_window** 的序列（否则 sliding mask 根本没生效）
- 至少两种分辨率/长宽比（测 aspect-ratio resize 与 padding patch）
- 至少一个含 `<boi>…<eoi>` / `<boa>…<eoa>` 的多模态序列（测 scatter 与数量校验）
- 音频要覆盖"长度不是 chunk 整数倍"（测 blocked mask 的 pad 逻辑）

## 常见"差一点点"

| 现象 | 大概率原因 |
| --- | --- |
| 1e-7 量级 | 只是 sdpa/eager 的归约顺序差异，可以接受 |
| 1e-3~1e-2，且**第一层就有**、逐层继续放大 | 先查 dtype 是否统一（参考默认吃 `config.dtype`）；确认统一后再怀疑 rope/mask |
| fp32 下 0.0、bf16 下第一层就差 1e-2，`cos` 逐位对但 `sin` 差 1 ulp | `inv_freq` 被注册成 buffer，`.to(bf16)` 把它舍入了（`bf16(0.1)=0.10009765625`）；改成普通属性存 fp32，forward 里再 upcast |
| 1e-3~1e-2，前几层为 0、某一层才起跳 | rope（theta/partial/proportional 的 inv_freq 拼法）、mask 边界差 1 |
| 只在长序列错 | sliding window 语义 / chunked attention 的 context 宽度 |
| 只在有 padding 时错 | mask 是否作用到 key 侧、是否用有限最小值 |
| 只在多模态错 | PLE 投影次数、scatter 顺序、soft token 数 |
| 前几层对、后面漂 | 某一层的 `layer_scalar` / 双倍宽 MLP / 共享 KV 取错层 |
| 怎么调都差 1e-1 | 参考实现自身分支问题（如 eager 音频 mask），换 `attn_implementation` 试 |

## 不要在测试里做的事

- 不要用 `strict=True` 加载（把"缺哪些键"这条最重要的信息丢掉）
- 不要 `torch.allclose` 一把梭（看不出是哪一层）
- 不要在带 padding 的 batch 上比 padded 行的 logits（本来就未定义）
- 不要把第一个真实 token 的 label 留着（它的预测来自全屏蔽行）
