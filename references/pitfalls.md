# 移植踩坑清单（按"最先查"排序）

每条都注明：症状 / 根因 / 处理。带 ★ 的是实测中真实踩到的（调出来的，不是从文档里抄的）。

## A. config 层

**★ A1 `config.to_dict()` 会丢字段**
症状：tiny 配置上 full-attention 层的 `head_dim` 变成默认值，logits 差 1e-1。
根因：`Gemma4TextConfig.__post_init__` 把 `global_head_dim` 从 kwargs 里 pop 进
`per_layer_config`，于是 `to_dict()` 里没有它（新版 transformers 的普遍现象：派生字段不序列化）。
处理：导出 tiny 配置后手动 `--set text_config.global_head_dim=16`；或直接以官方
`config.json` 为准（官方文件是带这个字段的）。

**★ A2 逐层异构：`config.head_dim` 会直接抛异常**
症状：`AmbiguousGlobalPerLayerAttributeError: 'head_dim' is a per-layer attribute`。
根因：transformers 5.x 的 `HeterogeneousConfigMixin` 禁止读"可能是逐层不同"的属性。
处理：用 `config.per_layer_config[i].head_dim`（也可用字符串 key：`per_layer_config["full_attention"]`）；
自己的实现里写 `layer_head_dim(i)` 这种访问器，别缓存成单一标量。

**A3 没写进 config.json 的派生态**
`layer_types`（默认 5:1 sliding:full，且**最后一层强制 full_attention**）、`rope_parameters`
（default/proportional 两套）、`sliding_window` 的 `use_bidirectional_attention == "all"` 特例
（会被折半）—— 都要在 `__post_init__` 里复刻。漏了不报错，只是数不对。

**A4 依赖 `num_*` 的派生尺寸**
`intermediate_size * 2`（双倍宽 MLP）、`num_hidden_layers * hidden_size_per_layer_input`
（PLE 打包表）、`vocab_size // 4 * channels`（音频 SSCP 展平维度）—— 全是"算错就 shape 报错或
静默错"的地方。shape 报错反而是好事，静默错才是灾难。

## B. 权重 / buffer

**★ B1 KV 共享层在官方权重里仍然存着 k/v**
症状：`load_state_dict` 报 6 个（或更多）unexpected keys。
根因：模型运行时 `k_proj/v_proj/k_norm` 根本不存在，但 checkpoint 里保留了它们。
处理：加载前按层类型剔除（等价于 `_keys_to_ignore_on_load_unexpected`）。

**B2 tied embedding**
`tie_word_embeddings=True` 的仓库一般不存 `lm_head.weight`，要从 `embed_tokens.weight` 补；
反之导出 tiny 权重时也要**主动删掉** `lm_head.weight`，否则 safetensors 会因为共享内存报错。

**★ B3 buffer 也是可加载参数**
`layer_scalar`、`input_min/input_max/output_min/output_max`、`embed_scale` 都在 ckpt 里。
凡是 `nn.Buffer(...)` 默认 `persistent=True` 的，必须同名注册，否则 missing。
反之 `persistent=False` 的（`inv_freq`、`softcap`、`inv_timescales`）**不在** ckpt 里，别去加载。

**★ B5 `inv_freq` 绝对不能注册成 buffer（bf16 下才暴露的坑）**
`nn.Module._apply()`（也就是 `model.to(dtype)` / `.to(device)`）会强转**所有**浮点 buffer。
一旦 `inv_freq` 被转成 bf16，精度就永久丢了：`bf16(0.1) == 0.10009765625`，
之后再 `.float()` 也回不来，于是每个 RoPE 角度偏 ~1e-3 rad。

症状极具迷惑性：
- fp32 下 **0.0（逐位相同）**，bf16 下第一层就差 1.26e-2，逐层放大
- 用 hook 会比出 `cos` 逐位相同、`sin` 差 1 个 bf16 ulp（因为小角度下 cos≈1，舍入不敏感）
- 拦截 `torch.matmul`，**第一个** q@kᵀ 就已经不同

参考实现的做法是把 `inv_freq` 留在 fp32、在 `forward` 里 `inv_freq.to(dtype=torch.float)`。
移植时对应做法：**用普通属性存 fp32 张量，不注册 buffer**，forward 里 `.to(device=x.device, dtype=torch.float)`。
它本来也不该进 `state_dict`（由 config 推导），所以不注册 buffer 不损失任何东西。

**B4 初始化值不是 0**
`position_embedding_table` 初值是 **ones**；`per_dim_scale` 是 **zeros**；`layer_scalar` 是 ones；
`std_bias=0 / std_scale=1`。移植时随手 `torch.empty` 就会得到一个"看起来正常、结果全错"的模型。

## C. 数值 / dtype

**★ C0 先统一 dtype，再谈 diff（最容易浪费一天的一条）**
`config.json` 顶层的 `"dtype": "bfloat16"` 会被 `from_pretrained()` 当作**默认加载精度**。
于是"我的实现 fp32 vs 参考 bf16"会看到 ~1e-2 的 logits 差异，而且是**第一层就已经有**
（逐层继续放大），看起来特别像"某个 op 写错了"。判断方法：看 `next(ref.parameters()).dtype`。

```python
# 做法 A：参考也压成 fp32（做逐位比对时用这个）
ref = AutoModel.from_pretrained(p, dtype=torch.float32)
# 做法 B：让自己跟随 checkpoint 精度（复现 from_pretrained 默认行为）
ours = load(p, dtype="auto")   # 读 config.dtype
```
真实 gemma-4-E2B 的 tiny 复现产物就是 bf16 存的；换成 fp32 后 logits diff 直接从 1.26e-2 变 0.0。

**★ C1 RMSNorm 的内部精度**
参考实现是 `x.float() → pow(mean+eps,-0.5) → *weight.float() → .type_as(x)`。
用 `F.rms_norm` 或 bf16 直算都会漂。同理注意用 `torch.pow(ms, -0.5)` 而不是 `rsqrt`
（上游注释明确说是为了和 JAX 对齐）。

**C2 注意力 scaling**
Gemma 4 文本/视觉注意力的 `scaling = 1.0`（因为前面有 QK-norm），不要套 `1/sqrt(d)`。

**C3 logit soft-capping**
`tanh(logits / c) * c`，c 可能来自 `final_logit_softcapping`（E2B 是 30）。放在 lm_head 之后、
loss 之前。

## D. mask

**D1 加性 mask 用有限最小值**
`torch.finfo(dtype).min` 而不是 `-inf`：左 padding 会产生"整行全屏蔽"，用 `-inf` 会 NaN。
（实测：padded 行本来就是未定义输出，sdpa/eager 给的值不同，比较时要跳过这些位置。）

**D2 sliding window 的确切语义**
`kv_idx > q_idx - sliding_window`（包含自身，窗口宽 = w）。差一个 1，短序列看不出来，
长序列才炸。

**★ D3 参考实现自己的 bug：音频 mask 类型**
`Gemma4AudioAttention.forward` 里是
`attn_weights.masked_fill(attention_mask.logical_not(), -1e9)`，
即它**期望 bool mask**（True=可用）。但 eager 路径的 `create_bidirectional_mask` 返回的是
0/-inf 的 **float** 加性 mask，于是 `logical_not()` 把"允许"的位置判成 True，
所有窗口内 logit 被覆盖成 -1e9 → 注意力退化成窗口内均匀平均。
sdpa 路径返回 bool mask，才是设计意图。
→ 结论：音频对齐要用 `attn_implementation="sdpa"`；自己的实现用 bool mask，别去复刻这个 bug。

## E. 多模态融合

**★ E1 PLE 只投影一次**
`Gemma4Model.forward` 拿到的是"token-identity"原始 per-layer embedding，
context 投影在 `Gemma4TextModel.forward` 里做（且用的是 **scatter 之后** 的 inputs_embeds）。
外层再投影一次 → 每层 hidden 差 1e-2，logits 差 5e-2，而且逐层放大。

**E2 占位符 token 在 embedding 里要被替换成 pad**
`llm_input_ids = where(multimodal_mask, pad_token_id, input_ids)`，再 embed；
soft token 随后 `masked_scatter` 进去。顺序反了（先 embed 再 where）会丢信息。

**E3 图像 soft token 数的动态性**
每张图 resize 后 patch 数不同 → soft token 数不同 → prompt 里的占位符个数必须**按图逐个**展开
（`<boi>` + `<image>` * n + `<eoi>`）。批量里每张图 n 不同时，padding 后 `masked_scatter` 仍然成立，
但 placeholders 总数必须等于 features 总数。

## F. 预处理

**F1 视觉：不做 normalize**
Gemma 4 直接 `rescale 1/255`，`do_normalize=False`，然后在 patch embedder 里做
`2*(x-0.5)`。套 ImageNet mean/std 就全错。

**F2 视觉：patchify 顺序**
`(C,H,W) → reshape(C, ph, P, pw, P) → transpose(1,3,2,4,0) → (ph*pw, P*P*C)`。
patch 内的通道是**最后一维**，和 patch embedding 是 `Linear(3*P*P, hidden)` 对应。

**F3 视觉：resize 的倍数约束**
aspect-ratio preserving，且宽高必须是 `pooling_kernel_size * patch_size` 的整数倍，
否则池化块不完整。边缘情况（某一维 round 到 0）参考实现有专门分支。

**F4 音频：semicausal padding + frame_length+1**
左边补 `frame_length//2` 个 0，unfold 窗口是 `frame_length+1` 再丢掉最后一列（或做 preemphasis）。
用 librosa/torchaudio 默认 STFT 一定对不上。mel 是 htk scale、`norm=None`、`mel_floor=1e-3`。

**F5 tokenizer：`tokenizer.json` 之外的东西**
special token 表（`<|image|>`/`<|image>`/`<image|>` 三个是**不同** token）、`add_bos_token`、
`padding_side`（gemma 是 left）、`decode(skip_special_tokens=True)`。
`chat_template` 可能既不在 `tokenizer_config.json` 也不在 `tokenizer.json` 里 ——
gemma-4-E2B 就是没有，需要自己按家族惯例实现并**在交付说明里写清楚**。

## G. 环境 / 网络（脚本跑不起来时先看这里）

**★ G1 `Tunnel connection failed: 502 Bad Gateway`**
Shell 里设了 `HTTPS_PROXY/HTTP_PROXY`，而代理拒绝了 CONNECT 隧道。
报错看起来像"网站挂了"，实际是本地代理的问题。两个脚本都支持 `--no-proxy`：

```bash
python scripts/peek_checkpoint.py google/gemma-4-E2B --endpoint https://hf-mirror.com --no-proxy
```

**G2 huggingface.co 不可达（内地网络）**
用 `HF_ENDPOINT=https://hf-mirror.com`，或给脚本传 `--endpoint https://hf-mirror.com`。
两者组合（镜像 + 绕过代理）是最常见的可用组合。

**G3 `huggingface_hub` 自己下载也挂**
`transformers` / `hf_hub_download` 不走脚本的参数，要设环境变量：
`HF_ENDPOINT=https://hf-mirror.com`，必要时 `NO_PROXY=hf-mirror.com`。

**G4 完全离线**
`peek_checkpoint.py` / `make_tiny_reference.py` 都接受**本地路径**（含 `config.json` 与
`*.safetensors` 的目录），断网也能做模块树核对和 tiny 参考模型。
只有 `fetch_minimal.py` 必须联网。
