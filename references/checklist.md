# 抽取清单（逐项打勾）

给定 `org/model`，按顺序填。左列是"看哪里"，右列是"产出什么"。

## 0. 前置

- [ ] 网络自检：能连 `huggingface.co` 吗？不能就 `--endpoint https://hf-mirror.com`；
      报 `502 Bad Gateway` 是本地代理问题，加 `--no-proxy`（`scripts/hf_net.py` 会直接给出诊断）
- [ ] `config.json` 里的 `architectures[0]` → 要移植的类名
- [ ] `model_type` → 源码目录 `transformers/models/<model_type>/`
- [ ] 是否 `trust_remote_code`（有 `modeling_*.py` 在仓库里 → 以仓库代码为准，不是 site-packages）
- [ ] 参数量/字节数（`peek_checkpoint.py`）→ 决定 dtype 与是否分片

## 1. 配置（`config.json` → 你的 dataclass）

- [ ] 顶层 `model_type` / `architectures` / `tie_word_embeddings` / `dtype`
- [ ] 子配置（`text_config` / `vision_config` / `audio_config` / `llm_config`…）各自的 `model_type`
- [ ] 特殊 token id：`image_token_id` / `audio_token_id` / `video_token_id` / `boi/eoi/boa/eoa`
- [ ] 派生态：`layer_types`、per-layer head_dim / kv_heads、rope 参数（按 layer_type 分组）
- [ ] 数值开关：`rms_norm_eps`、`final_logit_softcapping`、`sliding_window`、`partial_rotary_factor`
- [ ] 结构开关：`num_kv_shared_layers`、`use_double_wide_mlp`、`hidden_size_per_layer_input`、
      `enable_moe_block`、`attention_k_eq_v`、`standardize`、`use_clipped_linears`

## 2. 模块树（`peek_checkpoint.py --list`）

- [ ] 顶层：`model.` / `lm_head` / `model.vision_tower.` / `model.embed_vision.` / `model.audio_tower.` / `model.embed_audio.`
- [ ] 文本：`embed_tokens`、`embed_tokens_per_layer`、`per_layer_model_projection`、
      `per_layer_projection_norm`、`layers.N.{self_attn,mlp,*layernorm*}`、`norm`、`layer_scalar`
- [ ] 注意力里的 norm 有没有 weight（`v_norm` 常常没有 → `with_scale=False`）
- [ ] `Gemma4ClippableLinear` 这类"套一层"的模块：ckpt 里是 `xxx.linear.weight`
- [ ] 视觉：`patch_embedder.input_proj` / `position_embedding_table` / `encoder.layers.N.*` / `std_bias|std_scale`
- [ ] 音频：`subsample_conv_projection.*` / `layers.N.{feed_forward1,feed_forward2,self_attn,lconv1d,norm_*}` / `output_proj`
- [ ] 共享层缺哪些 key（`k_proj/v_proj/k_norm`）→ 运行时它们不需要
- [ ] 有哪些 `nn.Buffer` 是 persistent 的（`layer_scalar`、`input_min/max`、`output_min/max`）

## 3. forward 链（`modeling_*.py`）

- [ ] 每个 `__init__` 的 submodule 名与参数名逐字照抄
- [ ] `forward` 的算子顺序 + 残差位置 + norm 的 dtype 处理
- [ ] 注意力：scaling、QK-norm 应用位置、rope 的 `unsqueeze_dim`、`repeat_kv`
- [ ] mask：causal / sliding / bidirectional / chunked 各自的语义与取值
- [ ] 池化/下采样：核大小、取整方式、padding 位置
- [ ] 融合：占位符 → soft token 的替换顺序与数量校验
- [ ] 头：`lm_head`、softcap、loss（`ignore_index`、是否 shift、是否除 `num_items_in_batch`）

## 4. 预处理

- [ ] `tokenizer.json`（fast）或 `tokenizer.model` + `special_tokens_map.json` + `tokenizer_config.json`
- [ ] special token 全表 + `add_bos_token` + `padding_side` + `chat_template`（可能不存在！）
- [ ] 图像：resize 策略（固定尺寸 / aspect-preserving / patch 预算）、rescale、normalize、
      **输出形态**（`(B,C,H,W)` 还是已经 patchify 的 `(B,N,P*P*C)`）
- [ ] 音频：mel 参数（frame/hop/fft/mel_floor/mel_scale/norm）、semicausal padding、mask 构造
- [ ] 占位符展开规则（每图/每段音频的 soft token 数怎么算）

## 5. 权重加载

- [ ] 单文件 / `model.safetensors.index.json` 分片
- [ ] tied 权重补齐
- [ ] 冗余 key 剔除（KV 共享层）
- [ ] dtype 转换（`inf` buffer 不能变 NaN；建议 `model.to(dtype)` 后加载）
- [ ] **`inv_freq` 这类"必须保持 fp32"的量没有注册成 buffer**（否则 `.to(bf16)` 会舍入它）
- [ ] 加载后打印 `missing` / `unexpected` / `dropped` 三个列表

## 6. KV cache 与生成

- [ ] cache 只由**非共享层**写入；共享层从 `shared_kv_states` 读（它们没有 `k_proj/v_proj`）
- [ ] `store_full_length_kv` 的写入发生在 `cache.update()` **之后**（否则共享层拿半截历史）
- [ ] decode 时 q 的绝对位置是 `kv_len - q_len`，mask 里带偏移
- [ ] `position_ids = arange(seq_len) + past_seen`
- [ ] `attention_mask` 覆盖**全长**（`past + new`），不是只传新 token
- [ ] decode 阶段不重跑视觉/音频塔
- [ ] 验证：同一实现 `use_cache=True` vs `False` 的整段前向 → **逐位相同**
- [ ] 验证：增量 decode 拼接 vs 一次全序列 forward → **1e-6 量级**（不是 0，也不是 1e-2）
- [ ] 验证：greedy 生成的 token ids 与参考实现完全一致
- [ ] 序列长度要**超过 sliding window**，否则窗口裁剪那条路径根本没走到

## 7. 验证

- [ ] 两边 dtype 一致（`next(ref.parameters()).dtype` vs 自己的；参考默认吃 `config.dtype`）
- [ ] `make_tiny_reference.py` 产出 tiny ckpt（架构开关全保留）
- [ ] 加载 tiny ckpt：missing/unexpected 为空
- [ ] 预处理逐项对齐（ids / pixel_values / input_features）
- [ ] 逐层 hidden → logits → loss 对齐（见 `parity.md`）
- [ ] bf16 下再比一次（fp32 逐位相同 ≠ bf16 也相同：`model.to(bf16)` 会舍入浮点 buffer，
      比如 rope 用的 `inv_freq`）
- [ ] 交付 README 里写清：覆盖范围、未覆盖范围、逐项 diff 表、已知参考实现问题

## 8. 不要做的

- [ ] 不要重命名模块/参数（除非你愿意维护一层 key 映射）
- [ ] 不要把 model 拆成十几个文件
- [ ] 不要为了"看起来干净"去掉 buffer
- [ ] 不要在没对齐单个 layer 之前就去调 logits 容差
- [ ] 不要在 forward 没对齐之前先接 cache（同时面对两类 bug，没法定位）
