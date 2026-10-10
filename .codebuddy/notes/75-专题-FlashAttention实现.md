# 专题：Flash Attention 的实现方式

> `extensions/transformer_engine.py`、`transformer/dot_product_attention.py`、`transformer/attention.py`、`fusions/fused_softmax.py`

## 0. 一句话结论

**Megatron 不实现 Flash kernel 本身**，它是编排层：把正确格式、掩码语义、并行组喂给外部实现（TransformerEngine / flash-attn 库）。本地实现 `DotProductAttention` 是朴素 O(s²) attention。

## 1. 三条路径

```
                     ┌─ 路径1【训练主力】TEDotProductAttention
                     │   extensions/transformer_engine.py:2041
core_attention ──────┤   继承 te.pytorch.DotProductAttention，kernel 在 TE 内部
（spec 子模块）      │
                     ├─ 路径2【本地回退】DotProductAttention
                     │   transformer/dot_product_attention.py:26
                     │   裸 bmm + softmax，无 Flash
                     │
                     └─ 路径3【动态推理】直接调 flash-attn 库
                         transformer/attention.py:758/:1023
```

## 2. 路径 2：本地 `DotProductAttention`（`dot_product_attention.py:26`）

```
:169-175  GQA：K/V repeat_interleave 到与 Q 同 head 数（物化重复）
:194      baddbmm → 物化 [b, np, sq, sk] 完整 attention 矩阵
:210      scale_mask_softmax
:239      bmm(attention_probs, V)
```

硬限制：`assert context_parallel_size == 1`（`:57`）、`assert packed_seq_params is None`（`:153`）。

`--masked-softmax-fusion` 开的是 `FusedScaleMaskSoftmax`（`fused_softmax.py:179`）——融合 scale+mask+softmax 三步，**但矩阵仍物化，不是 Flash**。

它是选择性重计算的标定物（docstring `:28-32`，`attention.py:446`）：attention 矩阵是显存大头，重算最划算。

## 3. 路径 1：`TEDotProductAttention`（`transformer_engine.py:2041`）

kernel 调用只有一行 `super().forward(query, key, value, attention_mask, **_fa_kwargs)`（`:2322/:2351`），Flash/cuDNN 由 TE 内部选择。包装层工作：

| 行号 | 职责 |
|---|---|
| `:2091-2241` | 版本协商：GQA≥0.11、CP≥1.0、SWA≥1.2、softmax_type≥2.8、qk-clip≥2.9 |
| `:2307-2315` | 掩码映射：THD 时 `causal → padding_causal`、`no_mask → padding`（cuDNN 要求） |
| `:2191-2217` | `PackedSeqParams` 按 TE 版本过滤（`kept_packed_seq_params`） |
| `:2129-2153` | CP（Ring Attention）注入：`cp_group / cp_global_ranks / cp_stream / cp_comm_type` |
| `:2234-2238` | SP 旗标 + `CudaRNGStatesTracker` 递给 TE（dropout 语义一致） |
| `:2324-2345` | qk-clip：`return_max_logit=True` 拿 max logit，逐 batch 累积 running max |

`cp_stream` 是类变量（`:2050/:2134`，所有层共享）；forward 支持运行时换 CP 组（`:2258-2272`，hybrid CP 场景）。
spec 接线：`gpt_layer_specs.py:179` 填进 `SelfAttentionSubmodules.core_attention`。

## 4. 路径 3：动态推理直调 flash-attn（`attention.py`）

import 区（`:65-119`）即生态清单：`flash_attn_3`（`:65`）、`flashattn_hopper`（`:74`）、`flash_attn.cute` FA4（`:94`，`HAVE_FA4`）、`flash_mla`（`:104`）、`flash_attn` FA2（`:116`）。

| 行号 | 函数 | 说明 |
|---|---|---|
| `:1000` | `_resolve_flash_version` | pin 优先，否则 FA4 > FA3 > FA2 |
| **`:758`** | `flash_decode` | `flash_attn_with_kvcache`：一次 kernel 完成 RoPE + KV cache 更新 + attention |
| **`:1023`** | `flash_decode_and_prefill` | varlen（`cu_seqlens_q/k`）+ `block_table`（paged KV cache），混合 prefill/decode batch；FA4/FA3 缺失回退 FA2（`:1130`） |

### sink softmax 事后修正（`:853-900`）

flash kernel 只算 vanilla softmax，sink 变体用 LSE 事后重标定：

```
vanilla: out = Σ exp(qk_j)/Σ exp(qk_k) · v_j，lse = log Σ exp(qk_k)
sink:    分母 = exp(sink_h) + Σ exp(qk_k)
         ≡ out · sigmoid(lse - sink_h)
```

要求 kernel 返回 LSE（`return_softmax_lse=True`，`:806`）；FA3 走私有接口 `_flash_attn_forward`（`:902`）就为拿 `softmax_lse`。

### 确定性

`flash_decode_and_prefill` 的 `num_splits = 0 if not batch_invariant_mode else 1`（`:1103`）——多 split 引入非确定性归约顺序。

## 5. 与并行体系的交互

| 维度 | 谁负责 | 关键代码 |
|---|---|---|
| TP | heads 切分后各 rank 独立调 kernel，**kernel 内零通信** | linear_qkv（Column）→ core_attn → linear_proj（Row） |
| SP | 旗标传 TE，kernel 外围 g/rs | `transformer_engine.py:2234` |
| CP | Ring Attention 在 TE 内部 | `:2129-2153`；本地实现不支持（`dot_product_attention.py:57`） |
| GQA | TE/FA 原生支持；仅本地实现 repeat_interleave | `dot_product_attention.py:170` |
| 选择性重计算 | core_attention 是 recompute 区域 | `attention.py:446` |

## 6. 排查速查

```
transformer_impl == transformer_engine？
  → 是：TEDotProductAttention（TE 内部选 flash/cuDNN）
  → 否：local spec → DotProductAttention（朴素）
约束：thd → TE + padding_causal；CP>1 → TE；SWA → TE≥1.2 或 FA
推理：static engine 走 core_attention；dynamic engine 走 flash_decode(_and_prefill)
```
