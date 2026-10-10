# 阶段三：模型结构与 Spec 机制

> `transformer/spec_utils.py`（145 行）、`models/gpt/`、`models/common/`、`transformer/`（66 个 .py）

## 1. Spec 机制（`spec_utils.py`）

把"用哪个类 + 什么参数"数据化，运行时实例化，替代 if/else 分支爆炸（TE vs local、dense vs MoE、训练 vs 推理）。

| 行号 | 符号 | 说明 |
|---|---|---|
| **`:10`** | **`ModuleSpec`** | `module`（类或 `("路径","类名")` 延迟导入元组）、`params`、`submodules`、`metainfo` |
| `:31` | `__call__` | 即 `build_module(self, ...)` |
| `:41` | `import_module` | 动态导入元组路径 |
| **`:76`** | **`build_module`** | 分发 + `kwargs["submodules"]` 注入（`:118-119`）+ `spec.params` 合并（`:122-124`） |
| `:134` | `get_submodules` | 兼容 `partial` 与带 `.submodules` 属性的对象 |

层级心智模型：

```
ModuleSpec(module=TransformerLayer, submodules=TransformerLayerSubmodules(
    input_layernorm   = TENorm,
    self_attention    = ModuleSpec(module=SelfAttention,
                          submodules=SelfAttentionSubmodules(
                              linear_qkv     = TELayerNormColumnParallelLinear,
                              core_attention = TEDotProductAttention,
                              linear_proj    = TERowParallelLinear,
                              q_layernorm    = IdentityOp,    ← 关掉子模块
                              k_layernorm    = IdentityOp)),
    self_attn_bda     = get_bias_dropout_add,    ← submodule 可以是函数
    pre_mlp_layernorm = TENorm,
    mlp               = ModuleSpec(module=MLP, submodules=MLPSubmodules(
                              linear_fc1 = ..., linear_fc2 = ...)),
    mlp_bda           = get_bias_dropout_add))
```

`IdentityOp`（`identity_op.py:9`）避免 `if` 判断；TE 版与 local 版 spec 完全同构，只是类不同。

### Spec 工厂（`models/gpt/gpt_layer_specs.py`）

| 行号 | 函数 |
|---|---|
| **`:179`** | `get_gpt_layer_with_transformer_engine_submodules`（生产默认） |
| `:351` | `get_gpt_layer_with_transformer_engine_spec` |
| **`:359`** | `get_gpt_layer_local_submodules`（纯 PyTorch，学习首选） |
| `:467` | `get_gpt_layer_local_spec` |
| `:490` | `get_mlp_module_spec`（dense or MoE） |
| **`:563`** | `get_gpt_decoder_layer_specs` |
| **`:676`** | `get_gpt_decoder_block_spec` |
| `:722` | `get_gpt_mtp_block_spec`（MTP） |

## 2. 模型类层级

```
MegatronModule (transformer/module.py:30)
   ├── GraphableMegatronModule (:168)      CUDA Graph 捕获
   │      └── Float16Module (:479)         fp16 wrapper
   ├── LanguageModule (models/common/language_module/language_module.py:37)
   │      └── GPTModel (models/gpt/gpt_model.py:52)
   └── VisionModule (models/common/vision_module/vision_module.py:9)
```

`LanguageModule` 关键方法：`compute_language_model_loss:161`、`setup_embeddings_and_output_layer:208`、`shared_embedding_or_output_weight:345`、`tie_embeddings_..._state_dict:470`。

嵌入层：`LanguageModelEmbedding`（`embeddings/language_model_embedding.py:15`）、`RotaryEmbedding`（`rotary_pos_embedding.py:36`）、`YarnRotaryEmbedding:21`。

## 3. `GPTModel`（`gpt_model.py:52`）

| 行号 | 方法 |
|---|---|
| `:98` | `__init__`（建 embedding / decoder / output_layer） |
| **`:311`** | `set_input_tensor`（PP 胶水：接收上游 hidden_states） |
| `:327` | `_preprocess`（embedding + RoPE） |
| **`:566`** | **`forward`** |
| `:669` | `_postprocess`（output_layer + loss） |
| `:932` | `sharded_state_dict` |

`forward()`（`:566`）三段式：

```
:610  preproc = self._preprocess(input_ids, position_ids, ...)
:630  hidden_states = self.decoder(hidden_states=..., attention_mask=...,
                                    rotary_pos_emb=..., ...)
:645  return self._postprocess(hidden_states, labels=..., loss_mask=...)
```

PP 中间 stage：`set_input_tensor` 注入上游 tensor，`_postprocess` 只在末 stage 算 loss（对应 `pretrain_gpt.py:129` 早退）。

## 4. `TransformerBlock`（`transformer_block.py:267`）

| 行号 | 方法 |
|---|---|
| `:76` | `get_num_layers_to_build`（本 rank 建几层：PP/VPP/MTP 布局核心） |
| **`:336`** | `_build_layers`（`build_layer(spec)` 循环） |
| **`:506`** | `forward` |
| `:782` | `sharded_state_dict` |

定位：**层的容器 + PP 切分器**。只管局部层序列；全局层号映射由 `get_transformer_layer_offset`（`transformer_layer.py:65`）与 `PipelineParallelLayerLayout`（`pipeline_parallel_layer_layout.py:16`）处理。

```
forward: for layer_number in range(self.num_layers): layer(...)
         post_process 且 add_final_layernorm → final_layernorm（仅末 stage）
```

## 5. `TransformerLayer`（`transformer_layer.py:318`）

| 行号 | 方法 |
|---|---|
| `:251` | `TransformerLayerSubmodules` |
| `:325` | `__init__`（逐个 `build_module(submodules.xxx)`） |
| **`:675`** | **`_forward_attention`** |
| `:759` | `_run_input_layernorm`（支持 offload + `CheckpointWithoutOutput`） |
| `:798` | `_apply_self_attn_bda_step` |
| **`:862`** | **`forward`**（很短，复杂度拆进两个子流水线） |
| **`:993`** | **`_forward_mlp`**（dense MLP / MoE 分发） |
| `:1143` | `_apply_mlp_bda_step` |
| `:1227` | `sharded_state_dict` |
| `:1726` | `HyperConnectionTransformerLayer`（mHC，main 独有） |
| `:2061` | `MoETransformerLayer` |

`forward()`（`:862`）与 `_forward_attention`（`:675`）：

```
forward:
  hidden_states, context = self._forward_attention(*args, **kwargs)
  output, _              = self._forward_mlp(hidden_states, ...)

_forward_attention:
  :721  norm_out, residual, state = self._run_input_layernorm(hidden_states)
  :733  attn_out = self.self_attention(norm_out, ...)
  :747  norm 走 checkpoint → 丢弃输出并注册反传重算
  :754  hidden_states = self._apply_self_attn_bda_step(attn_out, residual)
```

Pre-Norm 对照：

```
x = x + Attn(Norm(x))  ↔  :721 norm → :733 attn → :754 bda(+残差)
x = x + MLP(Norm(x))   ↔  :937 norm → :993 mlp  → :1143 bda(+残差)
```

## 6. `Attention`（`attention.py`）

```
Attention (:289, ABC)
   ├── SelfAttention (:1682)      decoder-only 主力
   └── CrossAttention (:2100)     T5 类
MLASelfAttention (multi_latent_attention.py:471)     DeepSeek MLA
```

| 行号 | 方法 |
|---|---|
| `:296` | `__init__`（build linear_qkv / q&k norm / core_attention / linear_proj） |
| `:446` | `_checkpointed_attention_forward`（选择性重计算入口） |
| `:499` | `_run_core_attention` |
| `:742` | `get_query_key_value_tensors`（抽象） |
| **`:1306`** | **`forward`** |
| `:1851` | `SelfAttention.get_query_key_value_tensors`（QKV 投影 + RoPE） |

`forward` 骨架：

```
1. inference → KV cache 处理（:556）
2. q,k,v = self.get_query_key_value_tensors(hidden_states)   ← :1851
3. sequence_parallel → SP 变换
4. core = self._run_core_attention(...)   ← TEDotProductAttention
   或 dot_product_attention.py:26 DotProductAttention（selective recompute 时包 :446）
5. output, bias = self.linear_proj(context)   ← RowParallelLinear
```

TP 视角：`linear_qkv` Column（输出切分，各 rank 拿自己的 head）→ core_attention 零通信 → `linear_proj` Row（输出 all-reduce）。

## 7. `MLP`（`mlp.py:155`）

```
:172  __init__：linear_fc1（Column）+ linear_fc2（Row）
      （gated_linear_unit 时 fc1 输出翻倍：SwiGLU 的 gate）
:257  forward：fc1 → (gated ? gate * act(up) : act) → fc2
:385  backward_dw（延迟 wgrad，与通信重叠）
:361  sharded_state_dict
```

Protocol 类（`:51-130`）：`LinearFc1Interface` 等，spec 机制向接口契约演进的方向。

## 8. `transformer/` 目录分类

| 类别 | 文件:行号 |
|---|---|
| 核心层 | `module.py:30,168,479`、`transformer_block.py:267`、`transformer_layer.py:318,2061`、`mlp.py:155` |
| 注意力 | `attention.py:289,1682,2100`、`dot_product_attention.py:26`、`multi_latent_attention.py:471,1241` |
| MoE | `moe/` 子目录（见笔记 70） |
| Spec | `spec_utils.py:10,41,76`、`gpt_layer_specs.py` |
| 配置 | `transformer_config.py:57,3468`、`attention_layer_config.py:6`、`mla_layer_config.py:6`、`moe/moe_layer_config.py:6`、`pipeline_parallel_layer_layout.py:16` |
| 归一化 | `torch_norm.py:11,21,29,76`（`WrappedTorchNorm`） |
| 枚举 | `enums.py:8 ModelType, :17 LayerType, :33 AttnType, :40 AttnMaskType, :51 AttnBackend` |
| 工具 | `utils.py:38 get_default_causal_mask, :96 make_sharded_tensors_for_checkpoint, :252 sharded_state_dict_default` |
