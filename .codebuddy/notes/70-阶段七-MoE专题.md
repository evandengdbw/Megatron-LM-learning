# 阶段七：MoE 专题

> `transformer/moe/`（19 文件）。延伸阅读：`moe/README.md`

## 1. 前向五步

```
hidden_states
 ① router(hidden) → topk_probs, indices
 ② token_dispatcher.dispatch(hidden, probs)      # all-to-all 发给专家 rank
 ③ experts(hidden_local)                          # GroupedMLP / SequentialMLP
 ④ token_dispatcher.combine(output)               # all-to-all 送回
 ⑤ + shared_experts(hidden)（可选）
```

MoE 有独立的并行度：EP（专家切分）+ ETP（专家内 TP），对应 expert RankGenerator（笔记 20）。

## 2. `moe_layer.py`

| 行号 | 符号 |
|---|---|
| `:69-153` | Protocol 家族（新式类型化 spec） |
| `:154` | `MoESubmodules`（router / experts / shared_experts / token_dispatcher） |
| `:162` | `BaseMoELayer`（ABC） |
| **`:215`** | **`MoELayer`** |
| `:466` | `route` → `:630 router_and_preprocess` |
| **`:637`** | **`forward`**（五步编排） |
| `:744` | `backward_dw`（延迟 wgrad，dispatch/compute 重叠） |
| `:776` / `:799` | 用 autograd Function 做事件同步的技巧 |

## 3. `router.py`

| 行号 | 符号 |
|---|---|
| `:34` | `Router`（ABC）：`gating:94`、`routing:119`、`forward:132` |
| **`:148`** | **`TopKRouter`** |
| **`:750`** | **`routing`**（topk + sinkhorn / aux-loss 负载均衡） |
| `:890` | `InferenceTopKRouter` |

关键配置：`moe_aux_loss_coeff`（负载均衡辅助损失）、`moe_expert_capacity_factor`（token dropping）、`moe_router_load_balancing_type`（aux_loss / sinkhorn / none）。aux loss 由 `moe_utils.py` 的 `MoEAuxLossAutoScaler` 乘进主 loss 梯度；router 梯度的特殊归约见 `finalize_model_grads.py:278/:334`。

## 4. `token_dispatcher.py`

| 行号 | 类 | 适用 |
|---|---|---|
| `:64` | `MoETokenDispatcher`（基类） | |
| `:233` | `MoEAllGatherTokenDispatcher` | EP=1 |
| **`:375`** | **`MoEAlltoAllTokenDispatcher`** | EP>1 标准 |
| `:963` | `_DispatchManager`（ABC） | |
| `:1001` | `_HybridEPManager` | hybrid EP |
| **`:1225`** | **`_DeepepManager`** | DeepEP 后端 |
| `:1471` | `_NCCLEPManager` | NCCL all-to-all |
| `:1809` | `MoEFlexTokenDispatcher` | 统一入口，按 config 选 manager |

变长 all-to-all：每个 expert 的 token 数不同 → `mappings.py:118` 的不均匀 `output_split_sizes`。

## 5. `experts.py` 与其他

| 行号 | 类 |
|---|---|
| **`:189`** | **`TEGroupedMLP`**（Grouped GEMM，一个 kernel 打所有专家，主力） |
| **`:1480`** | **`SequentialMLP`**（逐专家 for 循环，无 grouped GEMM 时回退） |
| `:1136` | `InferenceGroupedMLP` |

其他文件：`shared_experts.py`（`SharedExpertMLP`）、`moe_utils.py`（permute/unpermute、AutoScaler）、`upcycling_utils.py`（dense→MoE 权重上采样）、`router_replay.py`/`router_trace.py`（路由回放调试）。

## 6. 与 TransformerLayer 的接缝

- `transformer_layer.py:898 _maybe_unflatten_for_moe`：packed sequence 时恢复 batch 维
- `gpt_layer_specs.py:490 get_mlp_module_spec`：按 `num_moe_experts>0` 返回 MoE 或 dense spec，**同一个 TransformerLayer 骨架无感切换**
- `transformer_layer.py:2061 MoETransformerLayer`
