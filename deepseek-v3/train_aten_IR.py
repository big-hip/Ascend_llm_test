import torch
import torch.nn as nn
import io
import sys
import math
import types
from transformers import AutoConfig, AutoModelForCausalLM
import Dist_IR 

# =============================================================================
# [System Config] 递归深度限制
# 即使层数减少了，由于 Graph Capture 需要递归展开所有专家分支，
# 设置稍微高一点的值仍然是安全的做法。
# =============================================================================
# sys.setrecursionlimit(5000)
# print(f">>> System recursion limit set to: {sys.getrecursionlimit()}")

# --- 1. 推理包装器 ---
class InferenceWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, input_ids):
        # use_cache=False: 避免生成 KV-Cache 相关的 Update 节点，让图更干净
        out = self.model(input_ids=input_ids, use_cache=False, return_dict=True)
        return out.logits

# --- 2. 自动热修补函数 (Dynamo Friendly MoE) ---
def patch_deepseek_moe_infer(model):
    """
    [Patch 说明]
    DeepSeek 原生 MoE 使用动态索引 (index_select)，导致编译器难以静态分析。
    这里替换为 'Masked Compute' 模式：
    1. 显式遍历所有专家 (Loop Unrolling)。
    2. 无论是否命中，都生成计算节点，然后用 Mask (0或1) 过滤结果。
    3. 这样生成的图是静态的、形状固定的，非常适合 Dist_IR 分析。
    """
    print("\n>>> [Patch] Replacing 'moe_infer' with Dynamo-friendly version...")
    
    found_class = False
    for layer in model.model.layers:
        # 寻找 MoE 模块的类定义
        if hasattr(layer, 'mlp') and 'MoE' in layer.mlp.__class__.__name__:
            MoEClass = layer.mlp.__class__
            
            def dynamo_safe_moe_infer(self, hidden_states, topk_idx, topk_weight):
                out = torch.zeros_like(hidden_states)
                
                # 遍历所有专家
                for i, expert in enumerate(self.experts):
                    # 计算 Mask: 检查当前专家 i 是否被选中
                    # any(-1) 处理 topk 维度
                    hit_mask = (topk_idx == i)
                    # 计算权重: 如果未命中，权重为 0
                    expert_weight = (topk_weight * hit_mask.float()).sum(dim=-1, keepdim=True)
                    
                    # 强制执行前向计算 (哪怕权重是0) -> 保证图中有节点
                    current_expert_out = expert(hidden_states)
                    
                    # 累加结果
                    out += current_expert_out * expert_weight.to(current_expert_out.dtype)
                return out

            MoEClass.moe_infer = dynamo_safe_moe_infer
            print(f">>> [Patch] Applied successfully to class: {MoEClass.__name__}")
            found_class = True
            break
    
    if not found_class:
        print(">>> [Patch Warning] No MoE layers found to patch (might be OK if config is pure Dense).")

# --- 3. 模型配置加载与修改 ---
print("\n>>> Initializing DeepSeek-V3 'Reasonable Mini' Config...")
model_path_or_identifier = './deepseek_v3_local' # 替换为你的本地路径

try:
    config = AutoConfig.from_pretrained(model_path_or_identifier, trust_remote_code=True)
    
    # =========================================================
    # [A] 合理的层级结构 (Reasonable Structure)
    # =========================================================
    # 总共 6 层：足以看清结构，又不会太庞大
    config.num_hidden_layers = 6  
    
    # 混合架构：前 2 层 Dense，后 4 层 MoE
    if hasattr(config, 'first_k_dense_replace'):
        config.first_k_dense_replace = 2
    if hasattr(config, 'num_dense_layers'):
        config.num_dense_layers = 2

    # =========================================================
    # [B] 维度配置 (Small but Readable)
    # =========================================================
    # 128 维度看起来比 64 更像正经模型，方便看 MatMul 形状
    config.hidden_size = 128             
    config.intermediate_size = 256      
    config.moe_intermediate_size = 64   
    
    config.num_attention_heads = 4
    config.num_key_value_heads = 4
    config.max_position_embeddings = 512

    # MLA (Multi-Head Latent Attention) 参数按比例缩小
    # 必须匹配 hidden_size，否则矩阵乘法维度不匹配
    if hasattr(config, 'kv_lora_rank'): config.kv_lora_rank = 32
    if hasattr(config, 'q_lora_rank'): config.q_lora_rank = 64
    if hasattr(config, 'qk_rope_head_dim'): config.qk_rope_head_dim = 16
    if hasattr(config, 'v_head_dim'): config.v_head_dim = 32
    if hasattr(config, 'qk_nope_head_dim'): config.qk_nope_head_dim = 16

    # =========================================================
    # [C] 专家配置 (All Experts Active)
    # =========================================================
    TOTAL_EXPERTS = 8
    
    # 1. 设定专家总数
    if hasattr(config, 'n_routed_experts'):
        config.n_routed_experts = TOTAL_EXPERTS

    # 2. [核心] 激活所有路由专家
    # 设为 8，意味着每个 Token 都会被路由到这 8 个专家
    if hasattr(config, 'num_experts_per_tok'):
        config.num_experts_per_tok = TOTAL_EXPERTS 

    # 3. [修复] 消除分组路由的副作用
    # 强制只有 1 个组，并选中这 1 个组。
    # 这样 8 个专家都在这个池子里，避免 k out of range 或选不全的问题
    if hasattr(config, 'n_group'):
        config.n_group = 1
    if hasattr(config, 'topk_group'):
        config.topk_group = 1
        
    # 4. 共享专家
    # 按照 V3 架构保留 1 个共享专家
    if hasattr(config, 'n_shared_experts'):
        config.n_shared_experts = 1
        
    # 5. 其他
    if hasattr(config, 'routed_scaling_factor'):
        config.routed_scaling_factor = 1.0
    if hasattr(config, 'num_nextn_predict_layers'):
        config.num_nextn_predict_layers = 0
    
    # 强制使用 Eager Attention (不用 FlashAttention，便于图捕获)
    config._attn_implementation = "eager"
    config.attn_implementation = "eager"

    print(f"    Config Ready: {config.num_hidden_layers} Layers (2 Dense + 4 MoE)")
    print(f"    Experts: {config.n_routed_experts} Routed (All Active) + {getattr(config, 'n_shared_experts', 0)} Shared")
    
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

except OSError as e:
    print(f"Error: Ensure '{model_path_or_identifier}' exists.")
    raise e

device = torch.device("cpu") 
model.to(device)
model.eval()

# [执行补丁]
patch_deepseek_moe_infer(model)

# --- 4. 验证层级结构 ---
print("\n>>> Verifying Architecture Topology...")
dense_cnt = 0
moe_cnt = 0
for i, layer in enumerate(model.model.layers):
    layer_type = layer.mlp.__class__.__name__
    if "MoE" in layer_type:
        moe_cnt += 1
    else:
        dense_cnt += 1
    print(f"    Layer {i}: {layer_type}")

print(f"    Total: {len(model.model.layers)} Layers. Dense: {dense_cnt}, MoE: {moe_cnt}")

if dense_cnt != 2 or moe_cnt != 4:
    print("    [Warning] Layer counts mismatch! Expected 2 Dense + 4 MoE.")

# --- 5. 包装与数据准备 ---
wrapped_model = InferenceWrapper(model).to(device)
# 序列长度设为 8，足够观察 Attention 模式
input_ids = torch.randint(0, config.vocab_size, (1, 8), device=device)

# --- 6. Eager Mode 预检查 ---
print("\n>>> Running Eager Mode pass...")
try:
    with torch.no_grad():
        logits = wrapped_model(input_ids)
    print(f">>> Pass successful. Output shape: {logits.shape}")
except Exception as e:
    print(f"\n[CRITICAL] Eager run failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# --- 7. Dist_IR 图捕获 ---
print("\n>>> Capturing graph with Dist_IR...")
try:
    with torch.no_grad():
        graph_capture = Dist_IR.GraphCapture(wrapped_model, input_ids)
        compiled_model = graph_capture.compile()
    print(">>> Capture complete.")
except Exception as e:
    print(f"Capture failed: {e}")
    raise e
# --- 8. 推理执行 ---
print("Running inference on compiled graph...")
with torch.no_grad():
    output = compiled_model(input_ids)
    print("Inference step complete.")
# --- 8. 保存结果 ---
filename = 'deepseek_v3_mini_all_experts.md'
print(f"Saving Forward graph to {filename}...")

with io.StringIO() as buf, open(filename, 'w') as f:
    sys.stdout = buf
    graph_capture.FW_gm.print_readable()
    sys.stdout = sys.__stdout__
    f.write(buf.getvalue())

# # --- 9. Pos 信息处理 ---
# print("Processing Positioning Info...")
# try:
#     bw_gm_arg = getattr(graph_capture, 'BW_gm', None)
#     # pass_instance = Dist_IR.Pos(model, graph_capture.FW_gm, bw_gm_arg)
#     # pass_instance.log_node_positioning_info()
#     print("Positioning info log generated.")
# except Exception as e:
#     print(f"Dist_IR.Pos skipped or failed: {e}")

print("\n>>> Done.")