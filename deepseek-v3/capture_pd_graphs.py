import torch
import torch.nn as nn
import io
import sys
import os
import shutil
import json
from transformers import AutoConfig, AutoModelForCausalLM

# --- 引入 Dist_IR 相关库 ---
import Dist_IR
from Dist_IR import InferenceGraphCapture, logger, KVCacheAnalyzer
from torch.fx.passes.graph_drawer import FxGraphDrawer
from Performance_Eval.Fake_Runtime.Simulation import Performance_Evaluation
# from schedule_irV.SR_version1 import ScheduleSpec, build, write_ndjson

# ==============================================================================
# 1. 辅助工具函数 (保留在全局作用域，供子进程 import)
# ==============================================================================

def save_inference_graph(graph_module, graph_name: str):
    """保存计算图为 MD 和 DOT 文件"""
    if not graph_module or not graph_module.graph.nodes:
        logger.info(f"Graph '{graph_name}' is empty, skipping save.")
        return

    output_dir = os.path.join('Dist_IR', graph_name)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存文本 IR
    md_path = os.path.join(output_dir, 'Dist_IR_FW.md')
    with open(md_path, 'w') as f:
        original_stdout = sys.stdout
        with io.StringIO() as buf:
            sys.stdout = buf
            print(graph_module.graph)
            sys.stdout = original_stdout
            f.write(buf.getvalue())

    # 保存 DOT 可视化
    dot_path = os.path.join(output_dir, 'Dist_IR_FW.dot')
    try:
        g = FxGraphDrawer(graph_module, graph_name, ignore_getattr=True)
        with open(dot_path, 'w') as f:
            f.write(str(g.get_dot_graph()))
        logger.info(f"[Save] Saved {graph_name} to {output_dir}")
    except Exception as e:
        logger.error(f"Failed to save DOT for {graph_name}: {e}")

def create_flat_kv_cache(config, batch_size, seq_len, device='cpu', dtype=torch.float32):
    """
    [修改版] 支持 DeepSeek MLA 结构的 Cache 生成
    生成格式: List[Tensor] -> [compressed_kv, k_pe, compressed_kv, k_pe, ...]
    """
    num_hidden_layers = config.num_hidden_layers
    
    # 检查是否开启了 MLA (看有没有 kv_lora_rank 参数)
    kv_lora_rank = getattr(config, 'kv_lora_rank', None)
    qk_rope_head_dim = getattr(config, 'qk_rope_head_dim', 64)

    is_mla = kv_lora_rank is not None

    if is_mla:
        print(f">> [Cache Info] Creating MLA Cache (Compressed):")
        print(f"   - Compressed KV: [B={batch_size}, S={seq_len}, Rank={kv_lora_rank}]")
        print(f"   - RoPE Key:      [B={batch_size}, 1, S={seq_len}, Dim={qk_rope_head_dim}]")
    else:
        # 回退到标准 MHA 逻辑
        num_key_value_heads = config.num_key_value_heads
        k_head_dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)
        v_head_dim = getattr(config, 'v_head_dim', k_head_dim)
        print(f">> [Cache Info] Creating Standard MHA Cache: [B={batch_size}, H={num_key_value_heads}, S={seq_len}]")

    flat_cache = []
    for _ in range(num_hidden_layers):
        if is_mla:
            # --- MLA Cache 构造 ---
            # 1. Compressed Latent Vector: 3D Tensor [Batch, Seq_Len, KV_Lora_Rank]
            # 注意：这里 dim=1 是 seq_len，因为 MLA 中不需要 Head 维度
            c_kv = torch.randn((batch_size, seq_len, kv_lora_rank), device=device, dtype=dtype)
            
            # 2. RoPE Key Part: 4D Tensor [Batch, 1, Seq_Len, Rope_Dim]
            # 注意：RoPE 部分通常保留 Head=1 的维度以便广播，且 dim=2 是 seq_len
            k_pe = torch.randn((batch_size, 1, seq_len, qk_rope_head_dim), device=device, dtype=dtype)
            
            flat_cache.append(c_kv)
            flat_cache.append(k_pe)
        else:
            # --- Standard Cache 构造 (原逻辑) ---
            k = torch.randn((batch_size, num_key_value_heads, seq_len, k_head_dim), device=device, dtype=dtype)
            v = torch.randn((batch_size, num_key_value_heads, seq_len, v_head_dim), device=device, dtype=dtype)
            flat_cache.append(k)
            flat_cache.append(v)
            
    return flat_cache

# ==============================================================================
# 2. 核心补丁: EP-Optimized MoE Patch
# ==============================================================================
def patch_deepseek_moe_infer(model):
    """
    [Patch 说明 - EP 优化版]
    将串行的 += 累加替换为 Stack + Sum，显式分离 Shared 和 Routed Experts。
    """
    print("\n>>> [Patch] Replacing 'moe_infer' with EP-Optimized version (Stack+Sum)...")
    
    found_class = False
    layers = model.model.layers if hasattr(model, 'model') else model.layers

    for layer in layers:
        if hasattr(layer, 'mlp') and 'MoE' in layer.mlp.__class__.__name__:
            MoEClass = layer.mlp.__class__
            
            def ep_optimized_moe_infer(self, hidden_states, topk_idx, topk_weight):
                # Path 1: Shared Experts
                if hasattr(self, 'shared_experts'):
                    shared_out = self.shared_experts(hidden_states)
                else:
                    shared_out = torch.zeros_like(hidden_states)

                # Path 2: Routed Experts
                routed_expert_outputs = []
                for i, expert in enumerate(self.experts):
                    hit_mask = (topk_idx == i)
                    weight_i = (topk_weight * hit_mask.float()).sum(dim=-1, keepdim=True)
                    
                    expert_out = expert(hidden_states)
                    weighted_res = expert_out * weight_i.to(expert_out.dtype)
                    routed_expert_outputs.append(weighted_res)

                # Path 3: Aggregation (Stack + Sum)
                if len(routed_expert_outputs) > 0:
                    stacked_outs = torch.stack(routed_expert_outputs, dim=0)
                    routed_out = stacked_outs.sum(dim=0)
                else:
                    routed_out = torch.zeros_like(hidden_states)
                
                return shared_out + routed_out

            MoEClass.moe_infer = ep_optimized_moe_infer
            print(f">>> [Patch] Applied successfully to class: {MoEClass.__name__}")
            found_class = True
            break
    
    if not found_class:
        print(">>> [Patch Warning] No MoE layers found to patch.")

# ==============================================================================
# 3. 模型包装器 (Wrapper)
# ==============================================================================
class DeepSeekWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model 

    def forward(self, input_ids, attention_mask, past_key_values=None):
        logits, new_cache = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True 
        )
        return logits, new_cache

# ==============================================================================
# MAIN EXECUTION BLOCK - 必须在 __name__ == "__main__" 下
# ==============================================================================
if __name__ == "__main__":
    
    # ==============================================================================
    # 4. 配置与初始化
    # ==============================================================================
    MODEL_NAME = "deepseek_v3"
    LOCAL_CONFIG_PATH = './deepseek_v3_local' 

    DEVICE = "cpu"

    BATCH_SIZE = 10
    PREFILL_SEQ_LEN = 30
    KV_CACHE_SEQ_LEN = PREFILL_SEQ_LEN 

    print(f"\n>> 初始化模型: Custom DeepSeek-V3 on {DEVICE}")

    try:
        print(">> Loading and Overriding config...")
        config = AutoConfig.from_pretrained(LOCAL_CONFIG_PATH, trust_remote_code=True)

        # --- 瘦身配置 ---
        config.num_hidden_layers = 1 
        # [关键修改] 强制让第 0 层就是 MoE 层 (默认可能是 1, 导致第 0 层是 Dense)
        config.first_k_dense_replace = 0 
        config.moe_layer_freq = 1  # 确保频率为 1
        config.hidden_size = 128            
        config.intermediate_size = 512      
        config.moe_intermediate_size = 64   
        config.num_attention_heads = 4
        config.num_key_value_heads = 4
        config.max_position_embeddings = 512
        
        # 强制设置 MLA 参数
        config.kv_lora_rank = 32 #k/v的维度
        config.q_lora_rank = 64 #q维度（通常为2*k）
        config.qk_rope_head_dim = 16 #需要位置编码的k
        config.qk_nope_head_dim = 16 #不需要位置编码的K
        config.v_head_dim = 32 
        
        TOTAL_EXPERTS = 8
        if hasattr(config, 'n_routed_experts'): config.n_routed_experts = TOTAL_EXPERTS
        if hasattr(config, 'num_experts_per_tok'): config.num_experts_per_tok = TOTAL_EXPERTS 
        if hasattr(config, 'n_group'): config.n_group = 1
        if hasattr(config, 'topk_group'): config.topk_group = 1
        if hasattr(config, 'n_shared_experts'): config.n_shared_experts = 1
        
        config.attn_implementation = "eager"
        config._attn_implementation = "eager"

        # 实例化模型
        raw_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True).to(DEVICE)
        
        # [关键步骤] 获取模型的真实 dtype (通常是 bfloat16 或 float32)
        # 这决定了后续所有 inputs 必须匹配这个类型
        MODEL_DTYPE = raw_model.dtype
        print(f">> Model initialized with dtype: {MODEL_DTYPE}")

        patch_deepseek_moe_infer(raw_model)
        model = DeepSeekWrapper(raw_model).to(DEVICE)
        model.eval()

    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(f"模型初始化失败: {e}")

    # ==============================================================================
    # 5. 图捕获 (Prefill & Decode)
    # ==============================================================================

    with torch.no_grad():

        # --- A. Prefill 阶段 ---
        print("\n" + "="*40)
        print("Step 1: 捕获 Prefill 图")
        
        # [修改] Mask 必须转为 MODEL_DTYPE，否则是 Float32
        _mask = torch.tril(torch.ones((PREFILL_SEQ_LEN, PREFILL_SEQ_LEN), device=DEVICE, dtype=MODEL_DTYPE))
        prefill_mask = _mask[None, None, :, :]
        
        prefill_inputs = { 
            'input_ids': torch.randint(0, config.vocab_size, (BATCH_SIZE, PREFILL_SEQ_LEN), device=DEVICE),
            'attention_mask': prefill_mask,
            'past_key_values': None 
        }
        
        prefill_capture = InferenceGraphCapture(model, filename_prefix="deepseek_prefill", **prefill_inputs)
        compiled_prefill = prefill_capture.compile()
        compiled_prefill(**prefill_inputs)
        
        # --- B. Decode 阶段 ---
        print("\n" + "="*40)
        print("Step 2: 捕获 Decode 图")
        
        # [修改] Mask 必须转为 MODEL_DTYPE
        total_len = KV_CACHE_SEQ_LEN + 1
        decode_mask = torch.ones((BATCH_SIZE, 1, 1, total_len), device=DEVICE, dtype=MODEL_DTYPE)

        # [修改] Cache 必须使用 MODEL_DTYPE
        dummy_cache = create_flat_kv_cache(config, BATCH_SIZE, KV_CACHE_SEQ_LEN, device=DEVICE, dtype=MODEL_DTYPE)
        
        decode_inputs = { 
            'input_ids': torch.randint(0, config.vocab_size, (BATCH_SIZE, 1), device=DEVICE),
            'attention_mask': decode_mask, 
            'past_key_values': dummy_cache
        }
        
        decode_capture = InferenceGraphCapture(model, filename_prefix="deepseek_decode", **decode_inputs)
        compiled_decode = decode_capture.compile()
        compiled_decode(**decode_inputs)

    # ==============================================================================
    # 6. 分析与并行切分
    # ==============================================================================

    print("\n" + "="*40)
    print("Step 3: 运行 Dist_IR Pass (Analysis & Positioning)")
    print("="*40)

    # --- A. Prefill ---
    if prefill_capture.FW_gm:
        print(">> Processing Prefill Graph...")
        Dist_IR.KVCacheAnalyzer(prefill_capture.FW_gm, "prefill", MODEL_NAME, config)
        pos_prefill = Dist_IR.Pos(raw_model, prefill_capture.FW_gm, None, model_name='deepseek')
        pos_prefill.positioning_for_graph()
        pos_prefill.log_node_positioning_info()
        
        prefill_graph=Dist_IR.Hybrid_Parallel_pass(
            prefill_capture.FW_gm, None, None, 
            Global_batch_size=BATCH_SIZE, inference_stage='prefill'
        )
        save_inference_graph(prefill_capture.FW_gm, "deepseek_prefill")

    # --- B. Decode ---
    if decode_capture.FW_gm:
        print("\n>> Processing Decode Graph...")
        Dist_IR.KVCacheAnalyzer(decode_capture.FW_gm, "decode", MODEL_NAME, config)
        pos_decode = Dist_IR.Pos(raw_model, decode_capture.FW_gm, None, model_name='deepseek')
        pos_decode.positioning_for_graph()
        
        decode_graph=Dist_IR.Hybrid_Parallel_pass(
            decode_capture.FW_gm, None, None, 
            Global_batch_size=BATCH_SIZE, inference_stage='decode'
        )
        save_inference_graph(decode_capture.FW_gm, "deepseek_decode")

    print("\nDone. DeepSeek-V3 (EP Optimized) graphs generated.")


    from Performance_Eval.Fake_Runtime.PD_separation import PD_Performance_Eval

    Eval = PD_Performance_Eval(prefill_capture.FW_gm, decode_capture.FW_gm, prefill_graph, decode_graph, PREFILL_SEQ_LEN, 2)
    time = Eval.Evaluate()

    print(f"最终结果: {time} us")

    # ==============================================================================
    # 7. 性能评估 (Evaluation) - 必须在 main 块内
    # ==============================================================================

    # Eval = Performance_Evaluation(decode_graph)
    # time,time_list = Eval.Evaluate()

    print(f"最终结果: {time} us")

    # # ==============================================================================
    # # 8. Schedule 生成与文件输出
    # # ==============================================================================

    # cfg = {
    #   "name": "pp_demo",
    #   "strategy": "gpipe",         
    #   "num_stages": 2,
    #   "num_mini": 2,
    #   "num_micro": 3,
    #   "lanes": {"compute": 0, "comm": 1, "custom": 2},
    #   "stage_durations": {"forward": [240, 240, 240], "backward": [240, 240, 240]},
    #   ## "meta": { "virtual_chunks": 2 } (当strategy是interleaved的时候，这个可以做chunk切分)
    #   "insertions": [
    #     {
    #       "name": "cast_activation",
    #       "op": "Cast",
    #       "phase": "forward",
    #       "anchor": "host_start",
    #       "offset": 0,
    #       "duration": {"ratio_of_host": 0.1},
    #       "lane": "custom", #“compute” or “comm”
    #       "stage_selector": "all",
    #       "micro_selector": "all",
    #       "mini_selector": "all"
    #     }
    #   ],
    #   "emit": {"include_comments": True}
    # }

    # prefill_cfg_str = os.getenv("dist_config")
    # if prefill_cfg_str is None:
    #     raise RuntimeError("环境变量 dist_config 没有设置")
    # prefill_cfg = json.loads(prefill_cfg_str)

    # pd = prefill_cfg["prefill"]["parallel_degree"]
    # D = pd["D"]
    # P = pd["P"]
    # T = pd["T"]
    # S = pd["S"]

    # list=[]
    # stage_numb = P
    # device_list=[]

    # for i in range(stage_numb):
    #     device_number= i * T * S 
    #     device_list.append(device_number)

    # for i in range(len(device_list)):
    #     if i == 0:
    #         temp0 = device_list[0]
    #         list.append(time_list[temp0])
    #     else :
    #         temp1 = device_list[i]
    #         temp2 = device_list[i-1]
    #         time_difference_stage= time_list[temp1]-time_list[temp2]
    #         list.append(time_difference_stage)

    # cfg["stage_durations"]["forward"] = list
    # cfg["stage_durations"]["backward"] = list
    # spec = ScheduleSpec(cfg)
    # events = build(spec)
    # with open("out.ndjson", "w", encoding="utf-8") as f:
    #     write_ndjson(f, events, include_comments=spec.emit.get("include_comments", True))

    # print("Wrote out.ndjson with", len(events), "events")