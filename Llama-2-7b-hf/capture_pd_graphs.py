import torch
import torch.nn as nn
import io
import sys
import os
import shutil
from torch.fx.passes.graph_drawer import FxGraphDrawer
from transformers import LlamaConfig, LlamaForCausalLM
import multiprocessing as mp # 引入 multiprocessing
# --- 假设您的库中包含了这些 ---
import Dist_IR
from Dist_IR import InferenceGraphCapture, logger, KVCacheAnalyzer
from Performance_Eval.Fake_Runtime.Simulation import Performance_Evaluation
# ==============================================================================
# 1. 辅助工具函数
# ==============================================================================

def save_inference_graph(graph_module, graph_name: str):
    """保存计算图为 MD 和 DOT 文件"""
    if not graph_module or not graph_module.graph.nodes:
        logger.info(f"Graph '{graph_name}' is empty, skipping save.")
        return

    output_dir = os.path.join('Dist_IR', graph_name)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir) # 清理旧文件
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存 .md
    md_path = os.path.join(output_dir, 'Dist_IR_FW.md')
    with open(md_path, 'w') as f:
        original_stdout = sys.stdout
        with io.StringIO() as buf:
            sys.stdout = buf
            print(graph_module.graph)
            sys.stdout = original_stdout
            f.write(buf.getvalue())

    # 保存 .dot
    dot_path = os.path.join(output_dir, 'Dist_IR_FW.dot')
    try:
        g = FxGraphDrawer(graph_module, graph_name)
        with open(dot_path, 'w') as f:
            f.write(str(g.get_dot_graph()))
        logger.info(f"[Save] Saved {graph_name} to {output_dir}")
    except Exception as e:
        logger.error(f"Failed to save DOT for {graph_name}: {e}")

def create_dummy_kv_cache(config, batch_size, seq_len, device='cpu'):
    """创建符合 Llama 结构的伪造 KV Cache"""
    num_heads = getattr(config, 'num_key_value_heads', config.num_attention_heads)
    head_dim = config.hidden_size // config.num_attention_heads
    num_layers = config.num_hidden_layers
    
    kv_cache = []
    for _ in range(num_layers):
        k = torch.randn((batch_size, num_heads, seq_len, head_dim), device=device, dtype=torch.float16)
        v = torch.randn((batch_size, num_heads, seq_len, head_dim), device=device, dtype=torch.float16)
        kv_cache.append((k, v))
    return tuple(kv_cache)

# ==============================================================================
# 2. 模型包装器 (Wrapper) - 【关键修改】
# ==============================================================================
class LlamaWrapper(nn.Module):
    """
    为了让 Stack Trace 与训练代码保持一致 (包含 'llama_model' 关键字)，
    我们将内部成员变量命名为 self.llama_model，而不是 self.model。
    """
    def __init__(self, model):
        super().__init__()
        # 【关键一致性修改】
        # 训练代码使用的是: self.llama_model = llama_model
        # 这里我们也用同样的名字！
        self.llama_model = model 

    def forward(self, input_ids, past_key_values=None):
        # 调用 self.llama_model，这样 FX 追踪到的路径就是 llama_model.xxx
        outputs = self.llama_model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=False 
        )
        return outputs[0], outputs[1]

# ==============================================================================
# 3. 配置与初始化
# ==============================================================================
if __name__ == "__main__":
    # --- 关键：设置 spawn 启动方式 ---
    try:
        mp.set_start_method('spawn', force=True)
        print("[Main] Multiprocessing start method set to 'spawn'.")
    except RuntimeError as e:
        print(f"[Warn] Cannot set start method: {e}")
    MODEL_NAME = "llama2"
    LOCAL_CONFIG_PATH = './Llama_2_7b_hf_local' 
    if not os.path.exists(LOCAL_CONFIG_PATH):
        sys.exit(f"错误：找不到模型路径 '{LOCAL_CONFIG_PATH}'。")

    gpu_index = 2
    DEVICE = f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu"
    # 保持与训练一致的 BatchSize
    BATCH_SIZE = 4 
    PREFILL_SEQ_LEN = 100 # 与训练 max_seq_length 保持一致
    KV_CACHE_SEQ_LEN = PREFILL_SEQ_LEN 

    print(f"\n>> 初始化模型: Llama-2 (Inference Mode) on {DEVICE}")

    try:
        config = LlamaConfig.from_pretrained(LOCAL_CONFIG_PATH)
        
        # --- [保持与训练一致的配置] ---
        config.num_hidden_layers = 1 #transformer block 数
        config.hidden_size = 512
        config.intermediate_size = 1376 
        config.num_attention_heads = 8 
        config.num_key_value_heads = 8 

        keys_to_clean = ['head_dim', 'kv_channels', 'n_head_kv'] # Qwen 常用字段
        for k in keys_to_clean:
            if hasattr(config, k):
                print(f"   [Config Clean] Removing stale key: {k} (Value: {getattr(config, k)})")
                delattr(config, k)
        
        # 实例化原始模型
        raw_model = LlamaForCausalLM(config).to(DEVICE)
        # 包装模型
        model = LlamaWrapper(raw_model).to(DEVICE)
        model.eval() 
        
        print(f"   Config: Layers={config.num_hidden_layers}, Hidden={config.hidden_size}")

    except Exception as e:
        sys.exit(f"模型初始化失败: {e}")

    # ==============================================================================
    # 4. 图捕获
    # ==============================================================================

    with torch.no_grad():

        # --- A. Prefill 阶段 ---
        print("\n" + "="*40)
        print("Step 1: 捕获 Prefill 图")
        
        prefill_inputs = { 
            'input_ids': torch.randint(0, config.vocab_size, (BATCH_SIZE, PREFILL_SEQ_LEN), device=DEVICE),  
            'past_key_values': None
        }
        
        prefill_capture = InferenceGraphCapture(model, filename_prefix="prefill", **prefill_inputs)
        compiled_prefill = prefill_capture.compile()
        compiled_prefill(**prefill_inputs)
        
        # --- B. Decode 阶段 ---
        print("\n" + "="*40)
        print("Step 2: 捕获 Decode 图")
        
        dummy_cache = create_dummy_kv_cache(config, BATCH_SIZE, KV_CACHE_SEQ_LEN, device=DEVICE)
        decode_inputs = { 
            'input_ids': torch.randint(0, config.vocab_size, (BATCH_SIZE, 1), device=DEVICE),  
            'past_key_values': dummy_cache
        }
        
        decode_capture = InferenceGraphCapture(model, filename_prefix="decode", **decode_inputs)
        compiled_decode = decode_capture.compile()
        compiled_decode(**decode_inputs)

    # ==============================================================================
    # 5. 分析与并行切分
    # ==============================================================================

    print("\n" + "="*40)
    print("Step 3: 运行 Dist_IR Pass")
    print("="*40)

    # --- A. 处理 Prefill ---
    print(">> Processing Prefill Graph...")
    if prefill_capture.FW_gm:

        Dist_IR.KVCacheAnalyzer(prefill_capture.FW_gm, "prefill", MODEL_NAME, config)

        # 【关键一致性】传入 raw_model (即 LlamaForCausalLM 实例)，这与训练代码传入 model 是一样的
        pos_prefill = Dist_IR.Pos(raw_model, prefill_capture.FW_gm, None, model_name='llama2')
        
        # 由于 Wrapper 属性名改为了 llama_model，这里的 positioning 应该能像训练一样工作
        pos_prefill.positioning_for_graph()
        
        # 记录日志 
        pos_prefill.log_node_positioning_info()
        

        prefill_graph=Dist_IR.Hybrid_Parallel_pass(
            prefill_capture.FW_gm, 
            None, None, 
            Global_batch_size=BATCH_SIZE,
            inference_stage='prefill'
        )
        save_inference_graph(prefill_capture.FW_gm, "_prefill")

    # --- B. 处理 Decode ---
    print("\n>> Processing Decode Graph...")
    if decode_capture.FW_gm:
        # 1. 定位 Pass
        Dist_IR.KVCacheAnalyzer(decode_capture.FW_gm, "decode", MODEL_NAME, config)

        pos_decode = Dist_IR.Pos(raw_model, decode_capture.FW_gm, None, model_name='llama2')
        pos_decode.positioning_for_graph()
        
        # 2. 混合并行 Pass
        decode_graph=Dist_IR.Hybrid_Parallel_pass(
            decode_capture.FW_gm, 
            None, None, 
            Global_batch_size=BATCH_SIZE,
            inference_stage='decode'
        )
        save_inference_graph(decode_capture.FW_gm, "_decode")

    print("\nDone.")
    from Performance_Eval.Fake_Runtime.PD_separation import PD_Performance_Eval

    Eval = PD_Performance_Eval(prefill_capture.FW_gm, decode_capture.FW_gm, prefill_graph, decode_graph, PREFILL_SEQ_LEN, 2)
    time = Eval.Evaluate()

    print(f"最终结果: {time} us")