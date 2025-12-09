import torch
import torch.nn as nn
import io
import sys
import os
import shutil
import multiprocessing as mp # 引入 multiprocessing
from torch.fx.passes.graph_drawer import FxGraphDrawer
from transformers import AutoConfig
from transformers import Qwen2Config 

# --- 引入修改后的本地模型文件 ---
sys.path.append(os.getcwd())
try:
    from modeling_qwen3_custom import Qwen3ForCausalLM
except ImportError as e:
    print(f"导入失败细节: {e}")
    sys.exit("错误：无法导入 modeling_qwen_custom。请检查步骤1是否执行正确。")

# --- 假设您的库中包含了这些 ---
import Dist_IR
from Dist_IR import InferenceGraphCapture, logger, KVCacheAnalyzer
# 将评估模块的导入提前，或者放在 main 块中均可，这里建议放顶层
from Performance_Eval.Fake_Runtime.PD_separation import PD_Performance_Eval

# ==============================================================================
# 1. 辅助工具函数 (保持在全局作用域，供子进程 pickle 使用)
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
    
    md_path = os.path.join(output_dir, 'Dist_IR_FW.md')
    with open(md_path, 'w') as f:
        original_stdout = sys.stdout
        with io.StringIO() as buf:
            sys.stdout = buf
            print(graph_module.graph)
            sys.stdout = original_stdout
            f.write(buf.getvalue())

    dot_path = os.path.join(output_dir, 'Dist_IR_FW.dot')
    try:
        g = FxGraphDrawer(graph_module, graph_name, ignore_getattr=True)
        with open(dot_path, 'w') as f:
            f.write(str(g.get_dot_graph()))
        logger.info(f"[Save] Saved {graph_name} to {output_dir}")
    except Exception as e:
        logger.error(f"Failed to save DOT for {graph_name}: {e}")

def create_flat_kv_cache(config, batch_size, seq_len, device='cpu'):
    """
    针对修改后的 Qwen 模型，创建扁平化的 List[Tensor]。
    结构: [layer0_k, layer0_v, layer1_k, layer1_v, ...]
    """
    num_hidden_layers = config.num_hidden_layers
    num_key_value_heads = config.num_key_value_heads
    head_dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)
    
    flat_cache = []
    for _ in range(num_hidden_layers):
        # Shape: [Batch, KV_Heads, Seq_Len, Dim]
        k = torch.randn((batch_size, num_key_value_heads, seq_len, head_dim), device=device, dtype=torch.float32)
        v = torch.randn((batch_size, num_key_value_heads, seq_len, head_dim), device=device, dtype=torch.float32)
        flat_cache.append(k)
        flat_cache.append(v)
        
    return flat_cache

# ==============================================================================
# 2. 模型包装器 (Wrapper) (保持在全局作用域)
# ==============================================================================
class QwenWrapper(nn.Module):
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
# 3. 主执行逻辑 (必须放入 if __name__ == "__main__":)
# ==============================================================================
if __name__ == "__main__":
    # --- 关键：设置 spawn 启动方式 ---
    try:
        mp.set_start_method('spawn', force=True)
        print("[Main] Multiprocessing start method set to 'spawn'.")
    except RuntimeError as e:
        print(f"[Warn] Cannot set start method: {e}")

    # ==============================================================================
    # 配置与初始化
    # ==============================================================================
    MODEL_NAME = "qwen"
    LOCAL_CONFIG_PATH = './Qwen_8B_Base_local'  # 替换为你实际的路径

    DEVICE =  "cpu"

    BATCH_SIZE = 10
    PREFILL_SEQ_LEN = 30
    KV_CACHE_SEQ_LEN = PREFILL_SEQ_LEN 

    print(f"\n>> 初始化模型: Custom Qwen (Inference Mode) on {DEVICE}")

    try:
        # 1. 加载配置
        config = Qwen2Config.from_pretrained(LOCAL_CONFIG_PATH)
        
        print(">> Overriding config to create a TINY model...")
        config.hidden_size = 512              
        config.intermediate_size = 1376        
        config.num_hidden_layers = 2         
        config.num_attention_heads = 8         
        config.num_key_value_heads = 8 
        config.max_position_embeddings = 512
        config.attn_implementation = "eager"
        config._attn_implementation = "eager"
        
        keys_to_clean = ['head_dim', 'kv_channels', 'n_head_kv'] 
        for k in keys_to_clean:
            if hasattr(config, k):
                print(f"   [Config Clean] Removing stale key: {k}")
                delattr(config, k)

        # 2. 实例化修改后的模型类
        raw_model = Qwen3ForCausalLM(config).to(DEVICE)
        
        # 3. 包装
        model = QwenWrapper(raw_model).to(DEVICE)
        model.eval()
        
        print(f"   Config: Layers={config.num_hidden_layers}, Hidden={config.hidden_size}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(f"模型初始化失败: {e}")

    # ==============================================================================
    # 4. 图捕获
    # ==============================================================================
    
    # 确保 Graph Capture 在主进程中完成
    with torch.no_grad():

        # --- A. Prefill 阶段 ---
        print("\n" + "="*40)
        print("Step 1: 捕获 Prefill 图")
        
        _mask = torch.tril(torch.ones((PREFILL_SEQ_LEN, PREFILL_SEQ_LEN), device=DEVICE))
        prefill_mask = _mask[None, None, :, :]
        
        prefill_inputs = { 
            'input_ids': torch.randint(0, config.vocab_size, (BATCH_SIZE, PREFILL_SEQ_LEN), device=DEVICE),
            'attention_mask': prefill_mask,
            'past_key_values': None 
        }
        
        prefill_capture = InferenceGraphCapture(model, filename_prefix="prefill", **prefill_inputs)
        compiled_prefill = prefill_capture.compile()
        compiled_prefill(**prefill_inputs)
        
        # --- B. Decode 阶段 ---
        print("\n" + "="*40)
        print("Step 2: 捕获 Decode 图")
        
        total_len = KV_CACHE_SEQ_LEN + 1
        decode_mask = torch.ones((BATCH_SIZE, 1, 1, total_len), device=DEVICE)

        dummy_cache = create_flat_kv_cache(config, BATCH_SIZE, KV_CACHE_SEQ_LEN, device=DEVICE)
        
        decode_inputs = { 
            'input_ids': torch.randint(0, config.vocab_size, (BATCH_SIZE, 1), device=DEVICE),
            'attention_mask': decode_mask, 
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
    prefill_graph = None
    if prefill_capture.FW_gm:
        Dist_IR.KVCacheAnalyzer(prefill_capture.FW_gm, "prefill", MODEL_NAME, config)

        pos_prefill = Dist_IR.Pos(raw_model, prefill_capture.FW_gm, None, model_name='qwen')
        pos_prefill.positioning_for_graph()
        pos_prefill.log_node_positioning_info()
        
        prefill_graph=Dist_IR.Hybrid_Parallel_pass(
            prefill_capture.FW_gm, 
            None, None, 
            Global_batch_size=BATCH_SIZE,
            inference_stage='prefill'
        )
        save_inference_graph(prefill_capture.FW_gm, "prefill")

    # --- B. 处理 Decode ---
    print("\n>> Processing Decode Graph...")
    decode_graph = None
    if decode_capture.FW_gm:
        Dist_IR.KVCacheAnalyzer(decode_capture.FW_gm, "decode", MODEL_NAME, config)

        pos_decode = Dist_IR.Pos(raw_model, decode_capture.FW_gm, None, model_name='qwen')
        pos_decode.positioning_for_graph()
        
        decode_graph=Dist_IR.Hybrid_Parallel_pass(
            decode_capture.FW_gm, 
            None, None, 
            Global_batch_size=BATCH_SIZE,
            inference_stage='decode'
        )
        save_inference_graph(decode_capture.FW_gm, "decode")

    print("\nDone. Starting Evaluation...")

    # ==============================================================================
    # 6. 运行评估 (这会触发多进程)
    # ==============================================================================
    # 只有在这里，当所有图都准备好后，才调用 Evaluate
    if prefill_graph and decode_graph:
        Eval = PD_Performance_Eval(prefill_capture.FW_gm, decode_capture.FW_gm, prefill_graph, decode_graph, PREFILL_SEQ_LEN, 2)
        
        # 这里的 Evaluate 内部会调用 manager.start() 和 Process.start()
        # 由于我们在 if __name__ == "__main__": 内部，
        # 子进程重新导入此文件时，不会再次执行这段代码，从而避免了无限递归。
        time_result = Eval.Evaluate()

        print(f"最终结果: {time_result} us")
    else:
        print("Error: Graph capture failed, cannot evaluate.")