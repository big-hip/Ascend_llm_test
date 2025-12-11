import torch
import torch.nn as nn
import io
import sys
import os
import shutil
from torch.fx.passes.graph_drawer import FxGraphDrawer
from transformers import LlamaConfig, LlamaForCausalLM

# --- 假设您的库中包含了这些 ---
import Dist_IR
from Dist_IR import InferenceGraphCapture, logger, KVCacheAnalyzer
import multiprocessing as mp # 引入 multiprocessing
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
    # 兼容 Llama 1 (Decapoda) 和 Llama 2 (GQA)
    # 如果配置中没有 num_key_value_heads，则默认等于 num_attention_heads
    num_heads = getattr(config, 'num_key_value_heads', config.num_attention_heads)
    head_dim = config.hidden_size // config.num_attention_heads
    num_layers = config.num_hidden_layers
    
    kv_cache = []
    for _ in range(num_layers):
        # Llama cache shape: [batch, n_kv_heads, seq_len, head_dim]
        k = torch.randn((batch_size, num_heads, seq_len, head_dim), device=device, dtype=torch.float16)
        v = torch.randn((batch_size, num_heads, seq_len, head_dim), device=device, dtype=torch.float16)
        kv_cache.append((k, v))
    return tuple(kv_cache)

# ==============================================================================
# 2. 模型包装器 (Wrapper) - 核心组件
# ==============================================================================
class LlamaWrapper(nn.Module):
    """
    Wrapper 的两个核心作用：
    1. 清洗输出：只返回 (logits, kv_cache) 元组，去除非 Tensor 的 dict 输出，利于图捕获。
    2. 对齐命名：将内部属性命名为 self.llama_model，使其 Stack Trace 与训练代码一致。
    """
    def __init__(self, model):
        super().__init__()
        # 【关键】命名为 llama_model 以适应 Positioning Pass 的正则匹配
        self.llama_model = model 

    def forward(self, input_ids, past_key_values=None):
        outputs = self.llama_model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=False # 强制返回 Tuple
        )
        # outputs[0]: logits
        # outputs[1]: past_key_values
        return outputs[0], outputs[1]

# ==============================================================================
# 3. 配置与初始化 (Tiny Mode)
# ==============================================================================
if __name__ == "__main__":
    # --- 关键：设置 spawn 启动方式 ---
    try:
        mp.set_start_method('spawn', force=True)
        print("[Main] Multiprocessing start method set to 'spawn'.")
    except RuntimeError as e:
        print(f"[Warn] Cannot set start method: {e}")
    MODEL_NAME = "decapoda_research_llama"
    LOCAL_CONFIG_PATH = './decapoda_research_llama_7B_hf_local'

    if not os.path.exists(LOCAL_CONFIG_PATH):
        sys.exit(f"错误：找不到模型路径 '{LOCAL_CONFIG_PATH}'。")

    gpu_index = 2
    DEVICE = f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 8 
    PREFILL_SEQ_LEN = 128 # 缩短以便快速调试
    KV_CACHE_SEQ_LEN = PREFILL_SEQ_LEN 

    print(f"\n>> 初始化模型: {MODEL_NAME} (Tiny Inference Mode) on {DEVICE}")

    try:
        # 加载配置
        config = LlamaConfig.from_pretrained(LOCAL_CONFIG_PATH)
        
        # ==========================================================================
        # [Tiny Mode] 修改配置以适配显存和调试
        # ==========================================================================
        config.num_hidden_layers = 2 
        config.hidden_size = 512
        config.num_attention_heads = 8 
        
        # Decapoda Llama 原配置没有 num_key_value_heads。
        # 当我们修改了 num_attention_heads 为 8 时，必须显式设置 num_key_value_heads 也为 8。
        # 否则它会默认保持原有的 32，导致 Broadcasting 形状不匹配错误。
        config.num_key_value_heads = 8 
        
        # 匹配 TP 策略 Priority 0 (512 * 2.6875 = 1376)
        config.intermediate_size = 1376 
        
        # 实例化原始模型 (随机权重)
        raw_model = LlamaForCausalLM(config).to(DEVICE)
        # 包装模型
        model = LlamaWrapper(raw_model).to(DEVICE)
        model.eval() # 必须设为 eval
        
        print(f"   Config: Layers={config.num_hidden_layers}, Hidden={config.hidden_size}, "
            f"Inter={config.intermediate_size}, Heads={config.num_attention_heads}")

    except Exception as e:
        sys.exit(f"模型初始化失败: {e}")

    # ==============================================================================
    # 4. 图捕获 (Graph Capture)
    # ==============================================================================

    # 全局禁用梯度，确保不生成反向图
    with torch.no_grad():

        # --- A. Prefill 阶段 ---
        print("\n" + "="*40)
        print("Step 1: 捕获 Prefill 图 (Prompt Phase)")
        print("="*40)
        
        prefill_inputs = { 
            'input_ids': torch.randint(0, config.vocab_size, (BATCH_SIZE, PREFILL_SEQ_LEN), device=DEVICE),  
            'past_key_values': None
        }
        
        prefill_capture = InferenceGraphCapture(model, filename_prefix="prefill", **prefill_inputs)
        compiled_prefill = prefill_capture.compile()
        compiled_prefill(**prefill_inputs) # 触发 Trace
        
        print("Prefill 图捕获成功。")

        # --- B. Decode 阶段 ---
        print("\n" + "="*40)
        print("Step 2: 捕获 Decode 图 (Generation Phase)")
        print("="*40)
        
        # 构造 Dummy KV Cache
        dummy_cache = create_dummy_kv_cache(config, BATCH_SIZE, KV_CACHE_SEQ_LEN, device=DEVICE)
        
        decode_inputs = { 
            'input_ids': torch.randint(0, config.vocab_size, (BATCH_SIZE, 1), device=DEVICE),  
            'past_key_values': dummy_cache
        }
        
        decode_capture = InferenceGraphCapture(model, filename_prefix="decode", **decode_inputs)
        compiled_decode = decode_capture.compile()
        compiled_decode(**decode_inputs) # 触发 Trace
        
        print("Decode 图捕获成功。")

    # ==============================================================================
    # 5. 分析与并行切分 (Dist_IR Passes)
    # ==============================================================================

    print("\n" + "="*40)
    print("Step 3: 运行 Dist_IR Pass (Positioning & Hybrid Parallel)")
    print("="*40)

    # --- A. 处理 Prefill ---
    print(">> Processing Prefill Graph...")
    if prefill_capture.FW_gm:
        # 1. 成本分析
        Dist_IR.KVCacheAnalyzer(prefill_capture.FW_gm, "prefill", MODEL_NAME, config)
        
        # 2. 定位 Pass (Positioning)
        # 传入 raw_model 以便访问 named_parameters
        pos_prefill = Dist_IR.Pos(raw_model, prefill_capture.FW_gm, None, model_name='llama2') # Decapoda 也是 Llama 结构
        pos_prefill.positioning_for_graph()
        pos_prefill.log_node_positioning_info()
        
        # 3. 混合并行 Pass (Hybrid Parallel)
        # 传入 inference_stage 以便生成独立文件夹
        prefill_graph = Dist_IR.Hybrid_Parallel_pass(
            prefill_capture.FW_gm, 
            None, None, 
            Global_batch_size=BATCH_SIZE,
            inference_stage="prefill" 
        )

        # --- B. 处理 Decode ---
        print("\n>> Processing Decode Graph...")
        if decode_capture.FW_gm:
            # 1. 成本分析
            Dist_IR.KVCacheAnalyzer(decode_capture.FW_gm, "decode", MODEL_NAME, config)
            
            # 2. 定位 Pass
            pos_decode = Dist_IR.Pos(raw_model, decode_capture.FW_gm, None, model_name='llama2')
            pos_decode.positioning_for_graph()
            
            # 3. 混合并行 Pass
            Dist_IR.Hybrid_Parallel_pass(
                decode_capture.FW_gm, 
                None, None, 
                Global_batch_size=BATCH_SIZE,
                inference_stage="decode"
            )

        print("\n" + "="*50)
        print("任务完成！请检查 'prefill/Dist_IR' 和 'decode/Dist_IR' 文件夹。")
        print("="*50)
        from Performance_Eval.Fake_Runtime.Simulation import Performance_Evaluation
        Eval = Performance_Evaluation(prefill_graph)
        time = Eval.Evaluate()
        print(f"最终结果: {time} us")