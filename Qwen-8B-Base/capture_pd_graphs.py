import torch
import torch.nn as nn
import io
import sys
import os
import shutil
from torch.fx.passes.graph_drawer import FxGraphDrawer
from transformers import AutoConfig
from transformers import Qwen2Config  # 显式导入 Config 类

# --- 引入修改后的本地模型文件 ---
# 1. 确保能从 transformers 导入标准 Config 类
from transformers import Qwen2Config 

# 2. 引入你的本地模型
# 只要 modeling_qwen_custom.py 按照步骤 1 改好了，这行就不会报错
sys.path.append(os.getcwd())
try:
    from modeling_qwen3_custom import Qwen3ForCausalLM
except ImportError as e:
    # 打印详细错误，方便调试
    print(f"导入失败细节: {e}")
    sys.exit("错误：无法导入 modeling_qwen_custom。请检查步骤1是否执行正确。")
# --- 假设您的库中包含了这些 ---
import Dist_IR
from Dist_IR import InferenceGraphCapture, logger, KVCacheAnalyzer

# ==============================================================================
# 1. 辅助工具函数
# ==============================================================================

def save_inference_graph(graph_module, graph_name: str):
    """保存计算图为 MD 和 DOT 文件 (保持不变)"""
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
    【关键修改】针对修改后的 Qwen 模型，创建扁平化的 List[Tensor]。
    结构: [layer0_k, layer0_v, layer1_k, layer1_v, ...]
    """
    num_hidden_layers = config.num_hidden_layers
    num_key_value_heads = config.num_key_value_heads
    # 如果 config 中没有 head_dim，手动计算
    head_dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)
    
    flat_cache = []
    for _ in range(num_hidden_layers):
        # Shape: [Batch, KV_Heads, Seq_Len, Dim]
        k = torch.randn((batch_size, num_key_value_heads, seq_len, head_dim), device=device, dtype=torch.float32) # export通常建议fp32 tracing
        v = torch.randn((batch_size, num_key_value_heads, seq_len, head_dim), device=device, dtype=torch.float32)
        flat_cache.append(k)
        flat_cache.append(v)
        
    return flat_cache

# ==============================================================================
# 2. 模型包装器 (Wrapper)
# ==============================================================================
class QwenWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        # 为了 Dist_IR 的 Pos 类能找到层，建议保持 self.model 命名
        # Qwen 原始结构通常是 model.model.layers
        self.model = model 

    def forward(self, input_ids, attention_mask, past_key_values=None):
        # 调用修改后的 Qwen forward
        # 注意：我们修改后的 forward 签名是明确的 Tensor 输入
        logits, new_cache = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True # 强制开启 cache
        )
        return logits, new_cache

# ==============================================================================
# 3. 配置与初始化
# ==============================================================================
MODEL_NAME = "qwen"
LOCAL_CONFIG_PATH = './Qwen_8B_Base_local'  # 替换为你实际的路径

DEVICE =  "cpu"

BATCH_SIZE = 10
PREFILL_SEQ_LEN = 30
KV_CACHE_SEQ_LEN = PREFILL_SEQ_LEN 

print(f"\n>> 初始化模型: Custom Qwen (Inference Mode) on {DEVICE}")

try:
    # 1. 加载配置 (瘦身版用于 Debug)
    config = Qwen2Config.from_pretrained(LOCAL_CONFIG_PATH)
    
    print(">> Overriding config to create a TINY model...")
    config.hidden_size = 512               
    config.intermediate_size = 1376         
    config.num_hidden_layers = 2 # 方便看图          
    config.num_attention_heads = 8         
    config.num_key_value_heads = 8 #GQA (Grouped Query Attention) 每几个Q共享一个Key和Value头，以节省显存   
    config.max_position_embeddings = 512
    # 强制 eager，避免 FlashAttn 引入的 CUDA 算子导致 Export 困难
    config.attn_implementation = "eager"
    config._attn_implementation = "eager"
    
    keys_to_clean = ['head_dim', 'kv_channels', 'n_head_kv'] # Qwen 常用字段
    for k in keys_to_clean:
        if hasattr(config, k):
            print(f"   [Config Clean] Removing stale key: {k} (Value: {getattr(config, k)})")
            delattr(config, k)

    # 2. 实例化修改后的模型类
    # 注意：这里不用 AutoModel，直接用我们 import 进来的类
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

with torch.no_grad():

    # --- A. Prefill 阶段 ---
    print("\n" + "="*40)
    print("Step 1: 捕获 Prefill 图")
    
    # 构造 Prefill Mask (下三角)
    # Shape: [Batch, 1, Seq, Seq] -> [1, 1, 32, 32]
    _mask = torch.tril(torch.ones((PREFILL_SEQ_LEN, PREFILL_SEQ_LEN), device=DEVICE))
    prefill_mask = _mask[None, None, :, :]
    
    prefill_inputs = { 
        'input_ids': torch.randint(0, config.vocab_size, (BATCH_SIZE, PREFILL_SEQ_LEN), device=DEVICE),
        'attention_mask': prefill_mask,
        'past_key_values': None # Prefill 初始无 cache
    }
    
    # 注意：InferenceGraphCapture 需要适配 QwenWrapper 的参数名
    prefill_capture = InferenceGraphCapture(model, filename_prefix="prefill", **prefill_inputs)
    compiled_prefill = prefill_capture.compile()
    # 试运行
    compiled_prefill(**prefill_inputs)
    
    # --- B. Decode 阶段 ---
    print("\n" + "="*40)
    print("Step 2: 捕获 Decode 图")
    
    # 构造 Decode Mask (全1)
    # Decode 时，input_len=1, past_len=32. Total=33.
    # Qwen 的 Attention 通常需要看到之前的 token。
    # Mask Shape 应该是: [Batch, 1, 1, Total_Len]
    total_len = KV_CACHE_SEQ_LEN + 1
    decode_mask = torch.ones((BATCH_SIZE, 1, 1, total_len), device=DEVICE)

    # 构造 Dummy Cache (List[Tensor])
    dummy_cache = create_flat_kv_cache(config, BATCH_SIZE, KV_CACHE_SEQ_LEN, device=DEVICE)
    
    decode_inputs = { 
        'input_ids': torch.randint(0, config.vocab_size, (BATCH_SIZE, 1), device=DEVICE),
        'attention_mask': decode_mask, 
        'past_key_values': dummy_cache
    }
    
    decode_capture = InferenceGraphCapture(model, filename_prefix="decode", **decode_inputs)
    compiled_decode = decode_capture.compile()
    # 试运行
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

    # 这一步 Analyzer 可能需要适配，因为它可能会去检查 inputs[1] 是否是 tuple
    # 由于我们改成了 List，KVCacheAnalyzer 内部可能需要留意一下（不过如果不报错就先跑着）
    Dist_IR.KVCacheAnalyzer(prefill_capture.FW_gm, "prefill", MODEL_NAME, config)

    # 定位
    # 注意：QwenWrapper.model 是 Qwen3ForCausalLM，它的主体是 .model
    # 这里的 raw_model 就是 Qwen3ForCausalLM
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

print("\nDone.")

from Performance_Eval.Fake_Runtime.PD_separation import PD_Performance_Eval

Eval = PD_Performance_Eval(prefill_capture.FW_gm, decode_capture.FW_gm, prefill_graph, decode_graph, PREFILL_SEQ_LEN, 5)
time = Eval.Evaluate()

print(f"最终结果: {time} us")