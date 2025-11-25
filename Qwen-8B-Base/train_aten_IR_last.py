import torch
import torch.nn as nn
import io
import sys

from transformers import AutoConfig, AutoModelForCausalLM
import Dist_IR 

# --- LossWrapper 保持不变 ---
class LossWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, input_ids, attention_mask, labels):
        out = self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            labels=labels, 
            use_cache=False, 
            return_dict=True
        )
        return out.loss

# --- 模型初始化 ---
print("Initializing Qwen model from configuration...")
model_path_or_identifier = './Qwen_8B_Base_local' 


config = AutoConfig.from_pretrained(model_path_or_identifier)
# except OSError:
#     print("Local path not found, using default Qwen/Qwen2-7B-beta (example)...")
#     config = AutoConfig.from_pretrained("Qwen/Qwen2-7B-beta")


# ================= 🛠️ 强制瘦身区域 =================
print("📉 Overriding config to create a TINY model for debugging...")
config.hidden_size = 128               # 原来通常是 4096 -> 变细
config.intermediate_size = 512         # 原来通常是 11008+ -> MLP变小
config.num_hidden_layers = 2           # 原来通常是 32 -> 变浅 (最关键！极大减少图的长度)
config.num_attention_heads = 4         # 原来通常是 32 -> 头数减少
config.num_key_value_heads = 2         # 保持 GQA 比例 (可选，设为 num_attention_heads 一样也可以)
config.max_position_embeddings = 128   # 序列长度上限减少
# ====================================================
# config.attn_implementation = "eager"
# config._attn_implementation = "eager"
# config.use_cache = True 
config.output_hidden_states = False
config.output_attentions = False

if hasattr(config, "sliding_window"):
    config.sliding_window = None
if hasattr(config, "window_size"):
    config.window_size = None

model = AutoModelForCausalLM.from_config(config)
device = torch.device("cpu") 
model.to(device)
model.train() 

wrapped_model = LossWrapper(model).to(device)

# --- 数据准备 ---
batch_size = 1
max_seq_length = 4
vocab_size = config.vocab_size
input_ids = torch.randint(0, vocab_size, (batch_size, max_seq_length), device=device)
labels = input_ids.clone()

# --- 4D Mask 构建 ---
causal_mask_bool = torch.tril(torch.ones((max_seq_length, max_seq_length), device=device))
causal_mask_bool = causal_mask_bool.view(1, 1, max_seq_length, max_seq_length).expand(batch_size, -1, -1, -1)
dtype = model.dtype if model.dtype is not None else torch.float32
min_value = torch.finfo(torch.float32).min if dtype == torch.float32 else -1e4
attention_mask = torch.zeros(causal_mask_bool.shape, dtype=dtype, device=device)
attention_mask = attention_mask.masked_fill(causal_mask_bool == 0, min_value)

# --- 优化器与编译 ---
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0001)
print("Capturing model graph with Dist_IR...")

graph_capture = Dist_IR.GraphCapture(wrapped_model, input_ids, attention_mask, labels)

try:
    compiled_model = graph_capture.compile()
    print("Graph capture and compilation complete.")
except Exception as e:
    print(f"Compilation failed: {e}")
    sys.exit(1)

# --- 训练循环 ---
loss = compiled_model(input_ids, attention_mask, labels)
optimizer.zero_grad()
loss.backward()
optimizer.step()
print("Optimization step complete.")

# --- 🛠️ 修改部分: 使用 Dist_IR.FxGraphDrawer 保存 ---
print("Saving forward and backward graph modules...")



# def save_graph_with_drawer(gm, filename):
#     try:
#         # 2. 实例化 Drawer
#         g = str(gm.graph)
        
#         # 3. 写入文件 (假设 str(g) 返回图的内容)
#         with open(filename, 'w') as f:
#             f.write(g)
#         print(f"Saved {filename}")
#     except Exception as e:
#         print(f"Failed to save {filename}: {e}")
#         import traceback
#         traceback.print_exc()

from torch.fx.passes.graph_drawer import FxGraphDrawer

def save_dot_via_stdout(gm, filename, mode='w'):
    """
    完全复刻你提供的逻辑：
    通过劫持 sys.stdout 来捕获 FxGraphDrawer 的输出，并写入文件。
    """
    print(f"Saving DOT to {filename} via stdout capture...")
    
    try:
        # 1. 实例化 Drawer
        # ignore_getattr=True 能让图更简洁
        g = FxGraphDrawer(gm, 'Qwen_Graph', ignore_getattr=True)
        
        # 2. 使用 StringIO 捕获 print 输出
        with io.StringIO() as buf:
            original_stdout = sys.stdout  # 备份原来的 stdout
            sys.stdout = buf              # 劫持 stdout 指向 buffer
            
            # print 会调用 pydot 对象的 __str__ 方法，将 DOT 内容输出到 buf
            print(g.get_dot_graph())
            
            sys.stdout = original_stdout  # 恢复原来的 stdout
            output = buf.getvalue()       # 获取捕获的字符串内容

        # 3. 写入文件
        # 注意：你原本的代码用了 'a' (追加模式)，但在单次脚本运行中
        # 为了避免文件内容重复堆叠，这里我默认设为 'w' (覆盖模式)。
        # 如果你确实需要追加，调用时传入 mode='a' 即可。
        with open(filename, mode) as file:
            file.write(output)
            
        print(f"✅ Saved successfully: {len(output)} characters written.")

    except Exception as e:
        # 恢复 stdout 以防出错后终端没有任何输出
        if sys.stdout != sys.__stdout__:
            sys.stdout = sys.__stdout__
        print(f"❌ Failed to save {filename}: {e}")
        import traceback
        traceback.print_exc()
# 保存 FW 和 BW
save_dot_via_stdout(graph_capture.FW_gm, 'aten_module_FW_after.md')
save_dot_via_stdout(graph_capture.BW_gm, 'aten_module_BW_after.md')


print("Graphs saved successfully.")