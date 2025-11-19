import torch
import torch.nn as nn
import io
import sys
import os

# --- 从 transformers 库导入 Llama-2 相关的类 ---
from transformers import LlamaConfig, LlamaForCausalLM

# 假设 Dist_IR 是您本地的库
import Dist_IR 
from Dist_IR.Optim_IR import RMSprop_Optimizer

# --- 模型包装器，简化模型输出 ---
class LlamaWrapper(nn.Module):
    def __init__(self, llama_model):
        super().__init__()
        self.llama_model = llama_model

    def forward(self, input_ids):
        # 调用原始 Llama 模型，强制返回元组
        # use_cache=False 在训练和图捕获时是必要的，可以避免不必要的输出
        outputs = self.llama_model(input_ids=input_ids, return_dict=False, use_cache=False)
        
        # Llama 模型的输出元组中，第一个元素就是 logits
        logits = outputs[0]
        
        # 只返回 torch.Tensor
        return logits

# --- 1. 模型配置和初始化 ---
print("Initializing Llama-2-7b model from configuration...")

local_model_path = './Llama_2_7b_hf_local' 
if not os.path.exists(local_model_path):
    sys.exit(f"错误：找不到模型路径 '{local_model_path}'。请先下载 Llama-2 的配置文件到该目录。")

# 使用 LlamaConfig 加载配置
config = LlamaConfig.from_pretrained(local_model_path)
# 从配置初始化模型（权重是随机的，因为我们只关心计算图）
model = LlamaForCausalLM(config) 

device = torch.device("cpu")
model.to(device)
print(f"Model moved to {device}.")

# --- 实例化新的包装器 ---
wrapped_model = LlamaWrapper(model).to(device)
wrapped_model.train()

# --- 2. 生成随机样本数据 ---
batch_size = 4
max_seq_length = 100
vocab_size = config.vocab_size
input_ids = torch.randint(0, vocab_size, (batch_size, max_seq_length), device=device)

# --- 3. 设置损失函数 ---
criterion = nn.CrossEntropyLoss(ignore_index=-100)

# --- 4. Dist_IR 初始化和编译 (移到循环外部以提高效率) ---
# 4.1 捕获前向和反向图
print("Capturing model graph with Dist_IR...")
graph_capture = Dist_IR.GraphCapture(wrapped_model, input_ids)
compiled_model = graph_capture.compile()
print("Graph capture and compilation complete.")

# 4.2 捕获和编译优化器图
print("Compiling custom RMSprop optimizer...")
# 优化器应作用于原始模型的参数上
dist_optimizer = RMSprop_Optimizer(list(model.parameters()), lr=0.0001)
optim_graph_capture = Dist_IR.OptimGraphCapture(dist_optimizer)
compiled_optimizer = optim_graph_capture.compile()
print("Optimizer compilation complete.")


# --- 5. 训练和优化流程 (只运行一次以捕获图) ---
print("\n--- Starting Training Loop to capture graphs ---")
for epoch in range(1): # 仅作演示，只运行一个 epoch
    print(f"Epoch: {epoch+1}")
    
    # 前向传播 (使用已编译的模型)
    logits = compiled_model(input_ids)
    
    # 计算损失
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    
    # ===================================================================
    # [修正] 这里是关键的修复点：将 --1 改为 -1
    # ===================================================================
    loss = criterion(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
    
    # 反向传播 (PyTorch 会自动填充每个参数的 .grad 属性)
    loss.backward()
    
    print(f"Loss: {loss.item()}")

    # 执行已编译的优化步骤
    # compiled_optimizer 会利用 .grad 属性来更新参数
    compiled_optimizer(list(model.parameters()))
    print("Optimization step complete.")
    print("Forward, backward, and optimizer graphs have been captured in memory.")


# --- 6. 执行图分析和转换 Pass (在捕获之后) ---
print("\nRunning positioning analysis and applying passes on captured graphs...")
pos = Dist_IR.Pos(model, graph_capture.FW_gm, graph_capture.BW_gm)
pos.positioning_for_graph()
pos.analyze_and_tag_graphs(model, graph_capture.FW_gm, graph_capture.BW_gm)
graph_capture.FW_gm = pos.FW_gm
graph_capture.BW_gm = pos.BW_gm

Dist_IR.Hybrid_Parallel_pass(graph_capture.FW_gm, graph_capture.BW_gm, optim_graph_capture.OPT_gm, batch_size)
print("Graph analysis and passes applied successfully.")


# --- 7. 保存最终的计算图 (经过Pass修改后的) ---
print("\nSaving final forward and backward graph modules...")
with io.StringIO() as buf, open('llama_aten_module_FW.md', 'w') as f:
    sys.stdout = buf
    graph_capture.FW_gm.print_readable()
    sys.stdout = sys.__stdout__
    f.write(buf.getvalue())

with io.StringIO() as buf, open('llama_aten_module_BW.md', 'w') as f:
    sys.stdout = buf
    graph_capture.BW_gm.print_readable()
    sys.stdout = sys.__stdout__
    f.write(buf.getvalue())
print("Final graphs saved to llama_aten_module_FW.md and llama_aten_module_BW.md.")