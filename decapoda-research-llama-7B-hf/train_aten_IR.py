import torch
import torch.nn as nn
import torch.optim as optim
import io
import sys

# 从 transformers 库导入 Llama 相关的类
from transformers import LlamaConfig, LlamaForCausalLM

# 假设 Dist_IR 是您本地的库
import Dist_IR 
from Dist_IR.Optim_IR import RMSprop_Optimizer

# --- 新增部分：模型包装器 ---
# 这个包装器的作用是“清理”模型的输出，使其符合 torch.export 的要求
class LlamaWrapper(nn.Module):
    def __init__(self, llama_model):
        super().__init__()
        self.llama_model = llama_model

    def forward(self, input_ids):
        # 1. 调用原始模型，但强制它返回一个元组 (tuple) 而不是自定义对象
        #     这是通过 `return_dict=False` 实现的
        outputs = self.llama_model(input_ids=input_ids, return_dict=False)
        
        # 2. 对于 Llama 的输出元组，第一个元素就是我们需要的 logits
        logits = outputs[0]
        
        # 3. 只返回纯粹的 torch.Tensor，这样 torch.export 就能处理了
        return logits

# --- 1. 模型配置和初始化 ---
print("Initializing Llama-7B model from configuration...")
# 从本地加载，避免网络问题
local_model_path = 'decapoda_research_llama_7B_hf_local'  # <-- 确保这个路径是正确的
config = LlamaConfig.from_pretrained(local_model_path)
model = LlamaForCausalLM(config) # 权重是随机的

# 检查 CUDA 是否可用
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu") # 警告：Llama-7B 在 CPU 上会非常慢
model.to(device)
print(f"Model moved to {device}.")


# --- 新增部分：实例化包装器 ---
# 我们将把这个 wrapped_model 传给 Dist_IR 进行图捕获
wrapped_model = LlamaWrapper(model).to(device)


# --- 2. 生成随机样本数据 ---
batch_size = 4
max_seq_length = 100
vocab_size = config.vocab_size
input_ids = torch.randint(0, vocab_size, (batch_size, max_seq_length), device=device)

# --- 3. 设置损失函数和优化器 ---
criterion = nn.CrossEntropyLoss(ignore_index=-100)
# 优化器仍然作用于原始模型的参数上
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

wrapped_model.train()

# --- 4. 使用 Dist_IR 捕获计算图 ---
print("Capturing model graph with Dist_IR...")
# 关键改动：传入 wrapped_model 而不是原始的 model
graph_capture = Dist_IR.GraphCapture(wrapped_model, input_ids)
compiled_model = graph_capture.compile()
print("Graph capture and compilation complete.")

# --- 5. 训练和优化流程 ---
for epoch in range(1):
    print(f"Starting Epoch: {epoch+1}")
    
    # 前向传播
    output = compiled_model(input_ids)
    
    # 关键改动：现在 output 直接就是 logits 张量
    logits = output

    # 计算损失
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    loss = criterion(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
    
    # 反向传播
    loss.backward()
    
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

    # 使用 Dist_IR 的自定义优化器
    print("Applying custom RMSprop optimizer...")
    # 优化器仍然作用于原始模型的参数上
    dist_optimizer = RMSprop_Optimizer(list(model.parameters()), lr=0.0001)
    optim_graph_capture = Dist_IR.OptimGraphCapture(dist_optimizer)
    compiled_optimizer = optim_graph_capture.compile()
    
    # 执行优化步骤
    compiled_optimizer(list(model.parameters()))
    print("Optimization step complete.")


# --- 6. 保存计算图 ---
print("Saving forward and backward graph modules...")
with io.StringIO() as buf, open('aten_module_FW.md', 'w') as f:
    sys.stdout = buf
    graph_capture.FW_gm.print_readable()
    sys.stdout = sys.__stdout__
    f.write(buf.getvalue())

with io.StringIO() as buf, open('aten_module_BW.md', 'w') as f:
    sys.stdout = buf
    graph_capture.BW_gm.print_readable()
    sys.stdout = sys.__stdout__
    f.write(buf.getvalue())
print("Graphs saved to aten_module_FW.md and aten_module_BW.md.")


# --- 7. Dist_IR 定位与 Pass ---
print("Running positioning analysis...")
# 注意：Pos 分析等后续步骤可能需要传入原始模型以正确访问参数和模块
pos = Dist_IR.Pos(model, graph_capture.FW_gm, graph_capture.BW_gm)
pos.positioning_for_graph()
pos.analyze_and_tag_graphs(model, graph_capture.FW_gm, graph_capture.BW_gm)

graph_capture.FW_gm = pos.FW_gm
graph_capture.BW_gm = pos.BW_gm
print("Positioning analysis complete.")

print("Applying Hybrid_Parallel_pass...")
Dist_IR.Hybrid_Parallel_pass(graph_capture.FW_gm, graph_capture.BW_gm, optim_graph_capture.OPT_gm, batch_size)
print("Hybrid_Parallel_pass applied successfully.")
