import torch
import torch.nn as nn
import torch.optim as optim
import io
import sys
import warnings
# 从 transformers 库导入通用的 AutoConfig 和 AutoModelForCausalLM
from transformers import AutoConfig, AutoModelForCausalLM

# 假设 Dist_IR 是您本地的库
import Dist_IR 
from Dist_IR.Optim_IR import RMSprop_Optimizer
from Performance_Eval.Fake_Runtime.Simulation import Performance_Evaluation
# from input_need import run_benchmark_from_hybridpass
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose = False
# warnings.filterwarnings("ignore", message="Graph break due to unsupported builtin")
# --- 新增部分：模型包装器 ---
# 这个包装器的作用是“清理”模型的输出，使其符合 torch.export 的要求
class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        # 1. 调用原始模型，但强制它返回一个元组 (tuple) 而不是自定义对象
        #    这是通过 `return_dict=False` 实现的
        #    适用于大多数 transformers Causal LM 模型
        outputs = self.model(input_ids=input_ids, return_dict=False)
        
        # 2. 对于 Causal LM 的输出元组，第一个元素就是我们需要的 logits
        logits = outputs[0]
        
        # 3. 只返回纯粹的 torch.Tensor，这样 torch.export 就能处理了
        return logits

# --- 1. 模型配置和初始化 ---
print("Initializing Qwen model from configuration...")

# *** 关键改动：修改为 Qwem_8B_Base_local 路径 ***
local_model_path = './Qwem_8B_Base_local' 

# *** 关键改动：使用 AutoConfig 加载配置 ***
# AutoConfig 会自动识别 config.json 中的 "model_type" 并加载正确的配置类
config = AutoConfig.from_pretrained(local_model_path)

# *** 关键改动：使用 AutoModelForCausalLM 和配置来初始化模型 ***
# .from_config(config) 会创建一个具有随机权重的模型实例
model = AutoModelForCausalLM.from_config(config)

# 检查 CUDA 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
model.to(device)
print(f"Model moved to {device}.")


# --- 新增部分：实例化包装器 ---
# *** 关键改动：使用新的 ModelWrapper ***
# 我们将把这个 wrapped_model 传给 Dist_IR 进行图捕获
# wrapped_model = ModelWrapper(model).to(device)


# --- 2. 生成随机样本数据 ---
batch_size = 4
max_seq_length = 100
# config.vocab_size 会从加载的 config.json 中正确获取词汇表大小
vocab_size = config.vocab_size 
input_ids = torch.randint(0, vocab_size, (batch_size, max_seq_length), device=device)
input_ids.to(device)
# --- 3. 设置损失函数和优化器 ---
criterion = nn.CrossEntropyLoss(ignore_index=-100)
# 优化器仍然作用于原始模型的参数上
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

# wrapped_model.train()

# model.train()
# # --- 4. 使用 Dist_IR 捕获计算图 ---
# print("Capturing model graph with Dist_IR...")
# # 关键改动：传入 wrapped_model
# graph_capture = Dist_IR.GraphCapture(wrapped_model, input_ids)
# compiled_model = graph_capture.compile()
# print("Graph capture and compilation complete.")
from IR_transform import aten_compile_capture
compiled_model = aten_compile_capture(model) 

output=compiled_model(input_ids).logits
output.sum().backward()
# --- 5. 训练和优化流程 ---
# for epoch in range(1):
#     print(f"Starting Epoch: {epoch+1}")
    
#     # 前向传播
#     output = compiled_model(input_ids)
    
#     # 关键改动：现在 output 直接就是 logits 张量，不再需要 .logits
#     logits = output.logits

#     # 计算损失 (适用于 Causal LM 的标准计算方式)
#     shift_logits = logits[..., :-1, :].contiguous()
#     shift_labels = input_ids[..., 1:].contiguous()
#     loss = criterion(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
    
#     # 反向传播
#     loss.backward()
    
#     print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

#     # 使用 Dist_IR 的自定义优化器
#     print("Applying custom RMSprop optimizer...")
#     # 优化器仍然作用于原始模型的参数上
#     dist_optimizer = RMSprop_Optimizer(list(model.parameters()), lr=0.0001)
#     optim_graph_capture = Dist_IR.OptimGraphCapture(dist_optimizer)
#     compiled_optimizer = optim_graph_capture.compile()
    
#     # 执行优化步骤
#     compiled_optimizer(list(model.parameters()))
#     print("Optimization step complete.")
