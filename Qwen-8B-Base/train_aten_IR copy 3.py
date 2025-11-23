import torch
import torch.nn as nn
import torch.optim as optim
import io
import sys

# 从 transformers 库导入 Qwen2 相关的类
# ！！！！！！ 1. 修改导入 ！！！！！！
from transformers import AutoConfig, AutoModelForCausalLM

# 假设 Dist_IR 是您本地的库
import Dist_IR 
from Dist_IR.Optim_IR import RMSprop_Optimizer
# from Performance_Eval.Fake_Runtime.Simulation import Performance_Evaluation
# from input_need import run_benchmark_from_hybridpass
# # 1. ！！！！！！在这里导入 dynamo ！！！！！！
# import torch._dynamo

# --- 2. ！！！！！！模型包装器 (重命名) ！！！！！！ ---
# 这个包装器的作用是“清理”模型的输出，使其符合 torch.export 的要求
# 这个逻辑对于 Qwen2ForCausalLM (当 return_dict=False 时) 同样适用
class LMHeadModelWrapper(nn.Module):
    def __init__(self, model): # 改为通用的 'model'
        super().__init__()
        self.model = model # 改为 self.model

    def forward(self, input_ids):
        # 1. 调用原始模型，但强制它返回一个元组 (tuple)
        #    这是通过 `return_dict=False` 实现的
        outputs = self.model(input_ids=input_ids, return_dict=False) # 使用 self.model
        
        # 2. 对于 Qwen2 的输出元组，第一个元素同样是 logits
        logits = outputs[0]
        
        # 3. 只返回纯粹的 torch.Tensor，这样 torch.export 就能处理了
        return logits

# --- 3. ！！！！！！模型配置和初始化 (修改为 Qwen) ！！！！！！ ---
print("Initializing Qwen model from configuration...")

# ！！！！！！！！！！！！ 关键修改 ！！！！！！！！！！！！
# 将 'Qwen/Qwen2-7B-Base' 替换为您本地的 Qwen3-8B 模型路径
#
# 注意：'Qwen3-8B' 可能不是一个标准的 Hugging Face 标识符。
# 请确保您有这个模型的本地副本。
model_path_or_identifier = './Qwen_8B_Base_local' # ！！！！！！ <--- 已修改为 Qwen3-8B 路径 ！！！！！！


# 尝试从路径加载配置
config = AutoConfig.from_pretrained(model_path_or_identifier)
print(f"Successfully loaded config from: {model_path_or_identifier}")

model = AutoModelForCausalLM.from_config(config)

# 检查 CUDA 是否可用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu") # 保持 CPU，因为 8B/7B 模型很大
model.to(device)
print(f"Model moved to {device}.")


# --- 4. ！！！！！！实例化包装器 (使用新名称) ！！！！！！ ---
# 我们将把这个 wrapped_model 传给 Dist_IR 进行图捕获
wrapped_model = LMHeadModelWrapper(model).to(device)


# --- 5. ！！！！！！生成随机样本数据 (调整大小) ！！！！！！ ---

# ！！！！！！ 注意 ！！！！！！
# Qwen-8B 模型比 GPT-2 大得多，可能需要大量内存。
# 如果遇到 OOM (Out-of-Memory) 错误，请尝试进一步减小 'batch_size' 或 'max_seq_length'。
batch_size = 2      # 从 4 降低到 2
max_seq_length = 64 # 从 100 降低到 64
vocab_size = config.vocab_size # 这将自动使用 Qwen 的 vocab_size

print(f"Using vocab_size: {vocab_size} (from config)")
print(f"Using batch_size: {batch_size}, max_seq_length: {max_seq_length}")
input_ids = torch.randint(0, vocab_size, (batch_size, max_seq_length), device=device)

# --- 6. 设置损失函数和优化器 ---
# (这部分无需更改，对于新模型仍然有效)
criterion = nn.CrossEntropyLoss(ignore_index=-100)
# 优化器仍然作用于原始模型的参数上
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

wrapped_model.train()


# # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
# # ！！！！！！        在这里插入 explain 诊断         ！！！！！！
# # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
# print("="*60)
# print("Running torch._dynamo.explain() to find graph breaks...")
#
# # 2. 调用 explain
# explain_output = torch._dynamo.explain(wrapped_model, input_ids)
#
# # 3. 打印详细报告
# print(explain_output)
# print("="*60)
# print("Explain complete. Continuing with normal script execution...")
# # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
# # ！！！！！！          诊断代码结束            ！！！！！！
# # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！


# --- 7. 使用 Dist_IR 捕获计算图 ---
# (这部分无需更改，它作用于 wrapped_model)
print("Capturing model graph with Dist_IR...")
# 关键改动：传入 wrapped_model 而不是原始的 model
graph_capture = Dist_IR.GraphCapture(wrapped_model, input_ids)
compiled_model = graph_capture.compile()
print("Graph capture and compilation complete.")

# ... 你的模型和数据加载代码 ...
print("Graph capture and compilation complete.")

# 关键改动 1：在这里（循环外）初始化标准优化器
# (这部分无需更改)
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0001)
# 或者使用 Adam，就像你之前注释掉的那样
# optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
print("Using standard PyTorch optimizer.")


# --- 8. 训练和优化流程 ---
# (这部分逻辑是模型无关的，无需更改)
for epoch in range(1):
    print(f"Starting Epoch: {epoch+1}")
    
    # 前向传播
    output = compiled_model(input_ids)
    
    # 关键改动：现在 output 直接就是 logits 张量，不再需要 .logits
    logits = output

    # 计算损失
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    
    # 在计算损失之前，清空旧的梯度（标准做法）
    optimizer.zero_grad() 
    
    loss = criterion(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
    
    # 反向传播
    loss.backward()
    
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

    # 关键改动 2：替换掉整个 Dist_IR 优化器编译块
    print("Applying standard optimizer step...")
    
    # 执行优化步骤
    optimizer.step()
    
    print("Optimization step complete.")

    # -----------------------------------------------------------------
    # 警告：下面的代码现在会失败，因为 optim_graph_capture 不再存在
    # -----------------------------------------------------------------
    # print("Applying Hybrid_Parallel_pass...")
    # hybridpass=Dist_IR.Hybrid_Parallel_pass(graph_capture.FW_gm, graph_capture.BW_gm, optim_graph_capture.OPT_gm, batch_size)
    # ...
    # -----------------------------------------------------------------

# --- 9. 保存计算图 ---
# (这部分无需更改)
print("Saving forward and backward graph modules...")
with io.StringIO() as buf, open('aten_module_FW_after.md', 'w') as f:
    sys.stdout = buf
    graph_capture.FW_gm.print_readable()
    sys.stdout = sys.__stdout__
    f.write(buf.getvalue())

with io.StringIO() as buf, open('aten_module_BW_after.md', 'w') as f:
    sys.stdout = buf
    graph_capture.BW_gm.print_readable()
    sys.stdout = sys.__stdout__
    f.write(buf.getvalue())
print("Graphs saved to aten_module_FW.md and aten_module_BW.md.")


# # --- 10. Dist_IR 定位与 Pass ---
# # (这部分无需更改，它作用于捕获的图)
# print("Running positioning analysis...")
# # 注意：Pos 分析等后续步骤可能需要传入原始模型以正确访问参数和模块
# pos = Dist_IR.Pos(model, graph_capture.FW_gm, graph_capture.BW_gm)
# pos.positioning_for_graph()
# pos.analyze_and_tag_graphs(model, graph_capture.FW_gm, graph_capture.BW_gm)

# graph_capture.FW_gm = pos.FW_gm
# graph_capture.BW_gm = pos.BW_gm
# print("Positioning analysis complete.")

# print("Applying Hybrid_Parallel_pass...")
# hybridpass=Dist_IR.Hybrid_Parallel_pass(graph_capture.FW_gm, graph_capture.BW_gm, graph_capture.BW_gm, batch_size)
# print("Hybrid_Parallel_pass applied successfully.")

# dist_ir = {"FW": fw_gm, "BW": bw_gm, ...}  # 你已有
# 直接调用（固定索引）：
# text = run_benchmark_from_hybridpass(hybridpass, md_path="fw_inputs_report.md")
# run_benchmark_from_hybridpass(hybridpass)
# print("已导出：fw_inputs_report.md")


# Eval = Performance_Evaluation(hybridpass)
# print(Eval.Evaluate())