import torch
import torch.nn as nn
import torch.optim as optim
import io
import sys

# 从 transformers 库导入 GPT-2 相关的类
from transformers import GPT2Config, GPT2LMHeadModel

# 假设 Dist_IR 是您本地的库
import Dist_IR 
from Dist_IR.Optim_IR import RMSprop_Optimizer
from Performance_Eval.Fake_Runtime.Simulation import Performance_Evaluation
# from input_need import run_benchmark_from_hybridpass
# # 1. ！！！！！！在这里导入 dynamo ！！！！！！
# import torch._dynamo
# --- 新增部分：模型包装器 ---
# 这个包装器的作用是“清理”模型的输出，使其符合 torch.export 的要求
class GPT2Wrapper(nn.Module):
    def __init__(self, gpt2_model):
        super().__init__()
        self.gpt2_model = gpt2_model

    def forward(self, input_ids):
        # 1. 调用原始模型，但强制它返回一个元组 (tuple) 而不是自定义对象
        #    这是通过 `return_dict=False` 实现的
        outputs = self.gpt2_model(input_ids=input_ids, return_dict=False)
        
        # 2. 对于 GPT-2 的输出元组，第一个元素就是我们需要的 logits
        logits = outputs[0]
        
        # 3. 只返回纯粹的 torch.Tensor，这样 torch.export 就能处理了
        return logits

# --- 1. 模型配置和初始化 ---
print("Initializing GPT-2 model from configuration...")
# 从本地加载，避免网络问题
local_model_path = './gpt2-local' 
config = GPT2Config.from_pretrained(local_model_path)
model = GPT2LMHeadModel(config) # 权重是随机的

# 检查 CUDA 是否可用
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
model.to(device)
print(f"Model moved to {device}.")


# --- 新增部分：实例化包装器 ---
# 我们将把这个 wrapped_model 传给 Dist_IR 进行图捕获
wrapped_model = GPT2Wrapper(model).to(device)


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


# # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
# # ！！！！！！        在这里插入 explain 诊断        ！！！！！！
# # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
# print("="*60)
# print("Running torch._dynamo.explain() to find graph breaks...")

# # 2. 调用 explain，它需要你的 Eager 模型和示例输入
# #    它会模拟完整的编译过程（包括 aot_autograd）并打印报告
# explain_output = torch._dynamo.explain(wrapped_model, input_ids)

# # 3. 打印详细报告
# print(explain_output)
# print("="*60)
# print("Explain complete. Continuing with normal script execution...")
# # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
# # ！！！！！！            诊断代码结束            ！！！！！！
# # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！


# --- 4. 使用 Dist_IR 捕获计算图 ---
print("Capturing model graph with Dist_IR...")
# 关键改动：传入 wrapped_model 而不是原始的 model
graph_capture = Dist_IR.GraphCapture(wrapped_model, input_ids)
compiled_model = graph_capture.compile()
print("Graph capture and compilation complete.")

# ... 你的模型和数据加载代码 ...
print("Graph capture and compilation complete.")

# 关键改动 1：在这里（循环外）初始化标准优化器
# 注意：我们仍然将原始 model.parameters() 传给它
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0001)
# 或者使用 Adam，就像你之前注释掉的那样
# optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
print("Using standard PyTorch optimizer.")


# --- 5. 训练和优化流程 ---
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
# --- 6. 保存计算图 ---
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
hybridpass=Dist_IR.Hybrid_Parallel_pass(graph_capture.FW_gm, graph_capture.BW_gm, graph_capture.BW_gm, batch_size)
print("Hybrid_Parallel_pass applied successfully.")

# dist_ir = {"FW": fw_gm, "BW": bw_gm, ...}  # 你已有
# 直接调用（固定索引）：
# text = run_benchmark_from_hybridpass(hybridpass, md_path="fw_inputs_report.md")
# run_benchmark_from_hybridpass(hybridpass)
# print("已导出：fw_inputs_report.md")


Eval = Performance_Evaluation(hybridpass)
print(Eval.Evaluate())


