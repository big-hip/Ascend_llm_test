import torch
import torch.nn as nn
import io
import sys

from transformers import AutoConfig, AutoModelForCausalLM
import Dist_IR 

# --- 修改后的损失包装器 ---
class LossWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, input_ids, labels):
        # Qwen2ForCausalLM 在提供 labels 时会自动计算损失
        out = self.model(input_ids=input_ids, labels=labels, return_dict=True)
        return out.loss

# --- 模型初始化 ---
print("Initializing Qwen model from configuration...")
model_path_or_identifier = './Qwen_8B_Base_local'

config = AutoConfig.from_pretrained(model_path_or_identifier)
print(f"Successfully loaded config from: {model_path_or_identifier}")

model = AutoModelForCausalLM.from_config(config)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu") 
model.to(device)
print(f"Model moved to {device}.")

# --- 使用 LossWrapper 包装 ---
wrapped_model = LossWrapper(model).to(device)

# --- 生成样本数据 ---
batch_size = 1
max_seq_length = 4
vocab_size = config.vocab_size

print(f"Using vocab_size: {vocab_size}")
print(f"Using batch_size: {batch_size}, max_seq_length: {max_seq_length}")

input_ids = torch.randint(0, vocab_size, (batch_size, max_seq_length), device=device)
# ⚠️ 关键：现在需要准备 labels
labels = input_ids.clone()  # 通常使用输入的移位版本作为标签

# --- 优化器 ---
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0001)

wrapped_model.train()

# --- 使用 Dist_IR 捕获计算图 ---
print("Capturing model graph with Dist_IR...")
# ⚠️ 关键：现在需要传入两个参数
graph_capture = Dist_IR.GraphCapture(wrapped_model, input_ids, labels)
compiled_model = graph_capture.compile()
print("Graph capture and compilation complete.")

# --- 训练循环 ---
for epoch in range(1):
    print(f"Starting Epoch: {epoch+1}")
    
    # ⚠️ 关键：调用时传入两个参数
    loss = compiled_model(input_ids, labels)
    
    # 现在直接得到损失，无需手动计算
    optimizer.zero_grad()
    loss.backward()
    
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")
    
    optimizer.step()
    print("Optimization step complete.")

# --- 保存计算图 ---
print("Saving forward and backward graph modules...")
with io.StringIO() as buf, open('aten_module_FW_after.md', 'w') as f:
    print("aten_module_FW_after.md")
    sys.stdout = buf
    graph_capture.FW_gm.print_readable()
    sys.stdout = sys.__stdout__
    f.write(buf.getvalue())

with io.StringIO() as buf, open('aten_module_BW_after.md', 'w') as f:
    sys.stdout = buf
    graph_capture.BW_gm.print_readable()
    sys.stdout = sys.__stdout__
    f.write(buf.getvalue())
    
# 1. 实例化 pass
pass_instance = Dist_IR.Pos(model, graph_capture.FW_gm, graph_capture.BW_gm, model_name = 'bert')


# 3. 运行核心的定位 pass
# 这会调用新的 add_source_info_to_node 来填充 'source_fn'
# pass_instance.positioning_for_graph()

# 2. 【可选】运行你的类型分析
# 注意：这会向 'source_fn' 添加 'tensor_type' 和 'grad_type'
# 最好在 positioning_for_graph 之前运行，这样 add_source_info_to_node
# 可以使用 setdefault 而不是覆盖
#pass_instance.analyze_and_tag_graphs(model, graph_capture.FW_gm, graph_capture.BW_gm)

# 4. 【关键】运行新的日志函数
# 这会生成 'node_positioning_log.txt' 文件
pass_instance.log_node_positioning_info()

print("Graphs saved successfully.")
