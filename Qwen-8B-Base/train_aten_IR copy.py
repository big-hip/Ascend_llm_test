import torch
import torch.nn as nn
import torch.optim as optim
import io
import sys
import torch._dynamo
from torch.fx.passes.graph_drawer import FxGraphDrawer
from transformers import AutoConfig, AutoModelForCausalLM

# 假设 Dist_IR 是您本地的库
import Dist_IR 
from Dist_IR.Optim_IR import RMSprop_Optimizer
from Performance_Eval.Fake_Runtime.Simulation import Performance_Evaluation


# --- 模型包装器 ---
class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        outputs = self.model(input_ids=input_ids, return_dict=False)
        logits = outputs[0]
        return logits


# --- 自定义 backend：捕获 FX 图 ---
def inspect_backend(gm, sample_inputs): 
    # 保存 graph IR
    with io.StringIO() as buf:
        sys.stdout = buf
        print(gm.graph)
        sys.stdout = sys.__stdout__
        output = buf.getvalue()
    with open('Fw_torch_IR.md', 'w') as f:
        f.write(output)

    # 保存可读 Module
    with io.StringIO() as buf:
        sys.stdout = buf
        gm.print_readable()
        sys.stdout = sys.__stdout__
        output = buf.getvalue()
    with open('Fw_torch_module.md', 'w') as f:
        f.write(output)

    print("✅ 已捕获 TorchDynamo 前向图，保存至 Fw_torch_IR.md 与 Fw_torch_module.md。")
    return gm.forward


def compile_capture(model):
    torch._dynamo.reset()
    compiled_model = torch.compile(model, backend=inspect_backend)
    return compiled_model


# --- 1. 初始化模型 ---
print("Initializing Qwen model from configuration...")
local_model_path = './Qwem_8B_Base_local' 
config = AutoConfig.from_pretrained(local_model_path)
model = AutoModelForCausalLM.from_config(config)

device = torch.device("cpu")
model.to(device)
print(f"Model moved to {device}.")

wrapped_model = ModelWrapper(model).to(device)


# --- 2. 生成随机输入 ---
batch_size = 4
max_seq_length = 100
vocab_size = config.vocab_size
input_ids = torch.randint(0, vocab_size, (batch_size, max_seq_length), device=device)


# --- 3. 设置损失函数与优化器 ---
criterion = nn.CrossEntropyLoss(ignore_index=-100)
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
wrapped_model.train()


# --- 4. 使用 torch.compile 捕获计算图 ---
print("Capturing model graph via torch.compile backend...")
compiled_model = compile_capture(wrapped_model)
output = compiled_model(input_ids)
print("✅ Torch.compile 图捕获完成。")


# --- 5. 继续使用 Dist_IR 进行后续分析 ---
print("Running Dist_IR compilation for further passes...")
graph_capture = Dist_IR.GraphCapture(wrapped_model, input_ids)
compiled_model = graph_capture.compile()
print("Dist_IR graph capture complete.")


# --- 6. 前向与反向传播 ---
for epoch in range(1):
    print(f"Starting Epoch: {epoch+1}")
    output = compiled_model(input_ids)
    logits = output
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    loss = criterion(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
    loss.backward()
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

    dist_optimizer = RMSprop_Optimizer(list(model.parameters()), lr=0.0001)
    optim_graph_capture = Dist_IR.OptimGraphCapture(dist_optimizer)
    compiled_optimizer = optim_graph_capture.compile()
    compiled_optimizer(list(model.parameters()))
    print("Optimization step complete.")


# --- 7. 保存计算图 ---
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


# # --- 8. Dist_IR 定位与 Pass ---
# print("Running positioning analysis...")
# pos = Dist_IR.Pos(model, graph_capture.FW_gm, graph_capture.BW_gm)
# pos.positioning_for_graph()
# pos.analyze_and_tag_graphs(model, graph_capture.FW_gm, graph_capture.BW_gm)
# graph_capture.FW_gm = pos.FW_gm
# graph_capture.BW_gm = pos.BW_gm
# print("Positioning analysis complete.")

# print("Applying Hybrid_Parallel_pass...")
# hybridpass = Dist_IR.Hybrid_Parallel_pass(graph_capture.FW_gm, graph_capture.BW_gm, optim_graph_capture.OPT_gm, batch_size)
# print("Hybrid_Parallel_pass applied successfully.")
