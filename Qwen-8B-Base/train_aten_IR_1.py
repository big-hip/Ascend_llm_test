import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoConfig, AutoModelForCausalLM
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose = True
# ---- 1) 模型与设备 ----
local_model_path = './Qwen_8B_Base_local'
config = AutoConfig.from_pretrained(local_model_path)
model = AutoModelForCausalLM.from_config(config)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()  # 记得进入训练模式

# ---- 2) 构造数据 ----
batch_size = 1
max_seq_length = 4
vocab_size = config.vocab_size
input_ids = torch.randint(0, vocab_size, (batch_size, max_seq_length), device=device, dtype=torch.long)

# 常见做法：自回归预训练/玩具训练里，labels = input_ids
# 模型会内部做 shift，所以不用你手动对齐
labels = input_ids.clone()

# ---- 3) 优化器 ----
optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)

# ---- 4) 用 Dist_IR 的 backend 捕获：建议编译一个“返回 loss 的包装器”
import torch.nn as nn
class LossWrapper(nn.Module):
    def __init__(self, mdl):
        super().__init__()
        self.mdl = mdl
    def forward(self, input_ids, labels):
        out = self.mdl(input_ids=input_ids, labels=labels, return_dict=True)
        # out.loss 是标量 Tensor（float32/float16），最利于 AOT/FX 捕获与 backward
        return out.loss

wrapped = LossWrapper(model).to(device)

from IR_transform import aten_compile_capture
compiled = aten_compile_capture(wrapped)  # 你的 backend 会在前/反向各捕获一次



# from Dist_IR import GraphCapture
# graph_capture = GraphCapture(wrapped, input_ids, labels)
# compiled = graph_capture.compile()

# ---- 5) 单步“真训练” ----
optimizer.zero_grad(set_to_none=True)
loss = compiled(input_ids, labels)   # 直接得到一个标量 Tensor
loss.backward()                      # 触发反向捕获
optimizer.step()

print(f"Training step done. loss={loss.item():.4f}")
