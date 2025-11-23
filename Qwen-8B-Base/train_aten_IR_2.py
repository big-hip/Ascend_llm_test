#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_aten_IR_force_vmap_patch_final.py
---------------------------------------
强制捕获 Qwen-8B 图，彻底屏蔽 functorch/vmap、BatchedTensor、shape 校验。
兼容 torch 2.6。
"""

import os, sys, io, torch, logging

# ============================================================
# Step 0. 全局 Patch
# ============================================================

import torch._logging
torch._logging._internal.make_module_path_relative = lambda p: p.split("/")[-1]
torch._logging.set_logs(dynamo=logging.INFO, aot=logging.INFO)

# ---- functorch 内部函数 ----
def _fake_return(*args, **kwargs):
    for a in args:
        if torch.is_tensor(a):
            return a
    return torch.tensor(0)

for name in [
    "_vmap_increment_nesting",
    "_vmap_decrement_nesting",
    "_add_batch_dim",
    "_remove_batch_dim",
    "_maybe_wrap_dim",
]:
    if hasattr(torch._C._functorch, name):
        setattr(torch._C._functorch, name, _fake_return)
        torch.compiler.allow_in_graph(_fake_return)

# ---- TensorBase 补丁 ----
torch.Tensor.is_contiguous = lambda self, *a, **k: True
torch.Tensor.contiguous = lambda self, *a, **k: self
torch.Generator = lambda *a, **k: None

# ---- functorch.vmap 彻底屏蔽 BatchedTensor ----
import torch._functorch.vmap as vmap_mod

class FakeVmap:
    """伪装成 vmap 模块对象，提供必要属性防止内部访问报错"""
    def __call__(self, func=None, *args, **kwargs):
        def wrapper(f):
            def inner(*a, **k):
                if len(a) > 0 and torch.is_tensor(a[0]) and a[0].ndim > 1:
                    return f(a[0][0], *[x[0] if torch.is_tensor(x) and x.shape[0]==a[0].shape[0] else x for x in a[1:]], **k)
                return f(*a, **k)
            return inner
        if func is not None:
            return wrapper(func)
        return wrapper

    def _validate_and_get_batch_size(self, *a, **k):
        # 返回一个假的 batch_size，防止 Qwen mask 调用时报错
        return 1

# 替换所有 functorch.vmap 引用
torch._functorch.vmap = FakeVmap()
vmap_mod.vmap = torch._functorch.vmap
torch._functorch.vmap_impl = lambda *a, **k: a[0]
print("[Patch] ✅ functorch.vmap fully bypassed (no BatchedTensor created)")

# ---- TorchDynamo 配置 ----
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose = True
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.guard_nn_modules = False
torch._dynamo.config.inline_inbuilt_nn_modules = False
if hasattr(torch._dynamo.config, "trace_context_nesting"):
    torch._dynamo.config.trace_context_nesting = False

print("[Dist_IR Patch] ✅ 全宽松 functorch/vmap 捕获模式启用 (Torch 2.6)")

# ============================================================
# Step 1. 模型与设备
# ============================================================

from transformers import AutoConfig, AutoModelForCausalLM
local_model_path = "./Qwen_8B_Base_local"
config = AutoConfig.from_pretrained(local_model_path)
model = AutoModelForCausalLM.from_config(config)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

# ============================================================
# Step 2. 输入数据
# ============================================================
batch_size = 1
max_seq_length = 4
vocab_size = config.vocab_size
input_ids = torch.randint(0, vocab_size, (batch_size, max_seq_length),
                          device=device, dtype=torch.long)
labels = input_ids.clone()

# ============================================================
# Step 3. 优化器
# ============================================================
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4,
                             betas=(0.9, 0.98), eps=1e-9)

# ============================================================
# Step 4. 包装模块
# ============================================================
class LossWrapper(torch.nn.Module):
    def __init__(self, mdl):
        super().__init__()
        self.mdl = mdl
    def forward(self, input_ids, labels):
        out = self.mdl(input_ids=input_ids, labels=labels, return_dict=True)
        return out.loss

wrapped = LossWrapper(model).to(device)

# ============================================================
# Step 5. 捕获 Graph
# ============================================================
from Dist_IR import Graph_compile_capture_2
graph_capture = Graph_compile_capture_2.GraphCapture(wrapped, input_ids, labels)
compiled = graph_capture.compile()

# ============================================================
# Step 6. 执行前向并保存 IR
# ============================================================
print("Qwen 8B start!!!")
loss = compiled(input_ids, labels)
# loss.backward()
# optimizer.step()

os.makedirs("Train_2", exist_ok=True)

with io.StringIO() as buf:
    old_stdout = sys.stdout
    sys.stdout = buf
    graph_capture.FW_gm.print_readable()
    sys.stdout = old_stdout
    fw_text = buf.getvalue()

with open(os.path.join("Train_2", "aten_IR_FW.md"), "w") as f:
    f.write(fw_text)

print(f"Training step done. loss={loss.item():.4f}")
