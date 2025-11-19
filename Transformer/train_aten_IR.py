from transformer import *
#from IR_transform import aten_compile_capture
import Dist_IR 
from torch.fx import symbolic_trace
from torch.nn.attention import sdpa_kernel, SDPBackend

from Dist_IR.Optim_IR import (SGD_Optimizer, 
    Adagrad_Optimizer, 
    RMSprop_Optimizer, 
    Adam_Optimizer)
src_vocab_size = 5000
tgt_vocab_size = 5000
d_model = 512
num_heads = 8
num_layers = 1 #原6  代表encoder层或decoder层中Transformer block的堆叠次数
d_ff = 2048 #FFN隐藏层维度 4*d_model
max_seq_length = 100
dropout = 0.1



transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
transformer.to(device)

# Generate random sample data
src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))# (batch_size, seq_length)
tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)
src_data = src_data.to(device)
tgt_data = tgt_data.to(device)

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

transformer.train()

# # transformer = aten_compile_capture(transformer)
# # 捕获图
# """ 用户参考 start"""
# torch_IR = Dist_IR.torch_IR_capture()
# torch._dynamo.reset()
# traced = torch.compile(transformer, backend=torch_IR.inspect_backend)
# for epoch in range(1):
#     optimizer.zero_grad()
#     output = traced(src_data, tgt_data[:, :-1])
#     loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
#     loss.backward()
#     optimizer.step()
#     print(f"Epoch: {epoch+1}, Loss: {loss.item()}")



# graph_capture = Dist_IR.GraphCapture()
# traced: torch.fx.GraphModule = symbolic_trace(transformer)
# torch_IR.FW_gm.print_readable()
# transformer = torch.compile(torch_IR.FW_gm.forward , backend=graph_capture.inspect_backend, dynamic=True)
import torch._dynamo as dynamo

# 1) 导出一个纯 Python‐API 级别的 FX GraphModule
# fx_mod, guards = dynamo.export(
#     transformer,
#     aten_graph=False,
# )(src_data, tgt_data[:, :-1])

# # print(fx_mod.code)
# # print(guards)
# graph_capture = Dist_IR.GraphCapture()
# # 2) 把这个 GraphModule 编译一次
# transformer  = torch.compile(
#     fx_mod,                        # 传入 GraphModule
#     backend=graph_capture.inspect_backend ,
#     dynamic=True                   # 如果你希望支持动态形状
# )
# graph_capture = Dist_IR.GraphCapture()
# transformer  = torch.compile(
#     transformer,                        # 传入 GraphModule
#     backend=graph_capture.inspect_backend ,
#     dynamic=True                   # 如果你希望支持动态形状
# )

graph_capture = Dist_IR.GraphCapture(transformer,src_data, tgt_data[:, :-1])
transformer = graph_capture.compile()

""" 用户参考 end"""

for epoch in range(1):
    # optimizer.zero_grad()
    output = transformer(src_data, tgt_data[:, :-1])
    loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
    loss.backward()
    #optimizer.step()
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")
    optimizer = RMSprop_Optimizer(list(transformer.parameters()), lr=0.0001)
    optim_graph_capture = Dist_IR.OptimGraphCapture(optimizer)
    optimizer = optim_graph_capture.compile()
    optimizer(list(transformer.parameters()))
#保存graph Module
import io,sys
output1 = io.StringIO()
with io.StringIO() as buf:
    original_stdout1 = sys.stdout
    sys.stdout = buf
    graph_capture.FW_gm.print_readable()
    sys.stdout = original_stdout1
    output1 = buf.getvalue()
with open( 'aten_module_FW.md', 'w') as f:
    # 写入捕获的输出
    f.write(output1)
# graph_capture.FW_gm.print_readable(colored = True)
for node in graph_capture.FW_gm.graph.nodes:
    if node.op == 'call_function':
        print(node.meta)
        print()

output1 = io.StringIO()
with io.StringIO() as buf:
    original_stdout1 = sys.stdout
    sys.stdout = buf
    graph_capture.BW_gm.print_readable()
    sys.stdout = original_stdout1
    output1 = buf.getvalue()
with open( 'aten_module_BW.md', 'w') as f:
    # 写入捕获的输出
    f.write(output1)

#创建pos实例
pos = Dist_IR.Pos(transformer, graph_capture.FW_gm, graph_capture.BW_gm)
#为算子添加source_fn基础定位信息
pos.positioning_for_graph()
#标定反向图中mm是计算dx还是dw
pos.analyze_and_tag_graphs(transformer, graph_capture.FW_gm, graph_capture.BW_gm)
pos.print_positioning_info()

#将修改过Meta信息的前向图和反向图储存下来，方便后续Pass使用
graph_capture.FW_gm = pos.FW_gm
graph_capture.BW_gm = pos.BW_gm


# 调用 pass
"""用户参考 start"""
Dist_IR.Hybrid_Parallel_pass(graph_capture.FW_gm, graph_capture.BW_gm,optim_graph_capture.OPT_gm,1000)
"""用户参考 end"""