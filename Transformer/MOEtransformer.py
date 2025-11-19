from transformer import Transformer
from transformer import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import Dist_IR
import torch._dynamo as dynamo
from Dist_IR.Optim_IR import (SGD_Optimizer, 
    Adagrad_Optimizer, 
    RMSprop_Optimizer, 
    Adam_Optimizer)

#启用GPU(CUDA可用情况)
# device = torch.device("cuda") if torch.cuda.is_available() else "cpu" 
# device_id = 3
# device = torch.device(f"cuda:{device_id}")

#启用CPU
device = 'cpu'
print(f"正在使用的设备是{device}")

class TopKRouter(nn.Module):
    def __init__(self, hidden_dim, num_experts, k=2, noise_epsilon=1e-2):
        super(TopKRouter, self).__init__()
        self.num_experts = num_experts
        self.k = k
        self.wg = nn.Linear(hidden_dim, num_experts)
        self.noise_epsilon = noise_epsilon

    def forward(self, h):
        # h: (B, T, hidden_dim)
        logits = self.wg(h)
        if self.training:
            noise = torch.randn_like(logits) * self.noise_epsilon
            logits = logits + noise
        topk_vals, topk_idx = torch.topk(logits, self.k, dim=-1)  # (B, T, k)
        topk_scores = F.softmax(topk_vals, dim=-1)  # (B, T, k)
        route_weights = torch.zeros_like(logits)  # (B, T, E)
        route_weights.scatter_(dim=-1, index=topk_idx, src=topk_scores)
        importance = logits.softmax(dim=-1).sum(dim=(0,1))
        load = (route_weights > 0).float().sum(dim=(0,1)) / (h.size(0)*h.size(1))
        aux_loss = (importance * load * (self.num_experts ** 2)).mean()
        return route_weights, aux_loss

class MoEWithTransformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model,
        num_heads,
        num_layers,
        d_ff,
        max_seq_length,
        dropout,
        hidden_dim,
        num_experts=4,
        padding_idx=0,
    ):
        super(MoEWithTransformer, self).__init__()
        self.hidden_dim = hidden_dim
        self.transformer = Transformer(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            max_seq_length=max_seq_length,
            dropout=dropout,
            padding_idx=padding_idx,
        )
        self.router = TopKRouter(d_model, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, d_model)
            ) for _ in range(num_experts)
        ])

    def forward(self, src, tgt):
        # Generate masks if needed (or set to None)
        src_mask = None
        tgt_mask = None
        memory_mask = None

        # 1) Embedding + positional encoding
        enc = self.transformer.encoder_embedding(src)
        enc = self.transformer.positional_encoding1(enc)
        enc = self.transformer.dropout1(enc)

        # 2) Global MoE routing before encoder layers
        route_weights, aux_loss = self.router(enc)
        expert_outs = torch.stack([expert(enc) for expert in self.experts], dim=-1)
        enc = (expert_outs * route_weights.unsqueeze(2)).sum(-1)

        # 3) Pass through encoder layers with mask
        for layer in self.transformer.encoder_layers:
            enc = layer(enc, src_mask)

        # 4) Decoder embedding + pos encoding
        dec = self.transformer.decoder_embedding(tgt)
        dec = self.transformer.positional_encoding2(dec)
        dec = self.transformer.dropout2(dec)

        # Pass through decoder layers, supplying padding masks if required
        for layer in self.transformer.decoder_layers:
            # signature: layer(x, memory, tgt_mask=None, memory_mask=None, tgt_padding_mask=None)
            dec = layer(dec, enc, tgt_mask, memory_mask, None)

        # 5) Output projection
        out = self.transformer.fc(dec)
        return out, aux_loss

# Testing
# Hyperparameters
batch_size, seq_len = 128, 256
learning_rate = 1e-3
num_epochs = 1


# Model, loss, optimizer
model = MoEWithTransformer(
    src_vocab_size=5000,
    tgt_vocab_size=5000,
    d_model=512,
    num_heads=8,
    num_layers=1,
    d_ff=2048,
    max_seq_length=1024,
    dropout=0.1,
    hidden_dim=512,
    num_experts=4,
    padding_idx=0,
)
model.to(device)

# Dummy data (replace with real dataset loader)
src = torch.randint(0, 5000, (batch_size, seq_len))
tgt = torch.randint(0, 5000, (batch_size, seq_len))
# For language modeling, targets shifted by one or same as input as placeholder
src = src.to(device)
tgt = tgt.to(device)
targets = tgt.clone()

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

model.train()  # set train mode

graph_capture = Dist_IR.GraphCapture(model,src, tgt[:, :-1])
model = graph_capture.compile()




for epoch in range(1, num_epochs+1):
    # optimizer.zero_grad()
    # Forward pass
    logits, aux_loss = model(src, tgt)
    # reshape for loss: (batch*seq_len, vocab_size)
    batch, length, vocab = logits.size()
    logits_flat = logits.view(batch*length, vocab)
    targets_flat = targets.view(batch*length)
    # compute main loss
    loss_main = criterion(logits_flat, targets_flat)
    # total loss with auxiliary
    loss = loss_main + aux_loss
    # Backward and optimize
    loss.backward()
    optimizer = RMSprop_Optimizer(list(model.parameters()), lr=learning_rate)
    optim_graph_capture = Dist_IR.OptimGraphCapture(optimizer)
    optimizer = optim_graph_capture.compile()
    optimizer(list(model.parameters()))
    # optimizer.step()

    

    print(f"Epoch {epoch}/{num_epochs}")

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
pos = Dist_IR.Pos(model, graph_capture.FW_gm, graph_capture.BW_gm)
#为算子添加source_fn基础定位信息
pos.positioning_for_graph()
#标定反向图中mm是计算dx还是dw
pos.analyze_and_tag_graphs(model, graph_capture.FW_gm, graph_capture.BW_gm)
# pos.print_positioning_info()

#将修改过Meta信息的前向图和反向图储存下来，方便后续Pass使用
graph_capture.FW_gm = pos.FW_gm
graph_capture.BW_gm = pos.BW_gm


Dist_IR.Hybrid_Parallel_pass(graph_capture.FW_gm, graph_capture.BW_gm, optim_graph_capture.OPT_gm, batch_size)