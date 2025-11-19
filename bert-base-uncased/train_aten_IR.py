import torch
import torch.nn as nn
import torch.optim as optim
import io
import sys

# --- 核心改动：从 transformers 库导入 BERT 相关的类 ---
from transformers import BertConfig, BertForMaskedLM

# 假设 Dist_IR 是您本地的库
import Dist_IR
from Dist_IR.Optim_IR import RMSprop_Optimizer

# --- 适用于 BERT 的模型包装器 ---
# BERT 模型只需要一个主要的输入 (input_ids)，因此包装器更简单
class BertWrapper(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.bert_model = bert_model

    def forward(self, input_ids):
        # 1. 调用原始模型
        #    我们使用 BertForMaskedLM，它的输出对象中直接包含了用于预测被遮盖词语的 logits
        outputs = self.bert_model(input_ids=input_ids, return_dict=True)
        
        # 2. 提取 logits
        logits = outputs.logits
        
        # 3. 返回纯粹的 torch.Tensor
        return logits

# --- 1. 模型配置和初始化 ---
print("Initializing BERT model from configuration...")
# --- 核心改动：使用 bert-base-uncased 的本地路径 ---
# 请确保您已将 'bert-base-uncased' 模型下载到这个路径下
local_model_path = './bert_base_uncased_local' 
config = BertConfig.from_pretrained(local_model_path)
# --- 核心改动：使用 BertForMaskedLM，它自带了预测头，方便计算损失 ---
model = BertForMaskedLM(config) # 权重是随机的

device = torch.device("cpu")
model.to(device)
print(f"Model moved to {device}.")

# --- 实例化新的包装器 ---
wrapped_model = BertWrapper(model).to(device)


# --- 2. 为“掩码语言模型”(MLM)任务生成随机样本数据 ---
batch_size = 4
max_seq_length = 128 # BERT 通常使用 128 或 512
vocab_size = config.vocab_size
mask_token_id = 103  # bert-base-uncased 的 [MASK] 标记ID是 103

# 生成一批随机的 token ID 作为原始输入
input_ids = torch.randint(1000, 29000, (batch_size, max_seq_length), device=device)

# 1. 将原始输入复制一份，用作计算损失时的“正确答案” (labels)
labels = input_ids.clone()

# 2. 创建一个概率矩阵，决定哪些 token 将被遮盖 (这里是 15% 的概率)
probability_matrix = torch.full(labels.shape, 0.15)
masked_indices = torch.bernoulli(probability_matrix).bool()

# 3. 在“正确答案”中，将所有未被遮盖的 token 设置为 -100
#    这样损失函数在计算时就会忽略它们
labels[~masked_indices] = -100

# 4. 在模型的输入中，将被选中的 token 替换为 [MASK] 标记
input_ids[masked_indices] = mask_token_id


# --- 3. 设置损失函数和优化器 ---
criterion = nn.CrossEntropyLoss() # PyTorch 的交叉熵损失默认会忽略 index 为 -100 的标签
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

wrapped_model.train()

# --- 4. 使用 Dist_IR 捕获计算图 ---
print("Capturing model graph with Dist_IR...")
# --- 核心改动：传入单个 input_ids 张量 ---
graph_capture = Dist_IR.GraphCapture(wrapped_model, input_ids)
compiled_model = graph_capture.compile()
print("Graph capture and compilation complete.")

# --- 5. 训练和优化流程 ---
for epoch in range(1):
    print(f"Starting Epoch: {epoch+1}")
    
    # --- 核心改动：前向传播只传入被遮盖过的 input_ids ---
    output = compiled_model(input_ids)
    
    logits = output

    # --- 核心改动：计算 MLM 损失 ---
    # 比较模型对被遮盖位置的预测 (logits) 和原始的正确答案 (labels)
    loss = criterion(logits.view(-1, vocab_size), labels.view(-1))
    
    # 反向传播
    loss.backward()
    
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

    # 使用 Dist_IR 的自定义优化器
    print("Applying custom RMSprop optimizer...")
    dist_optimizer = RMSprop_Optimizer(list(model.parameters()), lr=0.0001)
    optim_graph_capture = Dist_IR.OptimGraphCapture(dist_optimizer)
    compiled_optimizer = optim_graph_capture.compile()
    
    # 执行优化步骤
    compiled_optimizer(list(model.parameters()))
    print("Optimization step complete.")


# --- 6. 保存计算图 ---
print("Saving forward and backward graph modules...")
with io.StringIO() as buf, open('aten_module_FW_bert.md', 'w') as f:
    sys.stdout = buf
    graph_capture.FW_gm.print_readable()
    sys.stdout = sys.__stdout__
    f.write(buf.getvalue())

with io.StringIO() as buf, open('aten_module_BW_bert.md', 'w') as f:
    sys.stdout = buf
    graph_capture.BW_gm.print_readable()
    sys.stdout = sys.__stdout__
    f.write(buf.getvalue())
print("Graphs saved to aten_module_FW_bert.md and aten_module_BW_bert.md.")


# --- 7. Dist_IR 定位与 Pass ---
print("Running positioning analysis...")
pos = Dist_IR.Pos(model, graph_capture.FW_gm, graph_capture.BW_gm)
pos.positioning_for_graph()
pos.analyze_and_tag_graphs(model, graph_capture.FW_gm, graph_capture.BW_gm)

graph_capture.FW_gm = pos.FW_gm
graph_capture.BW_gm = pos.BW_gm
print("Positioning analysis complete.")

print("Applying Hybrid_Parallel_pass...")
Dist_IR.Hybrid_Parallel_pass(graph_capture.FW_gm, graph_capture.BW_gm, optim_graph_capture.OPT_gm, batch_size)
print("Hybrid_Parallel_pass applied successfully.")
