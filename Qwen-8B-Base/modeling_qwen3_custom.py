from typing import Callable, Optional, Union, List, Tuple # ### [修改] 添加 List, Tuple

import torch
from torch import nn

# ### [修改] 将所有相对引用改为绝对引用，确保独立运行时不报错 ###
# 假设 Qwen3 结构兼容 Qwen2，我们使用 Qwen2 的组件
from transformers.activations import ACT2FN
# form transformers.cache_utils import Cache, DynamicCache # ### [修改] 删除：不再需要 Cache 对象
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
# from transformers.masking_utils import create_causal_mask # ### [修改] 删除：Mask 由外部传入
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.utils import (
    auto_docstring, 
    can_return_tuple
)
from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils.generic import check_model_inputs

# ### [修改] 配置类处理 ###
# 直接使用 transformers 库中的 Qwen2Config 作为 Qwen3Config 的替身
# 因为 Qwen3 的 config.json 里的字段（hidden_size 等）Qwen2Config 都能识别
from transformers import Qwen2Config as Qwen3Config

# 为了兼容 Unpack，如果环境不支持简单定义一个
if "Unpack" not in globals():
    from typing import Any
    Unpack = Any 


@use_kernel_forward_from_hub("RMSNorm")
class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        # ### [修改] 简化 Mask 逻辑，直接相加，假设传入的 mask 形状已经正确 ###
        # 原始代码: causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        # 此时 key_states 的 seq_len 已经是包含历史的完整长度
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class Qwen3Attention(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    # ### [修改] forward 签名完全改变 ###
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, # 接收 Tuple Tensor
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        
        input_shape = hidden_states.shape[:-1]
        bsz, q_len = input_shape
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # ### [修改] 核心：手动 Cat 替代 Cache.update ###
        current_k = key_states
        current_v = value_states
        
        if past_key_value is not None:
            past_k, past_v = past_key_value
            # Decode 阶段：将过去 (B, H, S_past, D) 和 现在 (B, H, 1, D) 拼接
            key_states = torch.cat([past_k, current_k], dim=2)
            value_states = torch.cat([past_v, current_v], dim=2)
        
        # 准备返回的 Cache (Tuple of Tensors)
        past_key_value_out = (key_states, value_states)

        # 调用 Attention 计算
        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states, # 这里的 key_states 已经是完整的了
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        
        # ### [修改] 返回三个值：Out, Weights, NewCacheTuple
        return attn_output, attn_weights, past_key_value_out


class Qwen3DecoderLayer(nn.Module): # 移除了 GradientCheckpointingLayer 继承，简化
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    # ### [修改] 接收 Tuple Cache ###
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self Attention
        hidden_states, _, new_past_kv = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings, # 传入 RoPE embedding
            past_key_value=past_key_value,           # 传入 Tuple
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        # 返回 output 和 新的 kv tuple
        return hidden_states, new_past_kv


@auto_docstring
class Qwen3PreTrainedModel(PreTrainedModel):
    config_class = Qwen3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    # _supports_flex_attn = True # 如果报错找不到 flex_attn 也可以注释掉这行

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()



class Qwen3RotaryEmbedding(nn.Module):
    # ... (保持原样，省略大部分代码，这里不影响 Export) ...
    def __init__(self, config: Qwen3Config, device=None):
        super().__init__()
        self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def forward(self, x, position_ids):
        # 简化一下，去掉 dynamic update 那些复杂逻辑，直接算
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


@auto_docstring
class Qwen3Model(Qwen3PreTrainedModel):
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.post_init()

    # ### [修改] 核心入口 ###
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None, # 接收 List[Tensor]
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ): # -> Tuple[Tensor, List[Tensor]]: # 明确返回类型
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        hidden_states = inputs_embeds
        
        # 1. 计算 RoPE
        # 如果没有传 position_ids，这里需要简单生成 (Export 时最好强行传入)
        if position_ids is None:
             seq_len = hidden_states.shape[1]
             position_ids = torch.arange(seq_len, dtype=torch.long, device=hidden_states.device)
             position_ids = position_ids.unsqueeze(0)
             
        # 注意：rotary_emb 只需要 shape，不需要知道是 q 还是 k
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        position_embeddings = (cos, sin)

        # 2. 准备收集新的 Cache
        next_decoder_cache = []

        # 3. 循环层
        for idx, decoder_layer in enumerate(self.layers):
            # 解包 List[Tensor] -> Tuple[Tensor, Tensor]
            layer_past = None
            if past_key_values is not None:
                layer_past = (past_key_values[2 * idx], past_key_values[2 * idx + 1])
            
            hidden_states, new_kv = decoder_layer(
                hidden_states,
                attention_mask=attention_mask, # 直接传透
                position_ids=position_ids,
                past_key_value=layer_past,     # 传入 Tuple
                use_cache=use_cache,
                position_embeddings=position_embeddings,
            )
            
            if use_cache:
                next_decoder_cache.append(new_kv[0]) # k
                next_decoder_cache.append(new_kv[1]) # v

        hidden_states = self.norm(hidden_states)
        
        # ### [修改] 始终返回 Tuple (Tensor, List[Tensor])，为了 Export 友好
        return hidden_states, next_decoder_cache


@auto_docstring
class Qwen3ForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    # ### [修改] 适配 Model 的新接口 ###
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None,
        use_cache: bool = True, # 默认为 True
        **kwargs,
    ):
        # 调用修改后的 Model
        hidden_states, new_cache = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        logits = self.lm_head(hidden_states)

        # 直接返回 Tuple，不使用 CausalLMOutputWithPast，这对 Export 最干净
        # 返回: (logits, flattened_kv_list)
        return logits, new_cache