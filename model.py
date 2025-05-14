import math
import struct
import inspect
import time

from Config import LLMConfig
from typing import Any, Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

# RMSNorm层：实现均方根归一化
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps  # 防止除零的小常数
        self.weight = nn.Parameter(torch.ones(dim))  # 学习参数，初始全1
        
    def forward(self, x):
        # 计算x每个元素的均方根归一化，再乘以学习参数
        return self.weight * (x.float() * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)).type_as(x)

# 预先计算位置编码（旋转位置编码，RoPE）的复数形式
def precompute_pos_cis(dim: int, end: int = int(32 * 1024), theta: float = 1e6):
    # 计算频率因子
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 生成时间步长向量
    t = torch.arange(end, device=freqs.device)  # type: ignore
    # 计算外积，得到每个时间步对应的角度
    freqs = torch.outer(t, freqs).float()  # type: ignore
    # 将幅度固定为1，通过极坐标得到复数表示
    pos_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return pos_cis

# 应用旋转位置编码到查询和键（xq, xk）
def apply_rotary_emb(xq, xk, pos_cis):
    # 定义辅助函数，调整pos_cis的形状以匹配输入张量
    def unite_shape(pos_cis, x):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        # 确保pos_cis的形状为 (序列长度, head维度)
        assert pos_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return pos_cis.view(*shape)
    # 将xq和xk转换为复数形式，便于与位置编码相乘
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    pos_cis = unite_shape(pos_cis, xq_)
    # 应用旋转位置编码后再转换回实数表示，并展平最后一维
    xq_out = torch.view_as_real(xq_ * pos_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * pos_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

# 重复键和值，类似于torch.repeat_interleave实现，用于复制KV头
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

# 注意力机制模块
class Attention(nn.Module):
    def __init__(self, args: LLMConfig):
        super().__init__()
        # 如果未指定KV头数，则默认与heads数相同
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # 确保heads数量能被KV头数整除
        assert args.n_heads % self.n_kv_heads == 0
        self.n_heads = args.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads  # 每个KV头复制的次数
        self.head_dim = args.dim // args.n_heads  # 每个头的维度
        # 定义查询、键、值线性变换
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        # 输出投影
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        # 注意力和残差的dropout
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        # 构建因果掩码（上三角矩阵），防止信息泄露到未来的token
        # 创建一个全为-inf的矩阵，大小为(batch_size, n_heads, args.max_seq_len, args.max_seq_len)
        mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        pos_cis: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache=False,
    ):
        bsz, seq_len, _ = x.shape
        # 计算查询、键和值
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        # 调整形状以适应多头计算
        xq = xq.view(bsz, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        # 应用旋转位置编码
        xq, xk = apply_rotary_emb(xq, xk, pos_cis)
        # 如果提供了历史KV缓存，则拼接当前KV
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None
        # 调整查询、键和值的维度，为多头注意力做准备；对于键和值，需重复复制n_rep次
        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2),
        )
        # 计算注意力得分，缩放因子为sqrt(head_dim)
        scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # 加入因果掩码
        scores += self.mask[:, :, :seq_len, :seq_len]
        # softmax归一化
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        scores = self.attn_dropout(scores)
        # 计算注意力输出
        output = scores @ xv

        # 恢复输出形状并进行输出投影及残差dropout
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.wo(output))
        return output, past_kv

# 前馈神经网络模块
class FeedForward(nn.Module):
     def __init__(self, config: LLMConfig):
        super().__init__()
        # 如果hidden_dim未指定，则根据输入维度计算默认值
        if config.hidden_dim is None:
            hidden_dim = 4 * config.dim
            hidden_dim = int(2 * hidden_dim / 3)
            config.hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)
        # 两个线性变换以及一个辅助的线性变换
        self.w1 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.w2 = nn.Linear(config.hidden_dim, config.dim, bias=False)
        self.w3 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)
     
     def forward(self, x):
        # 使用SiLU激活函数，并结合w1、w3进行非线性变换后经过w2，还加上dropout
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
     
# Transformer层：包含注意力和前馈网络（这里称为SpongeBobBlock）
class SpongeBobBlock(nn.Module):
     def __init__(self, layer_id: int, config: LLMConfig):
          super().__init__()
          self.n_heads = config.n_heads
          self.dim = config.dim
          self.head_dim = config.dim // config.n_heads
          # 注意力子层
          self.attention = Attention(config)
          self.layer_id = layer_id
          # 两个归一化层：一个用于注意力前，一个用于前馈网络前
          self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
          self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
          # 前馈网络
          self.feed_forward = FeedForward(config)
     
     def forward(self, x, pos_cis, past_key_value=None, use_cache=False):
          # 先经过归一化和注意力计算
          h_attn, past_kv = self.attention(
            self.attention_norm(x),
            pos_cis,
            past_key_value=past_key_value,
            use_cache=use_cache
          )
          # 残差连接
          h = x + h_attn
          # 再经过前馈网络及归一化后加上残差连接
          out = h + self.feed_forward(self.ffn_norm(h))
          return out, past_kv

# 主模型类，继承自PreTrainedModel
class SpongeBob(PreTrainedModel):
    config_class = LLMConfig

    def __init__(self, params: LLMConfig = None):
        self.params = params or LLMConfig()
        super().__init__(params)
        # 初始化词表大小和层数
        self.vocab_size, self.n_layers = params.vocab_size, params.n_layers
        # 词嵌入层
        self.tok_embeddings = nn.Embedding(self.vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)
        # 多层Transformer结构
        self.layers = nn.ModuleList(
            [SpongeBobBlock(l, params) for l in range(self.n_layers)]
        )
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        # 输出线性层
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)
        # 权重共享：将词嵌入层权重和输出层权重绑定
        self.tok_embeddings.weight = self.output.weight
        # 预先计算位置编码，存储为buffer，不参与训练
        self.register_buffer(
            "pos_cis",
            precompute_pos_cis(
                dim=params.dim // params.n_heads, theta=params.rope_theta
            ),
            persistent=False,
        )
        # 主要用于处理因果语言模型的输出。它扩展了基本的模型输出，增加了对过去键值对（past key-value pairs）的支持
        # 输出结构:
        # logits：模型对每个输入 token 的预测分数，通常用于计算损失或进行采样。
        # past_key_values：用于存储先前计算的键值对。
        self.OUT = CausalLMOutputWithPast()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        **args,
    ):
        # 如果没有传入KV缓存，则置为None列表，第一个token没有KV缓存
        past_key_values = past_key_values or [None] * len(self.layers) # 因为每一层都有KV缓存
        # 获取开始位置，默认为0
        start_pos = args.get("start_pos", 0)

        # 词嵌入 + dropout
        h = self.dropout(self.tok_embeddings(input_ids))
        # 根据输入序列长度获取对应位置编码
        pos_cis = self.pos_cis[start_pos : start_pos + input_ids.size(1)]
        past_kvs = []

        # 逐层传递数据，并收集KV缓存（如果需要）
        for l, layer in enumerate(self.layers):
            h, past_kv = layer(
                h, pos_cis, past_key_value=past_key_values[l], use_cache=use_cache
            )
            past_kvs.append(past_kv)
        # 最后经过归一化和输出线性层得到logits
        # 将隐藏状态h先进行归一化处理，然后通过输出层映射到词表大小的logits
        logits = self.output(self.norm(h))
        
        # OUT是CausalLMOutputWithPast类的实例，是transformers库中用于因果语言模型的标准输出格式
        # 它提供了一个统一的接口来存储和访问模型的输出结果
        
        # 存储logits - 这是模型的主要预测输出，表示下一个token的概率分布
        self.OUT.__setitem__("logits", logits)
        
        # 存储past_key_values - 这些是transformer各层的注意力缓存
        # 在自回归生成时，可以重用这些缓存来加速后续token的生成，避免重复计算之前的token
        self.OUT.__setitem__("past_key_values", past_kvs)

        return self.OUT

    @torch.inference_mode()
    # 生成函数：支持流式生成与一次性生成
    def generate(
        self,
        input_ids,
        eos_token_id=2,
        max_new_tokens=1024,
        temperature=0.75,
        top_p=0.90,
        stream=False,
        rp=1.0,
        use_cache=True,
        pad_token_id=0,
        **args,
    ):
        return self._stream(
            input_ids,
            eos_token_id,
            max_new_tokens,
            temperature,
            top_p,
            rp,
            use_cache,
            **args,
        )

    # 内部流式生成函数
    def _stream(
        self,
        input_ids,
        eos_token_id,
        max_new_tokens,
        temperature,
        top_p,
        rp,
        use_cache,
        **args,
    ):
        start, first_seq, past_kvs = input_ids.shape[1], True, None  # input_ids.shape[1]是seq_len
        while input_ids.shape[1] < max_new_tokens - 1:
            # 首次调用或未使用缓存时，传入整个序列
            if first_seq or not use_cache:
                out, first_seq = (
                    self(
                        input_ids, past_key_values=past_kvs, use_cache=use_cache, **args
                    ),
                    False,
                )
            else:
                # 仅传入最后一个token，同时更新start_pos
                out = self(
                    input_ids[:, -1:],
                    past_key_values=past_kvs,
                    use_cache=use_cache,
                    start_pos=input_ids.shape[1] - 1,
                    **args,
                )
            # 取出最后一步的logits及更新后的KV缓存
            logits, past_kvs = out.logits[:, -1, :], out.past_key_values
            # 对已经生成的token进行惩罚，防止重复生成
            logits[:, list(set(input_ids.tolist()[0]))] /= rp
            # 温度缩放
            logits /= temperature + 1e-9
            # 如果设置了top_p采样，则进行核采样处理
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    logits, descending=True, dim=-1
                )
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[
                    :, :-1
                ].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = -float("Inf")
            # 根据采样后的概率分布选取下一个token
            input_ids_next = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            # 将新token拼接到已有序列上
            input_ids = torch.cat((input_ids, input_ids_next), dim=1)
            # 生成器返回新生成部分
            yield input_ids[:, start:]
            # 若生成的token为结束符，则停止生成
            if input_ids_next.item() == eos_token_id:
                break
