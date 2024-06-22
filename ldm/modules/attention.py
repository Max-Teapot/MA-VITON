from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from typing import Optional, Any
import os

from ldm.modules.diffusionmodules.util import checkpoint

try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False

# CrossAttn precision handling
import os
_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")

def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0., **kwargs):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        

    def forward(self, x, context=None, mask=None):
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        if _ATTN_PRECISION =="fp32":
            with torch.autocast(enabled=False, device_type = 'cuda'):
                q, k = q.float(), k.float()
                sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        else:
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        
        del q, k
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)
               
        sim = sim.softmax(dim=-1) 
                
        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)




class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0, zero_init=False, **kwargs):
        super().__init__()
        print(f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
              f"{heads} heads.")
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head
        if not zero_init:
            self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
            self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
            self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        else:
            self.to_q = zero_module(nn.Linear(query_dim, inner_dim, bias=False))
            self.to_k = zero_module(nn.Linear(context_dim, inner_dim, bias=False))
            self.to_v = zero_module(nn.Linear(context_dim, inner_dim, bias=False))

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        self.attention_op: Optional[Any] = None
            

    def forward(self, x, context=None, mask=None, **kwargs):
        # q k v (1, 192, 1280)
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)
        b, _, _ = q.shape
        # 输出qkv为 (8, 192, 160)
        q, k, v = map(  
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )   
        # out (8, 192, 160)
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)
        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)
    


# checkpoint参数默认为 True
class BasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention
    }
    def __init__(
            self, 
            dim, 
            n_heads, 
            d_head, 
            dropout=0., 
            context_dim=None, 
            gated_ff=True, 
            checkpoint=True,
            disable_self_attn=False
        ):
        super().__init__()
        attn_mode = "softmax-xformers" if XFORMERS_IS_AVAILBLE else "softmax"
        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                              context_dim=context_dim if self.disable_self_attn else None)
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(query_dim=dim, context_dim=context_dim,
                              heads=n_heads, dim_head=d_head, dropout=dropout)  
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None,hint=None):
        if hint is None:
            return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)
        else:
            return checkpoint(self._forward, (x, context, hint), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None,hint=None):
        # context(1,192,1280)     x(1,192,1280)    self.disable_self_attn = False
        x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None,hint=hint) + x
        # 输出x(1,192,1280)
        x = self.attn2(self.norm2(x), context=context) + x
        # 输出x(1,192,1280)
        x = self.ff(self.norm3(x)) + x
        # 输出x(1,192,1280)
        return x






class SpatialTransformer(nn.Module):
    """
        用于图像数据的Transformer块。
        首先，对输入进行投影（也称为嵌入），并重新形状为b、t、d。
        然后应用标准的Transformer操作。
        最后，重新形状为图像。
        新特性：使用线性层（use_linear）以提高效率，而不是使用1x1的卷积核。
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False, use_linear=False,
                 use_checkpoint=True):
        super().__init__()
        # 如果存在上下文维度，将其转换为列表形式
        if context_dim is not None and not isinstance(context_dim, list):
            context_dim = [context_dim]
        # 输入通道数
        self.in_channels = in_channels
        # 计算内部维度，等于注意力头数目乘以每个头的维度
        inner_dim = n_heads * d_head
        # 标准化模块
        self.norm = Normalize(in_channels)
        # 根据 use_linear 参数选择投影层是卷积层还是线性层
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1,
                                     stride=1, padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)
        # 堆叠的 Transformer 块（一般是1个）
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d],
                                   disable_self_attn=disable_self_attn, checkpoint=use_checkpoint)
                for d in range(depth)]
        )
        # 根据 use_linear 参数选择输出投影层是卷积层还是线性层
        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim,in_channels,kernel_size=1,
                                                  stride=1,padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        # 记录是否使用线性层
        self.use_linear = use_linear



    # 前向传播方法
    def forward(self, x, context=None, hint=None):
        # 注意：如果没有提供上下文信息，交叉注意力默认为自注意力
        if not isinstance(context, list):
            context = [context]

        # 获取输入数据的形状
        b, c, h, w = x.shape

        # 保存原始输入数据
        x_in = x

        # 对输入数据进行标准化
        x = self.norm(x)

        # 如果不使用线性层，则对输入数据进行卷积
        if not self.use_linear:
            x = self.proj_in(x)

        # 对输入数据进行形状重塑，将其变为二维张量
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()

        # 如果使用线性层，则对数据进行线性投影
        if self.use_linear:
            x = self.proj_in(x)

        # 遍历堆叠的 Transformer 块
        # x为模特特征，context为衣服特征，hint为None
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context[i], hint=hint)

        # 如果使用线性层，对输出数据进行卷积
        if self.use_linear:
            x = self.proj_out(x)

        # 将输出数据重塑为图像形状
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()

        # 如果不使用线性层，再次对输出数据进行线性投影
        if not self.use_linear:
            x = self.proj_out(x)

        # 返回最终结果，加上原始输入数据（残差连接）
        return x + x_in