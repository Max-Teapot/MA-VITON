import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import math
from inspect import isfunction


def local_conv(dim):
    return nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim)

def exists(val):
    return val is not None


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
        project_in = nn.Sequential( nn.Linear(dim, inner_dim),nn.GELU() ) if not glu else GEGLU(dim, inner_dim)
        self.net = nn.Sequential( project_in,nn.Dropout(dropout),nn.Linear(inner_dim, dim_out) )

    def forward(self, x):
        return self.net(x)



def window_partition(x, window_size, H, W):
    B, num_heads, N, C = x.shape
    x = x.contiguous().view(B * num_heads, N, C).contiguous().view(B * num_heads, H, W, C)
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C). \
        view(-1, window_size * window_size, C)
    return windows


def window_reverse(windows, window_size, H, W, head):
    Bhead = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(Bhead, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(Bhead, H, W, -1).view(Bhead // head, head, H, W, -1) \
        .contiguous().permute(0, 2, 3, 1, 4).contiguous().view(Bhead // head, H, W, -1).view(Bhead // head, H * W, -1)
    return x






class SG_Self_Attention(nn.Module):
    def __init__(self, context_dim, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=2, linear=False):
        super().__init__()
        assert context_dim % num_heads == 0, f"dim {context_dim} should be divided by num_heads {num_heads}."

        self.dim = context_dim
        self.num_heads = num_heads
        head_dim = context_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.sr_ratio = sr_ratio

        self.sr = nn.Conv2d(context_dim, context_dim, kernel_size=sr_ratio, stride=sr_ratio)
        self.norm = nn.LayerNorm(context_dim)
        self.act = nn.GELU()
        self.q1 = nn.Linear(context_dim, context_dim // 2, bias=qkv_bias)
        self.kv1 = nn.Linear(context_dim, context_dim, bias=qkv_bias)
        self.q2 = nn.Linear(context_dim, context_dim // 2, bias=qkv_bias)
        self.kv2 = nn.Linear(context_dim, context_dim, bias=qkv_bias)

        self.lepe_linear = nn.Linear(context_dim, context_dim)
        self.lepe_conv = local_conv(context_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(context_dim, context_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        lepe = self.lepe_conv(self.lepe_linear(x).transpose(1, 2).view(B, C, H, W)).view(B, C, -1).transpose(-1, -2)

        q1 = self.q1(x).reshape(B, N, self.num_heads // 2, C // self.num_heads).permute(0, 2, 1, 3)
        x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
        x_1 = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
        x_1 = self.act(self.norm(x_1))
        kv1 = self.kv1(x_1).reshape(B, -1, 2, self.num_heads // 2, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k1, v1 = kv1[0], kv1[1]

        attn1 = (q1 @ k1.transpose(-2, -1)) * self.scale
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn_drop(attn1)
        x1 = (attn1 @ v1).transpose(1, 2).reshape(B, N,C // 2)

        global_mask_value = torch.mean(attn1.detach().mean(1), dim=1)  # B Nk  #max ?  mean ?
        global_mask_value = F.interpolate(global_mask_value.view(B, 1, H // self.sr_ratio, W // self.sr_ratio),
                                          (H, W), mode='nearest')[:, 0]

        q2 = self.q2(x).reshape(B, N, self.num_heads // 2, C // self.num_heads).permute(0, 2, 1,3)
        kv2 = self.kv2(x_.reshape(B, C, -1).permute(0, 2, 1)).reshape(B, -1, 2, self.num_heads // 2,
                                                                      C // self.num_heads).permute(2, 0, 3, 1,4)
        k2, v2 = kv2[0], kv2[1]
        q_window = 4
        window_size = 4
        q2, k2, v2 = window_partition(q2, q_window, H, W), window_partition(k2, window_size, H, W), window_partition(v2, window_size, H, W)
        attn2 = (q2 @ k2.transpose(-2, -1)) * self.scale

        attn2 = attn2.softmax(dim=-1)
        attn2 = self.attn_drop(attn2)
        x2 = (attn2 @ v2)
        x2 = window_reverse(x2, q_window, H, W, self.num_heads // 2)

        local_mask_value = torch.mean(attn2.detach().view(B, self.num_heads // 2, H // window_size * W // window_size,
                                    window_size * window_size, window_size * window_size).mean(1), dim=2)
        local_mask_value = local_mask_value.view(B, H // window_size, W // window_size, window_size,window_size)
        local_mask_value = local_mask_value.permute(0, 1, 3, 2, 4).contiguous().view(B, H, W)

        x = torch.cat([x1, x2], dim=-1)
        x = self.proj(x + lepe)
        x = self.proj_drop(x) + x
        mask = local_mask_value + global_mask_value
        mask_1 = mask.view(B, H * W)
        mask_2 = mask.permute(0, 2, 1).reshape(B, H * W)
        mask = [mask_1, mask_2]

        return x, mask









class SG_Cross_Attention(nn.Module):
    def __init__(self, dim, context_dim, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, linear=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.sr_ratio = sr_ratio

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv1 = nn.Linear(context_dim, dim, bias=qkv_bias)
        self.kv2 = nn.Linear(context_dim, dim, bias=qkv_bias)
        if self.sr_ratio == 8:
            f1, f2 = 64, 16
        elif self.sr_ratio == 4:
            f1, f2 = 48, 12
        elif self.sr_ratio == 2:
            f1, f2 = 2, 1
        self.f1 = nn.Linear(f1, 1)
        self.f2 = nn.Linear(f2, 1)

        self.lepe_linear = nn.Linear(dim, dim)
        self.lepe_conv = local_conv(dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


    def forward(self, x, context, H, W, mask):
        B, N, C = x.shape
        _, _, context_C = context.shape
        lepe = self.lepe_conv(self.lepe_linear(x).transpose(1, 2).view(B, C, H, W)).view(B, C, -1).transpose(-1, -2)

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        mask_1, mask_2 = mask
        mask_sort1, mask_sort_index1 = torch.sort(mask_1, dim=1)
        mask_sort2, mask_sort_index2 = torch.sort(mask_2, dim=1)
        if self.sr_ratio == 8:
            token1, token2, token3 = H * W // 64, H * W // 16, H * W // 1
            token1, token2, token3 = token1 // 4, token2 // 2, token3 // 4
        elif self.sr_ratio == 4:
            token1, token2, token3 = H * W // 48, H * W // 12, H * W // 1
            token1, token2, token3 = token1 // 4, token2 // 2, token3 // 4
        elif self.sr_ratio == 2:
            token1, token2 = H * W // 2, H * W // 1
            token1, token2 = token1 // 2, token2 // 2
        if self.sr_ratio == 4 or self.sr_ratio == 8:
            p1 = torch.gather(context, 1,mask_sort_index1[:, :H * W // 4].unsqueeze(-1).repeat(1, 1, context_C))
            p2 = torch.gather(context, 1,mask_sort_index1[:, H * W // 4:H * W // 4 * 3].unsqueeze(-1).repeat(1, 1, context_C))
            p3 = torch.gather(context, 1, mask_sort_index1[:, H * W // 4 * 3:].unsqueeze(-1).repeat(1, 1, context_C))
            seq1 = torch.cat([self.f1(p1.permute(0, 2, 1).reshape(B, context_C, token1, -1)).squeeze(-1),
                              self.f2(p2.permute(0, 2, 1).reshape(B, context_C, token2, -1)).squeeze(-1),
                              p3.permute(0, 2, 1).reshape(B, context_C, token3, -1).squeeze(-1)],
                             dim=-1).permute(0, 2, 1)

            context_ = context.view(B, H, W, context_C).permute(0, 2, 1, 3).reshape(B, H * W, context_C)
            p1_ = torch.gather(context_, 1,mask_sort_index2[:, :H * W // 4].unsqueeze(-1).repeat(1, 1, context_C))
            p2_ = torch.gather(context_, 1,mask_sort_index2[:, H * W // 4:H * W // 4 * 3].unsqueeze(-1).repeat(1, 1, context_C))
            p3_ = torch.gather(context_, 1, mask_sort_index2[:, H * W // 4 * 3:].unsqueeze(-1).repeat(1, 1, context_C))
            seq2 = torch.cat([self.f1(p1_.permute(0, 2, 1).reshape(B, context_C, token1, -1)).squeeze(-1),
                              self.f2(p2_.permute(0, 2, 1).reshape(B, context_C, token2, -1)).squeeze(-1),
                              p3_.permute(0, 2, 1).reshape(B, context_C, token3, -1).squeeze(-1)],
                             dim=-1).permute(0, 2, 1)
        elif self.sr_ratio == 2:
            p1 = torch.gather(context, 1,mask_sort_index1[:, :H * W // 2].unsqueeze(-1).repeat(1, 1, context_C))
            p2 = torch.gather(context, 1, mask_sort_index1[:, H * W // 2:].unsqueeze(-1).repeat(1, 1, context_C))
            seq1 = torch.cat([self.f1(p1.permute(0, 2, 1).reshape(B, context_C, token1, -1)).squeeze(-1),
                              self.f2(p2.permute(0, 2, 1).reshape(B, context_C, token2, -1)).squeeze(-1)],
                             dim=-1).permute(0, 2, 1)

            context_ = context.view(B, H, W, context_C).permute(0, 2, 1, 3).reshape(B, H * W, context_C)
            p1_ = torch.gather(context_, 1, mask_sort_index2[:, :H * W // 2].unsqueeze(-1).repeat(1, 1, context_C))
            p2_ = torch.gather(context_, 1, mask_sort_index2[:, H * W // 2:].unsqueeze(-1).repeat(1, 1, context_C))
            seq2 = torch.cat([self.f1(p1_.permute(0, 2, 1).reshape(B, context_C, token1, -1)).squeeze(-1),
                              self.f2(p2_.permute(0, 2, 1).reshape(B, context_C, token2, -1)).squeeze(-1)],
                             dim=-1).permute(0, 2, 1)

        kv1 = self.kv1(seq1).reshape(B, -1, 2, self.num_heads // 2, C // self.num_heads).permute(2, 0, 3, 1, 4)
        kv2 = self.kv2(seq2).reshape(B, -1, 2, self.num_heads // 2, C // self.num_heads).permute(2, 0, 3, 1, 4)
        kv = torch.cat([kv1, kv2], dim=2)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x + lepe)
        x = self.proj_drop(x)

        return x













if __name__ == '__main__':
    # 生成随机张量
    # hint [1, 192, 1280]，x [1, 1280, 16, 12]
    # hint [1, 192, 1280]，x [1, 1280, 16, 12]
    # hint [1, 192, 640]，x [1, 1280, 16, 12]
    # hint [1, 768, 640])，x [1, 1280, 32, 24]
    # hint [1, 768, 640]，x [1, 640, 32, 24]
    # hint [1, 768, 320]，x [1, 640, 32, 24]
    # hint [1, 3072, 320])，x [1, 640, 64, 48]
    # hint [1, 3072, 320]，x [1, 320, 64, 48]
    # hint [1, 3072, 320]，x [1, 320, 64, 48]

    hint1, x1 = torch.randn(1, 192, 1280), torch.randn(1, 192, 1280)
    hint2, x2 = torch.randn(1, 192, 1280), torch.randn(1, 192, 1280)
    hint3, x3 = torch.randn(1, 192, 640), torch.randn(1, 192, 1280)
    hint4, x4 = torch.randn(1, 768, 640), torch.randn(1, 768, 1280)
    hint5, x5 = torch.randn(1, 768, 640), torch.randn(1, 768, 640)
    hint6, x6 = torch.randn(1, 768, 320), torch.randn(1, 768, 640)
    hint7, x7 = torch.randn(1, 3072, 320), torch.randn(1, 3072, 640)
    hint8, x8 = torch.randn(1, 3072, 320), torch.randn(1, 3072, 320)
    hint9, x9 = torch.randn(1, 3072, 320), torch.randn(1, 3072, 320)



    # 创建模型
    # model1_self = SG_Self_Attention(dim=640, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0.,proj_drop=0., sr_ratio=2)
    # model1_cross = SG_Cross_Attention(dim=1280, context_dim=640, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0.,proj_drop=0., sr_ratio=2)
    # out1_1, mask = model1_self(hint3, 16, 12)
    # out1_2,_ = model1_cross(x3, out1_1, 16, 12, mask)
    # print(out1_1.shape)
    # print(out1_2.shape)



    # model5_self = SG_Self_Attention(context_dim=640, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0.,proj_drop=0., sr_ratio=4)
    # model5_cross = SG_Cross_Attention(dim=640, context_dim=640, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0.,proj_drop=0., sr_ratio=4)
    # out5_1, mask = model5_self(hint5, 32, 24)
    # out5_2 = model5_cross(x5, out5_1, 32, 24, mask)
    # print(out5_1.shape)
    # print(out5_2.shape)



    model7_self = SG_Self_Attention(context_dim=320, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0.,proj_drop=0., sr_ratio=8)
    model7_cross = SG_Cross_Attention(dim=640, context_dim=320, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0.,proj_drop=0., sr_ratio=8)
    out7_1, mask = model7_self(hint7, 64, 48)
    out7_2 = model7_cross(x7, out7_1, 64, 48, mask)
    print(out7_1.shape)
    print(out7_2.shape)






