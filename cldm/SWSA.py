import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F


def with_pos_embed(x, pos):
    return x if pos is None else x + pos


class CrossAttentionLayer(nn.Module):
    def __init__(self, channels, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(channels, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

        self.activation = F.relu
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, 
                     q, 
                     kv,
                     q_pos = None, 
                     k_pos = None,
                     mask = None,):
        h = q
        q = with_pos_embed(q, q_pos).transpose(0, 1)
        k = with_pos_embed(kv, k_pos).transpose(0, 1)
        v = kv.transpose(0, 1)
        h1 = self.multihead_attn(q, k, v, attn_mask=mask)[0]
        h1 = h1.transpose(0, 1)
        h = h + self.dropout(h1)
        h = self.norm(h)
        return h



class SWSA(nn.Module):
    def __init__(self):
        super().__init__()
        self.garment_conv = nn.ModuleList()
        self.garment_conv.append(nn.Conv2d(320, 320, kernel_size=1, stride=1, padding=0))
        self.garment_conv.append(nn.Conv2d(320, 320, kernel_size=1, stride=1, padding=0))
        self.garment_conv.append(nn.Conv2d(320, 320, kernel_size=1, stride=1, padding=0))
        self.garment_conv.append(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1))
        self.garment_conv.append(nn.Conv2d(320, 640, kernel_size=1, stride=1, padding=0))
        self.garment_conv.append(nn.Conv2d(640, 640, kernel_size=1, stride=1, padding=0))
        self.garment_conv.append(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1))
        self.garment_conv.append(nn.Conv2d(640, 1280, kernel_size=1, stride=1, padding=0))
        self.garment_conv.append(nn.Conv2d(1280, 1280, kernel_size=1, stride=1, padding=0))
        self.channel_list = [320, 320, 320, 320, 640, 640, 640, 1280, 1280]
        self.channel_list2 = [320, 320, 320, 320, 320, 640, 640, 640, 1280]
        self.seq_list = [3072, 3072, 3072, 768, 768, 768, 192, 192, 192]

        self.people_query = [nn.Embedding(self.channel_list[i], self.seq_list[i]) for i in range(9)]
        self.garment_query = [nn.Embedding(self.channel_list[i], self.seq_list[i]) for i in range(9)]
        self.people_query_cross = nn.ModuleList([CrossAttentionLayer(self.channel_list[i], 8, 0.1) for i in range(9)])
        self.garment_query_cross = nn.ModuleList([CrossAttentionLayer(self.channel_list[i], 8, 0.1) for i in range(9)])
        self.people_aliment_cross = nn.ModuleList([CrossAttentionLayer(self.channel_list[i], 8, 0.1) for i in range(9)])
        self.garment_aliment_cross = nn.ModuleList([CrossAttentionLayer(self.channel_list[i], 8, 0.1) for i in range(9)])
        self.FFW_module = nn.ModuleList([nn.Conv2d(self.channel_list[i], self.channel_list2[i], 1, 1, 0) for i in range(9)])
        self.relu = nn.ReLU()



    def forward(self, outs, hint):
        garment_hints = []
        output_fea = []
        for module in self.garment_conv:
            hint = module(hint)
            garment_hints.append(hint)

        for i in range(len(self.garment_query)):
            bs, c, h, w = outs[i].shape
            p_query = self.people_query[i].weight.unsqueeze(0).repeat(bs, 1, 1)
            g_query = self.garment_query[i].weight.unsqueeze(0).repeat(bs, 1, 1)
            people_fea = outs[i].reshape(bs, self.channel_list[i], -1)
            garment_fea = garment_hints[i].reshape(bs, self.channel_list[i], -1)

            people_local_fea = self.people_query_cross[i](p_query.permute(2,0,1), people_fea.permute(2,0,1))
            garment_local_fea = self.garment_query_cross[i](g_query.permute(2,0,1), garment_fea.permute(2,0,1))
            people_aliment_fea = self.people_aliment_cross[i](garment_local_fea, people_local_fea)
            garment_aliment_fea = self.garment_aliment_cross[i](people_local_fea, garment_local_fea)
            people_aliment_fea = people_aliment_fea.permute(2,0,1).reshape(bs, c, h, w) + outs[i]
            garment_aliment_fea = garment_aliment_fea.permute(2,0,1).reshape(bs, c, h, w) + garment_hints[i]

            summ_fea = self.relu(self.FFW_module[i](people_aliment_fea + garment_aliment_fea))
            output_fea.append(summ_fea)
        
        output_fea.append(outs[9])
        output_fea.append(outs[10])
        output_fea.append(outs[11])

        return output_fea







