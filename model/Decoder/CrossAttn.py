from typing import List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import pytorch_lightning as pl
from einops import rearrange, repeat

class CrossAttn(pl.LightningModule):
    def __init__(self, in_channels=256):
        super(CrossAttn, self).__init__()
        self.in_channels = in_channels
        self.scale: float = in_channels ** -0.5    # 1/16

        self.Wq = nn.Linear(in_channels, in_channels)
        self.Wk = nn.Linear(in_channels, in_channels)
        self.Wv = nn.Linear(in_channels, in_channels)
        self.LN = nn.LayerNorm(in_channels)
 
        self.proj_out = nn.Sequential(
            nn.Linear(in_channels, in_channels*2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(in_channels*2, in_channels), 
            nn.Dropout(0.1),
            nn.LayerNorm(in_channels),
        )

    def forward(self, x, x_s, ym_s=None, ym=None, pad_mask=None):
        '''
        :param x: [batch_size, c, h, w], Q
        :param x_s: [batch_size, c, h, w], KV
        :param ym_s: [batch_size, 1, h, w]
        :param pad_mask: [batch_size, seq_len, seq_len]
        :return: [batch_size, c, h, w]
        '''
        b, c, h, w = x.shape

        x = rearrange(x, 'b c h w -> b (h w) c')
        x_s = rearrange(x_s, 'b c h w -> b (h w) c')

        Q = self.Wq(x)  # [batch_size, h*w, emb_dim] = [b, 900, 256]
        K_s = self.Wk(x_s)  # [batch_size, seq_len=h*w, emb_dim] = [b, 900, 256]
        V = self.Wv(x_s)
 
        # [batch_size, h*w (x), h*w (x_s)]
        att_weights = torch.einsum('bid,bjd -> bij', Q, K_s)  # [b, 900, 900]
        att_weights = att_weights * self.scale

        if pad_mask is not None:
            # [batch_size, h*w, seq_len]
            att_weights = att_weights.masked_fill(pad_mask, -1e9)

        # query branch
        affinity_mask = torch.zeros_like(att_weights, device=att_weights.device)
        
        if not self.training:
            ym = None
        affinity_mask, mask_ratio = I2R_mask(att_weights, ym_s, ym)
        # affinity_mask, mask_ratio = DaMa_mask(att_weights, ym_s, ym)
        # affinity_mask = Official_cyctr_mask(att_weights, ym_s)

        att_weights_q = F.softmax(att_weights+affinity_mask, dim=-1)
        out = torch.einsum('bij, bjd -> bid', att_weights_q, V)   # [b, 900, 256]

        out = self.LN(out) + x
        out = out + self.proj_out(out)   # [batch_size, hw, c]
        out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)   # [b, 256, h, w]

        return out, mask_ratio
    
def check_max_value_mask(max_value_index: torch.Tensor, mask: torch.Tensor):
    '''
    :param: max_value, [b, hw]
    :param: mask, [b, hw]
    '''
    # 使用gather
    mask = rearrange(mask, 'b c h w -> b (c h w)')
    mask_values = torch.gather(mask, 1, max_value_index)  # 沿着第三个维度收集
    return mask_values

def count_mask(affinity_mask: torch.Tensor):
    count_neg_inf = (affinity_mask == float('-inf')).sum(dim=(1, 2))
    total_elements = affinity_mask.size(1) * affinity_mask.size(2)
    mask_ratio = count_neg_inf.float() / total_elements
    return mask_ratio

def I2R_mask(attn_weights: torch.Tensor, ym_s: torch.Tensor, ym: Union[torch.Tensor,None] = None, ):
    '''
    :param: attn_weights: [batch_size, h\*w (x), h\*w (x_s)]
    :param: ym_s and ym, [batch_size, 1, h, w]
    :return: attn_mask: [batch_size, h\*w (x), h\*w (x_s)], ym: [batch_size, 1, h, w]
    '''
    b, c, h, w = ym_s.shape
    affinity_mask = torch.zeros((b, h*w*h*w), device=attn_weights.device)

    k2q_sim_idx = attn_weights.max(1)[1] # [bs, hw]      # max Q's index

    if ym is not None:
        K_index_mask = rearrange(ym_s, 'b c h w -> b (c h w)')  # [b, h*w]
        
        # 生成 key→query 掩码条件
        maxQ_mask = check_max_value_mask(k2q_sim_idx, ym)
        K1_xor_Q = K_index_mask ^ maxQ_mask  # key 与 query 掩码差异
    
        # 初始化 affinity_list
        affinity_list = torch.zeros((b, h*w), dtype=torch.int64, device=attn_weights.device)
        
        # Case 1: key→query 验证
        mask_k2q = (K1_xor_Q != 0)
        if mask_k2q.any():
            j_indices = repeat(torch.arange(h*w, device=attn_weights.device), 'c -> b c', b=b)
            selected_j = j_indices[mask_k2q]
            affinity_list[mask_k2q] = k2q_sim_idx[mask_k2q] * h*w + selected_j

        # 合并掩码
        affinity_mask = affinity_mask.scatter(1, affinity_list, -torch.inf)

    affinity_mask = repeat(affinity_mask, 'b (q k) -> b k q', q=h*w, k=h*w)

    return affinity_mask, count_mask(affinity_mask)

def DaMa_mask(attn_weights: torch.Tensor, ym_s: torch.Tensor, ym: Union[torch.Tensor,None] = None, ):
    '''
    :param: attn_weights: [batch_size, h\*w (x), h\*w (x_s)]
    :param: ym_s and ym, [batch_size, 1, h, w]
    :return: attn_mask: [batch_size, h\*w (x), h\*w (x_s)], ym: [batch_size, 1, h, w]
    '''
    b, c, h, w = ym_s.shape
    affinity_mask = torch.zeros((b, h*w*h*w), device=attn_weights.device)

    k2q_sim_idx = attn_weights.max(1)[1] # [bs, hw]      # max Q's index
    q2k_sim_idx = attn_weights.max(2)[1] # [bs, hw]      
    argmax_index_y  = torch.gather(q2k_sim_idx, 1, k2q_sim_idx)     # max K's index of max Q's index

    if ym is not None:
        K_index_mask = rearrange(ym_s, 'b c h w -> b (c h w)')
        maxQ_mask = check_max_value_mask(k2q_sim_idx, ym)
        maxK_mask = check_max_value_mask(argmax_index_y, ym_s)

        K1_xor_Q = K_index_mask ^ maxQ_mask
        Q_xor_K2 = maxK_mask ^ maxQ_mask
        
        # start_S is different from Q but Q is same as end_S
        affinity_list = torch.zeros((b, h*w), dtype=torch.int64, device=attn_weights.device)
        mask = (K1_xor_Q != 0) & (Q_xor_K2 == 0)
        j_indices = repeat(torch.arange(h*w, device=attn_weights.device), 'c -> b c', b=b)
        selected_j_indices = j_indices[mask]
        affinity_list[mask] = k2q_sim_idx[mask] * h*w + selected_j_indices

        # start_S is different from Q but Q is same as end_S
        mask = (K1_xor_Q == 0) & (Q_xor_K2 != 0)
        affinity_list[mask] = k2q_sim_idx[mask] * h*w + argmax_index_y[mask]

        affinity_mask = affinity_mask.scatter(1, affinity_list, -torch.inf)
        for i in range(len(affinity_mask)):
            if (K_index_mask[i, 0]==maxK_mask[i, 0]) \
                or (K1_xor_Q[i, 0]!=0 and Q_xor_K2[i, 0]==0 and k2q_sim_idx[i, 0]!=0) \
                    or (K1_xor_Q[i, 0]==0 and Q_xor_K2[i, 0]!=0 and k2q_sim_idx[i, 0]!=0 and argmax_index_y[i, 0]!=0):
                affinity_mask[i,0] = 0

        # both different from Q
        # mask = (K1_xor_Q != 0) & (Q_xor_K2 != 0)
        # ym = rearrange(ym, 'b c h w -> b (c h w)')
        # y_mask_copy = ym.clone()
        # y_mask_replace_index = torch.where(mask, k2q_sim_idx, 0)
        # new_y_mask = ym.scatter_(1, y_mask_replace_index, ~ym.gather(1, y_mask_replace_index))
        # for i in range(len(affinity_mask)):
        #     if K1_xor_Q[i,0]==0 or Q_xor_K2[i,0]==0:
        #         new_y_mask[i,0] = y_mask_copy[i,0]

        # affinity_mask = repeat(affinity_mask, 'b (a c) -> b a c', a=h*w, c=h*w)
        # new_y_mask = repeat(new_y_mask, 'b (c a d) -> b c a d', c=1, a=h, d=w)
    
    # else:
    #     K_index_mask = rearrange(ym_s, 'b c h w -> b (c h w)')
    #     maxK_mask = check_max_value_mask(argmax_index_y, ym_s)  # [b, hw]

    #     K1_xor_K2 = K_index_mask ^ maxK_mask        # same 0, diff 1
        
    #     # start_S is different from end_S
    #     mask = (K1_xor_K2 != 0)
        
        # affinity_mask[mask,:] = -torch.inf

    # slower 50%, but more readable
    # for i in range(b):
    #     for j in range(h*w):
    #         if K1_xor_Q[i, j]!=0 and Q_xor_K2[i, j]==0:     # start_S is different from Q but Q is same as end_S
    #             affinity_mask[i, k2q_sim_idx[i, j], j] = -torch.inf

    #         elif K1_xor_Q[i, j]==0 and Q_xor_K2[i, j]!=0:     # start_S is same from Q but Q is different as end_S
    #             affinity_mask[i, k2q_sim_idx[i, j], argmax_index_y[i, j]] = -torch.inf
            
    #         elif K1_xor_Q[i, j]!=0 and Q_xor_K2[i, j]!=0:     # modify Q's mask
    #             new_y_mask[i, 0, k2q_sim_idx[i, j]//w, k2q_sim_idx[i, j]%w] = ym_s[i, 0, j//w, j%w]

    affinity_mask = repeat(affinity_mask, 'b (a c) -> b a c', a=h*w, c=h*w)

    return affinity_mask, count_mask(affinity_mask)

def Official_cyctr_mask(attn_weights: torch.Tensor, ym_s: torch.Tensor):
    '''
    :param: attn_weights: [batch_size, h\*w (x), h\*w (x_s)]
    :param: ym_s, [batch_size, 1, h, w]
    :return: attn_mask: [batch_size, h\*w (x), h\*w (x_s)]
    '''
    b, c, h, w = ym_s.shape
    affinity_mask = torch.zeros((b, h*w, h*w), device=attn_weights.device)

    k2q_sim_idx = attn_weights.max(1)[1] # [bs, hw]      # max Q's index
    q2k_sim_idx = attn_weights.max(2)[1] # [bs, hw]      
    argmax_index_y  = torch.gather(q2k_sim_idx, 1, k2q_sim_idx)     # max K's index of max Q's index

    K_index_mask = rearrange(ym_s, 'b c h w -> b (c h w)')
    maxK_mask = check_max_value_mask(argmax_index_y, ym_s)

    K1_xor_K2 = K_index_mask ^ maxK_mask

    # start_S is different from Q but Q is same as end_S
    mask = (K1_xor_K2 != 0)
    affinity_mask[mask,:] = -torch.inf
    affinity_mask = rearrange(affinity_mask, 'b a c -> b c a')

    # slower 50%, but more readable
    # affinity_mask = torch.zeros((b, h*w, h*w), device=attn_weights.device)
    # for i in range(b):
    #     for j in range(h*w):
    #         if K1_xor_K2[i, j]!=0:
    #             affinity_mask[i,:,j] = -torch.inf

    return affinity_mask