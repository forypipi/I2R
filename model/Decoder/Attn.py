from typing import List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import pytorch_lightning as pl
from einops import rearrange, repeat

class SelfAttn(pl.LightningModule):
    def __init__(self, in_channels=256, embedding_channels=384, head=12, dropout=0.5, project_out=True):
        super(SelfAttn, self).__init__()
        self.in_channels = in_channels
        self.embedding_channels = embedding_channels
        self.head = head
        self.scale: float = (self.embedding_channels//head) ** -0.5    # 1/16
  
        self.Wq = nn.Linear(self.in_channels, self.embedding_channels)
        self.Wk = nn.Linear(self.in_channels, self.embedding_channels)
        self.Wv = nn.Linear(self.in_channels, self.embedding_channels)
        self.LN1 = nn.LayerNorm(self.in_channels)
 
        self.ffn = nn.Sequential(
                    nn.LayerNorm(self.in_channels),
                    nn.Linear(self.in_channels, self.in_channels*2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(self.in_channels*2, self.in_channels),
                    nn.Dropout(dropout),
                )
        
        self.to_out = nn.Sequential(
                    nn.Linear(self.embedding_channels, self.in_channels),
                    nn.Dropout(dropout)
                ) if project_out else nn.Identity()


    def forward(self, x):
        '''
        :param x: [batch_size, c, h, w], Q
        :param x_s: [batch_size, c, h, w], KV
        :param ym_s: [batch_size, 1, h, w]
        :param pad_mask: [batch_size, seq_len, seq_len]
        :return: [batch_size, c, h, w]
        '''
        b, c, h, w = x.shape

        x = rearrange(x, 'b c h w -> b (h w) c')
        x_Q = self.LN1(x)   # [batch_size, h*w, c] = [b, 900, 256]


        Q = repeat(self.Wq(x_Q), 'b e (h c) ->b h e c', h=self.head)  # [batch_size, head, hw, emb_dim//head] = [b, 8, 900, 32]
        V = repeat(self.Wv(x_Q), 'b e (h c) ->b h e c', h=self.head)
        K = repeat(self.Wk(x_Q), 'b e (h c) ->b h e c', h=self.head)  # [batch_size, head, hw, emb_dim//head] = [b, 8, 900, 32]
 
        # [batch_size, head, h*w (x), h*w (x_s)]
        att_weights = torch.einsum('bhid,bhjd -> bhij', Q, K)  # [b, 8, 900, 900]
        att_weights = att_weights * self.scale

        att_weights_q = F.softmax(att_weights, dim=-1)
        out = torch.einsum('bhij, bhjd -> bhid', att_weights_q, V)   # [b, 8, 900, 32]

        out = self.to_out(rearrange(out, 'b n h c -> b h (n c)'))

        out = out + x
        out = out + self.ffn(out)   # [batch_size, hw, c]
        out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)   # [b, 256, h, w]

        return out
