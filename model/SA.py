import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import pytorch_lightning as pl
from einops import rearrange, repeat

class SA(pl.LightningModule):
    def __init__(self, in_channels=256):
        super(SA, self).__init__()
        self.in_channels = in_channels
        self.scale: float = in_channels ** -0.5    # 1/16

        self.Wq = nn.Linear(in_channels, in_channels)
        self.Wk = nn.Linear(in_channels, in_channels)
        self.Wv = nn.Linear(in_channels, in_channels)
        self.LN_attn = nn.LayerNorm(in_channels)
        self.LN_ffn = nn.LayerNorm(in_channels)
        self.dropout = nn.Dropout(0.1)
 
        self.proj_out = nn.Sequential(
            nn.Linear(in_channels, in_channels*2),
            nn.GELU(),
            nn.Linear(in_channels*2, in_channels), 
            nn.Dropout(0.1),
        )

    def forward(self, x):
        '''
        :param x: [batch_size, c, h, w]
        :return: [batch_size, c, h, w]
        '''
        b, c, h, w = x.shape

        x = rearrange(x, 'b c h w -> b (h w) c')

        Q = self.Wq(x)  # [batch_size, h*w, emb_dim] = [b, 900, 256]
        K_s = self.Wk(x)  # [batch_size, seq_len=h*w, emb_dim] = [b, 900, 256]
        V = self.Wv(x)
 
        # [batch_size, h*w (x), h*w (x)]
        att_weights = torch.einsum('bid,bjd -> bij', Q, K_s)  # [b, 900, 900]
        att_weights = att_weights * self.scale

        att_weights_q = F.softmax(att_weights, dim=-1)
        out = torch.einsum('bij, bjd -> bid', att_weights_q, V)   # [b, 900, 256]

        out = self.dropout(out)
        out = self.LN_attn(out + x)

        out = self.LN_ffn(out + self.proj_out(out))   # [batch_size, hw, c]

        
        out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)   # [b, 256, h, w]

        return out