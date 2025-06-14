import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma      = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, W, H = x.size()
        proj_query = self.query_conv(x).view(B, -1, W * H).permute(0, 2, 1)
        proj_key   = self.key_conv(x).view(B, -1, W * H)
        attention  = torch.bmm(proj_query, proj_key)  # (B, N, N)
        attention  = F.softmax(attention, dim=-1)
        proj_value = self.value_conv(x).view(B, -1, W * H)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1)).view(B, C, W, H)
        out = self.gamma * out + x
        return out
