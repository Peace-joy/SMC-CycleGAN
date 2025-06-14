import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self, dim, n_levels=4):
        super().__init__()
        self.n_levels = n_levels
        chunk_dim = dim // n_levels

        # Multi-scale convolution layers with BatchNorm added
        self.mfr = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(chunk_dim, chunk_dim, 3, 1, 1, groups=chunk_dim),
                nn.BatchNorm2d(chunk_dim)  # Add BatchNorm
            ) for _ in range(self.n_levels)
        ])

        # Feature Aggregation
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)

        # Using SiLU (Swish) as activation function instead of GELU
        self.act = nn.SiLU()  # Swish activation function

    def forward(self, x):
        h, w = x.size()[-2:]
        xc = x.chunk(self.n_levels, dim=1)  # Chunk the input into n_levels parts

        out = []
        for i in range(self.n_levels):
            if i > 0:
                p_size = (h // 2 ** i, w // 2 ** i)  # Calculate pool size for each level
                s = F.adaptive_max_pool2d(xc[i], p_size)  # Adaptive max pooling
                s = self.mfr[i](s)  # Apply 3x3 convolution
                s = F.interpolate(s, size=(h, w), mode='bilinear', align_corners=True)  # Use bilinear interpolation
            else:
                s = self.mfr[i](xc[i])  # For the first level, no downsampling
            out.append(s)

        # Aggregate features from all levels
        out = self.aggr(torch.cat(out, dim=1))
        # Apply activation function and modulate the input
        out = self.act(out) * x  # Spatially adaptive feature modulation
        return out


# Weight initialization function
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)



