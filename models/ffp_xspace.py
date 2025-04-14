import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FourierFeatureEncoder(nn.Module):
    def __init__(self, in_dim=2, mapping_size=64, scale=10):
        super().__init__()
        self.B = torch.randn((mapping_size, in_dim)) * scale

    def forward(self, x):
        x_proj = 2 * math.pi * x @ self.B.T
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class ViTWithFFPInput(nn.Module):
    def __init__(
        self,
        coord_dim=2,
        mapping_size=64,
        emb_dim=256,
        depth=6,
        num_heads=8,
        mlp_dim=512,
        out_dim=3,
        dropout=0.1,
        use_cls_token=False,
    ):
        super().__init__()
        self.ffp = FourierFeatureEncoder(in_dim=coord_dim, mapping_size=mapping_size)
        self.input_dim = 2 * mapping_size
        self.token_embedding = nn.Linear(self.input_dim, emb_dim)

        self.use_cls_token = use_cls_token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))

        self.pos_embedding = None  # 初始化后按输入 size 创建

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=num_heads, dim_feedforward=mlp_dim, dropout=dropout, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.output_head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, out_dim)
        )

    def forward(self, coords):
        """
        coords: (B, N, 2) where N is number of coordinates (e.g. H*W)
        return: (B, N, 3) - predicted RGB value per pixel
        """
        B, N, _ = coords.shape
        x = self.ffp(coords.view(-1, 2)).view(B, N, -1)      # (B, N, FFP_dim)
        x = self.token_embedding(x)                          # (B, N, emb_dim)

        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)    # (B, 1, emb_dim)
            x = torch.cat((cls_tokens, x), dim=1)            # (B, N+1, emb_dim)

        # Position embedding
        if self.pos_embedding is None or self.pos_embedding.shape[1] != x.shape[1]:
            self.pos_embedding = nn.Parameter(torch.zeros(1, x.shape[1], x.shape[2]))
            nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        x = x + self.pos_embedding                          # (B, N, emb_dim)
        x = x.permute(1, 0, 2)                               # Transformer expects (N, B, E)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)                               # (B, N, emb_dim)

        if self.use_cls_token:
            x = x[:, 1:, :]  # Skip the class token

        out = self.output_head(x)                            # (B, N, 3)
        return torch.sigmoid(out)                            # RGB in [0, 1]

# H, W = 128, 128
# coords = torch.stack(torch.meshgrid(
#     torch.linspace(-1, 1, H),
#     torch.linspace(-1, 1, W),
# ), dim=-1).reshape(1, -1, 2).cuda()  # (1, H*W, 2)

# model = ViTWithFFPInput(use_cls_token=False).cuda()
# output = model(coords)  # (1, H*W, 3)
# img = output.view(H, W, 3).detach().cpu()

