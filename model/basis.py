import torch
import torch.nn as nn

class EnhancedCrossAttention(nn.Module):
    def __init__(self, image_dim, text_dim, num_heads=8, num_layers=3):
        super().__init__()
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.image_proj = nn.Linear(image_dim, image_dim)
        self.text_proj = nn.Linear(text_dim, text_dim)

        self.txt_to_img_attention = nn.ModuleList([nn.MultiheadAttention(image_dim, num_heads) for _ in range(num_layers)])
        self.img_to_txt_attention = nn.ModuleList([nn.MultiheadAttention(text_dim, num_heads) for _ in range(num_layers)])

        self.image_ffn = nn.Sequential(
            nn.Linear(image_dim, image_dim * 4),
            nn.ReLU(),
            nn.Linear(image_dim * 4, image_dim)
        )
        self.text_ffn = nn.Sequential(
            nn.Linear(text_dim, text_dim * 4),
            nn.ReLU(),
            nn.Linear(text_dim * 4, text_dim)
        )

        self.layer_norms = nn.ModuleList([nn.LayerNorm(image_dim) for _ in range(num_layers * 2)])

        self.gate = nn.Linear(image_dim * 2, image_dim)

        self.feature_enhancer = nn.Sequential(
            nn.Linear(image_dim, image_dim * 2),
            nn.ReLU(),
            nn.Linear(image_dim * 2, image_dim)
        )

    def forward(self, image_features, text_features):
        image_proj = self.image_proj(image_features)
        text_proj = self.text_proj(text_features)

        image_proj = image_proj.unsqueeze(1)  # [B, 1, D]
        text_proj = text_proj.unsqueeze(1)    # [B, 1, D]

        for i in range(self.num_layers):
            attn_img_to_txt, _ = self.img_to_txt_attention[i](
                query=image_proj, key=text_proj, value=text_proj
            )
            attn_img_to_txt = self.layer_norms[i*2](attn_img_to_txt)

            attn_txt_to_img, _ = self.txt_to_img_attention[i](
                query=text_proj, key=image_proj, value=image_proj
            )
            attn_txt_to_img = self.layer_norms[i*2+1](attn_txt_to_img)

            image_proj = image_proj + self.image_ffn(attn_txt_to_img)
            text_proj = text_proj + self.text_ffn(attn_img_to_txt)

        gate = torch.sigmoid(self.gate(torch.cat([image_proj, text_proj], dim=-1)))
        fused = gate * image_proj + (1 - gate) * text_proj

        enhanced = self.feature_enhancer(fused)

        return enhanced.squeeze(1)  # [B, D]