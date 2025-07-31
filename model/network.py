import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .clip import clip
from .clip.model import VisionTransformer, ModifiedResNet
from .basis import EnhancedCrossAttention
from .restormer import Restormer

class SlimEnhancer(nn.Module):
    def __init__(self, args):
        super(SlimEnhancer, self).__init__()
        self.cd = 256
        self.cnode = args.cnode
        self.cnum = args.cnum
        self.fusion_method = args.fusion_method
        self.use_image_features = args.use_image_features
        self.use_text_features = args.use_text_features

        self.spatial_net = Restormer(inp_channels=3, out_channels=self.cnum)

        # Load CLIP model and convert to full precision
        self.clip_model, self.preprocess = clip.load(args.clip_model_path + "/" + args.clip_model + ".pt", device='cuda')
        self.clip_model = self.clip_model.float()  # Convert to full precision
        
        # Use vision encoder from CLIP
        self.clip_visual = self.clip_model.visual

        if isinstance(self.clip_visual, VisionTransformer):
            self.clip_output_dim = self.clip_visual.output_dim
        elif isinstance(self.clip_visual, ModifiedResNet):
            self.clip_output_dim = self.clip_visual.output_dim
        else:
            raise ValueError(f"Unknown CLIP visual encoder type: {type(self.clip_visual)}")

        self.text_output_dim = self.clip_model.text_projection.shape[1]
        print(f"CLIP model: {args.clip_model}, Visual output dimension: {self.clip_output_dim}, Text output dimension: {self.text_output_dim}")

        if self.use_image_features and self.use_text_features:
            if self.fusion_method == 'concat':
                self.text_output_dim = 7
                self.combined_dim = self.clip_output_dim + self.text_output_dim
            elif self.fusion_method in ['add', 'multiply']:
                self.combined_dim = self.clip_output_dim
            elif self.fusion_method == 'attention':
                print(f'enhanced_attention is used')
                self.combined_dim = self.clip_output_dim
                self.enhanced_cross_attention = EnhancedCrossAttention(self.clip_output_dim, self.text_output_dim)
            elif self.fusion_method == 'cross_attention':
                print(f'cross_attention is used')
                self.combined_dim = self.clip_output_dim
                self.cross_attention = nn.MultiheadAttention(embed_dim=self.clip_output_dim, num_heads=8, batch_first=True)
        elif self.use_image_features:
            self.combined_dim = self.clip_output_dim
        elif self.use_text_features:
            self.combined_dim = self.text_output_dim
        else:
            raise ValueError("At least one of use_image_features or use_text_features must be True")

        # Modify curve_mapper to output 3 curves per group (RGB)
        self.curve_mapper = nn.Linear(self.combined_dim, self.cnode * self.cnum * 3)

    def fuse_features(self, image_features, text_features):
        if self.fusion_method == 'concat':
            return torch.cat([image_features, text_features], dim=1)
        elif self.fusion_method == 'add':
            return image_features + text_features
        elif self.fusion_method == 'multiply':
            return image_features * text_features
        elif self.fusion_method == 'attention':
            fused_features = self.enhanced_cross_attention(image_features, text_features)
            return fused_features
        elif self.fusion_method == 'cross_attention':
            # Q: text_features, K/V: image_features
            Q = text_features.unsqueeze(1)  # (B, 1, D)
            K = image_features.unsqueeze(1)  # (B, 1, D)
            V = image_features.unsqueeze(1)  # (B, 1, D)
            attn_output, _ = self.cross_attention(Q, K, V)
            return attn_output.squeeze(1)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

    def curve(self, x, func, depth):
        x_ind = x * (depth - 1)
        x_ind = x_ind.long().flatten(2).detach()
        out = torch.gather(func, 2, x_ind)
        return out.reshape(x.size())

    def forward(self, x, text, return_weights=False):
        B, _, H, W = x.size()

        # Get spatial weights from U-Net
        spatial_weights = self.spatial_net(x)  # [B, num_groups, H, W]
        spatial_weights = F.softmax(spatial_weights, dim=1)  # Normalize weights

        combined_features = None

        if self.use_image_features:
            # Resize images to match CLIP input size
            clip_input_size = self.clip_visual.input_resolution
            x_resized = F.interpolate(x, size=(clip_input_size, clip_input_size), mode='bilinear', align_corners=False)
            # Get CLIP image features
            clip_image_features = self.clip_visual(x_resized)
            combined_features = clip_image_features

        if self.use_text_features:
            # Process text input
            text_tokens = clip.tokenize(text).to(x.device)
            text_features = self.clip_model.encode_text(text_tokens)
            if combined_features is None:
                combined_features = text_features
            elif self.use_image_features:
                combined_features = self.fuse_features(clip_image_features, text_features)

        # Generate curve parameters from combined features
        params = self.curve_mapper(combined_features).view(B, self.cnum*3, self.cnode, 1)  # [B, num_groups, 3, nodes, 1]

        if self.cnode != self.cd:
            curves = F.interpolate(
                params, (self.cd, 1), 
                mode='bicubic', align_corners=True
            ).squeeze(3)
        else:
            curves = params.squeeze(3)
        curves = curves.reshape(B, self.cnum, 3, self.cd)
        
        # Apply curves to each channel separately
        x_output = torch.zeros_like(x)
        for c in range(3):  # For each RGB channel
            channel_curves = curves[:, :, c, :]  # [B, num_groups, nodes]
            channel_input = x[:, c:c+1, :, :]  # [B, 1, H, W]
            
            # Apply curves
            channel_output = self.curve(
                channel_input.repeat(1, self.cnum, 1, 1),  # [B, num_groups, H, W]
                channel_curves,  # [B, num_groups, nodes]
                self.cd
            )  # [B, num_groups, H, W]
            
            # Weight the outputs by spatial weights
            weighted_output = (channel_output * spatial_weights).sum(dim=1, keepdim=True)  # [B, 1, H, W]
            x_output[:, c:c+1, :, :] = weighted_output

        if return_weights:
            return x + x_output, spatial_weights
        return x + x_output