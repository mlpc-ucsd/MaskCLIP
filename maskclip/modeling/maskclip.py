from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

from .attention import MultiheadAttention


def gelu(x):
    return x * torch.sigmoid(1.702 * x)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return gelu(x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, clip_patch_size: int, need_masks_embed: bool):
        super().__init__()

        self.n_head = n_head
        self.clip_patch_size = clip_patch_size
        self.attn = MultiheadAttention(d_model, n_head, need_masks_embed=need_masks_embed)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)

    def attention(self, x: torch.Tensor, attn_mask, masks_embed):
        x, _, masks_res = self.attn(x, x, x, need_weights=False, attn_mask=attn_mask, masks_embed=masks_embed)
        return x, masks_res

    def forward(self, x: torch.Tensor, masks: torch.Tensor, masks_embed: torch.Tensor = None):
        l, b, d = x.shape
        _, q, _, _ = masks.shape
        masks = (masks.sigmoid() >= 0.5).float()
        masks = F.max_pool2d(masks, self.clip_patch_size).flatten(2)

        attn_mask = torch.empty((b, l, l), device=x.device, dtype=torch.bool)
        attn_mask[:, :, :] = False
        attn_mask[:, :, :q] = True
        attn_mask[:, :q, q+1:] = masks == 0.
        
        attn_mask = torch.repeat_interleave(attn_mask, self.n_head, dim=0)

        x_res, masks_res = self.attention(self.ln_1(x), attn_mask=attn_mask, masks_embed=masks_embed)
        x = x + x_res
        x = x + self.mlp(self.ln_2(x))
        return x, masks_res


class Transformer(nn.Module):
    def __init__(
            self, width: int, layers: int, heads: int,
            clip_input_resolution, clip_patch_size, clip_width, clip_layers, clip_heads
        ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, clip_patch_size, (i + 1) % 6 == 0) 
            for i in range(layers)]
        )

        self.clip_input_resolution = clip_input_resolution
        self.clip_patch_size = clip_patch_size
        self.clip_width = clip_width
        self.clip_layers = clip_layers
        self.clip_heads = clip_heads
        self.clip_num_patches_dim = clip_input_resolution // clip_patch_size
        self.clip_num_patches = self.clip_num_patches_dim ** 2

        self.idxs = [5, 11, 17, 23]
        self.conv1_added_params = nn.Sequential(
            *[nn.Conv2d(1, clip_width, clip_patch_size, clip_patch_size, bias=True) 
                for i in range(len(self.idxs))]
        )
        self.conv3_added_params = nn.Sequential(
            *[nn.Conv2d(clip_width, clip_patch_size ** 2, 1, 1, bias=True) 
                for i in range(len(self.idxs))]
        )
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            nn.init.uniform_(m.bias)

    def forward(self, x: torch.Tensor, masks: torch.Tensor):

        masks_list = []
        for i, block in enumerate(list(self.resblocks.modules())[0]):

            if i in self.idxs:
                masks_embed = self.conv1_added_params[i//6](masks.tanh().unsqueeze(2).flatten(0, 1)) #(Bxq)x1xhxw
                masks_embed = masks_embed.reshape(
                    x.shape[1], masks.shape[1], self.clip_width, self.clip_num_patches
                ).permute(1, 3, 0, 2)
                x, masks_res = block(x, masks, masks_embed)  # masks_res: Bx100x256x1024

                masks_res = masks_res.permute(0, 1, 3, 2).reshape(
                    x.shape[1], 100, self.clip_width, self.clip_num_patches_dim, self.clip_num_patches_dim
                ).flatten(0, 1)
                masks_res = gelu(masks_res)
                masks_res = self.conv3_added_params[i//6](masks_res)
                masks_res = masks_res.reshape(
                    -1, self.clip_patch_size, self.clip_patch_size, self.clip_num_patches_dim, self.clip_num_patches_dim
                ).permute(0,3,1,4,2)
                masks_res = masks_res.flatten(1, 2).flatten(2, 3)
                masks_res = masks_res.squeeze(1).reshape(
                    x.shape[1], 100, self.clip_input_resolution, self.clip_input_resolution
                )

                masks += masks_res
                masks_list.append(masks)
            else:
                x, _ = block(x, masks)

        return x, masks_list


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, input_resolution, patch_size, width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, masks: torch.Tensor):
        q = masks.shape[1]

        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        cls_embed = x[0:1]
        cls_embed = cls_embed.repeat(q, 1, 1)
        x = torch.cat([cls_embed, x], dim=0)

        x, masks_list = self.transformer(x, masks)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, :q, :])

        if self.proj is not None:
            x = x @ self.proj

        return x, masks_list


class MaskCLIP(nn.Module):
    def __init__(
            self, 
            # initialize CLIP
            clip_model_name,
            input_resolution,
            patch_size,
            width,
            layers,
            heads,
            output_dim,
            temperature
        ):
        super().__init__()

        self.temperature = temperature

        self.visual = VisionTransformer(
                input_resolution=input_resolution,
                patch_size=patch_size,
                width=width,
                layers=layers,
                heads=heads,
                output_dim=output_dim
            )

        clip_, _ = clip.load(clip_model_name, device='cpu')
        self.visual.load_state_dict(clip_.visual.state_dict(), strict=False)

        del clip_

    def forward(self, x, masks, txt_embed):
        outputs = {}

        img_fet, masks_list = self.visual(x, masks)

        logits = torch.einsum('b q c, n c -> b q n', img_fet / img_fet.norm(dim=-1, keepdim=True), 
            txt_embed.to(img_fet.device)) / self.temperature
        outputs['pred_logits'] = logits
        outputs['pred_masks'] = masks_list[-1]
        outputs['aux_outputs'] = []
        for i in range(len(masks_list) - 1):
            outputs['aux_outputs'].append({'pred_logits': logits, 'pred_masks': masks_list[i]})

        return outputs

