import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from misc import torchutils
from net import resnet50
from net.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
import math


class Net(VisionTransformer):
    def __init__(self, pretrained_cfg=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = 20 #HC alert
        self.head = nn.Conv2d(self.embed_dim, self.num_classes, kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head.apply(self._init_weights)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, self.num_classes, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_classes, self.embed_dim))

        self.backbone = nn.ModuleList([])

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        print(self.training)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - self.num_classes
        N = self.pos_embed.shape[1] - self.num_classes
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0:self.num_classes]
        patch_pos_embed = self.pos_embed[:, self.num_classes:]
        dim = x.shape[-1]

        w0 = w // self.patch_embed.patch_size[0]
        h0 = h // self.patch_embed.patch_size[0]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward_features(self, x, n=12):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        x = self.pos_drop(x)
        attn_weights = []

        for i, blk in enumerate(self.blocks):
            x, weights_i = blk(x)
            attn_weights.append(weights_i)

        return x[:, 0:self.num_classes], x[:, self.num_classes:], attn_weights

    def forward(self, x, return_att=False, n_layers=12, attention_type='fused'):
        w, h = x.shape[2:]
        x_cls, x_patch, attn_weights = self.forward_features(x)
        n, p, c = x_patch.shape
        if w != h:
            w0 = w // self.patch_embed.patch_size[0]
            h0 = h // self.patch_embed.patch_size[0]
            x_patch = torch.reshape(x_patch, [n, w0, h0, c])
        else:
            x_patch = torch.reshape(x_patch, [n, int(p ** 0.5), int(p ** 0.5), c])
        x_patch = x_patch.permute([0, 3, 1, 2])
        x_patch = x_patch.contiguous()
        x_patch = self.head(x_patch)
        x_patch_logits = self.avgpool(x_patch).squeeze(3).squeeze(2)

        attn_weights = torch.stack(attn_weights)  # 12 * B * H * N * N
        attn_weights = torch.mean(attn_weights, dim=2)  # 12 * B * N * N

        feature_map = x_patch.detach().clone()  # B * C * 14 * 14
        feature_map = F.relu(feature_map)

        n, c, h, w = feature_map.shape

        mtatt = attn_weights[-n_layers:].sum(0)[:, 0:self.num_classes, self.num_classes:].reshape([n, c, h, w])

        if attention_type == 'fused':
            cams = mtatt * feature_map  # B * C * 14 * 14
        elif attention_type == 'patchcam':
            cams = feature_map
        else:
            cams = mtatt

        patch_attn = attn_weights[:, :, self.num_classes:, self.num_classes:]

        x_cls_logits = x_cls.mean(-1)

        return x_cls_logits
    
    def trainable_parameters(self):
        params = list(self.parameters())
        return params

class Net_CAM(Net):
    def __init__(self, pretrained_cfg=None, *args, **kwargs):
        super().__init__(pretrained_cfg=None, *args, **kwargs)
    
    def forward(self, x, return_att=False, n_layers=12, attention_type='fused'):
        w, h = x.shape[2:]
        x_cls, x_patch, attn_weights = self.forward_features(x)
        n, p, c = x_patch.shape
        if w != h:
            w0 = w // self.patch_embed.patch_size[0]
            h0 = h // self.patch_embed.patch_size[0]
            x_patch = torch.reshape(x_patch, [n, w0, h0, c])
        else:
            x_patch = torch.reshape(x_patch, [n, int(p ** 0.5), int(p ** 0.5), c])
        x_patch = x_patch.permute([0, 3, 1, 2])
        x_patch = x_patch.contiguous()
        x_patch = self.head(x_patch)
        x_patch_logits = self.avgpool(x_patch).squeeze(3).squeeze(2)

        attn_weights = torch.stack(attn_weights)  # 12 * B * H * N * N
        attn_weights = torch.mean(attn_weights, dim=2)  # 12 * B * N * N

        feature_map = x_patch.detach().clone()  # B * C * 14 * 14
        feature_map = F.relu(feature_map)

        n, c, h, w = feature_map.shape

        mtatt = attn_weights[-n_layers:].sum(0)[:, 0:self.num_classes, self.num_classes:].reshape([n, c, h, w])

        if attention_type == 'fused':
            cams = mtatt * feature_map  # B * C * 14 * 14
        elif attention_type == 'patchcam':
            cams = feature_map
        else:
            cams = mtatt

        patch_attn = attn_weights[:, :, self.num_classes:, self.num_classes:]

        x_cls_logits = x_cls.mean(-1)

        return x_cls_logits, cams, patch_attn

class CAM(Net):

    def __init__(self, pretrained_cfg=None, *args, **kwargs):
        super().__init__(pretrained_cfg=None, *args, **kwargs)
    
    def forward(self, x, return_att=False, n_layers=12, attention_type='fused'):
        print(x.shape)
        w, h = x.shape[2:]
        x_cls, x_patch, attn_weights = self.forward_features(x)
        n, p, c = x_patch.shape
        if w != h:
            w0 = w // self.patch_embed.patch_size[0]
            h0 = h // self.patch_embed.patch_size[0]
            x_patch = torch.reshape(x_patch, [n, w0, h0, c])
        else:
            x_patch = torch.reshape(x_patch, [n, int(p ** 0.5), int(p ** 0.5), c])
        x_patch = x_patch.permute([0, 3, 1, 2])
        x_patch = x_patch.contiguous()
        x_patch = self.head(x_patch)
        x_patch_logits = self.avgpool(x_patch).squeeze(3).squeeze(2)

        attn_weights = torch.stack(attn_weights)  # 12 * B * H * N * N
        attn_weights = torch.mean(attn_weights, dim=2)  # 12 * B * N * N

        feature_map = x_patch.detach().clone()  # B * C * 14 * 14
        feature_map = F.relu(feature_map)

        n, c, h, w = feature_map.shape

        mtatt = attn_weights[-n_layers:].sum(0)[:, 0:self.num_classes, self.num_classes:].reshape([n, c, h, w])

        if attention_type == 'fused':
            cams = mtatt * feature_map  # B * C * 14 * 14
        elif attention_type == 'patchcam':
            cams = feature_map
        else:
            cams = mtatt

        patch_attn = attn_weights[:, :, self.num_classes:, self.num_classes:]

        x_cls_logits = x_cls.mean(-1)

        cams = cams[0] + cams[1].flip(-1) #added to remove extra dim
        return cams