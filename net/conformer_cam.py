import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, trunc_normal_
from net.trans_utils import *

class Net(nn.Module):

    def __init__(self, patch_size=16, in_chans=3, num_classes=1000, base_channel=64, channel_ratio=4, num_med_block=0,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):

        # Transformer
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        assert depth % 3 == 0

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.trans_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        # Classifier head
        self.trans_norm = nn.LayerNorm(embed_dim)
        self.trans_cls_head = nn.Linear(embed_dim, self.num_classes)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.conv_cls_head = nn.Conv2d(int(256 * channel_ratio), self.num_classes, kernel_size=3, stride=1, padding=1)

        # Stem stage: get the feature maps by conv block (copied form resnet.py)
        self.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 1 / 2 [112, 112]
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 1 / 4 [56, 56]

        # 1 stage
        stage_1_channel = int(base_channel * channel_ratio)
        trans_dw_stride = patch_size // 4
        self.conv_1 = ConvBlock(inplanes=64, outplanes=stage_1_channel, res_conv=True, stride=1)
        self.trans_patch_conv = nn.Conv2d(64, embed_dim, kernel_size=trans_dw_stride, stride=trans_dw_stride, padding=0)
        self.trans_1 = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                             qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.trans_dpr[0],
                             )

        # 2~4 stage
        init_stage = 2
        fin_stage = depth // 3 + 1
        for i in range(init_stage, fin_stage):
            self.add_module('conv_trans_' + str(i),
                    ConvTransBlock(
                        stage_1_channel, stage_1_channel, False, 1, dw_stride=trans_dw_stride, embed_dim=embed_dim,
                        num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=self.trans_dpr[i-1],
                        num_med_block=num_med_block
                    )
            )


        stage_2_channel = int(base_channel * channel_ratio * 2)
        # 5~8 stage
        init_stage = fin_stage # 5
        fin_stage = fin_stage + depth // 3 # 9
        for i in range(init_stage, fin_stage):
            s = 2 if i == init_stage else 1
            in_channel = stage_1_channel if i == init_stage else stage_2_channel
            res_conv = True if i == init_stage else False
            self.add_module('conv_trans_' + str(i),
                    ConvTransBlock(
                        in_channel, stage_2_channel, res_conv, s, dw_stride=trans_dw_stride // 2, embed_dim=embed_dim,
                        num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=self.trans_dpr[i-1],
                        num_med_block=num_med_block
                    )
            )

        stage_3_channel = int(base_channel * channel_ratio * 2 * 2)
        # 9~12 stage
        init_stage = fin_stage  # 9
        fin_stage = fin_stage + depth // 3  # 13
        for i in range(init_stage, fin_stage):
            s = 2 if i == init_stage else 1
            in_channel = stage_2_channel if i == init_stage else stage_3_channel
            res_conv = True if i == init_stage else False
            last_fusion = True if i == depth else False
            self.add_module('conv_trans_' + str(i),
                    ConvTransBlock(
                        in_channel, stage_3_channel, res_conv, s, dw_stride=trans_dw_stride // 4, embed_dim=embed_dim,
                        num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=self.trans_dpr[i-1],
                        num_med_block=num_med_block, last_fusion=last_fusion
                    )
            )
        self.fin_stage = fin_stage

        trunc_normal_(self.cls_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}


    def forward(self, x, method="transcam"):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        # pdb.set_trace()
        # stem stage [N, 3, 224, 224] -> [N, 64, 56, 56]
        x_base = self.maxpool(self.act1(self.bn1(self.conv1(x))))

        attn_weights = []

        # 1 stage
        x = self.conv_1(x_base, return_x_2=False)

        x_t = self.trans_patch_conv(x_base).flatten(2).transpose(1, 2)
        x_t = torch.cat([cls_tokens, x_t], dim=1)
        x_t, attn_weight = self.trans_1(x_t)
        attn_weights.append(attn_weight)

        # 2 ~ final
        for i in range(2, self.fin_stage):
            x, x_t, attn_weight = eval('self.conv_trans_' + str(i))(x, x_t)
            attn_weights.append(attn_weight)

        # conv classification
        x_conv = self.conv_cls_head(x)
        conv_cls = self.pooling(x_conv).flatten(1)

        # trans classification
        x_t = self.trans_norm(x_t)

        x_patch = x_t[:, 1:]
        n, p, c = x_patch.shape
        x_patch = torch.reshape(x_patch, [n, int(p ** 0.5), int(p ** 0.5), c])
        x_patch = x_patch.permute([0, 3, 1, 2]).contiguous()
        conv_logits = conv_cls
        trans_logits = self.trans_cls_head(x_t[:, 0])

        attn_weights = torch.stack(attn_weights)
        attn_weights = torch.mean(attn_weights, dim=2)  # 12 * B * N * N
        feature_map = x_patch.detach().clone()    # B * C * 14 * 14
        n, c, h, w = feature_map.shape
        attn_weights_no_cls = attn_weights.sum(0)[:, 1:, 1:]
        conv_flat = x_conv.flatten(start_dim=2)

        if method == 'convcam':
            cams = x_conv

        if method == 'clsattn':
            cams = attn_weights.sum(0)[:, 0, 1:].reshape([n, h, w]).unsqueeze(1) * x_conv

        if method == 'transcam':
            cams = torch.einsum('nhw,ncw->nch', attn_weights_no_cls, conv_flat).reshape([n, 21, h, w])

        if method == 'attnagg':
            cams = torch.einsum('nhw,ncw->nch', attn_weights_no_cls.transpose(1, 2), conv_flat).reshape([n, 21, h, w])

        return conv_logits + trans_logits

    def trainable_parameters(self):
        params = list(self.parameters())
        return params

class Net_CAM(Net):
    def __init__(self, patch_size=16, in_chans=3, num_classes=1000, base_channel=64, channel_ratio=4, num_med_block=0,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super(Net_CAM, self).__init__(self, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes, base_channel=base_channel,
                 channel_ratio=channel_ratio, num_med_block=num_med_block, embed_dim=embed_dim, depth=depth, num_heads=num_heads, 
                 mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                 drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate)

    def forward(self, x, method="transcam"):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        # pdb.set_trace()
        # stem stage [N, 3, 224, 224] -> [N, 64, 56, 56]
        x_base = self.maxpool(self.act1(self.bn1(self.conv1(x))))

        attn_weights = []

        # 1 stage
        x = self.conv_1(x_base, return_x_2=False)

        x_t = self.trans_patch_conv(x_base).flatten(2).transpose(1, 2)
        x_t = torch.cat([cls_tokens, x_t], dim=1)
        x_t, attn_weight = self.trans_1(x_t)
        attn_weights.append(attn_weight)

        # 2 ~ final
        for i in range(2, self.fin_stage):
            x, x_t, attn_weight = eval('self.conv_trans_' + str(i))(x, x_t)
            attn_weights.append(attn_weight)

        # conv classification
        x_conv = self.conv_cls_head(x)
        conv_cls = self.pooling(x_conv).flatten(1)

        # trans classification
        x_t = self.trans_norm(x_t)

        x_patch = x_t[:, 1:]
        n, p, c = x_patch.shape
        x_patch = torch.reshape(x_patch, [n, int(p ** 0.5), int(p ** 0.5), c])
        x_patch = x_patch.permute([0, 3, 1, 2]).contiguous()
        conv_logits = conv_cls
        trans_logits = self.trans_cls_head(x_t[:, 0])

        attn_weights = torch.stack(attn_weights)
        attn_weights = torch.mean(attn_weights, dim=2)  # 12 * B * N * N
        feature_map = x_patch.detach().clone()    # B * C * 14 * 14
        n, c, h, w = feature_map.shape
        attn_weights_no_cls = attn_weights.sum(0)[:, 1:, 1:]
        conv_flat = x_conv.flatten(start_dim=2)

        if method == 'convcam':
            cams = x_conv

        if method == 'clsattn':
            cams = attn_weights.sum(0)[:, 0, 1:].reshape([n, h, w]).unsqueeze(1) * x_conv

        if method == 'transcam':
            cams = torch.einsum('nhw,ncw->nch', attn_weights_no_cls, conv_flat).reshape([n, 21, h, w])

        if method == 'attnagg':
            cams = torch.einsum('nhw,ncw->nch', attn_weights_no_cls.transpose(1, 2), conv_flat).reshape([n, 21, h, w])

        return conv_logits + trans_logits, cams

class CAM(Net):
    def __init__(self, patch_size=16, in_chans=3, num_classes=1000, base_channel=64, channel_ratio=4, num_med_block=0,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__(patch_size, in_chans, num_classes, base_channel, channel_ratio, num_med_block, embed_dim, depth, num_heads, 
                mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate)
    
    def forward(self, x, method="transcam"):
        B = x.shape[0]
        print(x.shape)
        cls_tokens = self.cls_token.expand(B, -1, -1)

        # pdb.set_trace()
        # stem stage [N, 3, 224, 224] -> [N, 64, 56, 56]
        x_base = self.maxpool(self.act1(self.bn1(self.conv1(x))))

        attn_weights = []

        # 1 stage
        x = self.conv_1(x_base, return_x_2=False)

        x_t = self.trans_patch_conv(x_base).flatten(2).transpose(1, 2)
        x_t = torch.cat([cls_tokens, x_t], dim=1)
        x_t, attn_weight = self.trans_1(x_t)
        attn_weights.append(attn_weight)

        # 2 ~ final
        for i in range(2, self.fin_stage):
            print(x.shape, len(attn_weights), attn_weights[0].shape)
            x, x_t, attn_weight = eval('self.conv_trans_' + str(i))(x, x_t)
            attn_weights.append(attn_weight)

        # conv classification
        x_conv = self.conv_cls_head(x)
        conv_cls = self.pooling(x_conv).flatten(1)

        # trans classification
        x_t = self.trans_norm(x_t)

        x_patch = x_t[:, 1:]
        n, p, c = x_patch.shape
        x_patch = torch.reshape(x_patch, [n, int(p ** 0.5), int(p ** 0.5), c])
        x_patch = x_patch.permute([0, 3, 1, 2]).contiguous()
        conv_logits = conv_cls
        trans_logits = self.trans_cls_head(x_t[:, 0])

        attn_weights = torch.stack(attn_weights)
        attn_weights = torch.mean(attn_weights, dim=2)  # 12 * B * N * N
        feature_map = x_patch.detach().clone()    # B * C * 14 * 14
        n, c, h, w = feature_map.shape
        attn_weights_no_cls = attn_weights.sum(0)[:, 1:, 1:]
        conv_flat = x_conv.flatten(start_dim=2)

        if method == 'convcam':
            cams = x_conv

        if method == 'clsattn':
            cams = attn_weights.sum(0)[:, 0, 1:].reshape([n, h, w]).unsqueeze(1) * x_conv

        if method == 'transcam':
            cams = torch.einsum('nhw,ncw->nch', attn_weights_no_cls, conv_flat).reshape([n, 21, h, w])

        if method == 'attnagg':
            cams = torch.einsum('nhw,ncw->nch', attn_weights_no_cls.transpose(1, 2), conv_flat).reshape([n, 21, h, w])

        return cams



