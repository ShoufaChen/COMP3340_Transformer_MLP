import torch
import torch.nn as nn

from mmcv.cnn import NORM_LAYERS, build_activation_layer, build_norm_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.runner.base_module import BaseModule, ModuleList

from ..builder import BACKBONES
from ..utils import PatchEmbed, to_2tuple
from .base_backbone import BaseBackbone

@NORM_LAYERS.register_module()
class Affine(BaseModule):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones([1, 1, dim]))
        self.beta = nn.Parameter(torch.zeros([1, 1, dim]))

    def forward(self, x):
        return x * self.alpha + self.beta

class ResTokenMix(BaseModule):
    def __init__(self, num_tokens, init_values=1e-4, drop_path_rate=0., init_cfg=None):
        super(ResTokenMix, self).__init__(init_cfg=init_cfg)
        self.linear_tokens =nn.Linear(num_tokens,num_tokens)

        self.ls = nn.Parameter(init_values * torch.ones(num_tokens))

        if drop_path_rate>0:
            dropout_layer = dict(type='DropPath', drop_prob=drop_path_rate)
        else:
            dropout_layer = None
        self.dropout_layer = build_dropout(dropout_layer) if dropout_layer else torch.nn.Identity()

    def forward(self,x):
        return self.dropout_layer(self.ls * self.linear_tokens(x.transpose(1, 2)).transpose(1, 2))


class ResMlpChannel(BaseModule):
    def __init__(self, embed_dim, channel_dim, act_cfg=dict(type='GELU'), drop=0., init_cfg=None):
        super(ResMlpChannel, self).__init__(init_cfg=init_cfg)

        self.fc1 = nn.Linear(embed_dim, channel_dim)
        self.act_cfg = act_cfg
        self.act = build_activation_layer(act_cfg)
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(channel_dim, embed_dim)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x



class ResBlock(BaseModule):
    """ResMLP Basic Block.

    Base module of 'ResMLP : Feedforward networks for image classification with data-efficient training'
    https://arxiv.org/pdf/2105.03404.pdf


    Pipeline: Affine -> Transpose -> Linear -> Transpose -> Affine + skip connection->
            -> Affine -> Linear -> GELU -> Linear -> Affine ->


    Args:


    """
    def __init__(self,
                 num_tokens,
                 embed_dims,
                 mlp_ratio=4,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 init_values=1e-4,
                 init_cfg=None):
        super(ResBlock, self).__init__(init_cfg= init_cfg)

        channels_dims =mlp_ratio * embed_dims
        affine_norm_cfg = dict(type='Affine', dim=num_tokens)
        self.affine_norm1_name, affine_norm1 = build_norm_layer(affine_norm_cfg,embed_dims, postfix=1)
        self.add_module(self.affine_norm1_name, affine_norm1)

        self.token_mix = ResTokenMix(num_tokens, init_values=init_values, drop=drop_rate,
                                     drop_path=drop_path_rate, init_cfg= init_cfg)

        self.affine_norm2_name, affine_norm2 = build_norm_layer(affine_norm_cfg,embed_dims, postfix=2)
        self.add_module(self.affine_norm2_name, affine_norm2)

        self.mlp_channel = ResMlpChannel(embed_dims, channels_dims, act_cfg = act_cfg, drop= drop_rate)

        self.ls = nn.Parameter(init_values * torch.ones(embed_dims))
        if drop_path_rate>0:
            dropout_layer = dict(type='DropPath', drop_prob=drop_path_rate)
        else:
            dropout_layer = None
        self.dropout_layer = build_dropout(dropout_layer) if dropout_layer else torch.nn.Identity()


    @property
    def affine_norm1(self):
        return getattr(self, self.affine_norm1_name)

    @property
    def affine_norm2(self):
        return getattr(self, self.affine_norm2_name)

    def init_weights(self):
        pass

    def forward(self, x):
        x = x + self.token_mix(self.affine_norm1(x))
        x = x + self.dropout_layer(self.ls * self.mlp_channel(self.affine_norm2(x)))
        return x


@BACKBONES.register_module()
class ResMlp(BaseBackbone):
    def __init__(self):
        pass
    def forward(self,x):
        pass
