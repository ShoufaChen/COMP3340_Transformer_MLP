import torch
import torch.nn as nn

from mmcv.cnn import NORM_LAYERS, build_activation_layer, build_norm_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import FFN
from mmcv.runner.base_module import BaseModule, ModuleList

from ..builder import BACKBONES
from ..utils import PatchEmbed, to_2tuple
from .base_backbone import BaseBackbone

class Affine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return x * self.alpha + self.beta


class ResMlpChannel(BaseModule):
    """
    Same as FFN
    """
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



class ResMLPBlock(BaseModule):
    """ResMLP Basic Block.

    Base module of 'ResMLP : Feedforward networks for image classification with data-efficient training'
    https://arxiv.org/pdf/2105.03404.pdf

    Args:
        num_tokens (int): The number of patched tokens
        embed_dims (int): The feature dimension
        channels_mlp_dims (int): The hidden dimension for channels FFNs
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        act_cfg (dict): The activation config for FFNs.
            Defaluts to ``dict(type='GELU')``.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.


    """
    def __init__(self,
                 num_tokens,
                 embed_dims,
                 channels_mlp_dims,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 init_values=1e-4,
                 init_cfg=None):
        super(ResMLPBlock, self).__init__(init_cfg= init_cfg)

        self.affine_norm1 = Affine(dim=embed_dims)
        self.token_mix = nn.Linear(num_tokens,num_tokens)
        self.affine_norm2 = Affine(dim=embed_dims)

        self.channel_mix = FFN(embed_dims=embed_dims,
                                feedforward_channels=channels_mlp_dims,
                                num_fcs=2,
                                ffn_drop=drop_rate,
                                add_identity=False,
                                act_cfg=act_cfg)

        self.gamma1 = nn.Parameter(init_values * torch.ones(embed_dims),requires_grad=True)
        self.gamma2 = nn.Parameter(init_values * torch.ones(embed_dims), requires_grad=True)
        if drop_path_rate>0:
            dropout_layer = dict(type='DropPath', drop_prob=drop_path_rate)
        else:
            dropout_layer = None
        self.dropout_layer = build_dropout(dropout_layer) if dropout_layer else torch.nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self,m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight,std=0.02)
            if m.bias is not None:
                nn.init.constant(m.bias,0)

    def forward(self, x):
        x = x + self.dropout_layer(self.gamma1 * self.token_mix(self.affine_norm1(x).transpose(1,2)).transpose(1,2))
        x = x + self.dropout_layer(self.gamma2 * self.channel_mix(self.affine_norm2(x)))
        return x


@BACKBONES.register_module()
class ResMlp(BaseBackbone):
    """ResMLP Backbone.

    Base module of 'ResMLP : Feedforward networks for image classification with data-efficient training'
    https://arxiv.org/pdf/2105.03404.pdf

    Args:
        arch (str | dict): MLP Mixer architecture
            Defaults to 'b'.
        img_size (int | tuple): Input image size.
        patch_size (int | tuple): The patch size.
        out_indices (Sequence | int): Output from which layer.
            Defaults to -1, means the last layer.
        drop_rate (float): Probability of an element to be zeroed.
            Defaults to 0.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        act_cfg (dict): The activation config for FFNs. Default GELU.
        patch_cfg (dict): Configs of patch embeding. Defaults to an empty dict.
        layer_cfgs (Sequence | dict): Configs of each resmlp block layer.
            Defaults to an empty dict.
        global_pool (str): Global pooling
            Defaults to "avg"
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.

    """
    # change embed_dim params
    arch_zoo = {
        **dict.fromkeys(
            ['s', 'small'], {
                'embed_dims': 384,
                'num_layers': 12,
                'channels_mlp_dims': 1536,
            }),
        **dict.fromkeys(
            ['b', 'base'], {
                'embed_dims': 384,
                'num_layers': 24,
                'channels_mlp_dims': 1536,
            }),
        **dict.fromkeys(
            ['s768', 'small-768'], {
                'embed_dims': 768,
                'num_layers': 12,
                'channels_mlp_dims': 3072,
            }),
        **dict.fromkeys(
            ['b768', 'base-768'], {
                'embed_dims': 768,
                'num_layers': 24,
                'channels_mlp_dims': 3072,
            }),
    }

    def __init__(self,
                 arch='b',
                 img_size=224,
                 patch_size=16,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 norm_cfg=dict(type='LN'),
                 act_cfg=dict(type='GELU'),
                 patch_cfg=dict(),
                 layer_cfgs=dict(),
                 init_cfg=None):
        super(ResMlp, self).__init__(init_cfg)

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {
                'embed_dims', 'num_layers'
            }
            assert isinstance(arch, dict) and set(arch) == essential_keys, \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch

        self.embed_dims = self.arch_settings['embed_dims']
        self.num_layers = self.arch_settings['num_layers']
        self.channels_mlp_dims = self.arch_settings['channels_mlp_dims']

        self.img_size = to_2tuple(img_size)

        _patch_cfg = dict(
            img_size=img_size,
            embed_dims=self.embed_dims,
            conv_cfg=dict(
                type='Conv2d', kernel_size=patch_size, stride=patch_size),
        )
        _patch_cfg.update(patch_cfg)
        self.patch_embed = PatchEmbed(**_patch_cfg)
        num_patches = self.patch_embed.num_patches

        self.layers = ModuleList()
        if isinstance(layer_cfgs, dict):
            layer_cfgs = [layer_cfgs] * self.num_layers
        for i in range(self.num_layers):
            _layer_cfg = dict(
                num_tokens=num_patches,
                embed_dims=self.embed_dims,
                channels_mlp_dims=self.channels_mlp_dims,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                act_cfg=act_cfg,
            )
            _layer_cfg.update(layer_cfgs[i])
            self.layers.append(ResMLPBlock(**_layer_cfg))

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, self.embed_dims, postfix=1)
        self.add_module(self.norm1_name, norm1)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def forward(self, x):
        x = self.patch_embed(x)

        for i, layer in enumerate(self.layers):
            x = layer(x)

            if i == len(self.layers) - 1:
                x = self.norm1(x)

        return x
