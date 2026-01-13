import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.misc import DropPath


class FFN(nn.Module):
    """
    mmcv.cnn.bricks.transformer.FFN 的 PyTorch 替代品
    """

    def __init__(
        self,
        embed_dims,
        feedforward_channels,
        num_fcs=2,
        act_cfg=dict(type="GELU"),
        ffn_drop=0.0,
        dropout_layer=None,
        add_identity=True,
        init_cfg=None,
    ):  # init_cfg 未使用, 仅为 API 兼容
        super().__init__()
        assert num_fcs == 2, "This simplified FFN only supports num_fcs=2"
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.add_identity = add_identity

        #
        if act_cfg["type"] == "GELU":
            self.activate = nn.GELU()
        elif act_cfg["type"] == "ReLU":
            self.activate = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation type: {act_cfg['type']}")

        self.layers = nn.Sequential(
            nn.Linear(embed_dims, feedforward_channels),
            self.activate,
            nn.Dropout(ffn_drop),
            nn.Linear(feedforward_channels, embed_dims),
            nn.Dropout(ffn_drop),
        )

        self.dropout_layer = None
        if dropout_layer and dropout_layer["drop_prob"] > 0.0:
            if dropout_layer["type"] == "DropPath":
                self.dropout_layer = DropPath(dropout_layer["drop_prob"])
            else:
                warnings.warn(
                    f"Unsupported dropout layer: {dropout_layer['type']}. Using nn.Identity()"
                )
                self.dropout_layer = nn.Identity()
        else:
            self.dropout_layer = nn.Identity()

    def forward(self, x, identity=None):
        out = self.layers(x)
        out = self.dropout_layer(out)

        if self.add_identity:
            if identity is None:
                identity = x
            return identity + out
        return out
