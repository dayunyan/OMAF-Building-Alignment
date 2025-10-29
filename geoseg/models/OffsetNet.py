import torch
from torch import nn
import torch.nn.functional as F
from .UNetFormerOutDict import UNetFormerOutDict, Decoder


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        dilation=1,
        stride=1,
        norm_layer=nn.BatchNorm2d,
        bias=False,
    ):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                bias=bias,
                dilation=dilation,
                stride=stride,
                padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
            ),
            norm_layer(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )


class OffsetNet(UNetFormerOutDict):
    def __init__(self, **unetformerconfig):
        super().__init__(**unetformerconfig)

        # old_conv = self.backbone.conv1
        # self.backbone.conv1 = nn.Conv2d(
        #     in_channels,
        #     old_conv.out_channels,
        #     kernel_size=old_conv.kernel_size,
        #     stride=old_conv.stride,
        #     padding=old_conv.padding,
        #     bias=old_conv.bias,
        # )
        encoder_channels = self.backbone.feature_info.channels()
        offset_module = []
        for i in range(4):
            offset_module.append(
                ConvBNReLU(
                    encoder_channels[3 - i] + 32 if i else encoder_channels[3 - i],
                    32,
                    kernel_size=3,
                    dilation=1,
                    stride=1,
                    bias=False,
                )
            )
        self.offset_module = nn.ModuleList(offset_module)
        self.post_conv = nn.Conv2d(32 + 3, 2, kernel_size=1, bias=True)
        self.tanh = nn.Tanh()
        self.init_weight()

    def forward(self, x):
        h, w = x.size()[-2:]
        res1, res2, res3, res4 = self.backbone(x)
        if self.training:
            o = self.offset_module[0](res4)
            o = F.interpolate(o, size=res3.size()[-2:], mode="bilinear")
            o = self.offset_module[1](torch.cat([o, res3], dim=1))
            o = F.interpolate(o, size=res2.size()[-2:], mode="bilinear")
            o = self.offset_module[2](torch.cat([o, res2], dim=1))
            o = F.interpolate(o, size=res1.size()[-2:], mode="bilinear")
            o = self.offset_module[3](torch.cat([o, res1], dim=1))
            o = F.interpolate(o, size=(h, w), mode="bilinear")
            o = self.post_conv(torch.cat([o, x], dim=1))
            o = self.tanh(o)
            out = self.decoder(res1, res2, res3, res4, h, w)
            out["offset"] = o
            return out
        else:
            o = self.offset_module[0](res4)
            o = F.interpolate(o, size=res3.size()[-2:], mode="bilinear")
            o = self.offset_module[1](torch.cat([o, res3], dim=1))
            o = F.interpolate(o, size=res2.size()[-2:], mode="bilinear")
            o = self.offset_module[2](torch.cat([o, res2], dim=1))
            o = F.interpolate(o, size=res1.size()[-2:], mode="bilinear")
            o = self.offset_module[3](torch.cat([o, res1], dim=1))
            o = F.interpolate(o, size=(h, w), mode="bilinear")
            o = self.post_conv(torch.cat([o, x], dim=1))
            out = self.decoder(res1, res2, res3, res4, h, w)
            o = self.tanh(o)
            out["offset"] = o
            return out

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class OffsetNet_v2(UNetFormerOutDict):
    def __init__(self, **unetformerconfig):
        super().__init__(**unetformerconfig)

        decode_channels = unetformerconfig.get("decode_channels", 64)
        dropout = unetformerconfig.get("dropout", 0.1)
        window_size = unetformerconfig.get("window_size", 8)
        offset_num = 2

        encoder_channels = self.backbone.feature_info.channels()
        self.offset = Decoder(
            encoder_channels, decode_channels, dropout, window_size, offset_num
        )
        # self.bn = nn.BatchNorm2d(2)
        self.tanh = nn.Tanh()
        self.init_weight()

    def forward(self, x):
        h, w = x.size()[-2:]
        res1, res2, res3, res4 = self.backbone(x)

        offset = self.offset(res1, res2, res3, res4, h, w)
        offset = self.tanh(offset["logits"])
        out = self.decoder(res1, res2, res3, res4, h, w)
        out["offset"] = offset
        out["feature"] = torch.cat(
            [
                F.interpolate(r, size=(h, w), mode="bilinear")
                for r in [res1, res2, res3, res4]
            ],
            dim=1,
        )
        return out

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
