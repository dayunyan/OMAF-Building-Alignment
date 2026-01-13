from typing import Dict
from collections import defaultdict

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

from geoseg.models.segman.segman_encoder import SegMANEncoder_b
from geoseg.models.segman.segman_decoder import SegMANDecoder


class SegMAN(nn.Module):
    """
    Deeplabv3plus implememts
    This module has five components:

    self.backbone
    self.aspp
    self.projector: an 1x1 conv for lowlevel feature projection
    self.preclassifier: an 3x3 conv for feature mixing, before final classification
    self.classifier: last 1x1 conv for output classification results

    Args:
        backbone: Dict, configs for backbone
        decoder: Dict, configs for decoder

    NOTE: The bottleneck has only one 3x3 conv by default, some implements stack
        two 3x3 convs
    """

    def __init__(self, backbone: Dict, decoder: Dict, pretrained=None) -> None:
        super(SegMAN, self).__init__()

        self.align_corners = decoder["align_corners"]
        self.backbone = SegMANEncoder_b(**backbone)
        self.decode_head = SegMANDecoder(**decoder)
        # channel = decoder.get("in_channels", [512])[-1]
        # num_classes = decoder.get("num_classes")
        # self.decoder = nn.Conv2d(channel, num_classes, kernel_size=1)
        self.load_pretrained(pretrained)

    def load_pretrained(self, ckpt=None, key="state_dict"):
        if ckpt is None:
            print("No checkpoint provided, skip loading.")
            return

        try:
            # 加载 ckpt 并提取 state_dict
            _ckpt = torch.load(
                open(ckpt, "rb"), map_location=torch.device("cpu"), weights_only=False
            )
            if key not in _ckpt:
                raise KeyError(
                    f"Checkpoint has no key '{key}', available keys: {list(_ckpt.keys())}"
                )

            ckpt_state = _ckpt[key]
            model_state = self.state_dict()  # 模型自身的参数状态

            del (
                ckpt_state["decode_head.conv_seg.bias"],
                ckpt_state["decode_head.conv_seg.weight"],
            )
            print(f"Successfully load ckpt {ckpt}")
            print(f"\n=== Parameter Shape Matching Check ===")

            # 分类存储检查结果
            match_results = defaultdict(list)  # key: 结果类型, value: 参数名列表
            shape_mismatch_detail = []  # 尺寸不匹配的详细信息

            # 1. 获取参数名称集合
            model_keys = set(model_state.keys())
            ckpt_keys = set(ckpt_state.keys())

            # 2. 检查名称匹配的参数（交集）
            common_keys = model_keys & ckpt_keys
            print(f"\n1. Common parameters (name matched): {len(common_keys)}")
            for param_name in sorted(common_keys):
                model_param = model_state[param_name]
                ckpt_param = ckpt_state[param_name]

                # 对比尺寸
                if model_param.shape == ckpt_param.shape:
                    match_results["shape_match"].append(param_name)
                    # 可选：输出匹配成功的详细信息（注释掉可减少输出）
                    # print(f"  ✅ {param_name}: model_shape={model_param.shape}, ckpt_shape={ckpt_param.shape}")
                else:
                    match_results["shape_mismatch"].append(param_name)
                    shape_mismatch_detail.append(
                        f"  ❌ {param_name}:\n"
                        f"      model_shape={model_param.shape}, ckpt_shape={ckpt_param.shape}"
                    )

            # 3. 输出尺寸不匹配的详细信息
            if shape_mismatch_detail:
                print(f"\n2. Shape mismatch parameters ({len(shape_mismatch_detail)}):")
                for detail in shape_mismatch_detail:
                    print(detail)
            else:
                print(
                    f"\n2. Shape mismatch parameters: 0 (all common parameters match)"
                )

            # 4. 输出模型有但 ckpt 缺失的参数
            missing_in_ckpt = model_keys - ckpt_keys
            if missing_in_ckpt:
                print(
                    f"\n3. Parameters in model but missing in ckpt ({len(missing_in_ckpt)}):"
                )
                for key in sorted(missing_in_ckpt):
                    print(f"  ⚠️ {key} (model_shape={model_state[key].shape})")
            else:
                print(f"\n3. Parameters in model but missing in ckpt: 0")

            # 5. 输出 ckpt 有但模型没有的参数
            extra_in_ckpt = ckpt_keys - model_keys
            if extra_in_ckpt:
                print(
                    f"\n4. Parameters in ckpt but missing in model ({len(extra_in_ckpt)}):"
                )
                for key in sorted(extra_in_ckpt):
                    print(f"  ⚠️ {key} (ckpt_shape={ckpt_state[key].shape})")
            else:
                print(f"\n4. Parameters in ckpt but missing in model: 0")

            # 6. 原有加载逻辑（strict=False 忽略不匹配的参数）
            print(f"\n=== Start loading checkpoint ===")
            incompatibleKeys = self.load_state_dict(ckpt_state, strict=False)
            print(f"\nLoad result:")
            print(
                f"  Missing keys (model has, ckpt no): {len(incompatibleKeys.missing_keys)}"
            )
            print(
                f"  Unexpected keys (ckpt has, model no): {len(incompatibleKeys.unexpected_keys)}"
            )
            if incompatibleKeys.missing_keys:
                print(
                    f"    Missing list: {incompatibleKeys.missing_keys[:5]}..."
                )  # 只显示前5个，避免输出过长
            if incompatibleKeys.unexpected_keys:
                print(f"    Unexpected list: {incompatibleKeys.unexpected_keys[:5]}...")

        except Exception as e:
            print(f"\n❌ Failed loading checkpoint from {ckpt}: {e}")
            import traceback

            traceback.print_exc()  # 可选：打印详细异常栈，便于调试

    def forward(self, x: Tensor) -> Tensor:
        size = (x.shape[2], x.shape[3])
        output = self.backbone.forward_layer_feat(x)
        output = self.decode_head(output)

        out = {}
        out["logits"] = F.interpolate(
            output, size=size, mode="bilinear", align_corners=self.align_corners
        )
        return out
