import torch
import torch.nn.functional as F
import onnx
from onnx2torch import convert


class BEU_Net(torch.nn.Module):
    """自动适配ONNX结构的模型基类"""

    def __init__(self, onnx_path):
        super().__init__()
        self.base_model = convert(onnx_path)  # 动态加载结构

    def forward(self, x):
        h, w = x.shape[-2:]
        if h != 256 or w != 256:
            x = F.interpolate(x, size=(256, 256), mode="bilinear")
        return self.base_model(x).permute(0, 3, 1, 2).contiguous()

    def load_ckpt(self, ckpt_pth):
        self.base_model.load_state_dict(torch.load(ckpt_pth, map_location="cpu"))


if __name__ == "__main__":
    # 使用方式
    model = BEU_Net(
        "/workspace/zjj/xjd/GeoSeg-main/pretrained_weights/building_estimation/model.onnx"
    )
    model.load_ckpt(
        "/workspace/zjj/xjd/GeoSeg-main/pretrained_weights/building_estimation/model.pth"
    )  # 加载参数
    model.eval()
    model.cuda()

    inputs = torch.zeros((1, 3, 256, 256), dtype=torch.float32).cuda()
    outputs = model(inputs)
    print(f"outputs.shape: {outputs.shape}")
