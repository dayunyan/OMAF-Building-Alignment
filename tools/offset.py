import numpy as np
import cv2
import torch
from torch.nn import functional as F


def compute_distance_transform(mask):
    """计算二值mask的距离变换图"""
    # 输入：二值mask（0为背景，255为房顶）
    # 输出：距离变换图（归一化到[0,1]）
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    return dist / (dist.max() + 1e-8)


def offset_tensor(tensor: torch.Tensor, offset: torch.Tensor):
    """
    此函数使用 offset 张量对 tensor 张量中的像素位置进行移动。

    参数:
    tensor (torch.Tensor): 形状为 [B, C, H, W] 的张量，表示模型输出的 tensor
    offset (torch.Tensor): 形状为 [B, 2, H, W] 的张量，其中两个通道分别表示在 x 和 y 方向上的偏移量

    返回:
    torch.Tensor: 移动后的 tensor 张量
    """
    B, C, H, W = tensor.shape
    pred = tensor.argmax(dim=1, keepdim=True).expand(-1, 2, -1, -1)
    # 检查输入张量的形状是否符合预期
    assert offset.shape == (B, 2, H, W), "Offset 张量的形状应为 [B, 2, H, W]"
    # new_offset = torch.where(pred == 1, offset, torch.zeros_like(offset))
    new_offset = offset
    # 生成坐标网格
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    y = y.unsqueeze(0).expand(B, -1, -1)
    x = x.unsqueeze(0).expand(B, -1, -1)
    # 计算新的坐标
    new_y = (y - torch.round(H * new_offset[:, 1, :, :])).to(torch.int64)
    new_x = (x - torch.round(W * new_offset[:, 0, :, :])).to(torch.int64)
    # 确保新坐标在合法范围内
    new_y = torch.clamp(new_y, 0, H - 1).unsqueeze(1).expand(-1, C, -1, -1)
    new_x = torch.clamp(new_x, 0, W - 1).unsqueeze(1).expand(-1, C, -1, -1)

    # 初始化 shifted_tensor
    shifted_tensor_y = torch.zeros_like(tensor)
    # 使用广播机制更新 shifted_tensor
    shifted_tensor_y = torch.scatter(
        shifted_tensor_y,
        2,
        new_y,
        tensor,  # .gather(2, new_y.unsqueeze(1).expand(-1, C, -1, -1)),
    )

    new_x_shifted = torch.zeros_like(new_x)
    new_x_shifted = torch.scatter(new_x_shifted, 2, new_y, new_x)

    shifted_tensor_xy = torch.zeros_like(tensor)
    shifted_tensor_xy = torch.scatter(
        shifted_tensor_xy,
        3,
        new_x_shifted,
        shifted_tensor_y,  # .gather(3, new_x.unsqueeze(1).expand(-1, C, -1, -1)),
    )

    return shifted_tensor_xy


def offset_tensor_v2(
    tensor: torch.Tensor,
    offset: torch.Tensor,
    size_factor: float = 1.0,
    mode: str = "bilinear",
):
    """
    对 tensor 张量进行偏移操作

    参数:
    tensor (torch.Tensor): 形状为 [B, C, H, W] 的张量，表示预测的 tensor
    offset (torch.Tensor): 形状为 [B, 2, H, W] 的张量，表示每个像素的偏移量

    返回:
    torch.Tensor: 形状为 [B, C, H, W] 的张量，表示偏移后的 tensor
    """
    B, C, H, W = tensor.shape
    device = tensor.device
    dtype = tensor.dtype

    if size_factor != 1.0:
        original_h, original_w = H, W
        H = int(H * size_factor)
        W = int(W * size_factor)
        offset = F.interpolate(
            offset, size=(H, W), mode="bilinear", align_corners=False
        )
        tensor = F.interpolate(
            tensor.to(torch.float32), size=(H, W), mode="bilinear", align_corners=False
        )

    new_tensor = torch.zeros_like(tensor).to(device)
    for idx in torch.nonzero(offset):
        x = idx[3] - torch.round(W * offset[idx[0], 0, idx[2], idx[3]]).to(torch.int32)
        y = idx[2] - torch.round(H * offset[idx[0], 1, idx[2], idx[3]]).to(torch.int32)
        if x >= 0 and x < W and y >= 0 and y < H:
            new_tensor[
                idx[0],
                :,
                y,
                x,
            ] = tensor[idx[0], :, idx[2], idx[3]]

    if size_factor != 1.0:
        new_tensor = F.interpolate(
            new_tensor,
            size=(original_h, original_w),
            mode="bilinear",
            align_corners=False,
        )
        if mode == "linear":
            new_tensor = torch.where(new_tensor > 0.5, 1, 0.0)
        new_tensor = new_tensor.to(dtype)
    return new_tensor


def offset_tensor_v3(
    tensor: torch.Tensor, offset: torch.Tensor, sample_mode: str = "bilinear"
):
    """
    使用反向映射和双线性插值对 tensor 进行偏移。

    参数:
    tensor (torch.Tensor): 形状为 [B, C, H, W] 的张量
    offset (torch.Tensor): 形状为 [B, 2, H, W] 的张量，x和y方向偏移量

    返回:
    torch.Tensor: 偏移后的 tensor 张量
    """
    B, C, H, W = tensor.shape
    device = tensor.device
    dtype = tensor.dtype
    # if tensor.dtype != torch.float64:
    #     tensor = tensor.to(torch.float64)
    # if offset.dtype != torch.float64:
    #     offset = offset.to(torch.float64)

    # 生成归一化的基础网格 [B, H, W, 2]
    y_grid, x_grid = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing="ij",
    )
    # 将offset转换为标准化位移（注意方向取反）
    # offset_norm = -offset  # [B, 2, H, W]

    # 构建采样网格 [B, H, W, 2]
    sample_grid = torch.stack((x_grid, y_grid), dim=-1).unsqueeze(0)
    sample_grid = sample_grid.expand(B, -1, -1, -1) + offset.permute(0, 2, 3, 1) * 2
    sample_grid = torch.clamp(sample_grid, -1, 1)  # 确保网格在[-1, 1]范围内

    # 双线性插值采样（解决覆盖问题）
    shifted_tensor = torch.nn.functional.grid_sample(
        tensor,
        sample_grid,
        mode=sample_mode,
        padding_mode="zeros",
        align_corners=False,
    )

    return shifted_tensor.to(dtype)


if __name__ == "__main__":

    # def test_grad():
    #     B, C, H, W = 2, 3, 32, 32
    #     tensor = torch.randn(B, C, H, W, dtype=torch.float64, requires_grad=True)
    #     offset = torch.randn(B, 2, H, W, dtype=torch.float64, requires_grad=True)
    #     assert torch.autograd.gradcheck(offset_tensor_v3, (tensor, offset))

    # test_grad()

    def test_grad_with_net():
        # 初始化生成offset的神经网络（示例）
        class OffsetNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 2, kernel_size=3, padding=1)

            def forward(self, x):
                return self.conv(x)

        # 输入模拟
        img = torch.randn(4, 3, 256, 256, requires_grad=True)
        net = OffsetNet()
        optimizer = torch.optim.Adam(net.parameters())

        # 前向+反向传播检查
        offset = net(img)
        shifted_img = offset_tensor_v3(img, offset)
        loss = shifted_img.mean()
        loss.backward()

        # 检查梯度是否存在于offset生成网络的参数中
        assert net.conv.weight.grad is not None  # 应返回True

    def test_offset_tensor():
        label = torch.from_numpy(
            np.asarray(
                cv2.imread(
                    "/root/workspace/zjj/xjd/data/segmentation/Turkey/Islahiye/pre/test/labels/9216_10240.png",
                    cv2.IMREAD_GRAYSCALE,
                ),
                dtype=np.float32,
            )
        )
        tensor = label.unsqueeze(0).unsqueeze(0)
        offset = label.unsqueeze(0).unsqueeze(0) / 255.0  # 模拟偏移量
        offset = offset[:, :, 10:, 10:]  # 移除前10行和前10列
        offset = F.pad(
            offset, (0, 10, 0, 10), mode="constant", value=0
        )  # 在尾部添加10个0值元素
        offset = torch.cat((offset, offset), dim=1)
        print(f"{tensor.shape=}, {offset.shape=} {offset.max()=}, {offset.min()=}")
        shifted_tensor = offset_tensor_v3(tensor, offset / 100)

        cv2.imwrite("shifted_tensor.png", shifted_tensor[0, 0].cpu().numpy())
        cv2.imwrite("offset_tensor.png", offset[0, 0].cpu().numpy() * 255)
        cv2.imwrite("original_tensor.png", label.cpu().numpy())

    test_offset_tensor()
