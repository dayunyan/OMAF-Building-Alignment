from math import ceil
from typing import List
import torch
from torch import nn, Tensor
import torch.nn.functional as F

import geomloss


def tensor2pointcloud(im: Tensor, patch_size, norm=True, scale_factor=0.5):
    if im.ndim == 4:
        im = im.argmax(dim=1, keepdim=True)
    if im.ndim == 3:
        im = im.unsqueeze(1)

    assert im.ndim == 4, NotImplementedError("ndim of the input must be 4")

    im = F.interpolate(
        im.to(dtype=torch.float32), scale_factor=scale_factor, mode="nearest"
    ).squeeze(
        1
    )  # avoid OOM ERROR
    batch_size, height, width = im.shape

    signature = []
    num_points_list = []  # 新增，用于记录每个样本的点数
    for b in range(batch_size):
        for h in range(0, height, patch_size[0]):
            for w in range(0, width, patch_size[1]):
                patch = im[b, h : h + patch_size[0], w : w + patch_size[1]]
                coords = torch.nonzero(patch == 1).to(dtype=torch.float32)
                num_points = coords.shape[0]  # 获取当前样本对应的点数
                if norm:
                    coords -= coords.mean(dim=0, keepdim=True)
                    # coords /= (
                    #     torch.tensor([patch_size[0], patch_size[1]])
                    #     .view(1, 2)
                    #     .to(im.device)
                    #     / 2
                    # )
                signature.append(coords)
                num_points_list.append(num_points)
    return signature, num_points_list


def split_and_convert(im: torch.Tensor, patch_size, norm=True, scale_factor=0.5):
    """
    将输入的语义标签张量切割成多个小块，并将每个小块作为单独的样本转成点云结构。

    参数:
    - im: 输入的语义标签张量，形状为 (batch_size, channels, height, width)，一般channels为1
    - patch_size: 切割的小块尺寸，例如 (patch_h, patch_w)
    - norm: 是否进行归一化，默认为True
    - scale_factor: 下采样的比例因子，默认为0.5

    返回:
    - all_signatures: 包含所有小块样本点云结构的列表，每个元素对应一个小块样本的坐标信息
    - all_num_points_lists: 包含所有小块样本点数的列表，每个元素对应一个小块样本的点数
    """
    assert im.ndim == 4, "输入张量的维度必须为4"
    batch_size, _, height, width = im.shape
    all_signatures = []
    all_num_points_lists = []
    for b in range(batch_size):
        for h in range(0, height, patch_size[0]):
            for w in range(0, width, patch_size[1]):
                patch = im[b : b + 1, :, h : h + patch_size[0], w : w + patch_size[1]]
                patch_signature, patch_num_points_list = tensor2pointcloud(
                    patch, norm=norm, scale_factor=scale_factor
                )
                all_signatures.extend(patch_signature)
                all_num_points_lists.extend(patch_num_points_list)
    return all_signatures, all_num_points_lists


def group_samples_by_num_points(num_points_list, intervals: List[tuple]):
    groups = {interval: [] for interval in intervals}
    for idx, num_points_per_sample in enumerate(num_points_list):
        for i, interval in enumerate(intervals):
            if all([interval[0] <= num_points_per_sample <= interval[1]]):
                groups[interval].append(idx)
    return groups


def pad_group(group_sig):
    max_num_points = max([s.shape[0] for s in group_sig])
    padded_sig = []
    num_points_list = []
    for sample in group_sig:
        num_points = sample.shape[0]
        num_points_list.append(num_points)
        pad_size = max_num_points - num_points
        padded_sample = torch.cat(
            [
                sample,
                torch.zeros(
                    (pad_size, sample.shape[1]),
                    dtype=sample.dtype,
                    device=sample.device,
                ),
            ],
            dim=0,
        )
        padded_sig.append(padded_sample)
    return torch.stack(padded_sig), num_points_list


def unpad_loss(group_loss, logit_mask, target_mask):
    valid_loss = []
    for sample_loss, sample_logit_mask, sample_target_mask in zip(
        group_loss, logit_mask, target_mask
    ):
        valid_sample_loss = sample_loss[sample_logit_mask][:, sample_target_mask]
        valid_loss.append(torch.sum(valid_sample_loss))
    return valid_loss


def soft_emd_loss(
    tensor1: Tensor,
    tensor2: Tensor,
    scale_factor=0.5,
    reduction="mean",
):
    if tensor1.ndim == 3:
        tensor1 = tensor1.unsqueeze(1)
    if tensor2.ndim == 3:
        tensor2 = tensor2.unsqueeze(1)

    assert tensor1.ndim == 4 and tensor2.ndim == 4, NotImplementedError(
        "ndim of the input must be 4"
    )
    # if tensor1.dtype != torch.float64:
    #     tensor1 = tensor1.to(torch.float64)
    # if tensor2.dtype != torch.float64:
    #     tensor2 = tensor2.to(torch.float64)
    tensor1 = F.interpolate(
        tensor1, scale_factor=scale_factor, mode="bilinear"
    ).squeeze(1)
    tensor2 = F.interpolate(
        tensor2, scale_factor=scale_factor, mode="bilinear"
    ).squeeze(1)

    # to probability, sum to 1
    tensor1 = tensor1 / (torch.sum(tensor1, dim=(-2, -1), keepdim=True) + 1e-16)
    tensor2 = tensor2 / (torch.sum(tensor2, dim=(-2, -1), keepdim=True) + 1e-16)
    # k = 10
    # tensor1 = torch.sigmoid((tensor1 - 0.5) * k)
    # tensor2 = torch.sigmoid((tensor2 - 0.5) * k)

    batch_size, height, width = tensor1.shape
    coords = torch.meshgrid(
        torch.arange(height, dtype=tensor1.dtype, device=tensor1.device),
        torch.arange(width, dtype=tensor1.dtype, device=tensor1.device),
        indexing="ij",
    )
    coords = torch.stack(coords, dim=-1)  # shape: [H,W,2]
    coords = coords / torch.tensor([height - 1, width - 1], device=tensor1.device)
    # coords = (
    #     (coords / torch.tensor([height, width]))
    #     .view(-1, 2)
    #     .unsqueeze(0)
    #     .repeat(batch_size, 1, 1)
    #     .to(tensor1.device)
    # )  # shape: [B,H*W,2]

    point_cloud1 = tensor1.unsqueeze(-1) * coords.unsqueeze(0)  # [B,H,W,2]
    point_cloud1 = point_cloud1.view(batch_size, -1, 2)  # [B,N,2]
    point_cloud2 = tensor2.unsqueeze(-1) * coords.unsqueeze(0)
    point_cloud2 = point_cloud2.view(batch_size, -1, 2)

    # # 随机采样点云（每样本最多K个点）
    # K = 8192  # 根据显存调整
    # B, N, _ = point_cloud1.shape
    # num_samples = min(height * width, K)
    # # 生成随机索引
    # prob = tensor2.view(B, -1).clamp(min=0.0)  # [B, H*W]
    # print(f"{prob.sum()=}")
    # if torch.isnan(prob).any() or (prob < 0).any():
    #     raise ValueError("Invalid probability distribution")
    # indices = torch.multinomial(prob.cpu(), num_samples, replacement=False)
    # print(f"{indices.shape=}")
    # point_cloud1 = point_cloud1[:, indices, :]  # [B, K, 2]
    # point_cloud2 = point_cloud2[:, indices, :]

    p = 1
    entreg = 0.1
    OTLoss = geomloss.SamplesLoss(
        loss="sinkhorn",
        p=p,  # 对于p=1或p=2的情形
        cost=geomloss.utils.distances if p == 1 else geomloss.utils.squared_distances,
        blur=entreg ** (1 / p),
        backend="tensorized",
        scaling=0.5,
    )
    pW = OTLoss(point_cloud1, point_cloud2)

    return pW.mean() if reduction == "mean" else pW.sum()


def diff_num_emdloss(
    logits: Tensor,
    targets: Tensor,
    is_patch: bool = False,
    patch_size=(128, 128),
    scale_factor=0.5,
    crop_size=None,
    eps=0.01,
    max_iter=100,
    thresh=1e-9,
    reduction="none",
):
    if crop_size is not None:
        # crop tensor to crop_size with center crop
        OH, OW = logits.shape[-2:]
        center_h, center_w = OH // 2, OW // 2
        crop_h, crop_w = int(crop_size[0] * OH), int(crop_size[1] * OW)
        if crop_h % 2 == 1:
            crop_h += 1
        if crop_w % 2 == 1:
            crop_w += 1
        if crop_h > OH or crop_w > OW:
            raise ValueError("crop size is larger than original size")
        start_h = center_h - crop_h // 2
        start_w = center_w - crop_w // 2
        end_h = center_h + crop_h // 2
        end_w = center_w + crop_w // 2
        logits = logits[..., start_h:end_h, start_w:end_w]
        targets = targets[..., start_h:end_h, start_w:end_w]
    B = logits.shape[0]
    H, W = logits.shape[-2], logits.shape[-1]
    logit_sig, logit_num_points = tensor2pointcloud(
        logits,
        patch_size=(
            patch_size if is_patch else (ceil(H * scale_factor), ceil(W * scale_factor))
        ),
        norm=True,
        scale_factor=scale_factor,
    )  # [B*(H*W*(scale_factor**2)//(128*128)), N, 2], [B*(H*W*(scale_factor**2)//(128*128)), N]
    target_sig, target_num_points = tensor2pointcloud(
        targets,
        patch_size=(
            patch_size if is_patch else (int(H * scale_factor), int(W * scale_factor))
        ),
        norm=True,
        scale_factor=scale_factor,
    )  # [B*(H*W*(scale_factor**2)//(128*128)), N, 2], [B*(H*W*(scale_factor**2)//(128*128)), N]

    # 设定点数区间
    # inter_ends = [_ * scale_factor for _ in [0, 64, 128, 256, 512]]
    # intervals = [
    #     (st**2 + 1, ed**2)
    #     for st, ed in zip(
    #         inter_ends[: len(inter_ends) - 1], inter_ends[1 : len(inter_ends)]
    #     )
    # ]
    # intervals = [
    #     (1, 64 * 64),
    #     (64 * 64 + 1, 128 * 128),
    #     (128 * 128 + 1, 256 * 256),
    #     (256 * 256 + 1, 512 * 512),
    # ]
    if is_patch:
        intervals = [
            (0, 32 * 32),
            (32 * 32 + 1, 64 * 64),
            (64 * 64 + 1, 128 * 128),
        ]
    else:
        intervals = [
            (0, H * scale_factor * W * scale_factor),
        ]
    sample_groups = group_samples_by_num_points(target_num_points, intervals)

    all_losses = []
    for interval, group_indices in sample_groups.items():
        if not group_indices:  # 如果该组没有样本，跳过
            continue
        group_logit_sig = [logit_sig[i] for i in group_indices]  # [[N, 2]]
        group_target_sig = [target_sig[i] for i in group_indices]  # [[N, 2]]

        # 填充并获取有效点数
        padded_logit, logit_n = pad_group(group_logit_sig)
        padded_target, target_n = pad_group(group_target_sig)

        EMD = EMDLoss(eps, max_iter, thresh, reduction="none")
        group_loss, _, _ = EMD(padded_logit, padded_target, logit_n, target_n)

        all_losses.append(group_loss)

    total_loss = torch.cat(all_losses)
    return total_loss.mean() if reduction == "mean" else total_loss.sum()


class EMDLoss(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """

    def __init__(self, eps=0.01, max_iter=100, thresh=1e-3, reduction="none"):
        super().__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction
        self.thresh = thresh

    # def forward(
    #     self, x: Tensor, y: Tensor, x_num_points: List[int], y_num_points: List[int]
    # ):
    def forward(self, x: Tensor, y: Tensor, prob_x: Tensor, prob_y: Tensor):
        # generate the weights for the marginals
        # mu = self._create_marginals(x, x_num_points)
        # nu = self._create_marginals(y, y_num_points)
        mu = prob_x
        nu = prob_y

        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y).to(x.device)  # Wasserstein cost function

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u, v = self.update(u, v, C, mu, nu)
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < self.thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        # cost = torch.sum(pi * C, dim=(-2, -1))
        cost = pi * C
        # print(f"cost shape: {cost.shape}")

        if self.reduction == "mean":
            cost = torch.sum(cost, dim=(-2, -1))
            cost = cost.mean()
        elif self.reduction == "sum":
            cost = cost.sum()

        return cost, pi, C

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    def update(self, u, v, C, mu, nu):
        u_new = (
            self.eps
            * (torch.log(mu + 1e-16) - torch.logsumexp(self.M(C, u, v), dim=-1))
            + u
        )
        v_new = (
            self.eps
            * (
                torch.log(nu + 1e-16)
                - torch.logsumexp(self.M(C, u_new, v).transpose(-2, -1), dim=-1)
            )
            + v
        )
        return u_new, v_new

    # def _create_marginals(self, points, num_points):
    #     marginals = torch.zeros_like(points[..., 0], requires_grad=False)
    #     for i, n in enumerate(num_points):
    #         if n > 0:
    #             marginals[i, :n] = 1.0 / n
    #     return marginals

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1


if __name__ == "__main__":

    # def test_emd_position_invariance():
    #     """验证EMD损失是否与绝对位置无关"""
    #     # 生成两个仅位置不同的点云
    #     # 情况1：相同分块内的平移
    #     patch_size = (128, 128)
    #     scale_factor = 1.0

    #     # 生成基础形状（中心方块）
    #     base = torch.zeros(1, 256, 256)
    #     base[:, 112:144, 112:144] = 1  # 32x32方块居中

    #     # 生成平移后的形状（向右平移64像素）
    #     shifted = torch.zeros(1, 256, 256)
    #     shifted[:, 56:144, 176:208] = 1  # 保持32x32尺寸

    #     # # 转换为点云
    #     base_sig, base_point_num = tensor2pointcloud(
    #         base, patch_size, scale_factor=scale_factor
    #     )
    #     shifted_sig, shifted_point_num = tensor2pointcloud(
    #         shifted, patch_size, scale_factor=scale_factor
    #     )

    #     # # 情况1验证：同一分块内的平移（应产生相同损失）
    #     # emd = EMDLoss(reduction="mean")
    #     # loss_same_patch, _, _ = emd(
    #     #     base_sig[0].unsqueeze(0),
    #     #     base_sig[0].unsqueeze(0),
    #     #     [base_point_num[0]],
    #     #     [base_point_num[0]],
    #     # )
    #     loss_same_patch = diff_num_emdloss(
    #         base, base, scale_factor=scale_factor, reduction="sum"
    #     )
    #     print(f"相同点云损失: {loss_same_patch.item()} (应为0)")

    #     # 情况2验证：不同分块内的相同形状
    #     # loss_diff_patch, _, _ = emd(
    #     #     base_sig[0].unsqueeze(0),
    #     #     shifted_sig[0].unsqueeze(0),
    #     #     [base_point_num[0]],
    #     #     [shifted_point_num[0]],
    #     # )
    #     loss_diff_patch = diff_num_emdloss(
    #         base, shifted, scale_factor=scale_factor, reduction="sum"
    #     )
    #     print(f"跨分块相同形状损失: {loss_diff_patch.item()} (应接近0但实际非零)")

    #     # 可视化归一化后的坐标
    #     print("\n归一化坐标对比：")
    #     print("原始点云坐标（分块内）：\n", base_sig[0][:5])
    #     print("平移后点云坐标（不同分块）：\n", shifted_sig[0][:5])

    # # 执行测试
    # test_emd_position_invariance()

    # def test_emd_loss_grad():
    #     # 测试EMD损失的梯度计算
    #     B, C, H, W = 2, 1, 32, 32
    #     tensor1 = torch.rand(B, C, H, W, dtype=torch.float64, requires_grad=True)
    #     tensor2 = torch.rand(B, C, H, W, dtype=torch.float64, requires_grad=True)
    #     assert torch.autograd.gradcheck(
    #         soft_emd_loss,
    #         (tensor1, tensor2, 0.125),
    #         eps=1e-6,
    #         atol=1e-4,
    #         rtol=1e-4,
    #     )

    # test_emd_loss_grad()

    # def test_grad_with_net():
    #     import sys

    #     sys.path.append("../..")
    #     from tools.offset import offset_tensor_v3

    #     # 初始化生成offset的神经网络（示例）
    #     class OffsetNet(torch.nn.Module):
    #         def __init__(self):
    #             super().__init__()
    #             self.conv = torch.nn.Conv2d(3, 2, kernel_size=3, padding=1)

    #         def forward(self, x):
    #             return self.conv(x)

    #     # 输入模拟
    #     img = torch.randn(4, 3, 256, 256, requires_grad=True)
    #     label = torch.randn(4, 1, 256, 256, requires_grad=True, dtype=torch.float32)
    #     net = OffsetNet()
    #     optimizer = torch.optim.Adam(net.parameters())

    #     # 前向+反向传播检查
    #     offset = net(img)
    #     shifted_img = offset_tensor_v3(label, offset)
    #     print(f"Shifted image shape: {shifted_img.shape}")
    #     loss = soft_emd_loss(shifted_img, label, scale_factor=0.25)
    #     print(f"Loss: {loss.item()}")
    #     loss.backward()

    #     # 检查梯度是否存在于offset生成网络的参数中
    #     assert net.conv.weight.grad is not None  # 应返回True
    #     print(f"Gradient exists: {net.conv.weight.grad}")

    # test_grad_with_net()

    def test_emd_position_invariance():
        """验证EMD损失是否与绝对位置无关"""
        # 生成两个仅位置不同的点云
        # 情况1：相同分块内的平移
        patch_size = (128, 128)
        scale_factor = 1.0

        # 生成基础形状（中心方块）
        base = torch.zeros(1, 256, 256)
        base[:, 112:144, 112:144] = 1  # 32x32方块居中

        # 生成平移后的形状（向右平移64像素）
        shifted = torch.zeros(1, 256, 256)
        shifted[:, 56:144, 176:208] = 1  # 保持32x32尺寸

        loss = soft_emd_loss(base, shifted, scale_factor=scale_factor, reduction="mean")

        print(f"相同形状损失: {loss.item()} (应为0)")

        triangle = torch.zeros(1, 256, 256)
        # 生成一个三角形区域
        for i in range(112, 144):
            triangle[:, i, 112 : 112 + (i - 112 + 1)] = (
                1  # 生成一个右上角为直角的三角形
            )

        loss = soft_emd_loss(
            base, triangle, scale_factor=scale_factor, reduction="mean"
        )
        print(f"不同形状损失: {loss.item()} (应接近0但实际非零)")

    # 执行测试
    test_emd_position_invariance()
