import os
import cv2
import numpy as np
import torch
from pathlib import Path


def save_by_cv2(path, image):
    cv2.imwrite(path, image)


def apply_styled_overlay(
    origin_img,
    colored_mask,
    alpha=0.6,
    contour_thickness=1,
    contour_color=(255, 255, 255),
):
    """
    应用参考图风格的遮罩：
    1. 保持原图亮度。
    2. 仅在 Mask 区域半透明叠加颜色。
    3.
    Args:
        origin_img (np.array): 原始 BGR 图像。
        colored_mask (np.array): BGR 格式的彩色遮罩（例如R,G,Y或灰色），背景为黑色。
                                 必须与 origin_img 尺寸相同。
        alpha (float): 遮罩的透明度 (0.0 - 1.0)。
        contour_thickness (int): 轮廓线粗细，0 为不绘制。
        contour_color (tuple): 轮廓线 BGR 颜色。
    """

    # 1. 从彩色遮罩中提取二值轮廓
    # 任何非黑色像素都属于 mask
    mask_2d = cv2.cvtColor(colored_mask, cv2.COLOR_BGR2GRAY)
    _, mask_binary = cv2.threshold(mask_2d, 0, 255, cv2.THRESH_BINARY)

    # 2. 复制原图作为底板
    final_img = origin_img.copy()

    # 3. 仅在 Mask 区域应用半透明混合
    # 创建一个布尔索引
    mask_active = mask_binary > 0

    # 使用 NumPy 索引进行高效混合
    final_img[mask_active] = (
        origin_img[mask_active] * (1 - alpha) + colored_mask[mask_active] * alpha
    ).astype(np.uint8)

    # 4. 绘制轮廓
    if contour_thickness > 0:
        contours, _ = cv2.findContours(
            mask_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(
            final_img,
            contours,
            -1,
            contour_color,
            contour_thickness,
            lineType=cv2.LINE_AA,
        )

    return final_img


def generate_heatmap_overlay(
    origin_img,
    align_np,
    heatmap_alpha=0.5,
    contour_thickness=1,
    contour_color=(255, 255, 255),
):
    """生成带有效区域限制的热力图叠加图（新样式：带轮廓）"""
    origin_H, origin_W = origin_img.shape[:2]
    align_H, align_W = align_np.shape

    # 处理align：0区域设为NaN（不参与热力图）
    align_valid = align_np.copy()
    align_valid[align_valid <= 0] = np.nan
    valid_mask = ~np.isnan(align_valid)

    # 生成热力图（仅有效区域归一化）
    align_normalized = np.zeros_like(align_np, dtype=np.uint8)
    if np.any(valid_mask):
        align_min = np.nanmin(align_valid)
        align_max = np.nanmax(align_valid)
        align_normalized[valid_mask] = (
            (align_valid[valid_mask] - align_min) / (align_max - align_min + 1e-8) * 255
        ).astype(np.uint8)

    # 使用更美观的 Colormap (PLASMA)
    heatmap = cv2.applyColorMap(align_normalized, cv2.COLORMAP_PLASMA)

    # 生成热力图掩码（仅有效区域保留颜色）
    heatmap_mask = np.zeros((align_H, align_W, 3), dtype=np.uint8)
    heatmap_mask[valid_mask] = heatmap[valid_mask]

    # 缩放至原图尺寸
    heatmap_resized = cv2.resize(
        heatmap_mask, (origin_W, origin_H), interpolation=cv2.INTER_LINEAR
    )
    # 缩放二值有效区域 (用于轮廓和混合)
    valid_mask_2d_resized = cv2.resize(
        valid_mask.astype(np.uint8),
        (origin_W, origin_H),
        interpolation=cv2.INTER_NEAREST,
    )
    # 创建3通道布尔掩码
    mask_active_bool = valid_mask_2d_resized > 0

    # 叠加：仅有效区域混合热力图和原图
    overlay_img = origin_img.copy()
    overlay_img[mask_active_bool] = (
        origin_img[mask_active_bool] * (1 - heatmap_alpha)
        + heatmap_resized[mask_active_bool] * heatmap_alpha
    ).astype(np.uint8)

    # --- 新增：为热力图有效区域添加轮廓 ---
    contours, _ = cv2.findContours(
        valid_mask_2d_resized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(
        overlay_img,
        contours,
        -1,
        contour_color,
        contour_thickness,
        lineType=cv2.LINE_AA,
    )
    # --- 轮廓添加完毕 ---

    return overlay_img


def add_heatmap_legend(
    main_img,
    align_np,
    legend_bar_width=30,
    legend_text_width=60,
    colormap=cv2.COLORMAP_PLASMA,
):
    """给热力图添加类似 Matplotlib 风格的颜色图例（纵向排列在主图左侧）"""
    main_H, main_W = main_img.shape[:2]
    legend_H = main_H
    legend_W = legend_bar_width + legend_text_width  # 图例总宽度

    # 1. 创建白色图例背景板
    legend_bg = np.full((legend_H, legend_W, 3), 255, dtype=np.uint8)

    # 确定颜色条位置 (右侧)
    bar_x_start = legend_text_width
    bar_x_end = legend_W

    # 2. 生成纵向渐变条
    gradient = np.linspace(255, 0, legend_H, dtype=np.uint8).reshape(-1, 1)
    gradient = np.tile(gradient, (1, legend_bar_width))

    # 应用 colormap
    gradient_color = cv2.applyColorMap(gradient, colormap)

    # 3. 绘制渐变条到背景板
    legend_bg[:, bar_x_start:bar_x_end, :] = gradient_color

    # 4. 数值范围（固定0-1）
    min_val, max_val = 0, 1

    # 5. 添加数值标签和刻度线
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    font_color = (0, 0, 0)  # 黑色文字
    thickness = 1
    num_labels = 6

    # 计算标签位置（均匀分布，留边距）
    y_padding = 20
    y_positions = np.linspace(y_padding, legend_H - y_padding, num_labels, dtype=int)
    value_positions = np.linspace(max_val, min_val, num_labels)

    # 刻度线位置（颜色条左侧）
    tick_x_start = bar_x_start - 5
    tick_x_end = bar_x_start + 5

    for y, val in zip(y_positions, value_positions):
        text = f"{val:.1f}"
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        x_margin = 5
        text_x_pos = tick_x_start - x_margin - text_w
        text_y_pos = y + text_h // 2

        # 绘制文本和刻度线
        cv2.putText(
            legend_bg,
            text,
            (text_x_pos, text_y_pos),
            font,
            font_scale,
            font_color,
            thickness,
            cv2.LINE_AA,
        )
        cv2.line(legend_bg, (tick_x_start, y), (tick_x_end, y), font_color, thickness)

    # 6. 拼接图例和主图（图例在左）
    combined_img = np.hstack((legend_bg, main_img))
    return combined_img


def main():
    output_type = [
        "align_and_gt",
        "alignHeatmap_and_image",
        "corrected_pseudo_gt",
        "moved_corrected_pseudo_gt",
        "gt_overlay",  # 新增1：GT标签灰白色叠加原图
        "offset_label_overlay",  # 新增2：偏移标签（labels文件夹）灰白色叠加原图
    ]
    # ---------------------- 核心控制变量：选择生成类型 ----------------------
    output_type = output_type[3]  # 切换类型：4=gt_overlay,5=offset_label_overlay
    # ----------------------------------------------------------------------

    # 输入路径配置（新增偏移标签路径）
    image_paths = "../data/segmentation/Turkey/Islahiye/pre/test/images"
    align_paths = "../data/segmentation/Turkey/Islahiye/pre/test/align"
    gt_paths = "../data/segmentation/Turkey/Islahiye/pre/test/gt"
    corrected_label_paths = "../data/segmentation/Turkey/Islahiye/pre/test/pred_offsets"
    offset_label_paths = (
        "../data/segmentation/Turkey/Islahiye/pre/test/labels"  # 新增：偏移标签路径
    )

    # --- 新样式配置 ---
    STYLE_ALPHA = 0.6  # 遮罩透明度
    STYLE_CONTOUR_THICKNESS = 2  # 轮廓粗细 (0为关闭)
    STYLE_CONTOUR_COLOR = (255, 255, 255)  # 轮廓颜色 (BGR)
    # ---

    # 输出路径（按类型自动区分）
    if output_type == "align_and_gt":
        output_dir = "fig_results/ablation/align_and_gt"
        print(f"当前生成类型：伪彩色叠加图（新样式）")
    elif output_type == "alignHeatmap_and_image":
        output_dir = "fig_results/ablation/alignHeatmap_and_image"
        print(f"当前生成类型：热力图叠加图（新样式：带轮廓）")
    elif output_type == "corrected_pseudo_gt":
        output_dir = "fig_results/ablation/corrected_pseudo_gt"
        print(f"当前生成类型：矫正标签+GT伪彩色叠加图（新样式）")
    elif output_type == "moved_corrected_pseudo_gt":
        output_dir = "fig_results/ablation/moved_corrected_pseudo_gt"
        print(f"当前生成类型：【移动后】矫正标签+GT伪彩色叠加图（新样式）")
    elif output_type == "gt_overlay":
        output_dir = "fig_results/ablation/gt_overlay"
        print(f"当前生成类型：GT标签灰白色叠加图（新样式）")
    elif output_type == "offset_label_overlay":
        output_dir = "fig_results/ablation/offset_label_overlay"
        print(f"当前生成类型：偏移标签灰白色叠加图（新样式）")
    else:
        raise ValueError(f"无效的 output_type：{output_type}")
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有文件并排序（新增偏移标签文件读取）
    image_files = sorted(
        [
            f
            for f in Path(image_paths).glob("*.*")
            if f.suffix in [".png", ".jpg", ".jpeg"]
        ]
    )
    align_files = sorted([f for f in Path(align_paths).glob("*.pt") if f.is_file()])
    gt_files = sorted([f for f in Path(gt_paths).glob("*.png") if f.is_file()])
    corrected_label_files = sorted(
        [f for f in Path(corrected_label_paths).glob("*.png") if f.is_file()]
    )
    offset_label_files = sorted(
        [f for f in Path(offset_label_paths).glob("*.png") if f.is_file()]
    )  # 新增：读取偏移标签

    # 校验文件数量一致（根据输出类型判断需要校验的文件）
    required_files = [image_files]
    if output_type == "align_and_gt":
        required_files.extend([align_files, gt_files])
    elif output_type == "alignHeatmap_and_image":
        required_files.append(align_files)
    elif output_type in ["corrected_pseudo_gt", "moved_corrected_pseudo_gt"]:
        required_files.extend([gt_files, corrected_label_files])
    elif output_type == "gt_overlay":
        required_files.append(gt_files)
    elif output_type == "offset_label_overlay":
        required_files.append(offset_label_files)

    assert all(
        len(files) == len(image_files) for files in required_files
    ), f"文件数不匹配：图像({len(image_files)})、align({len(align_files)})、gt({len(gt_files)})、矫正标签({len(corrected_label_files)})、偏移标签({len(offset_label_files)})"
    print(f"共找到 {len(image_files)} 组文件，开始生成...")

    # 遍历处理每组文件
    for idx in range(len(image_files)):
        image_file = image_files[idx]

        # 1. 加载原图像（保持原始亮度）
        origin_img = cv2.imread(str(image_file))
        if origin_img is None:
            print(f"警告：无法读取原图像 {image_file.name}，跳过该文件")
            continue

        # 2. 按类型处理核心逻辑
        if output_type == "align_and_gt":
            align_tensor = torch.load(open(align_files[idx], "rb"), map_location="cpu")
            align_np = align_tensor.numpy().astype(np.float32)
            gt_np = cv2.imread(str(gt_files[idx]), cv2.IMREAD_GRAYSCALE)
            if gt_np.shape != align_np.shape:
                gt_np = cv2.resize(
                    gt_np,
                    (align_np.shape[1], align_np.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            # 伪彩色规则
            pseudo_mask = np.zeros(
                (align_np.shape[0], align_np.shape[1], 3), dtype=np.uint8
            )
            pseudo_mask[(align_np > 0) & (gt_np == 255)] = (0, 255, 255)  # 黄
            pseudo_mask[(align_np > 0) & (gt_np == 0)] = (0, 0, 255)  # 红
            pseudo_mask[(align_np <= 0) & (gt_np == 255)] = (0, 255, 0)  # 绿

            # 缩放
            pseudo_mask_resized = cv2.resize(
                pseudo_mask,
                (origin_img.shape[1], origin_img.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

            # --- 应用新样式 ---
            final_img = apply_styled_overlay(
                origin_img,
                pseudo_mask_resized,
                alpha=STYLE_ALPHA,
                contour_thickness=STYLE_CONTOUR_THICKNESS,
                contour_color=STYLE_CONTOUR_COLOR,
            )

        elif output_type == "alignHeatmap_and_image":
            # 加载align tensor
            align_tensor = torch.load(open(align_files[idx], "rb"), map_location="cpu")
            align_np = align_tensor.numpy().astype(np.float32)
            # 生成热力图叠加图（函数内部已更新为新样式）
            heatmap_overlay_img = generate_heatmap_overlay(
                origin_img,
                align_np,
                heatmap_alpha=STYLE_ALPHA,
                contour_thickness=STYLE_CONTOUR_THICKNESS,
                contour_color=STYLE_CONTOUR_COLOR,
            )
            # 添加颜色图例
            final_img = add_heatmap_legend(
                heatmap_overlay_img, align_np, colormap=cv2.COLORMAP_PLASMA
            )

        elif output_type == "corrected_pseudo_gt":
            gt_np = cv2.imread(str(gt_files[idx]), cv2.IMREAD_GRAYSCALE)
            corrected_label_np = cv2.imread(
                str(corrected_label_files[idx]), cv2.IMREAD_GRAYSCALE
            )
            if corrected_label_np.shape != gt_np.shape:
                corrected_label_np = cv2.resize(
                    corrected_label_np,
                    (gt_np.shape[1], gt_np.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            # 生成伪彩色掩码
            pseudo_mask = np.zeros((gt_np.shape[0], gt_np.shape[1], 3), dtype=np.uint8)
            pseudo_mask[(corrected_label_np > 0) & (gt_np == 0)] = (0, 0, 255)  # 红
            pseudo_mask[(corrected_label_np <= 0) & (gt_np > 0)] = (0, 255, 0)  # 绿
            pseudo_mask[(corrected_label_np > 0) & (gt_np > 0)] = (0, 255, 255)  # 黄

            # 缩放
            pseudo_mask_resized = cv2.resize(
                pseudo_mask,
                (origin_img.shape[1], origin_img.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

            # --- 应用新样式 ---
            final_img = apply_styled_overlay(
                origin_img,
                pseudo_mask_resized,
                alpha=STYLE_ALPHA,
                contour_thickness=STYLE_CONTOUR_THICKNESS,
                contour_color=STYLE_CONTOUR_COLOR,
            )

        elif output_type == "moved_corrected_pseudo_gt":
            gt_np = cv2.imread(str(gt_files[idx]), cv2.IMREAD_GRAYSCALE)
            corrected_label_np = cv2.imread(
                str(corrected_label_files[idx]), cv2.IMREAD_GRAYSCALE
            )
            _, corrected_label_binary = cv2.threshold(
                corrected_label_np, 0, 255, cv2.THRESH_BINARY
            )
            if corrected_label_binary.shape != gt_np.shape:
                corrected_label_binary = cv2.resize(
                    corrected_label_binary,
                    (gt_np.shape[1], gt_np.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            H, W = corrected_label_binary.shape
            moved_label_np = np.zeros_like(corrected_label_binary)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                corrected_label_binary, connectivity=8
            )
            for i in range(1, num_labels):
                x, y, w, h, area = stats[i]
                centroid_x, centroid_y = centroids[i]
                bbox_center_x = x + w / 2.0
                bbox_center_y = y + h / 2.0
                offset_x = centroid_x - bbox_center_x
                offset_y = centroid_y - bbox_center_y
                offset_distance = np.sqrt(offset_x**2 + offset_y**2)
                random_translation_dist = np.random.uniform(0, offset_distance)
                if offset_distance > 1e-6:
                    norm_reverse_dir_x = -offset_x / offset_distance
                    norm_reverse_dir_y = -offset_y / offset_distance
                else:
                    norm_reverse_dir_x = 0
                    norm_reverse_dir_y = 0
                tx = norm_reverse_dir_x * random_translation_dist
                ty = norm_reverse_dir_y * random_translation_dist
                M = np.float32([[1, 0, tx], [0, 1, ty]])
                instance_mask = np.uint8(labels == i) * 255
                translated_instance = cv2.warpAffine(
                    instance_mask, M, (W, H), flags=cv2.INTER_NEAREST
                )
                moved_label_np |= translated_instance

            # 生成伪彩色掩码
            pseudo_mask = np.zeros((gt_np.shape[0], gt_np.shape[1], 3), dtype=np.uint8)
            pseudo_mask[(moved_label_np > 0) & (gt_np == 0)] = (0, 0, 255)  # 红
            pseudo_mask[(moved_label_np <= 0) & (gt_np > 0)] = (0, 255, 0)  # 绿
            pseudo_mask[(moved_label_np > 0) & (gt_np > 0)] = (0, 255, 255)  # 黄

            # 缩放
            pseudo_mask_resized = cv2.resize(
                pseudo_mask,
                (origin_img.shape[1], origin_img.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

            # --- 应用新样式 ---
            final_img = apply_styled_overlay(
                origin_img,
                pseudo_mask_resized,
                alpha=STYLE_ALPHA,
                contour_thickness=STYLE_CONTOUR_THICKNESS,
                contour_color=STYLE_CONTOUR_COLOR,
            )

        elif output_type == "gt_overlay":  # GT标签灰白色叠加
            gt_np = cv2.imread(str(gt_files[idx]), cv2.IMREAD_GRAYSCALE)
            if gt_np.shape[:2] != origin_img.shape[:2]:
                gt_np = cv2.resize(
                    gt_np,
                    (origin_img.shape[1], origin_img.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            # 创建灰白色掩码
            gray_mask = np.zeros_like(origin_img, dtype=np.uint8)
            gray_mask[gt_np > 0] = (128, 128, 128)  # 灰白色

            # --- 应用新样式 ---
            final_img = apply_styled_overlay(
                origin_img,
                gray_mask,
                alpha=STYLE_ALPHA,
                contour_thickness=STYLE_CONTOUR_THICKNESS,
                contour_color=STYLE_CONTOUR_COLOR,
            )

        elif output_type == "offset_label_overlay":  # 偏移标签灰白色叠加
            offset_label_np = cv2.imread(
                str(offset_label_files[idx]), cv2.IMREAD_GRAYSCALE
            )
            if offset_label_np.shape[:2] != origin_img.shape[:2]:
                offset_label_np = cv2.resize(
                    offset_label_np,
                    (origin_img.shape[1], origin_img.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            # 创建灰白色掩码
            gray_mask = np.zeros_like(origin_img, dtype=np.uint8)
            gray_mask[offset_label_np > 0] = (128, 128, 128)  # 灰白色

            # --- 应用新样式 ---
            final_img = apply_styled_overlay(
                origin_img,
                gray_mask,
                alpha=STYLE_ALPHA,
                contour_thickness=STYLE_CONTOUR_THICKNESS,
                contour_color=STYLE_CONTOUR_COLOR,
            )

        # 保存最终图像
        save_path = os.path.join(output_dir, f"{image_file.stem}.png")
        save_by_cv2(save_path, final_img)
        print(f"[{idx+1}/{len(image_files)}] 已保存：{save_path}")

    print(f"\n所有文件处理完成！结果保存至：{output_dir}")


if __name__ == "__main__":
    main()
