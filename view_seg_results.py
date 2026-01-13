import os
import cv2
import numpy as np
from pathlib import Path

# ==============================================================================
# I. 样式化叠加函数 (沿用上一段对话的逻辑)
# ==============================================================================


def apply_styled_overlay(
    origin_img,
    colored_mask,
    alpha=0.6,
    contour_thickness=1,
    contour_color=(255, 255, 255),
):
    """
    应用参考图风格的遮罩：仅在 Mask 区域半透明叠加颜色，并添加轮廓。
    """
    # 1. 从彩色遮罩中提取二值轮廓
    mask_2d = cv2.cvtColor(colored_mask, cv2.COLOR_BGR2GRAY)
    # 确保 mask_2d 不是全黑，否则 cv2.threshold 可能出错
    if np.max(mask_2d) == 0:
        return origin_img.copy()
    _, mask_binary = cv2.threshold(mask_2d, 1, 255, cv2.THRESH_BINARY)

    # 2. 复制原图作为底板
    final_img = origin_img.copy()

    # 3. 仅在 Mask 区域应用半透明混合
    mask_active = mask_binary > 0

    # 创建一个3通道的布尔掩码用于索引 (或者直接使用 numpy 广播)
    mask_active_3d = np.stack([mask_active] * 3, axis=-1)

    # 使用 NumPy 索引进行高效混合
    # 注意：这里需要确保索引后的数组形状一致，所以使用3D掩码
    if np.any(mask_active):
        final_img[mask_active_3d] = (
            origin_img[mask_active_3d] * (1 - alpha)
            + colored_mask[mask_active_3d] * alpha
        ).astype(np.uint8)

    # 4. 绘制轮廓
    if contour_thickness > 0 and np.any(mask_active):
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


def colorize_binary_mask(binary_mask_np, color):
    """将二值掩码转换为指定颜色的3通道 BGR 图像"""
    color_mask = np.zeros((*binary_mask_np.shape, 3), dtype=np.uint8)
    # 检查是否是灰度图，如果不是则转为灰度图
    if binary_mask_np.ndim == 3:
        binary_mask_np = cv2.cvtColor(binary_mask_np, cv2.COLOR_BGR2GRAY)

    # 假设二值掩码中非零值代表前景
    mask_indices = binary_mask_np > 0
    color_mask[mask_indices] = color
    return color_mask


def load_and_crop(file_path, x, y, crop_size, is_mask=False):
    """加载、裁剪图像/掩码，并根据需要转换为灰度图"""
    if not os.path.exists(file_path):
        # 即使文件不存在，也要返回一个黑色占位符，保持布局
        print(f"警告：文件不存在 -> {file_path}")
        return np.zeros((crop_size, crop_size, 3), dtype=np.uint8)

    img = cv2.imread(
        file_path, cv2.IMREAD_COLOR if not is_mask else cv2.IMREAD_GRAYSCALE
    )
    if img is None:
        print(f"警告：无法读取图像 -> {file_path}")
        return np.zeros((crop_size, crop_size, 3), dtype=np.uint8)

    # 确保裁剪尺寸不超出图像范围
    H, W = img.shape[:2]
    x_end = min(x + crop_size, W)
    y_end = min(y + crop_size, H)

    # 裁剪图像
    cropped_img = img[y:y_end, x:x_end]

    # 如果裁剪后尺寸不足 (x_end - x < crop_size)，则填充黑色
    H_cropped, W_cropped = cropped_img.shape[:2]
    if H_cropped != crop_size or W_cropped != crop_size:
        # 确定需要填充的尺寸
        H_pad = crop_size - H_cropped
        W_pad = crop_size - W_cropped

        # 创建一个全黑的画布，然后将裁剪部分粘贴上去
        if is_mask:
            # Mask 返回单通道，后续需要上色
            canvas = np.zeros((crop_size, crop_size), dtype=np.uint8)
            canvas[:H_cropped, :W_cropped] = cropped_img
            return canvas
        else:
            # 图像返回3通道 BGR
            canvas = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
            canvas[:H_cropped, :W_cropped, :] = cropped_img
            return canvas

    # 如果是彩色图像，直接返回；如果是灰度 mask，返回灰度
    if is_mask:
        return cropped_img

    # 如果是彩色图像，但读取时是灰度图 (不应该发生)，则转为 BGR
    if cropped_img.ndim == 2:
        return cv2.cvtColor(cropped_img, cv2.COLOR_GRAY2BGR)

    return cropped_img


# ==============================================================================
# II. 文件分组与主逻辑
# ==============================================================================


def group_results(root_dir):
    """扫描文件夹并按基线名称分组"""
    # (代码与上一版本相同，返回 {baseline_name: (path_to_baseline, path_to_optimized)} )
    all_dirs = [d for d in Path(root_dir).iterdir() if d.is_dir()]
    algorithm_groups = {}

    for dir_path in all_dirs:
        dir_name = dir_path.name
        try:
            baseline_name = dir_name.split("-")[0]
        except:
            continue

        if baseline_name not in algorithm_groups:
            algorithm_groups[baseline_name] = [None, None]

        if "pred_offsets" in dir_name:
            algorithm_groups[baseline_name][1] = dir_path
        else:
            algorithm_groups[baseline_name][0] = dir_path

    complete_groups = {
        name: (b.as_posix(), o.as_posix())
        for name, (b, o) in algorithm_groups.items()
        if b and o
    }

    return complete_groups


def create_dashed_separator(
    height, width=10, line_thickness=3, color=(0, 0, 0), dash_ratio=0.1
):
    """
    创建居中、加粗的黑色短虚线分隔符。

    :param height: 裁剪图像的高度。
    :param width: 分隔符图像的总宽度 (应较小)。
    :param line_thickness: 虚线的粗细 (已增大)。
    :param dash_ratio: 虚线覆盖的高度比例 (例如 0.3 代表从 30% 到 70% 的高度)。
    """
    separator = np.full((height, width, 3), 255, dtype=np.uint8)  # 白色背景

    # 计算虚线覆盖的起始和结束 y 坐标
    start_y = int(height * dash_ratio)
    end_y = int(height * (1 - dash_ratio))

    dash_length = 8  # 虚线段长度 (可调)
    gap_length = 8  # 间隙长度 (可调)

    center_x = width // 2

    # 绘制虚线
    y = start_y
    while y < end_y:
        y_end = min(y + dash_length, end_y)
        # 使用 line_thickness 变量
        cv2.line(
            separator,
            (center_x, y),
            (center_x, y_end),
            color,
            line_thickness,
            lineType=cv2.LINE_AA,
        )
        y += dash_length + gap_length

    return separator


def create_comparison_mosaic(
    results_root_dir,
    image_name_stem,
    x_coord,
    y_coord,
    method_order,
    crop_size=256,
    inner_spacing_px=10,
    pair_spacing_px=40,  # 组间距，用于分隔符总宽度
    dashed_line_thickness=3,  # 新增：虚线粗细
    origin_image_path="../data/segmentation/Turkey/Islahiye/pre/test/images",
    gt_label_path="../data/segmentation/Turkey/Islahiye/pre/test/gt",
    output_dir="fig_results/comparison_mosaic",
):

    # ... (常量定义，保持不变)
    SEG_COLOR = (255, 255, 0)
    GT_COLOR = (0, 255, 255)
    STYLE_ALPHA = 0.6
    STYLE_CONTOUR_THICKNESS = 1
    STYLE_CONTOUR_COLOR = (255, 255, 255)

    # 1. 发现并分组结果
    grouped_algorithms = group_results(results_root_dir)

    # 2. 准备固定的输入图像 (原图, GT)
    image_filename = f"{image_name_stem}.png"
    origin_img_cropped = load_and_crop(
        os.path.join(origin_image_path, image_filename),
        x_coord,
        y_coord,
        crop_size,
        is_mask=False,
    )
    gt_mask_cropped_gray = load_and_crop(
        os.path.join(gt_label_path, image_filename),
        x_coord,
        y_coord,
        crop_size,
        is_mask=True,
    )
    gt_mask_colored = colorize_binary_mask(gt_mask_cropped_gray, GT_COLOR)
    gt_overlay = apply_styled_overlay(
        origin_img_cropped,
        gt_mask_colored,
        alpha=STYLE_ALPHA,
        contour_thickness=STYLE_CONTOUR_THICKNESS,
        contour_color=STYLE_CONTOUR_COLOR,
    )

    # 3. 准备间隔元素 (已更新)

    # 3.1. 内间距 (Base/Optimized之间)
    inner_spacer = np.full((crop_size, inner_spacing_px, 3), 255, dtype=np.uint8)

    # 3.2. 组间分隔符 (包含居中虚线)

    # 创建虚线本身
    DASHED_SEP_WIDTH = 7  # 定义虚线本身的固定宽度
    dashed_line = create_dashed_separator(
        crop_size, width=DASHED_SEP_WIDTH, line_thickness=dashed_line_thickness
    )

    # 计算两侧空白区域
    total_blank_width = pair_spacing_px - DASHED_SEP_WIDTH
    blank_spacer_width_L = total_blank_width // 2
    blank_spacer_width_R = (
        total_blank_width - blank_spacer_width_L
    )  # 确保总宽度为 pair_spacing_px

    # 创建两侧的空白间隔
    spacer_blank_L = np.full((crop_size, blank_spacer_width_L, 3), 255, dtype=np.uint8)
    spacer_blank_R = np.full((crop_size, blank_spacer_width_R, 3), 255, dtype=np.uint8)

    # 组合成完整的中心分隔符
    center_separator = np.hstack([spacer_blank_L, dashed_line, spacer_blank_R])

    # 4. 拼接图像

    # 初始元素: [原图, 内间距, GT 叠加图]
    combined_elements = [origin_img_cropped, inner_spacer, gt_overlay]

    print(f"将按照顺序 {method_order} 生成对比结果...")

    # 遍历方法数组，按指定顺序生成对比
    for algo_name in method_order:
        if algo_name not in grouped_algorithms:
            print(f"警告：找不到方法 '{algo_name}' 的完整结果，跳过。")
            continue

        # --- 组间分隔符 (居中虚线) ---
        # 放置在 Origin/GT 组和第一个算法组之间，以及算法组之间
        combined_elements.append(center_separator)

        baseline_path, optimized_path = grouped_algorithms[algo_name]

        # --- Baseline ---
        baseline_result_path = os.path.join(baseline_path, "logits_vis", image_filename)
        baseline_mask_cropped_gray = load_and_crop(
            baseline_result_path, x_coord, y_coord, crop_size, is_mask=True
        )
        baseline_mask_colored = colorize_binary_mask(
            baseline_mask_cropped_gray, SEG_COLOR
        )
        baseline_overlay = apply_styled_overlay(
            origin_img_cropped,
            baseline_mask_colored,
            alpha=STYLE_ALPHA,
            contour_thickness=STYLE_CONTOUR_THICKNESS,
            contour_color=STYLE_CONTOUR_COLOR,
        )
        combined_elements.append(baseline_overlay)

        # --- 内间距 ---
        combined_elements.append(inner_spacer)

        # --- Optimized ---
        optimized_result_path = os.path.join(
            optimized_path, "logits_vis", image_filename
        )
        optimized_mask_cropped_gray = load_and_crop(
            optimized_result_path, x_coord, y_coord, crop_size, is_mask=True
        )
        optimized_mask_colored = colorize_binary_mask(
            optimized_mask_cropped_gray, SEG_COLOR
        )
        optimized_overlay = apply_styled_overlay(
            origin_img_cropped,
            optimized_mask_colored,
            alpha=STYLE_ALPHA,
            contour_thickness=STYLE_CONTOUR_THICKNESS,
            contour_color=STYLE_CONTOUR_COLOR,
        )
        combined_elements.append(optimized_overlay)

    final_mosaic = np.hstack(combined_elements)

    # 5. 保存结果
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"comparison_{image_name_stem}_x{x_coord}_y{y_coord}.png"
    save_path = os.path.join(output_dir, output_filename)
    cv2.imwrite(save_path, final_mosaic)

    print(f"\n成功创建 {final_mosaic.shape[1]}x{final_mosaic.shape[0]} 像素的对比图。")
    print(f"结果已保存至：{save_path}")


# ==============================================================================
# III. 运行配置 (请根据您的实际路径和参数修改)
# ==============================================================================

if __name__ == "__main__":

    # --- 用户配置区 ---

    # 包含所有 'deeplab-...' 和 'deeplab-w-pred_offsets' 等结果文件夹的根目录
    RESULTS_ROOT_DIR = "fig_results/teq/Antakya"

    # 原始图像和 GT 标签所在的文件夹 (请根据实际路径修改)
    ORIGIN_IMAGE_PATH = "../data/segmentation/Turkey/Antakya/pre/test/images"
    GT_LABEL_PATH = "../data/segmentation/Turkey/Antakya/pre/test/gt"

    # 待裁剪的图像名称 (不带后缀)
    TARGET_IMAGE_NAME_STEM = "00001025"

    # 裁剪窗口的左上角坐标 (X, Y)
    CROP_X = 748
    CROP_Y = 630

    # 裁剪窗口大小
    CROP_SIZE = 256  # 256x256

    METHOD_ORDER = [
        "deeplab",
        "unetformer",
        "segformer",
        "feedformer",
        "vwformer",
        "vmamba",
        "segman",
    ]

    # 图像间距
    INNER_SPACING_PIXELS = 10  # Baseline/Optimized 之间的间距
    PAIR_SPACING_PIXELS = 40  # 算法对之间的间距 (已增大)
    DASHED_LINE_THICKNESS = 3

    # 结果输出文件夹
    OUTPUT_DIR = "fig_results/comparison_mosaic"

    # --- 调用主函数 ---

    create_comparison_mosaic(
        results_root_dir=RESULTS_ROOT_DIR,
        image_name_stem=TARGET_IMAGE_NAME_STEM,
        x_coord=CROP_X,
        y_coord=CROP_Y,
        method_order=METHOD_ORDER,
        crop_size=CROP_SIZE,
        inner_spacing_px=INNER_SPACING_PIXELS,
        pair_spacing_px=PAIR_SPACING_PIXELS,
        dashed_line_thickness=DASHED_LINE_THICKNESS,
        origin_image_path=ORIGIN_IMAGE_PATH,
        gt_label_path=GT_LABEL_PATH,
        output_dir=OUTPUT_DIR,
    )
