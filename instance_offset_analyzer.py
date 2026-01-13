import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from typing import List, Tuple, Dict, Optional
import pandas as pd
from pathlib import Path


class InstanceOffsetAnalyzer:
    """
    分析建筑物实例偏移量的工具类
    """

    def __init__(
        self,
        gt_dir: str,
        offset_dir: str,
        output_dir: str = "./vis_logs/offset_analysis",
        window_size: int = 256,
        step_size: int = 128,
    ):
        """
        初始化分析器

        参数:
            gt_dir: 真实标注图像目录
            offset_dir: 偏移标注图像目录
            output_dir: 输出结果目录
            window_size: 滑动窗口大小
            step_size: 滑动窗口步长
        """
        self.gt_dir = Path(gt_dir)
        self.offset_dir = Path(offset_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 滑动窗口参数
        self.window_size = window_size
        self.step_size = step_size

        # 存储所有偏移量数据
        self.all_offsets = []  # 每个实例的偏移量 (dx, dy)
        self.image_offsets = []  # 每张图像的偏移统计
        self.instance_data = []  # 详细的实例数据

    def find_matching_image_pairs(self) -> List[Tuple[Path, Path]]:
        """
        查找匹配的图像对
        """
        gt_images = sorted(
            [
                f
                for f in self.gt_dir.iterdir()
                if f.suffix.lower() in [".png", ".jpg", ".tif"]
            ]
        )
        offset_images = sorted(
            [
                f
                for f in self.offset_dir.iterdir()
                if f.suffix.lower() in [".png", ".jpg", ".tif"]
            ]
        )

        # 通过文件名匹配（去除扩展名）
        gt_names = {f.stem: f for f in gt_images}
        offset_names = {f.stem: f for f in offset_images}

        common_names = set(gt_names.keys()) & set(offset_names.keys())

        pairs = []
        for name in common_names:
            pairs.append((gt_names[name], offset_names[name]))

        print(f"找到 {len(pairs)} 对匹配图像")
        return pairs

    def load_and_validate_images(
        self, gt_path: Path, offset_path: Path
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载并验证图像对
        """
        # 加载图像
        gt_img = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        offset_img = cv2.imread(str(offset_path), cv2.IMREAD_GRAYSCALE)

        if gt_img is None:
            raise ValueError(f"无法加载图像: {gt_path}")
        if offset_img is None:
            raise ValueError(f"无法加载图像: {offset_path}")

        # 确保图像尺寸一致
        if gt_img.shape != offset_img.shape:
            print(
                f"警告: 图像尺寸不匹配 {gt_img.shape} vs {offset_img.shape}, 调整偏移图像尺寸"
            )
            offset_img = cv2.resize(
                offset_img,
                (gt_img.shape[1], gt_img.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        # 二值化图像
        _, gt_binary = cv2.threshold(gt_img, 127, 255, cv2.THRESH_BINARY)
        _, offset_binary = cv2.threshold(offset_img, 127, 255, cv2.THRESH_BINARY)

        return gt_binary, offset_binary

    def extract_instances(
        self, binary_image: np.ndarray, min_area: int = 5
    ) -> List[Dict]:
        """
        从二值图像中提取实例信息
        """
        # 连通组件分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_image, connectivity=8
        )

        instances = []
        for i in range(1, num_labels):  # 跳过背景（标签0）
            area = stats[i, cv2.CC_STAT_AREA]

            if area < min_area:
                continue  # 跳过小面积实例

            # 创建实例掩码
            mask = (labels == i).astype(np.uint8) * 255

            # 计算质心
            centroid_x, centroid_y = centroids[i]

            # 计算边界框
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            instances.append(
                {
                    "label": i,
                    "area": area,
                    "centroid": (centroid_x, centroid_y),
                    "bbox": (x, y, w, h),
                    "mask": mask,
                }
            )

        return instances

    def calculate_instance_offset(
        self, gt_instance: Dict, offset_instance: Dict
    ) -> Tuple[float, float]:
        """
        计算单个实例的偏移量
        """
        gt_x, gt_y = gt_instance["centroid"]
        offset_x, offset_y = offset_instance["centroid"]

        dx = gt_x - offset_x  # x方向偏移
        dy = gt_y - offset_y  # y方向偏移

        return dx, dy

    def sliding_window_match(
        self,
        gt_instances: List[Dict],
        offset_instances: List[Dict],
        image_shape: Tuple[int, int],
        max_distance: float = 50.0,
    ) -> List[Tuple[Dict, Dict, float, float]]:
        """
        使用滑动窗口进行实例匹配
        将图像分割为多个窗口，在每个窗口内独立进行匹配
        """
        H, W = image_shape
        all_matches = []
        used_gt_indices = set()
        used_offset_indices = set()

        # 计算窗口数量
        num_windows_x = max(1, (W - self.window_size) // self.step_size + 1)
        num_windows_y = max(1, (H - self.window_size) // self.step_size + 1)

        print(
            f"  使用滑动窗口: {num_windows_x}x{num_windows_y} 个窗口, 窗口大小: {self.window_size}, 步长: {self.step_size}"
        )

        for i in range(num_windows_y):
            for j in range(num_windows_x):
                # 计算当前窗口位置
                y_start = i * self.step_size
                y_end = min(y_start + self.window_size, H)
                x_start = j * self.step_size
                x_end = min(x_start + self.window_size, W)

                # 筛选在当前窗口内的实例
                window_gt_instances = []
                window_offset_instances = []

                # 筛选GT实例
                for idx, inst in enumerate(gt_instances):
                    if idx in used_gt_indices:
                        continue
                    x, y = inst["centroid"]
                    if x_start <= x < x_end and y_start <= y < y_end:
                        window_gt_instances.append((idx, inst))

                # 筛选偏移实例
                for idx, inst in enumerate(offset_instances):
                    if idx in used_offset_indices:
                        continue
                    x, y = inst["centroid"]
                    if x_start <= x < x_end and y_start <= y < y_end:
                        window_offset_instances.append((idx, inst))

                # 如果窗口内没有实例，跳过
                if not window_gt_instances or not window_offset_instances:
                    continue

                # 在当前窗口内进行匹配
                window_matches = self.match_instances_in_window(
                    window_gt_instances, window_offset_instances, max_distance
                )

                # 记录匹配结果并标记已使用的实例
                for gt_idx, offset_idx, dx, dy in window_matches:
                    if (
                        gt_idx not in used_gt_indices
                        and offset_idx not in used_offset_indices
                    ):
                        all_matches.append(
                            (gt_instances[gt_idx], offset_instances[offset_idx], dx, dy)
                        )
                        # used_gt_indices.add(gt_idx)
                        # used_offset_indices.add(offset_idx)

        print(f"  滑动窗口匹配完成: 共匹配 {len(all_matches)} 个实例")
        return all_matches

    def match_instances_in_window(
        self,
        window_gt_instances: List[Tuple[int, Dict]],
        window_offset_instances: List[Tuple[int, Dict]],
        max_distance: float = 50.0,
    ) -> List[Tuple[int, int, float, float]]:
        """
        在单个窗口内匹配实例
        """
        matches = []
        used_offset_indices = set()

        for gt_idx, gt_inst in window_gt_instances:
            best_match_idx = -1
            best_distance = float("inf")
            best_iou = 0.0

            gt_centroid = np.array(gt_inst["centroid"])
            gt_mask = gt_inst["mask"]

            for i, (offset_idx, offset_inst) in enumerate(window_offset_instances):
                if offset_idx in used_offset_indices:
                    continue

                offset_centroid = np.array(offset_inst["centroid"])

                # 计算质心距离
                distance = np.linalg.norm(gt_centroid - offset_centroid)

                # 计算IoU（交并比）作为形状相似性度量
                intersection = np.logical_and(
                    gt_mask > 0, offset_inst["mask"] > 0
                ).sum()
                union = np.logical_or(gt_mask > 0, offset_inst["mask"] > 0).sum()
                iou = intersection / union if union > 0 else 0

                # 综合评分：距离越近、IoU越大越好
                if distance < max_distance and (
                    distance < best_distance or iou > best_iou
                ):
                    best_match_idx = i
                    best_distance = distance
                    best_iou = iou

            if best_match_idx != -1:
                offset_idx = window_offset_instances[best_match_idx][0]
                offset_inst = window_offset_instances[best_match_idx][1]
                dx, dy = self.calculate_instance_offset(gt_inst, offset_inst)
                matches.append((gt_idx, offset_idx, dx, dy))
                used_offset_indices.add(offset_idx)

        return matches

    def analyze_image_pair(self, gt_path: Path, offset_path: Path) -> Dict:
        """
        分析单对图像的实例偏移
        """
        # 加载图像
        gt_img, offset_img = self.load_and_validate_images(gt_path, offset_path)

        # 提取实例
        gt_instances = self.extract_instances(gt_img)
        offset_instances = self.extract_instances(offset_img)

        print(
            f"图像 {gt_path.stem}: GT实例数={len(gt_instances)}, 偏移实例数={len(offset_instances)}"
        )

        # 使用滑动窗口匹配实例
        matches = self.sliding_window_match(
            gt_instances, offset_instances, gt_img.shape
        )

        # 计算偏移统计
        if matches:
            dx_values = [match[2] for match in matches]
            dy_values = [match[3] for match in matches]
            distances = [np.sqrt(dx**2 + dy**2) for dx, dy in zip(dx_values, dy_values)]

            image_stats = {
                "image_name": gt_path.stem,
                "num_gt_instances": len(gt_instances),
                "num_offset_instances": len(offset_instances),
                "num_matched_instances": len(matches),
                "mean_dx": np.mean(dx_values),
                "mean_dy": np.mean(dy_values),
                "mean_distance": np.mean(distances),
                "std_dx": np.std(dx_values),
                "std_dy": np.std(dy_values),
                "max_distance": np.max(distances),
                "matches": matches,
            }
        else:
            image_stats = {
                "image_name": gt_path.stem,
                "num_gt_instances": len(gt_instances),
                "num_offset_instances": len(offset_instances),
                "num_matched_instances": 0,
                "mean_dx": 0,
                "mean_dy": 0,
                "mean_distance": 0,
                "std_dx": 0,
                "std_dy": 0,
                "max_distance": 0,
                "matches": [],
            }

        return image_stats

    def run_analysis(self) -> None:
        """
        运行完整的偏移分析
        """
        # 查找匹配的图像对
        image_pairs = self.find_matching_image_pairs()

        if not image_pairs:
            print("未找到匹配的图像对！")
            return

        # 分析每对图像
        total_matches = 0

        for gt_path, offset_path in image_pairs:
            print(f"分析图像对: {gt_path.stem}")

            image_stats = self.analyze_image_pair(gt_path, offset_path)
            self.image_offsets.append(image_stats)

            # 收集所有实例的偏移数据
            for gt_inst, offset_inst, dx, dy in image_stats["matches"]:
                distance = np.sqrt(dx**2 + dy**2)

                instance_info = {
                    "image_name": gt_path.stem,
                    "gt_area": gt_inst["area"],
                    "offset_area": offset_inst["area"],
                    "dx": dx,
                    "dy": dy,
                    "distance": distance,
                    "gt_centroid": gt_inst["centroid"],
                    "offset_centroid": offset_inst["centroid"],
                }
                self.instance_data.append(instance_info)
                self.all_offsets.append((dx, dy))

            total_matches += image_stats["num_matched_instances"]
            print(f"  匹配实例数: {image_stats['num_matched_instances']}")

        print(f"\n分析完成！总共匹配了 {total_matches} 个实例")

        # 保存结果
        self.save_results()

        # 绘制统计图
        self.plot_statistics()

    def save_results(self) -> None:
        """
        保存分析结果到CSV文件
        """
        # 保存图像级别的统计
        if self.image_offsets:
            # 创建一个不包含'matches'列的副本用于保存
            image_data_for_save = []
            for stats in self.image_offsets:
                stats_copy = stats.copy()
                if "matches" in stats_copy:
                    del stats_copy["matches"]
                image_data_for_save.append(stats_copy)

            image_df = pd.DataFrame(image_data_for_save)
            image_df.to_csv(
                self.output_dir / "image_offset_statistics.csv", index=False
            )
            print(
                f"图像统计已保存到: {self.output_dir / 'image_offset_statistics.csv'}"
            )

        # 保存实例级别的详细数据
        if self.instance_data:
            instance_df = pd.DataFrame(self.instance_data)
            instance_df.to_csv(
                self.output_dir / "instance_offset_details.csv", index=False
            )
            print(
                f"实例详情已保存到: {self.output_dir / 'instance_offset_details.csv'}"
            )

    def plot_statistics(self) -> None:
        """
        绘制偏移量统计直方图
        """
        if not self.all_offsets:
            print("没有偏移数据可绘制！")
            return

        dx_values = [offset[0] for offset in self.all_offsets]
        dy_values = [offset[1] for offset in self.all_offsets]
        distances = [np.sqrt(dx**2 + dy**2) for dx, dy in self.all_offsets]

        # 创建图形 - 使用英文标签避免字体问题
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            "Building Instance Offset Analysis (Sliding Window)",
            fontsize=16,
            fontweight="bold",
        )

        # 1. X方向偏移直方图
        axes[0, 0].hist(
            dx_values, bins=50, alpha=0.7, color="skyblue", edgecolor="black"
        )
        axes[0, 0].axvline(
            np.mean(dx_values),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {np.mean(dx_values):.2f}",
        )
        axes[0, 0].set_xlabel("X Offset (pixels)")
        axes[0, 0].set_ylabel("Number of Instances")
        axes[0, 0].set_title("X Offset Distribution")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Y方向偏移直方图
        axes[0, 1].hist(
            dy_values, bins=50, alpha=0.7, color="lightcoral", edgecolor="black"
        )
        axes[0, 1].axvline(
            np.mean(dy_values),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {np.mean(dy_values):.2f}",
        )
        axes[0, 1].set_xlabel("Y Offset (pixels)")
        axes[0, 1].set_ylabel("Number of Instances")
        axes[0, 1].set_title("Y Offset Distribution")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 总偏移距离直方图
        axes[0, 2].hist(
            distances, bins=50, alpha=0.7, color="lightgreen", edgecolor="black"
        )
        axes[0, 2].axvline(
            np.mean(distances),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {np.mean(distances):.2f}",
        )
        axes[0, 2].set_xlabel("Total Offset Distance (pixels)")
        axes[0, 2].set_ylabel("Number of Instances")
        axes[0, 2].set_title("Total Offset Distance Distribution")
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # 4. 偏移方向散点图
        axes[1, 0].scatter(dx_values, dy_values, alpha=0.6, s=10, color="blue")
        axes[1, 0].axhline(0, color="black", linewidth=0.5)
        axes[1, 0].axvline(0, color="black", linewidth=0.5)
        axes[1, 0].set_xlabel("X Offset (pixels)")
        axes[1, 0].set_ylabel("Y Offset (pixels)")
        axes[1, 0].set_title("Offset Direction Distribution")
        axes[1, 0].grid(True, alpha=0.3)

        # 5. 二维直方图（热力图）
        hb = axes[1, 1].hist2d(dx_values, dy_values, bins=50, cmap="viridis")
        axes[1, 1].axhline(0, color="white", linewidth=0.5)
        axes[1, 1].axvline(0, color="white", linewidth=0.5)
        axes[1, 1].set_xlabel("X Offset (pixels)")
        axes[1, 1].set_ylabel("Y Offset (pixels)")
        axes[1, 1].set_title("2D Offset Distribution Heatmap")
        plt.colorbar(hb[3], ax=axes[1, 1])

        # 6. 统计摘要
        axes[1, 2].axis("off")
        stats_text = f"""
Statistical Summary (Sliding Window):
Total Instances: {len(self.all_offsets)}
X Offset:
  Mean: {np.mean(dx_values):.2f} px
  Std: {np.std(dx_values):.2f} px
  Range: [{np.min(dx_values):.2f}, {np.max(dx_values):.2f}] px

Y Offset:
  Mean: {np.mean(dy_values):.2f} px
  Std: {np.std(dy_values):.2f} px
  Range: [{np.min(dy_values):.2f}, {np.max(dy_values):.2f}] px

Total Distance:
  Mean: {np.mean(distances):.2f} px
  Std: {np.std(distances):.2f} px
  Max Distance: {np.max(distances):.2f} px
        """
        axes[1, 2].text(
            0.1,
            0.9,
            stats_text,
            transform=axes[1, 2].transAxes,
            fontsize=10,
            verticalalignment="top",
            family="monospace",
        )

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "offset_statistics_sliding_window.png",
            dpi=300,
            bbox_inches="tight",
        )
        # plt.show()

        # 绘制偏移方向的玫瑰图
        self.plot_direction_rose(dx_values, dy_values)

    def plot_direction_rose(
        self, dx_values: List[float], dy_values: List[float]
    ) -> None:
        """
        绘制偏移方向的玫瑰图（极坐标）
        """
        # 计算偏移角度（弧度）
        angles = np.arctan2(dy_values, dx_values)  # -π 到 π
        angles_deg = np.degrees(angles)  # 转换为角度
        angles_deg[angles_deg < 0] += 360  # 转换为 0-360°

        # 计算偏移距离
        distances = [np.sqrt(dx**2 + dy**2) for dx, dy in zip(dx_values, dy_values)]

        # 创建玫瑰图
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(15, 6), subplot_kw=dict(projection="polar")
        )
        fig.suptitle(
            "Offset Direction Analysis (Sliding Window)", fontsize=16, fontweight="bold"
        )

        # 1. 方向频率玫瑰图
        n_bins = 36  # 每10度一个bin
        hist, bin_edges = np.histogram(angles_deg, bins=n_bins, range=(0, 360))
        theta = np.deg2rad(np.arange(0, 360, 360 / n_bins))

        ax1.bar(
            theta,
            hist,
            width=2 * np.pi / n_bins,
            alpha=0.7,
            color="skyblue",
            edgecolor="black",
        )
        ax1.set_title("Offset Direction Frequency", pad=20)
        ax1.set_theta_zero_location("N")  # 0度在顶部
        ax1.set_theta_direction(-1)  # 顺时针

        # 2. 方向-距离玫瑰图
        # 按方向分组计算平均距离
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        mean_distances = []
        for i in range(len(bin_edges) - 1):
            mask = (angles_deg >= bin_edges[i]) & (angles_deg < bin_edges[i + 1])
            if mask.any():
                mean_distances.append(np.mean(np.array(distances)[mask]))
            else:
                mean_distances.append(0)

        ax2.bar(
            theta,
            mean_distances,
            width=2 * np.pi / n_bins,
            alpha=0.7,
            color="lightcoral",
            edgecolor="black",
        )
        ax2.set_title("Mean Offset Distance by Direction", pad=20)
        ax2.set_theta_zero_location("N")
        ax2.set_theta_direction(-1)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "offset_direction_rose_sliding_window.png",
            dpi=300,
            bbox_inches="tight",
        )
        # plt.show()

    def get_summary_statistics(self) -> Dict:
        """
        获取汇总统计信息
        """
        if not self.all_offsets:
            return {}

        dx_values = [offset[0] for offset in self.all_offsets]
        dy_values = [offset[1] for offset in self.all_offsets]
        distances = [np.sqrt(dx**2 + dy**2) for dx, dy in self.all_offsets]

        summary = {
            "total_instances": len(self.all_offsets),
            "dx_mean": float(np.mean(dx_values)),
            "dx_std": float(np.std(dx_values)),
            "dx_min": float(np.min(dx_values)),
            "dx_max": float(np.max(dx_values)),
            "dy_mean": float(np.mean(dy_values)),
            "dy_std": float(np.std(dy_values)),
            "dy_min": float(np.min(dy_values)),
            "dy_max": float(np.max(dy_values)),
            "distance_mean": float(np.mean(distances)),
            "distance_std": float(np.std(distances)),
            "distance_max": float(np.max(distances)),
        }

        return summary


def main():
    """
    主函数：运行偏移量分析
    """
    # 设置图像目录路径
    gt_dir = "../data/segmentation/Turkey/Antakya/pre/test/gt"  # 真实标注图像目录
    offset_dir = (
        "../data/segmentation/Turkey/Antakya/pre/test/labels"  # 偏移标注图像目录
    )
    output_dir = "./vis_logs/Antakya/offset_analysis"  # 结果输出目录

    # 创建分析器并运行分析（使用滑动窗口）
    analyzer = InstanceOffsetAnalyzer(
        gt_dir, offset_dir, output_dir, window_size=256, step_size=128
    )
    analyzer.run_analysis()

    # 打印汇总统计
    summary = analyzer.get_summary_statistics()
    if summary:
        print("\n=== 偏移量统计汇总 (滑动窗口方法) ===")
        for key, value in summary.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
