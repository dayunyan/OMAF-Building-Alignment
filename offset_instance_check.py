import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import json
from datetime import datetime


class DetailedPTChecker:
    """
    详细的PT文件检查器，会打印不合法的bbox坐标及其相关信息
    """

    def __init__(self, min_bbox_area=1.0, max_bbox_aspect_ratio=20.0):
        """
        初始化检查器

        参数:
            min_bbox_area: 最小bbox面积阈值
            max_bbox_aspect_ratio: 最大bbox宽高比阈值
        """
        self.min_bbox_area = min_bbox_area
        self.max_bbox_aspect_ratio = max_bbox_aspect_ratio
        self.stats = {
            "total_files": 0,
            "total_instances": 0,
            "valid_files": 0,
            "invalid_files": 0,
            "invalid_bboxes": 0,
            "fixed_files": 0,
            "errors": [],
            "invalid_bbox_details": [],  # 存储不合法的bbox详细信息
        }

    def check_bbox_validity_detailed(
        self, bboxes, centroids=None, offsets=None, confidences=None, file_path=""
    ):
        """
        详细检查bboxes的合法性，并记录不合法的bbox信息

        参数:
            bboxes: [N, 4] 张量，格式为[x1, y1, x2, y2]
            centroids: [N, 2] 质心张量
            offsets: [N, 2] 偏移量张量
            confidences: [N, 1] 置信度张量
            file_path: 文件路径，用于记录

        返回:
            valid: 是否合法
            issues: 问题列表
            invalid_indices: 不合法的bbox索引列表
        """
        issues = []
        invalid_indices = []

        if bboxes is None:
            issues.append("bboxes为None")
            return False, issues, invalid_indices

        if not isinstance(bboxes, torch.Tensor):
            issues.append(f"bboxes不是torch.Tensor，而是{type(bboxes)}")
            return False, issues, invalid_indices

        if bboxes.dim() != 2 or bboxes.shape[1] != 4:
            issues.append(f"bboxes形状不正确: {bboxes.shape}，应为[N, 4]")
            return False, issues, invalid_indices

        # 检查坐标顺序
        x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

        # 检查x1 < x2, y1 < y2
        invalid_x = x1 >= x2
        invalid_y = y1 >= y2
        invalid_coords = invalid_x | invalid_y

        if invalid_coords.any():
            invalid_count = invalid_coords.sum().item()
            issues.append(f"有{invalid_count}个bbox坐标顺序不正确")

            # 记录不合法的bbox详细信息
            for i in range(len(bboxes)):
                if invalid_coords[i]:
                    bbox_info = {
                        "file_path": str(file_path),
                        "bbox_index": i,
                        "bbox_coords": bboxes[i].tolist(),
                        "invalid_reason": [],
                    }

                    if invalid_x[i]:
                        bbox_info["invalid_reason"].append(
                            f"x1({x1[i].item():.2f}) >= x2({x2[i].item():.2f})"
                        )
                    if invalid_y[i]:
                        bbox_info["invalid_reason"].append(
                            f"y1({y1[i].item():.2f}) >= y2({y2[i].item():.2f})"
                        )

                    # 添加关联信息（如果存在）
                    if centroids is not None and i < len(centroids):
                        bbox_info["centroid"] = centroids[i].tolist()

                    if offsets is not None and i < len(offsets):
                        bbox_info["offset"] = offsets[i].tolist()

                    if confidences is not None and i < len(confidences):
                        bbox_info["confidence"] = confidences[i].tolist()

                    self.stats["invalid_bbox_details"].append(bbox_info)
                    invalid_indices.append(i)

            self.stats["invalid_bboxes"] += invalid_count

        # 检查非负坐标
        negative_coords = (bboxes < 0).any(dim=1)
        if negative_coords.any():
            negative_count = negative_coords.sum().item()
            issues.append(f"有{negative_count}个bbox包含负坐标")

            for i in range(len(bboxes)):
                if negative_coords[i]:
                    bbox_info = {
                        "file_path": str(file_path),
                        "bbox_index": i,
                        "bbox_coords": bboxes[i].tolist(),
                        "invalid_reason": [f"包含负坐标: {bboxes[i].tolist()}"],
                    }

                    # 添加关联信息
                    if centroids is not None and i < len(centroids):
                        bbox_info["centroid"] = centroids[i].tolist()

                    if offsets is not None and i < len(offsets):
                        bbox_info["offset"] = offsets[i].tolist()

                    if confidences is not None and i < len(confidences):
                        bbox_info["confidence"] = confidences[i].tolist()

                    self.stats["invalid_bbox_details"].append(bbox_info)
                    if i not in invalid_indices:
                        invalid_indices.append(i)

        # 检查bbox面积
        areas = (x2 - x1) * (y2 - y1)
        small_bboxes = areas < self.min_bbox_area
        if small_bboxes.any():
            small_count = small_bboxes.sum().item()
            issues.append(f"有{small_count}个bbox面积小于{self.min_bbox_area}")

            for i in range(len(bboxes)):
                if small_bboxes[i]:
                    bbox_info = {
                        "file_path": str(file_path),
                        "bbox_index": i,
                        "bbox_coords": bboxes[i].tolist(),
                        "invalid_reason": [
                            f"面积过小: {areas[i].item():.2f} < {self.min_bbox_area}"
                        ],
                    }

                    # 添加关联信息
                    if centroids is not None and i < len(centroids):
                        bbox_info["centroid"] = centroids[i].tolist()

                    if offsets is not None and i < len(offsets):
                        bbox_info["offset"] = offsets[i].tolist()

                    if confidences is not None and i < len(confidences):
                        bbox_info["confidence"] = confidences[i].tolist()

                    self.stats["invalid_bbox_details"].append(bbox_info)
                    if i not in invalid_indices:
                        invalid_indices.append(i)

        # 检查宽高比
        widths = x2 - x1
        heights = y2 - y1
        aspect_ratios = torch.max(widths / (heights + 1e-8), heights / (widths + 1e-8))
        extreme_aspect = aspect_ratios > self.max_bbox_aspect_ratio
        if extreme_aspect.any():
            extreme_count = extreme_aspect.sum().item()
            issues.append(
                f"有{extreme_count}个bbox宽高比超过{self.max_bbox_aspect_ratio}"
            )

            for i in range(len(bboxes)):
                if extreme_aspect[i]:
                    bbox_info = {
                        "file_path": str(file_path),
                        "bbox_index": i,
                        "bbox_coords": bboxes[i].tolist(),
                        "invalid_reason": [
                            f"宽高比过大: {aspect_ratios[i].item():.2f} > {self.max_bbox_aspect_ratio}"
                        ],
                    }

                    # 添加关联信息
                    if centroids is not None and i < len(centroids):
                        bbox_info["centroid"] = centroids[i].tolist()

                    if offsets is not None and i < len(offsets):
                        bbox_info["offset"] = offsets[i].tolist()

                    if confidences is not None and i < len(confidences):
                        bbox_info["confidence"] = confidences[i].tolist()

                    self.stats["invalid_bbox_details"].append(bbox_info)
                    if i not in invalid_indices:
                        invalid_indices.append(i)

        return len(invalid_indices) == 0, issues, invalid_indices

    def check_tensor_consistency(self, data_dict, file_path=""):
        """
        检查张量之间的一致性

        参数:
            data_dict: 包含bboxes, centroids, gt_offsets, gt_confidences的字典
            file_path: 文件路径，用于记录

        返回:
            valid: 是否一致
            issues: 问题列表
        """
        issues = []
        required_keys = ["bboxes", "centroids", "gt_offsets", "gt_confidences"]

        # 检查必需键是否存在
        for key in required_keys:
            if key not in data_dict:
                issues.append(f"缺少必需键: {key}")
                return False, issues

        # 检查实例数量一致性
        n_instances = len(data_dict["bboxes"])
        self.stats["total_instances"] += n_instances

        for key in ["centroids", "gt_offsets", "gt_confidences"]:
            if len(data_dict[key]) != n_instances:
                issues.append(
                    f"{key}的实例数量({len(data_dict[key])})与bboxes({n_instances})不匹配"
                )
                return False, issues

        # 检查centroids是否在bboxes内
        bboxes = data_dict["bboxes"]
        centroids = data_dict["centroids"]

        if n_instances > 0:
            x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
            cx, cy = centroids[:, 0], centroids[:, 1]

            outside_centroids = (cx < x1) | (cx > x2) | (cy < y1) | (cy > y2)
            if outside_centroids.any():
                outside_count = outside_centroids.sum().item()
                issues.append(f"有{outside_count}个centroid不在对应的bbox内")

                # 记录不在bbox内的centroid
                for i in range(n_instances):
                    if outside_centroids[i]:
                        centroid_info = {
                            "file_path": str(file_path),
                            "instance_index": i,
                            "bbox_coords": bboxes[i].tolist(),
                            "centroid": centroids[i].tolist(),
                            "issue": "centroid不在bbox内",
                        }

                        # 添加其他信息
                        if "gt_offsets" in data_dict and i < len(
                            data_dict["gt_offsets"]
                        ):
                            centroid_info["offset"] = data_dict["gt_offsets"][
                                i
                            ].tolist()

                        if "gt_confidences" in data_dict and i < len(
                            data_dict["gt_confidences"]
                        ):
                            centroid_info["confidence"] = data_dict["gt_confidences"][
                                i
                            ].tolist()

                        self.stats["invalid_bbox_details"].append(centroid_info)

        return len(issues) == 0, issues

    def move_tensors_to_cpu(self, data_dict):
        """
        将所有张量移动到CPU

        参数:
            data_dict: 数据字典

        返回:
            data_dict_cpu: 所有张量在CPU上的字典
            moved: 是否进行了移动
        """
        data_dict_cpu = {}
        moved = False

        for key, value in data_dict.items():
            if isinstance(value, torch.Tensor) and value.device.type != "cpu":
                data_dict_cpu[key] = value.cpu()
                moved = True
            else:
                data_dict_cpu[key] = value

        return data_dict_cpu, moved

    def print_invalid_bbox_details(self, max_display=10):
        """
        打印不合法的bbox详细信息

        参数:
            max_display: 最大显示数量
        """
        if not self.stats["invalid_bbox_details"]:
            print("没有发现不合法的bbox")
            return

        print(f"\n{'='*80}")
        print(f"发现 {len(self.stats['invalid_bbox_details'])} 个不合法的bbox实例")
        print(f"{'='*80}")

        # 限制显示数量
        display_count = min(max_display, len(self.stats["invalid_bbox_details"]))

        for i, detail in enumerate(self.stats["invalid_bbox_details"][:display_count]):
            print(f"\n{i+1}. 文件: {detail['file_path']}")
            print(
                f"   实例索引: {detail.get('bbox_index', detail.get('instance_index', 'N/A'))}"
            )
            print(f"   bbox坐标: {detail['bbox_coords']}")
            print(
                f"   问题: {', '.join(detail.get('invalid_reason', [detail.get('issue', '未知问题')]))}"
            )

            if "centroid" in detail:
                print(f"   质心: {detail['centroid']}")

            if "offset" in detail:
                print(f"   偏移量: {detail['offset']}")

            if "confidence" in detail:
                print(f"   置信度: {detail['confidence']}")

        if len(self.stats["invalid_bbox_details"]) > display_count:
            print(
                f"\n... 还有 {len(self.stats['invalid_bbox_details']) - display_count} 个未显示"
            )

    def check_single_file(self, file_path, fix_issues=False):
        """
        检查单个PT文件

        参数:
            file_path: PT文件路径
            fix_issues: 是否尝试修复问题

        返回:
            result: 检查结果字典
        """
        result = {
            "file_path": str(file_path),
            "is_valid": True,
            "issues": [],
            "fixed": False,
            "file_size_mb": 0,
            "num_instances": 0,
            "invalid_bbox_indices": [],
        }

        try:
            # 获取文件大小
            result["file_size_mb"] = os.path.getsize(file_path) / (1024 * 1024)

            # 加载数据
            data = torch.load(file_path, map_location="cpu")

            if not isinstance(data, dict):
                result["issues"].append("文件内容不是字典格式")
                result["is_valid"] = False
                return result

            # 检查必需键
            required_keys = ["bboxes", "centroids", "gt_offsets", "gt_confidences"]
            for key in required_keys:
                if key not in data:
                    result["issues"].append(f"缺少必需键: {key}")
                    result["is_valid"] = False

            if not result["is_valid"]:
                return result

            # 记录实例数量
            result["num_instances"] = len(data["bboxes"])

            # 检查张量设备并移动到CPU
            data_cpu, moved = self.move_tensors_to_cpu(data)
            if moved and fix_issues:
                torch.save(data_cpu, file_path)
                result["fixed"] = True
                data = data_cpu

            # 详细检查bboxes合法性
            bboxes_valid, bbox_issues, invalid_indices = (
                self.check_bbox_validity_detailed(
                    data["bboxes"],
                    data.get("centroids"),
                    data.get("gt_offsets"),
                    data.get("gt_confidences"),
                    file_path,
                )
            )

            if not bboxes_valid:
                result["issues"].extend(bbox_issues)
                result["is_valid"] = False
                result["invalid_bbox_indices"] = invalid_indices

            # 检查张量一致性
            consistency_valid, consistency_issues = self.check_tensor_consistency(
                data, file_path
            )
            if not consistency_valid:
                result["issues"].extend(consistency_issues)
                result["is_valid"] = False

            # 检查张量数据类型
            for key in required_keys:
                if not isinstance(data[key], torch.Tensor):
                    result["issues"].append(f"{key}不是torch.Tensor")
                    result["is_valid"] = False

            # 检查NaN和Inf值
            for key in required_keys:
                if isinstance(data[key], torch.Tensor):
                    if torch.isnan(data[key]).any():
                        result["issues"].append(f"{key}包含NaN值")
                        result["is_valid"] = False
                    if torch.isinf(data[key]).any():
                        result["issues"].append(f"{key}包含Inf值")
                        result["is_valid"] = False

        except Exception as e:
            result["issues"].append(f"加载文件时出错: {str(e)}")
            result["is_valid"] = False

        return result

    def check_directory(self, root_dir, fix_issues=False, file_pattern="*.pt"):
        """
        检查目录中的所有PT文件

        参数:
            root_dir: 根目录路径
            fix_issues: 是否尝试修复问题
            file_pattern: 文件匹配模式

        返回:
            results: 所有文件的检查结果列表
        """
        root_path = Path(root_dir)
        pt_files = list(root_path.rglob(file_pattern))

        print(f"在目录 {root_dir} 中找到 {len(pt_files)} 个PT文件")

        results = []
        for file_path in tqdm(pt_files, desc="检查PT文件"):
            result = self.check_single_file(file_path, fix_issues)
            results.append(result)

            # 更新统计信息
            self.stats["total_files"] += 1
            if result["is_valid"]:
                self.stats["valid_files"] += 1
            else:
                self.stats["invalid_files"] += 1
                self.stats["errors"].append(
                    {"file": str(file_path), "issues": result["issues"]}
                )

            if result["fixed"]:
                self.stats["fixed_files"] += 1

        return results

    def generate_report(self, results, output_file=None, save_details=False):
        """
        生成检查报告

        参数:
            results: 检查结果列表
            output_file: 输出文件路径（可选）
            save_details: 是否保存详细错误信息到JSON文件
        """
        report = []
        report.append("=" * 80)
        report.append("PT文件完整性检查报告")
        report.append("=" * 80)
        report.append(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"总文件数: {self.stats['total_files']}")
        report.append(f"总实例数: {self.stats['total_instances']}")
        report.append(f"有效文件: {self.stats['valid_files']}")
        report.append(f"无效文件: {self.stats['invalid_files']}")
        report.append(f"不合法bbox数: {self.stats['invalid_bboxes']}")
        report.append(f"修复文件: {self.stats['fixed_files']}")
        report.append("")

        # 打印不合法的bbox详细信息
        self.print_invalid_bbox_details(max_display=20)

        if self.stats["invalid_files"] > 0:
            report.append("\n无效文件详情:")
            report.append("-" * 40)

            for error in self.stats["errors"]:
                report.append(f"文件: {error['file']}")
                for issue in error["issues"]:
                    report.append(f"  - {issue}")
                report.append("")

        # 文件大小统计
        file_sizes = [r["file_size_mb"] for r in results if r["file_size_mb"] > 0]
        if file_sizes:
            report.append("文件大小统计:")
            report.append(f"  最小: {min(file_sizes):.2f} MB")
            report.append(f"  最大: {max(file_sizes):.2f} MB")
            report.append(f"  平均: {np.mean(file_sizes):.2f} MB")
            report.append(f"  中位数: {np.median(file_sizes):.2f} MB")

        report_text = "\n".join(report)
        print(report_text)

        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(report_text)
            print(f"报告已保存到: {output_file}")

        # 保存详细错误信息到JSON文件
        if save_details and self.stats["invalid_bbox_details"]:
            details_file = (
                output_file.replace(".txt", "_details.json")
                if output_file
                else "invalid_bbox_details.json"
            )
            with open(details_file, "w", encoding="utf-8") as f:
                json.dump(
                    self.stats["invalid_bbox_details"], f, indent=2, ensure_ascii=False
                )
            print(f"详细错误信息已保存到: {details_file}")

        return report_text


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="详细检查PT文件完整性")
    parser.add_argument("root_dir", type=str, help="要检查的根目录")
    parser.add_argument(
        "--fix", action="store_true", help="尝试修复问题（如移动到CPU）"
    )
    parser.add_argument("--pattern", type=str, default="*.pt", help="文件匹配模式")
    parser.add_argument("--output", type=str, help="报告输出文件路径")
    parser.add_argument(
        "--save-details", action="store_true", help="保存详细错误信息到JSON文件"
    )
    parser.add_argument("--min_area", type=float, default=1.0, help="最小bbox面积阈值")
    parser.add_argument(
        "--max_aspect", type=float, default=20.0, help="最大bbox宽高比阈值"
    )
    parser.add_argument(
        "--max_display", type=int, default=20, help="最大显示不合法的bbox数量"
    )

    args = parser.parse_args()

    if not os.path.exists(args.root_dir):
        print(f"错误: 目录不存在: {args.root_dir}")
        return

    # 创建检查器
    checker = DetailedPTChecker(
        min_bbox_area=args.min_area, max_bbox_aspect_ratio=args.max_aspect
    )

    # 执行检查
    results = checker.check_directory(args.root_dir, args.fix, args.pattern)

    # 生成报告
    checker.generate_report(results, args.output, args.save_details)

    # 如果有无效文件，返回非零退出码
    if checker.stats["invalid_files"] > 0:
        exit(1)
    else:
        exit(0)


# 直接使用的函数
def check_pt_files_detailed(
    root_dir,
    fix_issues=False,
    pattern="*.pt",
    min_area=1.0,
    max_aspect=20.0,
    max_display=20,
):
    """
    便捷函数：详细检查PT文件完整性

    参数:
        root_dir: 根目录路径
        fix_issues: 是否尝试修复问题
        pattern: 文件匹配模式
        min_area: 最小bbox面积
        max_aspect: 最大宽高比
        max_display: 最大显示不合法的bbox数量

    返回:
        bool: 是否所有文件都有效
    """
    checker = DetailedPTChecker(min_area, max_aspect)
    results = checker.check_directory(root_dir, fix_issues, pattern)
    checker.generate_report(results, max_display=max_display)

    return checker.stats["invalid_files"] == 0


if __name__ == "__main__":
    # 使用示例
    if False:  # 设置为True来测试
        # 测试单个目录
        root_path = "../data/segmentation/Turkey/Islahiye/pre"
        check_pt_files_detailed(
            root_path, fix_issues=True, min_area=1.0, max_aspect=20.0, max_display=20
        )
    else:
        # 使用命令行参数
        main()
