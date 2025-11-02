import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import curve_fit
import seaborn as sns
from pathlib import Path


def gaussian_1d(x, mu, sigma, amplitude):
    """一维高斯函数"""
    return amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def gaussian_2d(x, y, mu_x, mu_y, sigma_x, sigma_y, amplitude, rho=0):
    """二维高斯函数（考虑相关性）"""
    z = (
        ((x - mu_x) / sigma_x) ** 2
        - 2 * rho * (x - mu_x) * (y - mu_y) / (sigma_x * sigma_y)
        + ((y - mu_y) / sigma_y) ** 2
    )
    return amplitude * np.exp(-z / (2 * (1 - rho**2)))


def fit_gaussian_1d(data, bins=50):
    """拟合一维高斯分布"""
    # 计算直方图
    hist, bin_edges = np.histogram(data, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 初始参数估计
    mu_guess = np.mean(data)
    sigma_guess = np.std(data)
    amplitude_guess = np.max(hist)

    # 拟合高斯分布
    try:
        popt, pcov = curve_fit(
            gaussian_1d, bin_centers, hist, p0=[mu_guess, sigma_guess, amplitude_guess]
        )
        mu, sigma, amplitude = popt
        return mu, sigma, amplitude, bin_centers, hist
    except:
        # 如果拟合失败，使用矩估计
        mu = np.mean(data)
        sigma = np.std(data)
        amplitude = 1.0 / (sigma * np.sqrt(2 * np.pi))
        return mu, sigma, amplitude, bin_centers, hist


def fit_gaussian_2d(x_data, y_data):
    """拟合二维高斯分布"""
    # 计算均值和协方差矩阵
    data = np.vstack([x_data, y_data])
    mu = np.mean(data, axis=1)
    cov = np.cov(data)

    # 提取参数
    mu_x, mu_y = mu
    sigma_x = np.sqrt(cov[0, 0])
    sigma_y = np.sqrt(cov[1, 1])
    rho = cov[0, 1] / (sigma_x * sigma_y) if sigma_x * sigma_y > 0 else 0

    return mu_x, mu_y, sigma_x, sigma_y, rho, cov


def kl_divergence_1d(mu1, sigma1, mu2, sigma2):
    """计算两个一维高斯分布之间的KL散度"""
    # KL(P||Q) = log(σ2/σ1) + (σ1² + (μ1-μ2)²)/(2σ2²) - 0.5
    term1 = np.log(sigma2 / sigma1)
    term2 = (sigma1**2 + (mu1 - mu2) ** 2) / (2 * sigma2**2)
    kl = term1 + term2 - 0.5
    return kl


def kl_divergence_2d(mu1, cov1, mu2, cov2):
    """计算两个二维高斯分布之间的KL散度"""
    # KL(P||Q) = 0.5 * [log(|Σ2|/|Σ1|) + tr(Σ2^{-1}Σ1) + (μ2-μ1)^T Σ2^{-1} (μ2-μ1) - k]
    # 其中k是维度（这里为2）

    # 计算协方差矩阵的行列式
    det1 = np.linalg.det(cov1)
    det2 = np.linalg.det(cov2)

    # 计算Σ2的逆
    cov2_inv = np.linalg.inv(cov2)

    # 计算迹
    trace_term = np.trace(np.dot(cov2_inv, cov1))

    # 计算均值差项
    mu_diff = mu2 - mu1
    mu_term = np.dot(mu_diff.T, np.dot(cov2_inv, mu_diff))

    # 计算KL散度
    kl = 0.5 * (np.log(det2 / det1) + trace_term + mu_term - 2)
    return kl


def analyze_offset_distributions(file1_path, file2_path, output_dir="./kl_analysis"):
    """
    分析两个CSV文件中的偏移分布并计算KL散度

    参数:
        file1_path: 第一个CSV文件路径（单位：像素）
        file2_path: 第二个CSV文件路径（单位：需要乘以1024转换为像素）
        output_dir: 输出目录
    """
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 读取CSV文件
    print("读取CSV文件...")
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)

    # 对第二个文件的偏移量乘以1024进行单位转换
    df2["dx"] = df2["dx"] * 1024
    df2["dy"] = df2["dy"] * 1024
    df2["distance"] = np.sqrt(df2["dx"] ** 2 + df2["dy"] ** 2)

    print(f"文件1: {len(df1)} 个实例")
    print(f"文件2: {len(df2)} 个实例")

    # 提取偏移量数据
    dx1 = df1["dx"].values
    dy1 = df1["dy"].values
    distance1 = df1["distance"].values

    dx2 = df2["dx"].values
    dy2 = df2["dy"].values
    distance2 = df2["distance"].values

    # 拟合一维高斯分布（dx, dy, distance）
    print("\n拟合一维高斯分布...")

    # 文件1的一维拟合
    mu_dx1, sigma_dx1, amp_dx1, bin_centers_dx1, hist_dx1 = fit_gaussian_1d(dx1)
    mu_dy1, sigma_dy1, amp_dy1, bin_centers_dy1, hist_dy1 = fit_gaussian_1d(dy1)
    mu_dist1, sigma_dist1, amp_dist1, bin_centers_dist1, hist_dist1 = fit_gaussian_1d(
        distance1
    )

    # 文件2的一维拟合
    mu_dx2, sigma_dx2, amp_dx2, bin_centers_dx2, hist_dx2 = fit_gaussian_1d(dx2)
    mu_dy2, sigma_dy2, amp_dy2, bin_centers_dy2, hist_dy2 = fit_gaussian_1d(dy2)
    mu_dist2, sigma_dist2, amp_dist2, bin_centers_dist2, hist_dist2 = fit_gaussian_1d(
        distance2
    )

    # 拟合二维高斯分布（dx, dy）
    print("拟合二维高斯分布...")
    mu_x1, mu_y1, sigma_x1, sigma_y1, rho1, cov1 = fit_gaussian_2d(dx1, dy1)
    mu_x2, mu_y2, sigma_x2, sigma_y2, rho2, cov2 = fit_gaussian_2d(dx2, dy2)

    # 计算KL散度
    print("计算KL散度...")

    # 一维KL散度
    kl_dx = kl_divergence_1d(mu_dx1, sigma_dx1, mu_dx2, sigma_dx2)
    kl_dy = kl_divergence_1d(mu_dy1, sigma_dy1, mu_dy2, sigma_dy2)
    kl_dist = kl_divergence_1d(mu_dist1, sigma_dist1, mu_dist2, sigma_dist2)

    # 二维KL散度
    kl_2d = kl_divergence_2d(
        np.array([mu_x1, mu_y1]), cov1, np.array([mu_x2, mu_y2]), cov2
    )

    # 打印结果
    print("\n=== 高斯分布拟合结果 ===")
    print(f"文件1 - dx: μ={mu_dx1:.4f}, σ={sigma_dx1:.4f}")
    print(f"文件1 - dy: μ={mu_dy1:.4f}, σ={sigma_dy1:.4f}")
    print(f"文件1 - 距离: μ={mu_dist1:.4f}, σ={sigma_dist1:.4f}")
    print(
        f"文件1 - 二维: μ=({mu_x1:.4f}, {mu_y1:.4f}), σ=({sigma_x1:.4f}, {sigma_y1:.4f}), ρ={rho1:.4f}"
    )

    print(f"\n文件2 - dx: μ={mu_dx2:.4f}, σ={sigma_dx2:.4f}")
    print(f"文件2 - dy: μ={mu_dy2:.4f}, σ={sigma_dy2:.4f}")
    print(f"文件2 - 距离: μ={mu_dist2:.4f}, σ={sigma_dist2:.4f}")
    print(
        f"文件2 - 二维: μ=({mu_x2:.4f}, {mu_y2:.4f}), σ=({sigma_x2:.4f}, {sigma_y2:.4f}), ρ={rho2:.4f}"
    )

    print("\n=== KL散度结果 ===")
    print(f"dx方向的KL散度: {kl_dx:.6f}")
    print(f"dy方向的KL散度: {kl_dy:.6f}")
    print(f"总距离的KL散度: {kl_dist:.6f}")
    print(f"二维分布的KL散度: {kl_2d:.6f}")

    # 保存结果到文件
    results = {
        "file1_samples": len(df1),
        "file2_samples": len(df2),
        "file1_dx_mu": mu_dx1,
        "file1_dx_sigma": sigma_dx1,
        "file1_dy_mu": mu_dy1,
        "file1_dy_sigma": sigma_dy1,
        "file1_dist_mu": mu_dist1,
        "file1_dist_sigma": sigma_dist1,
        "file1_2d_mu_x": mu_x1,
        "file1_2d_mu_y": mu_y1,
        "file1_2d_sigma_x": sigma_x1,
        "file1_2d_sigma_y": sigma_y1,
        "file1_2d_rho": rho1,
        "file1_2d_cov": cov1.flatten().tolist(),
        "file2_dx_mu": mu_dx2,
        "file2_dx_sigma": sigma_dx2,
        "file2_dy_mu": mu_dy2,
        "file2_dy_sigma": sigma_dy2,
        "file2_dist_mu": mu_dist2,
        "file2_dist_sigma": sigma_dist2,
        "file2_2d_mu_x": mu_x2,
        "file2_2d_mu_y": mu_y2,
        "file2_2d_sigma_x": sigma_x2,
        "file2_2d_sigma_y": sigma_y2,
        "file2_2d_rho": rho2,
        "file2_2d_cov": cov2.flatten().tolist(),
        "kl_dx": kl_dx,
        "kl_dy": kl_dy,
        "kl_dist": kl_dist,
        "kl_2d": kl_2d,
    }

    results_df = pd.DataFrame([results])
    results_df.to_csv(output_path / "kl_analysis_results.csv", index=False)
    print(f"\n结果已保存到: {output_path / 'kl_analysis_results.csv'}")

    # 绘制分布图
    plot_distributions(
        dx1,
        dy1,
        distance1,
        dx2,
        dy2,
        distance2,
        mu_dx1,
        sigma_dx1,
        mu_dy1,
        sigma_dy1,
        mu_dist1,
        sigma_dist1,
        mu_dx2,
        sigma_dx2,
        mu_dy2,
        sigma_dy2,
        mu_dist2,
        sigma_dist2,
        kl_dx,
        kl_dy,
        kl_dist,
        kl_2d,
        output_path,
    )

    # 单独绘制二维分布等高线图
    plot_2d_contours(
        dx1,
        dy1,
        dx2,
        dy2,
        mu_x1,
        mu_y1,
        sigma_x1,
        sigma_y1,
        rho1,
        mu_x2,
        mu_y2,
        sigma_x2,
        sigma_y2,
        rho2,
        output_path,
    )

    return results


def plot_distributions(
    dx1,
    dy1,
    dist1,
    dx2,
    dy2,
    dist2,
    mu_dx1,
    sigma_dx1,
    mu_dy1,
    sigma_dy1,
    mu_dist1,
    sigma_dist1,
    mu_dx2,
    sigma_dx2,
    mu_dy2,
    sigma_dy2,
    mu_dist2,
    sigma_dist2,
    kl_dx,
    kl_dy,
    kl_dist,
    kl_2d,
    output_path,
):
    """绘制分布对比图"""
    # 设置中文字体（如果系统支持）
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # 创建图形
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        "Offset Distribution Comparison and KL Divergence Analysis",
        fontsize=16,
        fontweight="bold",
    )

    # 1. dx分布对比
    x_range = np.linspace(
        min(min(dx1), min(dx2)) - 5, max(max(dx1), max(dx2)) + 5, 1000
    )
    pdf1_dx = stats.norm.pdf(x_range, mu_dx1, sigma_dx1)
    pdf2_dx = stats.norm.pdf(x_range, mu_dx2, sigma_dx2)

    axes[0, 0].hist(
        dx1,
        bins=50,
        alpha=0.7,
        density=True,
        color="skyblue",
        label="File 1",
        edgecolor="black",
    )
    axes[0, 0].hist(
        dx2,
        bins=50,
        alpha=0.7,
        density=True,
        color="lightcoral",
        label="File 2",
        edgecolor="black",
    )
    axes[0, 0].plot(
        x_range,
        pdf1_dx,
        "b-",
        linewidth=2,
        label=f"File 1 Fit (μ={mu_dx1:.2f}, σ={sigma_dx1:.2f})",
    )
    axes[0, 0].plot(
        x_range,
        pdf2_dx,
        "r-",
        linewidth=2,
        label=f"File 2 Fit (μ={mu_dx2:.2f}, σ={sigma_dx2:.2f})",
    )
    axes[0, 0].set_xlabel("X Offset (pixels)")
    axes[0, 0].set_ylabel("Probability Density")
    axes[0, 0].set_title("X Offset Distribution Comparison")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. dy分布对比
    y_range = np.linspace(
        min(min(dy1), min(dy2)) - 5, max(max(dy1), max(dy2)) + 5, 1000
    )
    pdf1_dy = stats.norm.pdf(y_range, mu_dy1, sigma_dy1)
    pdf2_dy = stats.norm.pdf(y_range, mu_dy2, sigma_dy2)

    axes[0, 1].hist(
        dy1,
        bins=50,
        alpha=0.7,
        density=True,
        color="skyblue",
        label="File 1",
        edgecolor="black",
    )
    axes[0, 1].hist(
        dy2,
        bins=50,
        alpha=0.7,
        density=True,
        color="lightcoral",
        label="File 2",
        edgecolor="black",
    )
    axes[0, 1].plot(
        y_range,
        pdf1_dy,
        "b-",
        linewidth=2,
        label=f"File 1 Fit (μ={mu_dy1:.2f}, σ={sigma_dy1:.2f})",
    )
    axes[0, 1].plot(
        y_range,
        pdf2_dy,
        "r-",
        linewidth=2,
        label=f"File 2 Fit (μ={mu_dy2:.2f}, σ={sigma_dy2:.2f})",
    )
    axes[0, 1].set_xlabel("Y Offset (pixels)")
    axes[0, 1].set_ylabel("Probability Density")
    axes[0, 1].set_title("Y Offset Distribution Comparison")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 距离分布对比
    dist_range = np.linspace(0, max(max(dist1), max(dist2)) + 5, 1000)
    pdf1_dist = stats.norm.pdf(dist_range, mu_dist1, sigma_dist1)
    pdf2_dist = stats.norm.pdf(dist_range, mu_dist2, sigma_dist2)

    axes[0, 2].hist(
        dist1,
        bins=50,
        alpha=0.7,
        density=True,
        color="skyblue",
        label="File 1",
        edgecolor="black",
    )
    axes[0, 2].hist(
        dist2,
        bins=50,
        alpha=0.7,
        density=True,
        color="lightcoral",
        label="File 2",
        edgecolor="black",
    )
    axes[0, 2].plot(
        dist_range,
        pdf1_dist,
        "b-",
        linewidth=2,
        label=f"File 1 Fit (μ={mu_dist1:.2f}, σ={sigma_dist1:.2f})",
    )
    axes[0, 2].plot(
        dist_range,
        pdf2_dist,
        "r-",
        linewidth=2,
        label=f"File 2 Fit (μ={mu_dist2:.2f}, σ={sigma_dist2:.2f})",
    )
    axes[0, 2].set_xlabel("Total Offset Distance (pixels)")
    axes[0, 2].set_ylabel("Probability Density")
    axes[0, 2].set_title("Total Offset Distance Distribution Comparison")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # 4. 二维散点图
    axes[1, 0].scatter(dx1, dy1, alpha=0.5, s=10, color="blue", label="File 1")
    axes[1, 0].scatter(dx2, dy2, alpha=0.5, s=10, color="red", label="File 2")
    axes[1, 0].axhline(0, color="black", linewidth=0.5)
    axes[1, 0].axvline(0, color="black", linewidth=0.5)
    axes[1, 0].set_xlabel("X Offset (pixels)")
    axes[1, 0].set_ylabel("Y Offset (pixels)")
    axes[1, 0].set_title("2D Offset Distribution")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 5. 二维核密度估计
    # 使用seaborn绘制核密度估计图
    data1 = pd.DataFrame({"dx": dx1, "dy": dy1, "source": "File 1"})
    data2 = pd.DataFrame({"dx": dx2, "dy": dy2, "source": "File 2"})
    data_combined = pd.concat([data1, data2])

    sns.kdeplot(
        data=data_combined,
        x="dx",
        y="dy",
        hue="source",
        ax=axes[1, 1],
        alpha=0.7,
        fill=True,
        thresh=0.05,
    )
    axes[1, 1].set_xlabel("X Offset (pixels)")
    axes[1, 1].set_ylabel("Y Offset (pixels)")
    axes[1, 1].set_title("2D Kernel Density Estimation")
    axes[1, 1].grid(True, alpha=0.3)

    # 6. KL散度结果
    axes[1, 2].axis("off")
    kl_text = f"""
KL Divergence Results:
X Offset KL: {kl_dx:.6f}
Y Offset KL: {kl_dy:.6f}
Distance KL: {kl_dist:.6f}
2D Distribution KL: {kl_2d:.6f}

Interpretation:
- KL divergence measures how different two distributions are.
- Smaller values indicate more similar distributions.
- Values close to 0 suggest nearly identical distributions.
    """
    axes[1, 2].text(
        0.1,
        0.9,
        kl_text,
        transform=axes[1, 2].transAxes,
        fontsize=12,
        verticalalignment="top",
        family="monospace",
    )

    plt.tight_layout()
    plt.savefig(
        output_path / "distribution_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.show()


def plot_2d_contours(
    dx1,
    dy1,
    dx2,
    dy2,
    mu_x1,
    mu_y1,
    sigma_x1,
    sigma_y1,
    rho1,
    mu_x2,
    mu_y2,
    sigma_x2,
    sigma_y2,
    rho2,
    output_path,
):
    """绘制二维高斯分布的等高线图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 创建网格
    x_min = min(min(dx1), min(dx2)) - 10
    x_max = max(max(dx1), max(dx2)) + 10
    y_min = min(min(dy1), min(dy2)) - 10
    y_max = max(max(dy1), max(dy2)) + 10

    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x, y)

    # 计算文件1的二维高斯分布
    Z1 = np.zeros_like(X)
    for i in range(len(x)):
        for j in range(len(y)):
            z = (
                ((X[j, i] - mu_x1) / sigma_x1) ** 2
                - 2
                * rho1
                * (X[j, i] - mu_x1)
                * (Y[j, i] - mu_y1)
                / (sigma_x1 * sigma_y1)
                + ((Y[j, i] - mu_y1) / sigma_y1) ** 2
            )
            Z1[j, i] = np.exp(-z / (2 * (1 - rho1**2)))

    # 计算文件2的二维高斯分布
    Z2 = np.zeros_like(X)
    for i in range(len(x)):
        for j in range(len(y)):
            z = (
                ((X[j, i] - mu_x2) / sigma_x2) ** 2
                - 2
                * rho2
                * (X[j, i] - mu_x2)
                * (Y[j, i] - mu_y2)
                / (sigma_x2 * sigma_y2)
                + ((Y[j, i] - mu_y2) / sigma_y2) ** 2
            )
            Z2[j, i] = np.exp(-z / (2 * (1 - rho2**2)))

    # 绘制文件1的等高线
    contour1 = ax1.contour(X, Y, Z1, levels=5, colors="blue")
    ax1.scatter(dx1, dy1, alpha=0.3, s=5, color="blue")
    ax1.set_xlabel("X Offset (pixels)")
    ax1.set_ylabel("Y Offset (pixels)")
    ax1.set_title("File 1: 2D Gaussian Fit")
    ax1.grid(True, alpha=0.3)

    # 绘制文件2的等高线
    contour2 = ax2.contour(X, Y, Z2, levels=5, colors="red")
    ax2.scatter(dx2, dy2, alpha=0.3, s=5, color="red")
    ax2.set_xlabel("X Offset (pixels)")
    ax2.set_ylabel("Y Offset (pixels)")
    ax2.set_title("File 2: 2D Gaussian Fit")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "2d_gaussian_contours.png", dpi=300, bbox_inches="tight")
    plt.show()


def main():
    """
    主函数：运行KL散度分析
    """
    # 设置文件路径
    file1_path = (
        "./vis_logs/offset_analysis/instance_offset_details.csv"  # 第一个CSV文件
    )
    file2_path = (
        "./vis_logs/offset_analysis/instance_offset_emi_details.csv"  # 第二个CSV文件
    )
    output_dir = "./vis_logs/kl_analysis"  # 输出目录

    # 检查文件是否存在
    if not Path(file1_path).exists():
        print(f"错误: 文件 {file1_path} 不存在！")
        return

    if not Path(file2_path).exists():
        print(f"错误: 文件 {file2_path} 不存在！")
        return

    # 运行分析
    results = analyze_offset_distributions(file1_path, file2_path, output_dir)

    print("\n分析完成！")


if __name__ == "__main__":
    main()
