import numpy as np
import matplotlib.pyplot as plt

def load_pointcloud(npz_path):
    data = np.load(npz_path)
    points = data["all_points"]
    colors = data["all_colors"]
    return points, colors

def describe_pointcloud(name, points):
    center = np.mean(points, axis=0)
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    extent = max_bound - min_bound

    print(f"\n🔹 {name} 点云统计")
    print(f"总点数: {points.shape[0]}")
    print(f"中心坐标: {center}")
    print(f"X 范围: [{min_bound[0]:.3f}, {max_bound[0]:.3f}]，跨度: {extent[0]:.3f}")
    print(f"Y 范围: [{min_bound[1]:.3f}, {max_bound[1]:.3f}]，跨度: {extent[1]:.3f}")
    print(f"Z 范围: [{min_bound[2]:.3f}, {max_bound[2]:.3f}]，跨度: {extent[2]:.3f}")

def compare_pointclouds(p1, p2):
    print("\n🔸 点云比较")
    print(f"shape: {p1.shape} vs {p2.shape}")
    if p1.shape != p2.shape:
        print("❗ 点云维度不一致，无法直接比较")
        return

    is_equal = np.array_equal(p1, p2)
    is_close = np.allclose(p1, p2, atol=1e-6)
    diff = np.abs(p1 - p2)
    max_diff = np.max(diff)
    num_diff = np.count_nonzero(diff)

    print(f"是否完全相等: {is_equal}")
    print(f"是否近似相等 (误差 < 1e-6): {is_close}")
    print(f"最大差值: {max_diff:.6f}")
    print(f"有差异的值数量: {num_diff}")

def visualize_xy(p1, p2, title1="版本1", title2="版本2"):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(p1[:, 0], p1[:, 1], s=1, c='blue')
    plt.title(f"{title1} XY 投影")
    plt.axis("equal")

    plt.subplot(1, 2, 2)
    plt.scatter(p2[:, 0], p2[:, 1], s=1, c='red')
    plt.title(f"{title2} XY 投影")
    plt.axis("equal")

    plt.suptitle("点云 XY 分布对比")
    plt.tight_layout()
    plt.show()

# ============================== 主流程 ==============================

# 替换为你的文件路径
path1 = "version1_output.npz"
path2 = "version2_output.npz"

points1, colors1 = load_pointcloud(path1)
points2, colors2 = load_pointcloud(path2)

describe_pointcloud("版本1", points1)
describe_pointcloud("版本2", points2)

compare_pointclouds(points1, points2)

visualize_xy(points1, points2)
