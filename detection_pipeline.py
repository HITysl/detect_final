import time
import cv2
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List, Dict
import os
from datetime import datetime

from point_detector import PointDetector
from point_processor import PointProcessor
from point_adjuster import PointAdjuster
from utils import Visualizer, preprocess_point_cloud, create_boxes, assign_box_sides, create_tasks

# 配置参数
@dataclass
class FilterConfig:
    bin_width: float = 0.01
    y_range: float = 0.25
    outlier_nb_neighbors: int = 80
    outlier_std_ratio: float = 2.0

# 保存路径
SAVE_DIR = "images"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def filter_by_y_density_and_visualize(
        points: np.ndarray,
        colors: np.ndarray,
        visualizer: Visualizer,
        config: FilterConfig = FilterConfig()
) -> Tuple[np.ndarray, np.ndarray]:
    """
    根据 Y 值密度峰值过滤点云，并使用统计过滤去除离群点。
    保存并以非阻塞方式可视化保留的点云（使用原始颜色）。

    参数：
        points：3D 点云 (N, 3)。
        colors：对应的颜色 (N, 3)。
        visualizer：Visualizer 实例，用于显示和保存。
        config：过滤参数配置。

    返回：
        过滤后的点和颜色（NumPy 数组）。
    """
    y_vals = points[:, 1]

    # 构建 Y 值直方图
    hist, bin_edges = np.histogram(y_vals, bins=np.arange(y_vals.min(), y_vals.max(), config.bin_width))
    max_bin_idx = np.argmax(hist)

    # 找到主峰中心并设置保留范围
    y_center = (bin_edges[max_bin_idx] + bin_edges[max_bin_idx + 1]) / 2
    y_min, y_max = y_center - config.y_range / 2, y_center + config.y_range / 2

    print(f"保留 Y 值在 [{y_min:.4f}, {y_max:.4f}] 内的点")

    # 初始 Y 范围过滤
    mask = (y_vals >= y_min) & (y_vals <= y_max)
    filtered_points = points[mask]
    filtered_colors = colors[mask]

    # 创建临时点云用于过滤
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_points)
    pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

    # 统计离群点移除
    pcd_clean, ind = pcd.remove_statistical_outlier(
        nb_neighbors=config.outlier_nb_neighbors,
        std_ratio=config.outlier_std_ratio
    )

    print(f"统计过滤后的点数：{len(ind)} / 初始点数：{len(filtered_points)}")

    # 保存点云截图并以非阻塞方式显示
    visualizer.save_point_cloud_screenshot(pcd_clean, "y_peak_filtered_point_cloud")
    visualizer.display_point_cloud_non_blocking(pcd_clean, "Y Peak + Statistically Filtered Point Cloud (Original Colors)")

    return np.asarray(pcd_clean.points), np.asarray(pcd_clean.colors)

def process_detections(
        detector: PointDetector,
        processor: PointProcessor,
        adjuster: PointAdjuster,
        visualizer: Visualizer,
        color_low: str,
        depth_low: str,
        color_high: str,
        depth_high: str
) -> List[Dict]:
    """
    处理点云检测并为检测到的箱子生成任务。
    保存中间可视化结果并确保非阻塞显示。

    参数：
        detector：PointDetector 实例。
        processor：PointProcessor 实例。
        adjuster：PointAdjuster 实例。
        visualizer：Visualizer 实例。
        color_low, depth_low, color_high, depth_high：输入图像路径。

    返回：
        检测到的箱子的任务列表。
    """
    # 处理图像以获取点云
    high_points, high_colors, low_points, low_colors = detector.process_images(
        color_low, depth_low, color_high, depth_high
    )

    # 合并高低点云
    all_points = np.vstack((high_points, low_points))
    all_colors = np.vstack((high_colors, low_colors))

    # 预处理点云
    all_points, all_colors = preprocess_point_cloud(all_points, all_colors)

    # 可视化初始点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)
    visualizer.save_point_cloud_screenshot(pcd, "initial_point_cloud")
    visualizer.display_point_cloud_non_blocking(pcd, "Initial Point Cloud")

    # 过滤点云
    all_points, all_colors = filter_by_y_density_and_visualize(all_points, all_colors, visualizer)

    # 投影和检测
    all_detected_info, all_proj_img, all_display_img = detector._project_and_detect(all_points, all_colors, "All")
    #visualizer.display_image_non_blocking(all_proj_img, "All XZ Projection")
    visualizer.display_image_non_blocking(all_display_img, "All Detection Results")

    # 提取 3D 中心
    all_points = np.array([box['center_3d'] for box in all_detected_info])
    indices = list(range(len(all_detected_info)))

    # 可视化点
    visualizer.visualize_points(all_points, len(all_detected_info))

    # 聚类和排序点
    points_2d = all_points[:, [0, 2]]
    x_labels, z_labels = processor.cluster_points(points_2d)
    sorted_x, sorted_z = processor.sort_labels(points_2d, x_labels, z_labels)

    second_row = np.where(sorted_z == 1)[0]
    if not len(second_row):
        raise ValueError("第二行没有点")
    ref_idx = np.where(sorted_x[second_row] == 2)[0]
    if not len(ref_idx):
        raise ValueError("第二行的第三列没有点")
    ref_point = points_2d[second_row[ref_idx[0]]]
    # 调整点
    grid_points, origin_x, origin_z = adjuster.generate_grid_points(ref_point)
    adjusted_points = adjuster.adjust_points(all_points, points_2d, sorted_x, sorted_z, origin_x, origin_z)
    visualizer.plot_all_2d(points_2d, grid_points, adjusted_points, sorted_x, sorted_z)

    # 创建并分配箱子
    boxes = create_boxes(adjusted_points, all_detected_info, sorted_x, sorted_z, indices)
    assign_box_sides(boxes)

    # 生成任务
    tasks = create_tasks(boxes)

    # 确保 OpenCV 窗口关闭
    cv2.destroyAllWindows()

    return tasks