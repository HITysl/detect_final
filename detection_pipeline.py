import cv2
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt

from point_detector import PointDetector
from point_processor import PointProcessor
from point_adjuster import PointAdjuster
from utils import Visualizer, preprocess_point_cloud, create_boxes, assign_box_sides, create_tasks



def filter_by_y_density_and_visualize(all_points, all_colors,
                                      bin_width=0.005,
                                      y_range=0.02,
                                      outlier_nb_neighbors=20,
                                      outlier_std_ratio=2.0):
    """
    基于 y 值主峰进行点云过滤，并进一步通过统计滤波剔除离群点。
    显示保留的点云（保持原始颜色）。
    """

    y_vals = all_points[:, 1]

    # 1. 构建 y 值直方图
    hist, bin_edges = np.histogram(y_vals, bins=np.arange(y_vals.min(), y_vals.max(), bin_width))
    max_bin_idx = np.argmax(hist)

    # 2. 找到主峰中心 y 值，设定保留范围
    y_center = (bin_edges[max_bin_idx] + bin_edges[max_bin_idx + 1]) / 2
    y_min = y_center - y_range / 2
    y_max = y_center + y_range / 2

    print(f"保留 Y 值在 [{y_min:.4f}, {y_max:.4f}] 的点云")

    # 3. 初步筛选 y 范围内的点
    mask = (y_vals >= y_min) & (y_vals <= y_max)
    filtered_points = all_points[mask]
    filtered_colors = all_colors[mask]

    # 4. 构造临时点云对象用于滤波
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_points)
    pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

    # 5. 执行统计滤波
    pcd_clean, ind = pcd.remove_statistical_outlier(
        nb_neighbors=outlier_nb_neighbors,
        std_ratio=outlier_std_ratio
    )

    print(f"统计滤波后点数: {len(ind)} / 初始点数: {len(filtered_points)}")

    # 6. 可视化（使用原始颜色）
    o3d.visualization.draw_geometries(
        [pcd_clean],
        window_name="Y 主峰 + 统计滤波后的点云（原始颜色）"
    )

    # 7. 返回清洗后的 NumPy 数据
    final_points = np.asarray(pcd_clean.points)
    final_colors = np.asarray(pcd_clean.colors)

    return final_points, final_colors

def process_detections(detector, processor, adjuster, visualizer, color_low, depth_low, color_high, depth_high):
    high_points, high_colors, low_points, low_colors = detector.process_images(
        color_low, depth_low, color_high, depth_high
    )
    # high_points, high_colors, low_points, low_colors = detector.process_images(
    #     'E:\Desktop\detect_final\Low_colour.png', 'E:\Desktop\detect_final\Low_depth.png', 'E:\Desktop\detect_final\High_colour.png', 'E:\Desktop\detect_final\High_depth.png'
    # )
    all_points = np.vstack((high_points, low_points))
    all_colors = np.vstack((high_colors, low_colors))
    #np.savez("version2_output.npz", all_points=all_points, all_colors=all_colors)

    all_points, all_colors = preprocess_point_cloud(all_points, all_colors)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)

    o3d.visualization.draw_geometries([pcd])

    all_points, all_colors = filter_by_y_density_and_visualize(all_points, all_colors,
                                                 bin_width=0.01,
                                                 y_range=0.25,
                                                 outlier_nb_neighbors=80,
                                                 outlier_std_ratio=2.0)

    all_detected_info, all_proj_img, all_display_img = detector._project_and_detect(all_points, all_colors, "All")

    cv2.imshow("All XZ Projection", all_proj_img)
    cv2.imshow("All Detection Results", all_display_img)

    all_points = np.array([box['center_3d'] for box in all_detected_info])
    indices = list(range(len(all_detected_info)))

    visualizer.visualize_points(all_points, len(all_detected_info))

    points_2d = all_points[:, [0, 2]]
    x_labels, z_labels = processor.cluster_points(points_2d)
    sorted_x, sorted_z = processor.sort_labels(points_2d, x_labels, z_labels)

    second_row = np.where(sorted_z == 1)[0]
    if not len(second_row):
        raise ValueError("No points in first row")
    ref_idx = np.where(sorted_x[second_row] == 2)[0]
    if not len(ref_idx):
        raise ValueError("No points in third column of first row")
    ref_point = points_2d[second_row[ref_idx[0]]]

    grid_points, origin_x, origin_z = adjuster.generate_grid_points(ref_point)
    adjusted_points = adjuster.adjust_points(all_points, points_2d, sorted_x, sorted_z, origin_x, origin_z)
    visualizer.plot_all_2d(points_2d, grid_points, adjusted_points, sorted_x, sorted_z)

    boxes = create_boxes(adjusted_points, all_detected_info, sorted_x, sorted_z, indices)
    assign_box_sides(boxes)

    # print("\nDetected Boxes:")
    # for box in sorted(boxes, key=lambda b: b.id):
    #     print(box)

    tasks = create_tasks(boxes)
    # print("\nTask Summary:")
    # print(tasks)
    return tasks