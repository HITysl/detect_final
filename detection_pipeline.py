import time
import cv2
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List, Dict
import os
from datetime import datetime

from config import GRID_PARAMS
from point_detector import PointDetector
from point_processor import PointProcessor
from point_adjuster import PointAdjuster
from utils import Visualizer, preprocess_point_cloud, create_boxes, assign_box_sides, create_tasks, \
    filter_by_y_density_and_visualize

SAVE_DIR = "images"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

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

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(all_points)
    # pcd.colors = o3d.utility.Vector3dVector(all_colors)
    #
    # vis = o3d.visualization.VisualizerWithEditing()
    # vis.create_window()
    # vis.add_geometry(pcd)
    # vis.run()
    # vis.destroy_window()

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

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

    # 投影和检测
    all_detected_info, all_proj_img, all_display_img = detector._project_and_detect(all_points, all_colors, "All")
    visualizer.display_image_non_blocking(all_proj_img, "All XZ Projection")
    visualizer.display_image_non_blocking(all_display_img, "All Detection Results")

    # 提取每个箱子 3D 中心
    all_points = np.array([box['center_3d'] for box in all_detected_info])


    indices = list(range(len(all_detected_info)))
    # 聚类和排序点
    points_2d = all_points[:, [0, 2]]
    x_labels, z_labels = processor.cluster_points(points_2d)
    sorted_x, sorted_z = processor.sort_labels(points_2d, x_labels, z_labels)

    visualizer.visualize_points(all_points,sorted_x=sorted_x, sorted_z=sorted_z) # 带行列数的显示点

    second_row = np.where(sorted_z == 1)[0]
    if not len(second_row):
        raise ValueError("第二行没有点")
    ref_idx = np.where(sorted_x[second_row] == 2)[0]
    if not len(ref_idx):
        raise ValueError("第二行的第三列没有点")
    ref_point = points_2d[second_row[ref_idx[0]]]

    grid_points, origin_x, origin_z = adjuster.generate_grid_points(ref_point)
    adjusted_points = adjuster.adjust_points(all_points, points_2d, sorted_x, sorted_z, origin_x, origin_z)
    visualizer.plot_all_2d(points_2d, grid_points, adjusted_points, sorted_x, sorted_z)

    # 创建并分配箱子
    boxes = create_boxes(adjusted_points, all_detected_info, sorted_x, sorted_z, indices)
    assign_box_sides(boxes)

    if boxes:
        max_row = max(box.row for box in boxes)
        print(f"Max row: {max_row}")
        if max_row > 1:
            box_map = {(box.row, box.col): box for box in boxes}
            for box in boxes:
                if box.row == max_row:  # 最下面一行
                    above_box = box_map.get((max_row - 1, box.col))
                    if above_box:
                        # 更新aGraspPoint_Top，使用上方箱子的z - GRID_PARAMS['z_spacing']*500
                        box.aGraspPoint_Top = np.array([
                            np.clip(box.aGraspPoint_Side[0], -400, 400),
                            np.clip(box.aGraspPoint_Side[1] + GRID_PARAMS['y_spacing'] * 500, None, 1510),
                            300
                        ])
    # for box in boxes:
    #     print(f"[Right] ID: {box.id}, Row: {box.row}, Col: {box.col}")
    #     print(
    #         f"        Top Grasp Point : x={box.aGraspPoint_Top[0]:.1f}, y={box.aGraspPoint_Top[1]:.1f}, z={box.aGraspPoint_Top[2]:.1f}")
    #     print(
    #         f"        Side Grasp Point: x={box.aGraspPoint_Side[0]:.1f}, y={box.aGraspPoint_Side[1]:.1f}, z={box.aGraspPoint_Side[2]:.1f}")
    #     print(f"        Width: {box.width_3d:.1f} mm, Height: {box.height_3d:.1f} mm")

    tasks = create_tasks(boxes)

    # 确保 OpenCV 窗口关闭
    cv2.destroyAllWindows()

    return tasks