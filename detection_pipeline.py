import logging
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
from utils import Visualizer, preprocess_point_cloud, create_boxes, assign_box_sides, create_tasks

SAVE_DIR = "images"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
logger = logging.getLogger(__name__)
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

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
    # 预处理点云

    all_points, all_colors = preprocess_point_cloud(all_points, all_colors)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

    # 可视化初始点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)
    visualizer.save_point_cloud_screenshot(pcd, "initial_point_cloud")
    visualizer.display_point_cloud_non_blocking(pcd, "Initial Point Cloud")

    # 投影和检测
    all_detected_info, all_proj_img, all_display_img, box_account = detector._project_and_detect(all_points, all_colors, "All")
    visualizer.display_image_non_blocking(all_proj_img, "All XZ Projection")
    visualizer.display_image_non_blocking(all_display_img, "All Detection Results")

    # 提取每个箱子 3D 中心
    all_points = np.array([box['center_3d'] for box in all_detected_info])

    indices = list(range(len(all_detected_info)))
    # 聚类和排序点

    z_lbl = processor.cluster_points(all_points)

    sorted_x, sorted_y, sorted_z = processor.sort_labels(all_points, z_lbl)

    # 4. 可视化（标注 layer, row, col）
    visualizer.visualize_points(all_points,
                                sorted_x=sorted_x,
                                sorted_y=sorted_y,
                                sorted_z=sorted_z)

    boxes = create_boxes(all_points, all_detected_info, sorted_x, sorted_z, indices)
    assign_box_sides(boxes)

# 最下一行顶吸做下clip防止越界
    if boxes:
        max_row = max(box.row for box in boxes)
        print(f"Max row: {max_row}")
        if max_row > 1:
            box_map = {(box.row, box.col): box for box in boxes}
            for box in boxes:
                if box.row == max_row:  # 最下面一行
                    above_box = box_map.get((max_row - 1, box.col))
                    if above_box:
                        box.aGraspPoint_Top = np.array([
                            np.clip(box.aGraspPoint_Side[0], -400, 400),
                            np.clip(box.aGraspPoint_Side[1] + GRID_PARAMS['y_spacing'] * 500, None, 1510),
                            300
                        ])

    tasks = create_tasks(boxes, box_account)


    # 确保 OpenCV 窗口关闭
    cv2.destroyAllWindows()

    return tasks