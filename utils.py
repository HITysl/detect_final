import json
import logging
import os
import socket
import time
import threading
from dataclasses import dataclass
from typing import Tuple, List, Any, Dict, Union
import open3d as o3d
import cv2
import numpy as np
import pyads
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.cluster import KMeans
from class_define import Box, Tasks
from config import GRID_PARAMS
from datetime import datetime

@dataclass
class FilterConfig:
    bin_width: float = 0.01
    y_range: float = 0.25
    outlier_nb_neighbors: int = 80
    outlier_std_ratio: float = 2.0

logging.basicConfig(
    filename='plc_camera_log.txt',
    level=logging.INFO,
    format='%(asctime)s: %(levelname)s: %(message)s'
)

# Point Cloud Processing
def preprocess_point_cloud(points, colors, voxel_size=0.005, nb_neighbors=20, std_ratio=0.5, min_cluster_size=1000):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 体素下采样
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)

    if nb_neighbors > 0 and std_ratio > 0:
        filtered_pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        pcd = filtered_pcd

    if min_cluster_size > 0:
        labels = np.array(pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=False))
        if len(labels) > 0 and np.any(labels >= 0):
            largest_label = np.argmax(np.bincount(labels[labels >= 0]))  # 找到最大簇
            mask = labels == largest_label  # 只保留最大簇
            pcd = pcd.select_by_index(np.where(mask)[0])  # 直接过滤

    return np.asarray(pcd.points), np.asarray(pcd.colors)

def transmit_to_plc(tasks):
    # ========== 打印任务结构 ==========
    print("========== Task Summary ==========")
    print(tasks)  # 触发 Tasks.__str__()

    # ========== 打印每个箱子详细信息 ==========
    print("========== Left Boxes ==========")
    for box in tasks.aLeftBoxArray:
        print(f"[Left] ID: {box.id}, Row: {box.row}, Col: {box.col}")
        print(
            f"       Top Grasp Point : x={box.aGraspPoint_Top[0]:.1f}, y={box.aGraspPoint_Top[1]:.1f}, z={box.aGraspPoint_Top[2]:.1f}")
        print(
            f"       Side Grasp Point: x={box.aGraspPoint_Side[0]:.1f}, y={box.aGraspPoint_Side[1]:.1f}, z={box.aGraspPoint_Side[2]:.1f}")
        print(f"       Width: {box.width_3d:.1f} mm, Height: {box.height_3d:.1f} mm")

    print("========== Right Boxes ==========")
    for box in tasks.aRightBoxArray:
        print(f"[Right] ID: {box.id}, Row: {box.row}, Col: {box.col}")
        print(
            f"        Top Grasp Point : x={box.aGraspPoint_Top[0]:.1f}, y={box.aGraspPoint_Top[1]:.1f}, z={box.aGraspPoint_Top[2]:.1f}")
        print(
            f"        Side Grasp Point: x={box.aGraspPoint_Side[0]:.1f}, y={box.aGraspPoint_Side[1]:.1f}, z={box.aGraspPoint_Side[2]:.1f}")
        print(f"        Width: {box.width_3d:.1f} mm, Height: {box.height_3d:.1f} mm")

    logging.info("========== Task Summary ==========")
    logging.info(str(tasks))  # 触发 Tasks.__str__()

    logging.info("========== Left Boxes ==========")
    for box in tasks.aLeftBoxArray:
        logging.info(f"[Left] ID: {box.id}, Row: {box.row}, Col: {box.col}")
        logging.info(
            f"       Top Grasp Point : x={box.aGraspPoint_Top[0]:.1f}, y={box.aGraspPoint_Top[1]:.1f}, z={box.aGraspPoint_Top[2]:.1f}")
        logging.info(
            f"       Side Grasp Point: x={box.aGraspPoint_Side[0]:.1f}, y={box.aGraspPoint_Side[1]:.1f}, z={box.aGraspPoint_Side[2]:.1f}")
        logging.info(f"       Width: {box.width_3d:.1f} mm, Height: {box.height_3d:.1f} mm")

    logging.info("========== Right Boxes ==========")
    for box in tasks.aRightBoxArray:
        logging.info(f"[Right] ID: {box.id}, Row: {box.row}, Col: {box.col}")
        logging.info(
            f"        Top Grasp Point : x={box.aGraspPoint_Top[0]:.1f}, y={box.aGraspPoint_Top[1]:.1f}, z={box.aGraspPoint_Top[2]:.1f}")
        logging.info(
            f"        Side Grasp Point: x={box.aGraspPoint_Side[0]:.1f}, y={box.aGraspPoint_Side[1]:.1f}, z={box.aGraspPoint_Side[2]:.1f}")
        logging.info(f"        Width: {box.width_3d:.1f} mm, Height: {box.height_3d:.1f} mm")

    nbox_l = int(GRID_PARAMS['x_spacing'] * 1000)
    nbox_w = int(GRID_PARAMS['y_spacing'] * 1000)
    nbox_h = int(GRID_PARAMS['z_spacing'] * 1000)

    nLeftBoxCount = int(tasks.nLeftBoxCount)
    nRightBoxCount = int(tasks.nRightBoxCount)
    nTotalRow = int(tasks.nTotalRow)
    nTotalCol = int(tasks.nTotalCol)

    leftArm_Data = []
    rightArm_Data = []

    for box in tasks.aLeftBoxArray:
        flat_data = [
            int(box.id),
            int(box.row),
            int(box.col),
            int(box.aGraspPoint_Top[0]),
            int(box.aGraspPoint_Top[1]),
            int(box.aGraspPoint_Top[2]),
            int(box.aGraspPoint_Side[0]),
            int(box.aGraspPoint_Side[1]),
            int(box.aGraspPoint_Side[2])
        ]
        leftArm_Data.extend(flat_data)
    print("Left Box IDs:", [box.id for box in tasks.aLeftBoxArray])

    for box in tasks.aRightBoxArray:
        flat_data = [
            int(box.id),
            int(box.row),
            int(box.col),
            int(box.aGraspPoint_Top[0]),
            int(box.aGraspPoint_Top[1]),
            int(box.aGraspPoint_Top[2]),
            int(box.aGraspPoint_Side[0]),
            int(box.aGraspPoint_Side[1]),
            int(box.aGraspPoint_Side[2])
        ]
        rightArm_Data.extend(flat_data)
    print("Right Box IDs:", [box.id for box in tasks.aRightBoxArray])

    aHeightEachRow = [int(h) if h is not None else 0 for h in tasks.aHeightEachRow]

    max_retry_time = 60
    retry_interval = 0
    start_time = time.time()
    while True:
        try:
            plc = pyads.Connection("192.168.1.20.1.1", 851)
            plc.open()
            plc.write_by_name('Camera.nbox_l', nbox_l, pyads.PLCTYPE_UINT)
            plc.write_by_name('Camera.nbox_w', nbox_w, pyads.PLCTYPE_UINT)
            plc.write_by_name('Camera.nbox_h', nbox_h, pyads.PLCTYPE_UINT)
            plc.write_by_name('Camera.nLeftBoxCount', nLeftBoxCount, pyads.PLCTYPE_UINT)
            plc.write_by_name('Camera.nRightBoxCount', nRightBoxCount, pyads.PLCTYPE_UINT)
            plc.write_by_name('Camera.nTotalRow', nTotalRow, pyads.PLCTYPE_UINT)
            plc.write_by_name('Camera.nTotalCol', nTotalCol, pyads.PLCTYPE_UINT)
            plc.write_by_name('Camera.aHeightEachRow', aHeightEachRow, pyads.PLCTYPE_ARR_INT(nTotalRow))
            plc.write_by_name('Camera.aLeftBoxArrayFlat', leftArm_Data, pyads.PLCTYPE_ARR_INT(nLeftBoxCount * 9))
            plc.write_by_name('Camera.aRightBoxArrayFlat', rightArm_Data, pyads.PLCTYPE_ARR_INT(nRightBoxCount * 9))
            plc.write_by_name('Camera.bInspection_IPC_Done', True, pyads.PLCTYPE_BOOL)
            print("Data successfully written to PLC")
            logging.info("Data successfully written to PLC")
            logging.info("Camera.bInspection_IPC_Done set to True")
            break
        except pyads.ADSError as e:
            elapsed_time = time.time() - start_time
            print(f"PLC write error, retrying in {retry_interval} seconds: {e}")
            if elapsed_time > max_retry_time:
                print(f"PLC write failed after {max_retry_time} seconds.")
                break
            time.sleep(retry_interval)
        except Exception as e:
            print(f"Unexpected error: {e}")
            break
        finally:
            try:
                plc.close()
                print("PLC connection closed")
            except Exception as e:
                print(f"Error closing PLC connection: {e}")

class Visualizer:
    def __init__(self, images_dir="images", display_duration=5.0, enable_display=True):
        """
        初始化 Visualizer，设置默认图形大小、图像保存目录、显示时长和显示标志位。

        参数：
            images_dir (str)：保存图像的目录，默认为 'images'。
            display_duration (float)：非阻塞显示的持续时间（秒）。
            enable_display (bool)：是否启用可视化显示，默认为 True。
        """
        self.figsize = (10, 8)
        self.images_dir = images_dir
        self.display_duration = display_duration
        self.enable_display = enable_display
        os.makedirs(self.images_dir, exist_ok=True)
        if self.enable_display:
            plt.ion()  # 仅在启用显示时开启 Matplotlib 交互模式

    def _get_timestamped_filename(self, prefix):
        """
        生成带时间戳的文件名。

        参数：
            prefix (str)：文件名前缀（例如 'visualize_points'）。

        返回：
            str：带时间戳的完整文件路径（例如 'images/visualize_points_20250416_123456.png'）。
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.images_dir, f"{prefix}_{timestamp}.png")

    def _set_equal_aspect(self, ax, points):
        """
        为 3D 轴设置等比例显示，根据点云范围调整。

        参数：
            ax：Matplotlib 3D 轴对象。
            points (np.ndarray)：形状为 (N, 3) 的 3D 点。

        抛出：
            ValueError：如果 points 不是有效的 numpy 数组。
        """
        if not isinstance(points, np.ndarray) or points.shape[1] != 3:
            raise ValueError("Points must be a numpy array of shape (N, 3)")
        ax.set_box_aspect([1, 1, 1])
        max_range = max(points[:, i].ptp() for i in range(3)) / 2
        mid = [points[:, i].mean() for i in range(3)]
        for i, lim in enumerate(['xlim', 'ylim', 'zlim']):
            getattr(ax, f'set_{lim}')(mid[i] - max_range, mid[i] + max_range)

    def save_image(self, image: np.ndarray, img_type: str):
        """
        保存带时间戳的图像。

        参数：
            image：图像（NumPy 数组）。
            img_type：图像类型，用于文件名前缀。
        """
        filename = self._get_timestamped_filename(img_type)
        try:
            cv2.imwrite(filename, image)
            print(f"保存图像：{filename}")
        except Exception as e:
            print(f"保存图像到 {filename} 时出错：{e}")

    def save_point_cloud_screenshot(self, pcd: o3d.geometry.PointCloud, pcd_type: str):
        """
        保存点云从 X-O-Z 平面（沿 -Y 方向的前视图）的截图。

        参数：
            pcd：Open3D PointCloud 对象。
            pcd_type：点云类型，用于文件名前缀。
        """
        filename = self._get_timestamped_filename(pcd_type)
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=pcd_type, visible=False)
        vis.add_geometry(pcd)

        # 计算点云中心作为注视点
        points = np.asarray(pcd.points)
        center = np.mean(points, axis=0) if points.size else [0, 0, 0]

        # 设置相机从 Y 负方向看过去，Z 朝上（即 X-O-Z 平面）
        view_control = vis.get_view_control()
        view_control.set_front([0, -1, 0])
        view_control.set_up([0, 0, 1])
        view_control.set_lookat(center)
        view_control.set_zoom(0.7)

        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        try:
            vis.capture_screen_image(filename, do_render=True)
            print(f"保存 X-O-Z 前视图截图：{filename}")
        except Exception as e:
            print(f"保存点云截图到 {filename} 时出错：{e}")
        finally:
            vis.destroy_window()

    def display_point_cloud_non_blocking(self, pcd: o3d.geometry.PointCloud, window_name: str = "PointCloud"):
        """
        使用后台线程以非阻塞方式显示点云，仅当 enable_display 为 True 时执行。

        参数：
            pcd：Open3D PointCloud 对象。
            window_name：显示窗口的名称。
        """
        if not self.enable_display:
            return

        def visualize():
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name=window_name)
            vis.add_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
            start_time = time.time()
            while time.time() - start_time < self.display_duration:
                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.03)
            vis.destroy_window()

        thread = threading.Thread(target=visualize)
        thread.daemon = True
        thread.start()

    def display_image_non_blocking(self, image: np.ndarray, window_name: str):
        """
        以非阻塞方式显示图像并保存，仅当 enable_display 为 True 时显示。

        参数：
            image：图像（NumPy 数组）。
            window_name：显示窗口的名称。
        """
        self.save_image(image, window_name.lower().replace(" ", "_"))
        if not self.enable_display:
            return

        def show_image():
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(window_name, image)
            cv2.waitKey(int(self.display_duration * 1000))
            cv2.destroyAllWindows()

        thread = threading.Thread(target=show_image)
        thread.daemon = True
        thread.start()

    def visualize_points(self, points, sorted_x=None, sorted_z=None):
        """
        可视化 3D 点，并在每个点旁标注其 (row, col)，非阻塞显示并保存带时间戳的图像。

        参数：
            points (np.ndarray): 形状为 (N, 3) 的 3D 点。
            sorted_x (np.ndarray): 每个点对应的列标签。
            sorted_z (np.ndarray): 每个点对应的行标签。
        """
        if not isinstance(points, np.ndarray) or points.shape[1] != 3:
            raise ValueError("Points must be a numpy array of shape (N, 3)")
        if len(points) == 0:
            print("警告：没有点可可视化")
            return
        if (sorted_x is not None and len(sorted_x) != len(points)) or \
                (sorted_z is not None and len(sorted_z) != len(points)):
            raise ValueError("sorted_x and sorted_z must be the same length as points if provided")

        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', label='Points')

        # 添加文本标签 (row, col)
        if sorted_x is not None and sorted_z is not None:
            for i, (x, y, z) in enumerate(points):
                row = sorted_z[i]
                col = sorted_x[i]
                if row != -1 and col != -1:
                    ax.text(x, y, z, f"({row},{col})", fontsize=8, color='black')

        self._set_equal_aspect(ax, points)
        ax.set(xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)', title='3D Points Visualization with Labels')
        ax.legend()

        save_path = self._get_timestamped_filename("visualize_points")
        try:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"绘图保存到 {save_path}")
        except Exception as e:
            print(f"保存绘图到 {save_path} 时出错：{e}")

        if self.enable_display:
            plt.draw()
            plt.pause(self.display_duration)
        plt.close(fig)

    def visualize_comparison(self, original, filtered):
        """
        比较原始和过滤后的 3D 点云，显示并保存带时间戳的图像，仅当 enable_display 为 True 时显示。

        参数：
            original (np.ndarray)：原始 3D 点，形状为 (N, 3)。
            filtered (np.ndarray)：过滤后的 3D 点，形状为 (M, 3)。

        抛出：
            ValueError：如果输入不是有效的 numpy 数组。
        """
        if not isinstance(original, np.ndarray) or not isinstance(filtered, np.ndarray):
            raise ValueError("Original and filtered points must be numpy arrays")
        if original.shape[1] != 3 or filtered.shape[1] != 3:
            raise ValueError("Points must have shape (N, 3)")
        if not (len(original) and len(filtered)):
            print("警告：没有点可可视化")
            return

        fig = plt.figure(figsize=(12, 6))
        for i, (data, title, color) in enumerate([(original, 'Original Points', 'gray'),
                                                  (filtered, 'Filtered Points', 'blue')]):
            ax = fig.add_subplot(121 + i, projection='3d')
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=color, s=20 + 30 * i, alpha=0.5 - 0.3 * i)
            ax.set_title(title)
            self._set_equal_aspect(ax, data)
            ax.set(xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)')
            ax.legend([title])
        plt.tight_layout()

        # 保存绘图
        save_path = self._get_timestamped_filename("visualize_comparison")
        try:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"绘图保存到 {save_path}")
        except Exception as e:
            print(f"保存绘图到 {save_path} 时出错：{e}")

        # 非阻塞显示，仅当 enable_display 为 True 时执行
        if self.enable_display:
            plt.draw()
            plt.pause(self.display_duration)
        plt.close(fig)

    def plot_all_2d(self, points_2d, grid_points, adjusted_3d, x_labels, z_labels):
        """
        绘制原始、网格和调整后的点的 2D 比较图，非阻塞显示并保存带时间戳的图像，仅当 enable_display 为 True 时显示。

        参数：
            points_2d (np.ndarray)：形状为 (N, 2) 的 2D 点。
            grid_points (np.ndarray)：形状为 (M, 2) 的网格点。
            adjusted_3d (np.ndarray)：调整后的 3D 点，形状为 (K, 3)。
            x_labels (list)：X 坐标标签。
            z_labels (list)：Z 坐标标签。

        抛出：
            ValueError：如果输入无效或不一致。
        """
        if not all(isinstance(arr, np.ndarray) for arr in [points_2d, grid_points, adjusted_3d]):
            raise ValueError("All point arrays must be numpy arrays")
        if points_2d.shape[1] != 2 or grid_points.shape[1] != 2 or adjusted_3d.shape[1] != 3:
            raise ValueError("Invalid point array shapes")
        if not (len(points_2d) and len(grid_points) and len(adjusted_3d)):
            print("警告：没有点可可视化")
            return
        if len(x_labels) != len(points_2d) or len(z_labels) != len(points_2d):
            raise ValueError("Label lengths must match points_2d length")

        plt.figure(figsize=(12, 9))
        plt.scatter(points_2d[:, 0], points_2d[:, 1], c='blue', s=30, edgecolors='k', label='Original')
        plt.scatter(grid_points[:, 0], grid_points[:, 1], c='black', marker='x', s=50, label='Grid')
        plt.scatter(adjusted_3d[:, 0], adjusted_3d[:, 2], c='red', s=30, edgecolors='k', marker='s', label='Adjusted')
        for i, (x, z) in enumerate(points_2d):
            plt.text(x, z, f'x:{x_labels[i]},z:{z_labels[i]}', fontsize=8, ha='right', color='blue')
        for i, (x, z) in enumerate(adjusted_3d[:, [0, 2]]):
            plt.text(x, z, f'x:{x_labels[i]},z:{z_labels[i]}', fontsize=8, ha='left', color='red')
        for x in np.unique(grid_points[:, 0]):
            plt.axvline(x=x, color='grey', linestyle='--', alpha=0.5)
        for z in np.unique(grid_points[:, 1]):
            plt.axhline(y=z, color='grey', linestyle='--', alpha=0.5)
        plt.xlabel('X (m)')
        plt.ylabel('Z (m)')
        plt.title('Original vs Adjusted vs Grid Points (2D)')
        plt.legend()
        plt.grid(True)

        # 保存绘图
        save_path = self._get_timestamped_filename("plot_all_2d")
        try:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"绘图保存到 {save_path}")
        except Exception as e:
            print(f"保存绘图到 {save_path} 时出错：{e}")

        # 非阻塞显示，仅当 enable_display 为 True 时执行
        if self.enable_display:
            plt.draw()
            plt.pause(self.display_duration)
        plt.close()

    def plot_all_3d(self, original, adjusted):
        """
        绘制原始和调整后的点的 3D 比较图，非阻塞显示并保存带时间戳的图像，仅当 enable_display 为 True 时显示。

        参数：
            original (np.ndarray)：原始 3D 点，形状为 (N, 3)。
            adjusted (np.ndarray)：调整后的 3D 点，形状为 (M, 3)。

        抛出：
            ValueError：如果输入不是有效的 numpy 数组。
        """
        if not isinstance(original, np.ndarray) or not isinstance(adjusted, np.ndarray):
            raise ValueError("Original and adjusted points must be numpy arrays")
        if original.shape[1] != 3 or adjusted.shape[1] != 3:
            raise ValueError("Points must have shape (N, 3)")
        if not (len(original) and len(adjusted)):
            print("警告：没有点可可视化")
            return

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(original[:, 0], original[:, 1], original[:, 2], c='blue', label='Original')
        ax.scatter(adjusted[:, 0], adjusted[:, 1], adjusted[:, 2], c='red', label='Adjusted')
        self._set_equal_aspect(ax, np.vstack((original, adjusted)))
        ax.set(xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)', title='Original vs Adjusted 3D')
        ax.legend()

        # 保存绘图
        save_path = self._get_timestamped_filename("plot_all_3d")
        try:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"绘图保存到 {save_path}")
        except Exception as e:
            print(f"保存绘图到 {save_path} 时出错：{e}")

        # 非阻塞显示，仅当 enable_display 为 True 时执行
        if self.enable_display:
            plt.draw()
            plt.pause(self.display_duration)
        plt.close(fig)

# Box Utilities
def merge_boxes_by_row_col(box_list):
    box_groups = defaultdict(list)
    for box in box_list:
        key = (box.row, box.col)
        box_groups[key].append(box)
    for key, group in box_groups.items():
        if len(group) > 1:
            avg_coords = np.mean([box.aGraspPoint_Side for box in group], axis=0)
            for box in group:
                box.aGraspPoint_Side = avg_coords
                box.aGraspPoint_Top = np.array([
                    avg_coords[0],
                    avg_coords[1],
                    avg_coords[2] + int(GRID_PARAMS['z_spacing']//2)
                ])

def assign_box_sides(boxes):
    if len(boxes) >= 2:
        x_coords = np.array([box.aGraspPoint_Side[0] for box in boxes]).reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, random_state=0, n_init='auto').fit(x_coords)
        labels = kmeans.labels_
        cluster_centers = kmeans.cluster_centers_
        left_cluster = 0 if cluster_centers[0] < cluster_centers[1] else 1
        for i, box in enumerate(boxes):
            box.side = 'left' if labels[i] == left_cluster else 'right'
            if box.side == 'right':
                box.aGraspPoint_Side[0] -= 1200
                box.aGraspPoint_Top[0] -= 1200
    else:
        print("Warning: Not enough boxes to perform K-means clustering.")

def create_tasks(boxes):
    sorted_boxes = sorted(boxes, key=lambda b: b.id)
    merge_boxes_by_row_col(sorted_boxes)
    total_rows = max(box.row for box in boxes) if boxes else 0
    total_cols = max(box.col for box in boxes) if boxes else 0
    return Tasks(sorted_boxes, total_rows, total_cols)

def create_boxes(adjusted_points, all_detected_info, sorted_x, sorted_z, indices):
    boxes = []
    for i in range(len(adjusted_points)):
        if sorted_x[i] != -1 and sorted_z[i] != -1:
            row = sorted_z[i] + 1
            col = sorted_x[i] + 1
            id = (row - 1) * GRID_PARAMS['col_count'] + col
            coords_3d = adjusted_points[i] * 1000
            orig_idx = indices[i]
            width_3d = all_detected_info[orig_idx]['width_3d'] * 1000
            height_3d = all_detected_info[orig_idx]['height_3d'] * 1000
            box = Box(id, row, col, coords_3d, width_3d, height_3d)
            boxes.append(box)
    # if boxes:
    #     max_row = max(box.row for box in boxes)
    #     print(f"Max row: {max_row}")
    #     if max_row > 1:
    #         box_map = {(box.row, box.col): box for box in boxes}
    #         for box in boxes:
    #             if box.row == max_row:  # 最下面一行
    #                 above_box = box_map.get((max_row - 1, box.col))
    #                 if above_box:
    #                     # 更新aGraspPoint_Top，使用上方箱子的z - GRID_PARAMS['z_spacing']*500
    #                     box.aGraspPoint_Top = np.array([
    #                         box.aGraspPoint_Side[0],
    #                         box.aGraspPoint_Side[1] + GRID_PARAMS['y_spacing'] * 500,
    #                         300
    #                     ])
    #                     # 调试打印更新后的箱子
    #                     print(
    #                         f"Updated Box id={box.id}, row={box.row}, col={box.col}, Top z={box.aGraspPoint_Top[2]}, Above z={above_box.aGraspPoint_Side[2]}")
    return boxes


def filter_by_y_density_and_visualize(
        points: np.ndarray,
        colors: np.ndarray,
        visualizer: Visualizer,
        known_y_min: float = 1.2255,
        known_y_max: float = 1.4755,
        max_y_threshold: float = 1.6,
        config: FilterConfig = FilterConfig()
) -> Tuple[np.ndarray, np.ndarray]:
    """
    根据 Y 值密度峰值过滤点云，并使用统计过滤去除离群点。
    根据计算的 Y 最大值与经验值进行比对，如果超出阈值则使用已知的 Y 范围进行过滤。
    保存并以非阻塞方式可视化保留的点云（使用原始颜色）。

    参数：
        points：3D 点云 (N, 3)。
        colors：对应的颜色 (N, 3)。
        visualizer：Visualizer 实例，用于显示和保存。
        known_y_min：已知 Y 值范围的下限。
        known_y_max：已知 Y 值范围的上限。
        max_y_threshold：用于比较的 Y 最大值阈值。
        config：过滤参数配置。

    返回：
        过滤后的点和颜色（NumPy 数组）。
    """
    # 获取 Y 值
    y_vals = points[:, 1]

    # 构建 Y 值的直方图
    hist, bin_edges = np.histogram(y_vals, bins=np.arange(y_vals.min(), y_vals.max(), config.bin_width))
    max_bin_idx = np.argmax(hist)

    # 计算 Y 值的主峰中心和范围
    y_center = (bin_edges[max_bin_idx] + bin_edges[max_bin_idx + 1]) / 2
    y_min, y_max = y_center - config.y_range / 2, y_center + config.y_range / 2

    # 打印计算得到的 Y 值范围
    print(f"计算得到的 Y 值范围：[{y_min:.4f}, {y_max:.4f}]")

    # 如果计算的 y_max 大于 max_y_threshold，则使用已知的 Y 值范围
    if y_max > max_y_threshold:
        print(f"y_max > {max_y_threshold}, 使用已知的 Y 范围 [{known_y_min}, {known_y_max}] 进行过滤")
        y_min, y_max = known_y_min, known_y_max
    else:
        print(f"y_max 没有超出阈值，继续使用计算的 Y 范围")

    # Y 范围过滤
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

    # 打印统计信息
    print(f"统计过滤后的点数：{len(ind)} / 初始点数：{len(filtered_points)}")

    # 保存点云截图并以非阻塞方式显示
    visualizer.save_point_cloud_screenshot(pcd_clean, "Y Range + Statistically Filtered Point Cloud (Original Colors)")

    return np.asarray(pcd_clean.points), np.asarray(pcd_clean.colors)

from datetime import datetime
import socket
import json
import cv2
import numpy as np
from typing import Any, Dict, List, Union

def send_detections_Csharp(
    results: List[Any],
    box_3d_info: List[Dict[str, Union[List[float], float]]],
    img: np.ndarray,
    server_host: str,
    server_port: int
) -> None:
    """
    将 2D/3D 坐标打包到同一个 dict，中间字段顺序是：
    coord_2d, width_2d, height_2d, confidence, coord_3d, width_3d, height_3d
    然后连同 img 一起通过 TCP 发送给 C# 端。
    """
    # 1) 编码图像
    success, img_encoded = cv2.imencode('.jpg', img)
    if not success:
        raise RuntimeError("Failed to encode image")
    img_bytes = img_encoded.tobytes()

    # 2) 拉平所有 2D 框 和置信度
    bboxes_2d = results[0].boxes.xywh.cpu().numpy()   # (N,4)
    confs     = results[0].boxes.conf.cpu().numpy()   # (N,)

    # 3) 按顺序构造 detections
    detections = []
    for (bbox, conf), extra in zip(zip(bboxes_2d, confs), box_3d_info):
        coord_3d = extra["center_3d"]
        if isinstance(coord_3d, np.ndarray):
            coord_3d = coord_3d.tolist()
        cx, cy, w, h = map(float, bbox)
        det = {
            "coord_2d":   [cx, cy],
            "width_2d":   w,
            "height_2d":  h,
            "confidence": float(conf),
            "coord_3d":   coord_3d,
            "width_3d":   extra["width_3d"],
            "height_3d":  extra["height_3d"],
        }
        detections.append(det)

    # 4) （可选）按 2D y→x 排序
    detections.sort(key=lambda d: (d["coord_2d"][1], d["coord_2d"][0]))

    # 5) 打包 JSON
    payload = {
        "timestamp": datetime.now().isoformat(),
        "box_count": len(detections),
        "detections": detections
    }
    json_bytes = json.dumps(payload, indent=2).encode('utf-8')

    # 6) 发送 JSON + 图像
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((server_host, server_port))
        sock.sendall(len(json_bytes).to_bytes(4, 'big'))
        sock.sendall(json_bytes)
        sock.sendall(len(img_bytes).to_bytes(4, 'big'))
        sock.sendall(img_bytes)

    print(f"[send_detections] Sent {payload['box_count']} boxes at {payload['timestamp']}")
