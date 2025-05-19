import os
import random
import threading
from dataclasses import dataclass
from typing import Tuple, Optional, List, Any, Dict, Union
import open3d as o3d
import pyads
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from class_define import Box, Tasks
from config import GRID_PARAMS
import numpy as np
import socket
import json
import time
import logging
import cv2
from datetime import datetime
logger = logging.getLogger(__name__)

@dataclass
class FilterConfig:
    bin_width: float = 0.01             # 直方图 bin 宽度
    peak_prominence: float =20         # 峰值 prominence 阈值，可根据数据调节
    peak_width: float = 0.1             # 保留峰附近 Δy 窗口宽度
    outlier_nb_neighbors: int = 20      # 统计滤波 k 值
    outlier_std_ratio: float = 2.0      # 统计滤波 std_ratio
logger = logging.getLogger(__name__)

# Point Cloud Processing
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


def transmit_to_plc(tasks):
    # ========== 打印任务结构 ==========
    print("========== Task Summary ==========")
    logger.info("========== Task Summary ==========")
    logger.info(str(tasks))  # 触发 Tasks.__str__()
    print(tasks)  # 触发 Tasks.__str__()


    print("========== Orgin Boxes ==========")
    logger.info("========== Orgin Boxes ==========")
    for box in tasks.all_box_origin:
        print(f"[origin] ID: {box.id}, Row: {box.row}, Col: {box.col}")
        print(
            f"       Top Grasp Point : x={box.aGraspPoint_Top[0]:.1f}, y={box.aGraspPoint_Top[1]:.1f}, z={box.aGraspPoint_Top[2]:.1f}")
        print(
            f"       Side Grasp Point: x={box.aGraspPoint_Side[0]:.1f}, y={box.aGraspPoint_Side[1]:.1f}, z={box.aGraspPoint_Side[2]:.1f}")
        print(f"       Width: {box.width_3d:.1f} mm, Height: {box.height_3d:.1f} mm")
        logger.info(f"[origin] ID: {box.id}, Row: {box.row}, Col: {box.col}")
        logger.info(
            f"       Top Grasp Point : x={box.aGraspPoint_Top[0]:.1f}, y={box.aGraspPoint_Top[1]:.1f}, z={box.aGraspPoint_Top[2]:.1f}")
        logger.info(
            f"       Side Grasp Point: x={box.aGraspPoint_Side[0]:.1f}, y={box.aGraspPoint_Side[1]:.1f}, z={box.aGraspPoint_Side[2]:.1f}")
        logger.info(f"       Width: {box.width_3d:.1f} mm, Height: {box.height_3d:.1f} mm")




    print("========== Left Boxes ==========")
    logger.info("========== Left Boxes ==========")
    for box in tasks.aLeftBoxArray:
        print(f"[Left] ID: {box.id}, Row: {box.row}, Col: {box.col}")
        print(
            f"       Top Grasp Point : x={box.aGraspPoint_Top[0]:.1f}, y={box.aGraspPoint_Top[1]:.1f}, z={box.aGraspPoint_Top[2]:.1f}")
        print(
            f"       Side Grasp Point: x={box.aGraspPoint_Side[0]:.1f}, y={box.aGraspPoint_Side[1]:.1f}, z={box.aGraspPoint_Side[2]:.1f}")
        print(f"       Width: {box.width_3d:.1f} mm, Height: {box.height_3d:.1f} mm")
        logger.info(f"[Left] ID: {box.id}, Row: {box.row}, Col: {box.col}")
        logger.info(
            f"       Top Grasp Point : x={box.aGraspPoint_Top[0]:.1f}, y={box.aGraspPoint_Top[1]:.1f}, z={box.aGraspPoint_Top[2]:.1f}")
        logger.info(
            f"       Side Grasp Point: x={box.aGraspPoint_Side[0]:.1f}, y={box.aGraspPoint_Side[1]:.1f}, z={box.aGraspPoint_Side[2]:.1f}")
        logger.info(f"       Width: {box.width_3d:.1f} mm, Height: {box.height_3d:.1f} mm")

    print("========== Right Boxes ==========")
    logger.info("========== Right Boxes ==========")
    for box in tasks.aRightBoxArray:
        print(f"[Right] ID: {box.id}, Row: {box.row}, Col: {box.col}")
        print(
            f"        Top Grasp Point : x={box.aGraspPoint_Top[0]:.1f}, y={box.aGraspPoint_Top[1]:.1f}, z={box.aGraspPoint_Top[2]:.1f}")
        print(
            f"        Side Grasp Point: x={box.aGraspPoint_Side[0]:.1f}, y={box.aGraspPoint_Side[1]:.1f}, z={box.aGraspPoint_Side[2]:.1f}")
        print(f"        Width: {box.width_3d:.1f} mm, Height: {box.height_3d:.1f} mm")
        logger.info(f"[Right] ID: {box.id}, Row: {box.row}, Col: {box.col}")
        logger.info(
            f"        Top Grasp Point : x={box.aGraspPoint_Top[0]:.1f}, y={box.aGraspPoint_Top[1]:.1f}, z={box.aGraspPoint_Top[2]:.1f}")
        logger.info(
            f"        Side Grasp Point: x={box.aGraspPoint_Side[0]:.1f}, y={box.aGraspPoint_Side[1]:.1f}, z={box.aGraspPoint_Side[2]:.1f}")
        logger.info(f"        Width: {box.width_3d:.1f} mm, Height: {box.height_3d:.1f} mm")

    nbox_l = int(GRID_PARAMS['x_spacing'] * 1000)
    nbox_w = int(GRID_PARAMS['y_spacing'] * 1000)
    nbox_h = int(GRID_PARAMS['z_spacing'] * 1000)

    nLeftBoxCount = int(tasks.nLeftBoxCount)
    nRightBoxCount = int(tasks.nRightBoxCount)
    nTotalRow = int(tasks.nTotalRow)
    nTotalCol = int(tasks.nTotalCol)

    leftArm_Data = []
    rightArm_Data = []
    boxArray_Data = []
    for box in tasks.all_box_origin:
        flat = [
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
        boxArray_Data.extend(flat)
    logger.info(f"Flattened Original Boxes data length: {len(boxArray_Data)}")

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
            plc.write_by_name('Camera.aBoxArray',boxArray_Data,pyads.PLCTYPE_ARR_INT(len(boxArray_Data)))
            plc.write_by_name('Camera.bInspection_IPC_Done', True, pyads.PLCTYPE_BOOL)
            print("Data successfully written to PLC")
            logger.info("Data successfully written to PLC")
            logger.info("Camera.bInspection_IPC_Done set to True")
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

    def visualize_points(self, points, sorted_x=None, sorted_y=None, sorted_z=None):
        """
        在 3D 散点图里标注 (layer, row, col)。
        """
        if sorted_x is None or sorted_y is None or sorted_z is None:
            raise ValueError("必须同时提供 sorted_x, sorted_y, sorted_z")

        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:,0], points[:,1], points[:,2], c='b', label='Points')

        # 标注 (layer, row, col)
        for i, (x, y, z) in enumerate(points):
            l, r, c = sorted_y[i], sorted_z[i], sorted_x[i]
            if l != -1 and r != -1 and c != -1:
                ax.text(x, y, z, f"({l},{r},{c})", fontsize=8, color='black')

        self._set_equal_aspect(ax, points)
        ax.set(xlabel='X', ylabel='Y', zlabel='Z',
               title='3D Points with (layer,row,col)')
        ax.legend()

        base = self._get_timestamped_filename("visualize_with_layers")
        # 如果 base 已经包含 .png 后缀，就插在 .png 之前，否则随意拼一个后缀
        name, ext = os.path.splitext(base)
        rand_suffix = random.randint(0, 9999)
        save_path = f"{name}_{rand_suffix:04d}{ext}"

        plt.savefig(save_path, bbox_inches='tight')

        if self.enable_display:
            plt.draw(); plt.pause(self.display_duration)
        plt.close(fig)
        print(f"保存：{save_path}")

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
    for (row, col), group in box_groups.items():
        if len(group) > 1:
            msg = f"WARNING: 在 (row={row}, col={col}) 检测到 {len(group)} 个 box，已合并抓取点"
            logger.warning(msg)
            print(msg)
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
        logger.info("Warning: Not enough boxes to perform K-means clustering.")
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
    如果发送失败，最多重试 5 次；连续 5 次失败后抛出异常。
    """
    # 1) 编码图像
    success, img_encoded = cv2.imencode('.jpg', img)
    if not success:
        logger.error("Failed to encode image")
        raise RuntimeError("Failed to encode image")
    img_bytes = img_encoded.tobytes()

    # 2) 拉平所有 2D 框 和置信度
    bboxes_2d = results[0].boxes.xywh.cpu().numpy()   # (N,4)
    confs     = results[0].boxes.conf.cpu().numpy()   # (N,)

    # 3) 按顺序构造 detections
    detections = []
    for (bbox, conf), extra in zip(zip(bboxes_2d, confs), box_3d_info):
        coord_3d = extra["center_3d"]
        if hasattr(coord_3d, "tolist"):
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

    # 6) 发送 JSON + 图像，带重试逻辑
    max_retries = 5
    for attempt in range(1, max_retries + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect((server_host, server_port))
                sock.sendall(len(json_bytes).to_bytes(4, 'big'))
                sock.sendall(json_bytes)
                sock.sendall(len(img_bytes).to_bytes(4, 'big'))
                sock.sendall(img_bytes)
            msg = f"[send_detections] attempt {attempt}/{max_retries} Sent {payload['box_count']} boxes at {payload['timestamp']}"
            logger.info(msg)
            print(msg)
            break  # 发送成功，跳出重试
        except Exception as e:
            warn = f"[send_detections] attempt {attempt} failed: {e}"
            logger.warning(warn)
            print(warn)
            time.sleep(0.5)  # 间隔 500ms 再试
    else:
        # 连续 max_retries 次都失败
        err = f"[send_detections] failed after {max_retries} attempts"
        logger.error(err)
        print(err)
        raise RuntimeError(err)

def preprocess_point_cloud(points: np.ndarray,
                           colors: np.ndarray,
                           voxel_size: float = 0.009,
                           nb_neighbors: int = 30,
                           std_ratio: float = 0.5,

                           ) -> Tuple[np.ndarray, np.ndarray]:
    """
    1) 体素下采样
    2) 统计离群点去除
    3) 显示一次 Y 轴直方图
    4) 基于 Y 缺口过滤背景 + 二次统计滤波
    返回最终的点与颜色数组（保持与原来相同的 return 形式）
    """
    # 构建 PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 体素下采样
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)

    # 统计滤波去噪
    if nb_neighbors > 0 and std_ratio > 0:
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio
        )
    ys = np.asarray(pcd.points)[:, 1]
    plt.figure(figsize=(8, 4))
    plt.hist(ys, bins=100)
    plt.title("Point Cloud Y-Value Histogram (After Pre-filtering)")
    plt.xlabel("Y value")
    plt.ylabel("Point Count")
    plt.tight_layout()
    plt.show()

    filtered_pts, filtered_clrs = filter_by_y_peaks(
        points=np.asarray(pcd.points),
        colors=np.asarray(pcd.colors),
        visualizer=None
    )

    pcd.points = o3d.utility.Vector3dVector(filtered_pts)
    pcd.colors = o3d.utility.Vector3dVector(filtered_clrs)

    ys = np.asarray(pcd.points)[:, 1]
    plt.figure(figsize=(8, 4))
    plt.hist(ys, bins=100)
    plt.title("Point Cloud Y-Value Histogram (After Pre-filtering)")
    plt.xlabel("Y value")
    plt.ylabel("Point Count")
    plt.tight_layout()
    plt.show()

    return np.asarray(pcd.points), np.asarray(pcd.colors)

def filter_by_y_peaks(
    points: np.ndarray,
    colors: np.ndarray,
    visualizer: Optional[Visualizer] = None,
    config: FilterConfig = FilterConfig()
) -> Tuple[np.ndarray, np.ndarray]:
    """
    基于 Y 轴方向的直方图峰值筛选：
    1) 找峰，不用最高，只要明显高；
    2) 去除 Y 最大峰的那一簇；
    3) 保留剩余峰附近 Δy 窗口内的点。
    """
    y_vals = points[:, 1]

    # 1. 构造直方图
    bins = np.arange(y_vals.min(),
                     y_vals.max() + config.bin_width,
                     config.bin_width)
    hist, bin_edges = np.histogram(y_vals, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    hist_smooth = gaussian_filter1d(hist, sigma=0.5)
    # 2. 找峰值
    peaks, props = find_peaks(
        hist_smooth,
        prominence=config.peak_prominence,
        distance=10,
        height=5000
    )
    if len(peaks) == 0:
        print("未检测到明显峰值，返回原点云")
        return points, colors

    peak_ys = bin_centers[peaks]
    print(f"检测到峰值 Y：{peak_ys}")

    # 3. 确定并剔除 Y 最大峰
    max_peak_y = peak_ys.max()
    remove_mask = np.abs(y_vals - max_peak_y) <= config.peak_width
    keep_mask = ~remove_mask
    print(f"剔除 Y={max_peak_y:.4f} 峰附近 {np.sum(remove_mask)} 个点，剩余 {np.sum(keep_mask)} 个")

    # 4. 基于剩余点重新构造直方图
    y_vals_filtered = y_vals[keep_mask]
    hist_filtered, bin_edges_filtered = np.histogram(y_vals_filtered, bins=bins)
    hist_smooth_filtered = gaussian_filter1d(hist_filtered, sigma=0.5)

    # 5. 重新找峰值
    peaks_filtered, _ = find_peaks(
        hist_smooth_filtered,
        prominence=config.peak_prominence,
        height=5000
    )
    if len(peaks_filtered) == 0:
        print("剔除最大峰后未检测到其他峰值，返回剩余点云")
        filtered_points = points[keep_mask]
        filtered_colors = colors[keep_mask]
    else:
        peak_ys_filtered = bin_centers[peaks_filtered]
        print(f"剔除最大峰后检测到峰值 Y：{peak_ys_filtered}")

        # 6. 构建保留掩码：保留剩余峰附近 Δy 窗口内的点
        keep_mask_refined = np.zeros_like(y_vals, dtype=bool)
        for yp in peak_ys_filtered:
            keep_mask_refined[keep_mask] |= np.abs(y_vals_filtered - yp) <= config.peak_width

        print(f"保留 {len(peak_ys_filtered)} 个峰附近的点，共计 {np.sum(keep_mask_refined)} 个")

        # 7. 应用最终掩码
        filtered_points = points[keep_mask_refined]
        filtered_colors = colors[keep_mask_refined]

    # 8. 统计离群点剔除
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_points)
    pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
    pcd_clean, ind = pcd.remove_statistical_outlier(
        nb_neighbors=config.outlier_nb_neighbors,
        std_ratio=config.outlier_std_ratio
    )

    print(f"统计滤波后剩余：{len(ind)} / {len(filtered_points)}")
    if visualizer is not None:
        visualizer.save_point_cloud_screenshot(
            pcd_clean, "Y-Gap Filtered Point Cloud"
        )
    return np.asarray(pcd_clean.points), np.asarray(pcd_clean.colors)