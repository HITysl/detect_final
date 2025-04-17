import logging
import os
import time
from _ctypes import sizeof
import open3d as o3d
import numpy as np
import pyads
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.cluster import KMeans
from class_define import Box, Tasks
from config import GRID_PARAMS
from datetime import datetime
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

    # ========== 原逻辑不变 ==========
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

            print("Data successfully written to PLC")
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
    def __init__(self, images_dir="images"):
        """
        Initialize the Visualizer with default figure size and images directory.

        Args:
            images_dir (str): Directory to save images. Defaults to 'images'.
        """
        self.figsize = (10, 8)
        self.images_dir = images_dir
        os.makedirs(self.images_dir, exist_ok=True)  # Create images directory if it doesn't exist
        plt.ion()  # Enable interactive mode for non-blocking display

    def _set_equal_aspect(self, ax, points):
        """
        Set equal aspect ratio for 3D axes based on point cloud range.

        Args:
            ax: Matplotlib 3D axes object.
            points (np.ndarray): 3D points of shape (N, 3).

        Raises:
            ValueError: If points is not a valid numpy array.
        """
        if not isinstance(points, np.ndarray) or points.shape[1] != 3:
            raise ValueError("Points must be a numpy array of shape (N, 3)")
        ax.set_box_aspect([1, 1, 1])
        max_range = max(points[:, i].ptp() for i in range(3)) / 2
        mid = [points[:, i].mean() for i in range(3)]
        for i, lim in enumerate(['xlim', 'ylim', 'zlim']):
            getattr(ax, f'set_{lim}')(mid[i] - max_range, mid[i] + max_range)

    def _get_timestamped_filename(self, prefix):
        """
        Generate a filename with a timestamp.

        Args:
            prefix (str): Prefix for the filename (e.g., 'visualize_points').

        Returns:
            str: Full path to the timestamped file (e.g., 'images/visualize_points_20250416_123456.png').
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.images_dir, f"{prefix}_{timestamp}.png")

    def visualize_points(self, points, high_len):
        """
        Visualize 3D points, distinguishing high and low points, non-blocking, and save with timestamp.

        Args:
            points (np.ndarray): 3D points of shape (N, 3).
            high_len (int): Number of points classified as 'high'.

        Raises:
            ValueError: If points is not a valid numpy array or high_len is invalid.
        """
        if not isinstance(points, np.ndarray) or points.shape[1] != 3:
            raise ValueError("Points must be a numpy array of shape (N, 3)")
        if not isinstance(high_len, int) or high_len < 0 or high_len > len(points):
            raise ValueError("high_len must be a non-negative integer <= len(points)")
        if len(points) == 0:
            print("Warning: No points to visualize")
            return

        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:high_len, 0], points[:high_len, 1], points[:high_len, 2], c='r', label='High')
        ax.scatter(points[high_len:, 0], points[high_len:, 1], points[high_len:, 2], c='b', label='Low')
        self._set_equal_aspect(ax, points)
        ax.set(xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)', title='Detected 3D Points')
        ax.legend()

        # Save the plot
        save_path = self._get_timestamped_filename("visualize_points")
        try:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        except Exception as e:
            print(f"Error saving plot to {save_path}: {e}")

        # Non-blocking display
        plt.draw()
        plt.pause(0.001)  # Brief pause to update display

    def visualize_comparison(self, original, filtered):
        """
        Compare original and filtered 3D point clouds side by side, non-blocking, and save with timestamp.

        Args:
            original (np.ndarray): Original 3D points of shape (N, 3).
            filtered (np.ndarray): Filtered 3D points of shape (M, 3).

        Raises:
            ValueError: If inputs are not valid numpy arrays.
        """
        if not isinstance(original, np.ndarray) or not isinstance(filtered, np.ndarray):
            raise ValueError("Original and filtered points must be numpy arrays")
        if original.shape[1] != 3 or filtered.shape[1] != 3:
            raise ValueError("Points must have shape (N, 3)")
        if not (len(original) and len(filtered)):
            print("Warning: No points to visualize")
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

        # Save the plot
        save_path = self._get_timestamped_filename("visualize_comparison")
        try:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        except Exception as e:
            print(f"Error saving plot to {save_path}: {e}")

        # Non-blocking display
        plt.draw()
        plt.pause(0.001)

    def plot_all_2d(self, points_2d, grid_points, adjusted_3d, x_labels, z_labels):
        """
        Plot 2D comparison of original, grid, and adjusted points, non-blocking, and save with timestamp.

        Args:
            points_2d (np.ndarray): 2D points of shape (N, 2).
            grid_points (np.ndarray): Grid points of shape (M, 2).
            adjusted_3d (np.ndarray): Adjusted 3D points of shape (K, 3).
            x_labels (list): Labels for x-coordinates.
            z_labels (list): Labels for z-coordinates.

        Raises:
            ValueError: If inputs are not valid or inconsistent.
        """
        if not all(isinstance(arr, np.ndarray) for arr in [points_2d, grid_points, adjusted_3d]):
            raise ValueError("All point arrays must be numpy arrays")
        if points_2d.shape[1] != 2 or grid_points.shape[1] != 2 or adjusted_3d.shape[1] != 3:
            raise ValueError("Invalid point array shapes")
        if not (len(points_2d) and len(grid_points) and len(adjusted_3d)):
            print("Warning: No points to visualize")
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

        # Save the plot
        save_path = self._get_timestamped_filename("plot_all_2d")
        try:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        except Exception as e:
            print(f"Error saving plot to {save_path}: {e}")

        # Non-blocking display
        plt.draw()
        plt.pause(0.001)

    def plot_all_3d(self, original, adjusted):
        """
        Plot 3D comparison of original and adjusted points, non-blocking, and save with timestamp.

        Args:
            original (np.ndarray): Original 3D points of shape (N, 3).
            adjusted (np.ndarray): Adjusted 3D points of shape (M, 3).

        Raises:
            ValueError: If inputs are not valid numpy arrays.
        """
        if not isinstance(original, np.ndarray) or not isinstance(adjusted, np.ndarray):
            raise ValueError("Original and adjusted points must be numpy arrays")
        if original.shape[1] != 3 or adjusted.shape[1] != 3:
            raise ValueError("Points must have shape (N, 3)")
        if not (len(original) and len(adjusted)):
            print("Warning: No points to visualize")
            return

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(original[:, 0], original[:, 1], original[:, 2], c='blue', label='Original')
        ax.scatter(adjusted[:, 0], adjusted[:, 1], adjusted[:, 2], c='red', label='Adjusted')
        self._set_equal_aspect(ax, np.vstack((original, adjusted)))
        ax.set(xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)', title='Original vs Adjusted 3D')
        ax.legend()

        # Save the plot
        save_path = self._get_timestamped_filename("plot_all_3d")
        try:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        except Exception as e:
            print(f"Error saving plot to {save_path}: {e}")

        # Non-blocking display
        plt.draw()
        plt.pause(0.001)

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
        kmeans = KMeans(n_clusters=2, random_state=0).fit(x_coords)
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
                            box.aGraspPoint_Side[0],
                            box.aGraspPoint_Side[1] + GRID_PARAMS['y_spacing'] * 500,
                            300
                        ])
                        # 调试打印更新后的箱子
                        print(
                            f"Updated Box id={box.id}, row={box.row}, col={box.col}, Top z={box.aGraspPoint_Top[2]}, Above z={above_box.aGraspPoint_Side[2]}")
    return boxes
