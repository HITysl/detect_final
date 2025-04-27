import numpy as np
import hdbscan
from config import GRID_PARAMS
import logging

class PointProcessor:
    def __init__(self):
        pass  # 移除无意义的 self.eps 和 self.min_samples

    # def cluster_and_filter(self, points):
    #     """对点云进行聚类并返回簇中心"""
    #     # 动态计算最小簇大小
    #     min_cluster_size = max(2, int(len(points) * 0.05))  # 至少2点，比例为5%
    #     clustering = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size).fit(points)
    #     labels = clustering.labels_
    #     centers = [points[labels == label].mean(axis=0) for label in set(labels) if label != -1]
    #     if not centers:
    #         raise ValueError("未找到任何聚类中心")
    #     return np.array(centers)

    def cluster_points(self, points_2d):
        """对2D点云进行行和列聚类，返回 X 和 Z 标签"""
        # 自适应聚类参数
        min_cluster_size = max(2, int(len(points_2d) * 0.1))  # 动态最小簇大小

        # 使用 HDBSCAN 进行 X 轴（列）和 Z 轴（行）聚类
        clustering_x = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size).fit(points_2d[:, [0]])
        clustering_z = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size).fit(points_2d[:, [1]])
        x_labels = clustering_x.labels_
        z_labels = clustering_z.labels_

        # 计算行数和每行列数
        unique_z = np.unique(z_labels[z_labels != -1])
        detected_row_count = len(unique_z)
        detected_col_counts = {}

        for z_label in unique_z:
            row_mask = z_labels == z_label
            unique_x_in_row = np.unique(x_labels[row_mask])
            col_count = len(unique_x_in_row) - (-1 in unique_x_in_row)
            detected_col_counts[z_label] = col_count
            logging.info(f"第{z_label}行：检测到{col_count}列")

        # ✅ 检查是否所有行的列数一致
        if len(set(detected_col_counts.values())) != 1:
            raise ValueError(f"检测失败：不同的行检测到的列数不一致 {detected_col_counts}")

        # ✅ 如果 row_count 或 col_count 为 None，就自动赋值
        if GRID_PARAMS['row_count'] is None:
            GRID_PARAMS['row_count'] = detected_row_count
        elif detected_row_count > GRID_PARAMS['row_count']:
            raise ValueError(f"检测到行数 {detected_row_count}，超出配置的最大行数 {GRID_PARAMS['row_count']}")
        logging.info(f"检测到总行数：{detected_row_count}")

        if GRID_PARAMS['col_count'] is None:
            GRID_PARAMS['col_count'] = list(detected_col_counts.values())[0]
        else:
            for z_label, col_count in detected_col_counts.items():
                if col_count > GRID_PARAMS['col_count']:
                    raise ValueError(f"第{z_label}行检测到列数 {col_count}，超出配置的最大列数 {GRID_PARAMS['col_count']}")

        return x_labels, z_labels

    def sort_labels(self, points_2d, x_labels, z_labels):
        """对行和列标签进行排序，列从左到右，行从上到下"""
        # 按 Z 坐标均值从大到小排序行（从上到下）
        z_centroids = sorted(
            [(l, points_2d[z_labels == l, 1].mean()) for l in np.unique(z_labels) if l != -1],
            key=lambda x: -x[1]
        )
        z_map = {old: new for new, (old, _) in enumerate(z_centroids)}
        sorted_x = np.full_like(x_labels, -1)
        sorted_z = np.array([z_map.get(l, -1) for l in z_labels])

        # 每行独立排序 X 标签（从左到右）
        for z_label, row_idx in z_map.items():
            row_mask = z_labels == z_label
            x_labels_in_row = x_labels[row_mask]
            unique_x = np.unique(x_labels_in_row)
            x_centroids = sorted(
                [(l, points_2d[row_mask][x_labels_in_row == l, 0].mean()) for l in unique_x if l != -1],
                key=lambda x: x[1]
            )
            x_map = {old: new for new, (old, _) in enumerate(x_centroids)}
            sorted_x[row_mask] = np.array([x_map.get(l, -1) for l in x_labels_in_row])

        return sorted_x, sorted_z
