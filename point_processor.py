import numpy as np
import hdbscan
from config import GRID_PARAMS
import logging
logger = logging.getLogger(__name__)
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

    def cluster_points(self, points_3d):
        """
        对 X（列）、Y（层）、Z（行）三轴分别做 HDBSCAN 聚类，
        并把检测到的 layer_count、row_count、col_count 写入 GRID_PARAMS。
        返回：x_labels, y_labels, z_labels
        """
        N = len(points_3d)
        # 1. 自适应最小簇大小
        min_cluster_size = max(2, int(N * 0.1))

        # 2. 三轴聚类
        clustering_x = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                       cluster_selection_epsilon=0.2).fit(points_3d[:, [0]])
        clustering_y = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                       cluster_selection_epsilon=0.2).fit(points_3d[:, [1]])
        clustering_z = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                       cluster_selection_epsilon=0.2).fit(points_3d[:, [2]])

        x_labels = clustering_x.labels_
        y_labels = clustering_y.labels_
        z_labels = clustering_z.labels_

        # 3. 统计每个维度的簇数
        unique_y = np.unique(y_labels[y_labels != -1])
        detected_layer_count = len(unique_y)

        unique_z = np.unique(z_labels[z_labels != -1])
        detected_row_count = len(unique_z)

        # 每行有多少列
        detected_col_counts = {}
        for z_label in unique_z:
            mask = (z_labels == z_label)
            ux = np.unique(x_labels[mask])
            # 减去噪声标签
            col_count = len(ux) - (-1 in ux)
            detected_col_counts[z_label] = col_count
            logger.info(f"第{z_label}行：检测到{col_count}列")

        # 4. 更新 GRID_PARAMS（保留你原有的 row_count/col_count 逻辑，并新增 layer_count）
        # 层数
        if GRID_PARAMS.get('layer_count') is None:
            GRID_PARAMS['layer_count'] = detected_layer_count
        elif detected_layer_count > GRID_PARAMS['layer_count']:
            raise ValueError(f"检测到层数 {detected_layer_count} 超出配置 {GRID_PARAMS['layer_count']}")

        # 行数
        if GRID_PARAMS.get('row_count') is None:
            GRID_PARAMS['row_count'] = detected_row_count
        elif detected_row_count > GRID_PARAMS['row_count']:
            raise ValueError(f"检测到行数 {detected_row_count} 超出配置 {GRID_PARAMS['row_count']}")

        logger.info(f"检测到总层数：{detected_layer_count}，总行数：{detected_row_count}")

        # 列数（取第一行的列数作为参考）
        first_row_cols = list(detected_col_counts.values())[0]
        if GRID_PARAMS.get('col_count') is None:
            GRID_PARAMS['col_count'] = first_row_cols
        else:
            for z_label, cc in detected_col_counts.items():
                if cc > GRID_PARAMS['col_count']:
                    raise ValueError(f"第{z_label}行检测到列数 {cc} 超出配置 {GRID_PARAMS['col_count']}")

        # 返回三轴标签
        return x_labels, y_labels, z_labels

    def sort_labels(self, points_3d, x_labels, y_labels, z_labels):
        """
        分别对三组标签做重编码：
          - X 轴：从左到右（质心 X 小→大）
          - Y 轴：从近到远（质心 Y 小→大）
          - Z 轴：从上到下（质心 Z 大→小）
        返回：sorted_x, sorted_y, sorted_z
        """
        # --- Y 层排序（近→远） ---
        unique_y = [l for l in np.unique(y_labels) if l != -1]
        y_centroids = sorted(
            [(l, points_3d[y_labels == l, 1].mean()) for l in unique_y],
            key=lambda x: x[1]
        )
        y_map = {old: new for new, (old, _) in enumerate(y_centroids)}
        sorted_y = np.array([y_map.get(l, -1) for l in y_labels])

        # --- Z 行排序（上→下） ---
        unique_z = [l for l in np.unique(z_labels) if l != -1]
        z_centroids = sorted(
            [(l, points_3d[z_labels == l, 2].mean()) for l in unique_z],
            key=lambda x: -x[1]
        )
        z_map = {old: new for new, (old, _) in enumerate(z_centroids)}
        sorted_z = np.array([z_map.get(l, -1) for l in z_labels])

        # --- X 列排序（左→右），对每行独立排序 ---
        sorted_x = np.full_like(x_labels, -1)
        for old_z_label, new_row_idx in z_map.items():
            mask = z_labels == old_z_label
            xs = x_labels[mask]
            unique_x = [l for l in np.unique(xs) if l != -1]
            x_centroids = sorted(
                [(l, points_3d[mask][xs == l, 0].mean()) for l in unique_x],
                key=lambda x: x[1]
            )
            x_map = {old: new for new, (old, _) in enumerate(x_centroids)}
            sorted_x[mask] = [x_map.get(l, -1) for l in xs]

        return sorted_x, sorted_y, sorted_z
