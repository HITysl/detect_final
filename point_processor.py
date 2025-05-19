import numpy as np
import hdbscan
from config import GRID_PARAMS
import logging
from typing import Tuple
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

    def cluster_points(self, points_3d: np.ndarray) -> np.ndarray:
        """
        仅对 Z 轴（行）进行 HDBSCAN 聚类。
        更新 GRID_PARAMS 中的 row_count。
        返回：z_labels (行标签)
        """
        N = len(points_3d)
        if N == 0:
            logger.warning("cluster_points 接收到空点云数组")
            # 根据 GRID_PARAMS 的期望，决定是否设置或如何处理
            if GRID_PARAMS.get('row_count') is None:
                 GRID_PARAMS['row_count'] = 0
            return np.array([], dtype=int)

        # 1. 自适应最小簇大小
        min_cluster_size = max(2, int(N * 0.1))

        # 2. Z轴聚类 (行聚类)
        # 使用 reshape(-1, 1) 保证输入是二维的
        clustering_z = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                       cluster_selection_epsilon=0.2).fit(points_3d[:, [2]])
        z_labels = clustering_z.labels_

        # 3. 统计行数
        unique_z_clusters = np.unique(z_labels[z_labels != -1])
        detected_row_count = len(unique_z_clusters)

        logger.info(f"检测到总行数：{detected_row_count}")

        # 4. 更新 GRID_PARAMS 中的 row_count
        if GRID_PARAMS.get('row_count') is None:
            GRID_PARAMS['row_count'] = detected_row_count
        elif detected_row_count > GRID_PARAMS['row_count']:
            logger.warning(f"检测到行数 {detected_row_count} 超出配置 {GRID_PARAMS['row_count']}")
            GRID_PARAMS['row_count'] = detected_row_count

        # 5. 计算每行的箱子数并更新 GRID_PARAMS['col_count'] 为最大值
        actual_max_cols_in_any_row = 0
        row_box_counts_map = {} # 用于存储每个原始Z簇标签及其对应的箱子数
        if detected_row_count > 0:
            row_counts_summary = []
            for z_cluster_label in unique_z_clusters:
                # z_cluster_label 是 HDBSCAN 分配的原始簇标签 (0, 1, 2, ...)
                # 我们需要计算的是 z_labels 数组中，有多少个元素的等于当前的 z_cluster_label
                num_boxes_in_row = np.sum(z_labels == z_cluster_label)
                row_box_counts_map[z_cluster_label] = num_boxes_in_row # 存储，供后续验证使用
                row_counts_summary.append(f"  原始Z簇标签 {z_cluster_label}: {num_boxes_in_row} 个箱子")
                if num_boxes_in_row > actual_max_cols_in_any_row:
                    actual_max_cols_in_any_row = num_boxes_in_row
            
            logger.info("每个检测到的Z簇（行）的箱子数量:")
            for summary_line in row_counts_summary:
                logger.info(summary_line)
        else:
            logger.info("未从Z轴聚类中检测到有效行。")

        original_col_count = GRID_PARAMS.get('col_count')
        GRID_PARAMS['col_count'] = actual_max_cols_in_any_row
        logger.info(f"GRID_PARAMS['col_count'] 原为 {original_col_count}, 更新为检测到的最大列数: {GRID_PARAMS['col_count']}")

        # 6. 验证行箱子数量是否符合预期
        if detected_row_count >= 2:
            # 为行排序和验证准备数据：(原始Z簇标签, 平均Z值, 箱子数)
            row_data_for_sorting = []
            for z_cluster_label in unique_z_clusters: # unique_z_clusters 只包含非噪声标签
                # 确保 z_cluster_label 存在于 row_box_counts_map 中
                if z_cluster_label in row_box_counts_map:
                    mean_z = points_3d[z_labels == z_cluster_label, 2].mean()
                    num_boxes = row_box_counts_map[z_cluster_label]
                    row_data_for_sorting.append({'original_label': z_cluster_label, 'mean_z': mean_z, 'count': num_boxes})
                else:
                    #这种情况理论上不应发生，因为 unique_z_clusters 是从 z_labels 来的，row_box_counts_map 也基于它填充
                    logger.warning(f"警告：在验证准备阶段，原始Z簇标签 {z_cluster_label} 未在 row_box_counts_map 中找到。")
            
            if not row_data_for_sorting or len(row_data_for_sorting) < detected_row_count:
                # 如果实际收集到的用于排序的行数据少于检测到的行数（排除空簇等情况后）
                # 并且仍然至少有两行数据可供比较
                if len(row_data_for_sorting) >=2:
                    logger.warning(f"警告：为验证准备的行数据 ({len(row_data_for_sorting)}) 少于检测到的非噪声Z簇数 ({detected_row_count}). 继续使用可用数据进行验证。")
                else:
                    logger.info(f"为验证准备的有效行数据 ({len(row_data_for_sorting)}) 不足两行，跳过结构验证。")
                    # 在这种情况下，由于后续逻辑依赖于至少两行，我们应该直接返回
                    # 确保 z_labels 在此之前已定义并可返回
                    return z_labels 

            # 按平均Z值降序排列行 (Z越大，越靠上，越是顶行)
            sorted_rows_by_z = sorted(row_data_for_sorting, key=lambda x: x['mean_z'], reverse=True)

            # 如果排序后的行数仍然不足2，则无法进行比较
            if len(sorted_rows_by_z) < 2:
                logger.info(f"排序后有效行数 ({len(sorted_rows_by_z)}) 不足两行，跳过行箱数量结构验证。")
            else:
                topmost_row_data = sorted_rows_by_z[0]
                topmost_row_count = topmost_row_data['count']
                
                # 以第二行（排序后的索引为1的行）的箱子数作为其他行的标准箱子数
                standard_other_rows_count = sorted_rows_by_z[1]['count']
                
                validation_passed = True
                error_details = []

                # 规则 A: 检查所有"其他"行（从排序后索引1开始）的箱子数是否等于 standard_other_rows_count
                for i in range(1, len(sorted_rows_by_z)):
                    current_row_data = sorted_rows_by_z[i]
                    if current_row_data['count'] != standard_other_rows_count:
                        error_details.append(
                            f"  规则A失败：排序后行 {i} (原始Z簇 {current_row_data['original_label']}) 箱数 {current_row_data['count']} != 标准数 {standard_other_rows_count}."
                        )
                        validation_passed = False
                
                # 规则 B: 检查最顶行的箱子数是否小于或等于 standard_other_rows_count
                # 只有在规则A没有失败（即 standard_other_rows_count 是一个有效基准）或者所有其他行数量都一致时，此检查才有意义
                # 如果规则A失败，意味着 standard_other_rows_count 可能不是一个统一的标准，规则B的判断可能基于一个不一致的基准。
                # 但目前的逻辑是独立的，如果规则A失败，validation_passed已经是False。
                if topmost_row_count > standard_other_rows_count:
                    error_details.append(
                        f"  规则B失败：最顶行 (原始Z簇 {topmost_row_data['original_label']}) 箱数 {topmost_row_count} > 标准数 {standard_other_rows_count}."
                    )
                    validation_passed = False
                
                if not validation_passed:
                    full_error_message_parts = ["行箱数量验证失败。预期：除最顶行外，其余行箱数一致，且最顶行箱数不大于其余行。实际情况："]
                    for i, r_data in enumerate(sorted_rows_by_z):
                        full_error_message_parts.append(
                            f"  排序后行 {i} (原始Z簇 {r_data['original_label']}, 平均Z值 {r_data['mean_z']:.3f}): {r_data['count']} 个箱子"
                        )
                    full_error_message_parts.extend(error_details)
                    final_error_message = "\n".join(full_error_message_parts)
                    logger.error(final_error_message)
                    raise ValueError(final_error_message)
                else:
                    logger.info("行箱数量结构验证通过。")

        elif detected_row_count == 1:
            logger.info("仅检测到一行，跳过行箱数量结构验证。")
        # else detected_row_count == 0 (no valid rows found from clustering)
        #   logger.info("未从Z轴聚类中检测到有效行。") 已在前面记录

        return z_labels

    def sort_labels(self, points_3d: np.ndarray, z_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        对 Z 轴标签（行）进行排序（Z 大 -> 小，即上到下）。
        在每个已排序的行内，根据 X 坐标（X 小 -> 大，即左到右）对点进行编号。
        Y 轴标签（层）默认为0。
        返回：sorted_x, sorted_y, sorted_z
        """
        if len(points_3d) == 0:
            logger.warning("sort_labels 接收到空点云数组")
            empty_array = np.array([], dtype=int)
            return empty_array, empty_array, empty_array
        
        if len(points_3d) != len(z_labels):
            raise ValueError("points_3d 和 z_labels 的长度必须一致")

        # --- Y 层标签 (默认为0, 表示单层或Y轴不用于分层索引) ---
        sorted_y = np.zeros(len(points_3d), dtype=int)

        # --- Z 行排序（上→下，Z值大到小） ---
        unique_z_clusters = [l for l in np.unique(z_labels) if l != -1]
        if not unique_z_clusters: # 如果所有点都是噪声
            logger.warning("在 Z 轴聚类中未找到有效簇，所有点被视为噪声。")
            sorted_z = np.full(len(points_3d), -1, dtype=int)
            sorted_x = np.full(len(points_3d), -1, dtype=int)
            return sorted_x, sorted_y, sorted_z

        # 计算每个有效 Z 簇的质心 Z 值
        z_centroids_data = []
        for l_cluster in unique_z_clusters:
            mask = (z_labels == l_cluster)
            if np.any(mask): # 确保簇不为空
                 z_centroids_data.append((l_cluster, points_3d[mask, 2].mean()))
            else:
                logger.warning(f"Z-cluster {l_cluster} 在 z_labels 中标记，但在 points_3d 中找不到对应点。")


        if not z_centroids_data: # 如果所有有效簇实际上都没有点
            logger.warning("所有有效的Z轴簇都没有关联的点。")
            sorted_z = np.full(len(points_3d), -1, dtype=int) # 将所有原始标签设为-1
            for i, orig_label in enumerate(z_labels):
                if orig_label != -1 : # 如果原始标签不是噪声，但我们没能处理它
                     sorted_z[i] = -2 # 特殊标记，表示未处理的非噪声簇
                else:
                    sorted_z[i] = -1 # 噪声点保持-1
            sorted_x = np.full(len(points_3d), -1, dtype=int)
            return sorted_x, sorted_y, sorted_z

        # 根据质心Z值排序，Z值大的在前（从上到下）
        z_centroids_data.sort(key=lambda item: item[1], reverse=True)
        
        # 创建从旧 Z 簇标签到新排序后行索引的映射
        z_map = {old_label: new_row_idx for new_row_idx, (old_label, _) in enumerate(z_centroids_data)}
        
        # 应用映射，得到所有点的排序后的行索引 (sorted_z)
        # 对于原始标签为 -1 (噪声) 的点，其 sorted_z 值也应为 -1
        # 对于在 unique_z_clusters 中但未出现在 z_centroids_data (例如空簇) 的标签，也应视为 -1 或特殊值
        sorted_z = np.full(len(points_3d), -1, dtype=int)
        for i, old_label in enumerate(z_labels):
            sorted_z[i] = z_map.get(old_label, -1)


        # --- X 列编号（左→右），在每个已排序的行内进行 ---
        sorted_x = np.full(len(points_3d), -1, dtype=int)

        # 遍历每个新的行索引 (0, 1, 2...)
        for new_row_idx in range(len(z_centroids_data)):
            # 获取当前处理行对应的原始点在 points_3d 中的布尔掩码
            # 注意：这里应该用 sorted_z 来确定哪些点属于当前 new_row_idx
            row_points_mask = (sorted_z == new_row_idx)
            
            if not np.any(row_points_mask): # 如果该行没有点
                continue

            # 获取这些点在原始 points_3d 数组中的索引
            original_indices_in_row = np.where(row_points_mask)[0]
            
            # 获取这些点的X坐标
            x_coords_in_row = points_3d[row_points_mask, 0]

            # 根据X坐标排序，得到排序后的索引（相对于 original_indices_in_row 和 x_coords_in_row）
            # argsort 返回的是原始数组中元素在新排序数组中的位置的索引
            # 我们需要的是，如果 x_coords_in_row 排序后，每个元素来自原数组的哪个位置
            sorted_order_within_row = np.argsort(x_coords_in_row) # indexes into x_coords_in_row

            # 为这些排序后的点分配列号 (0, 1, 2...)
            for col_idx, order_idx in enumerate(sorted_order_within_row):
                # order_idx 是 x_coords_in_row[order_idx] 是第 col_idx 个元素
                # 它在 original_indices_in_row 中的位置也是 order_idx
                # 所以，它在最初的 points_3d 中的索引是 original_indices_in_row[order_idx]
                final_original_idx = original_indices_in_row[order_idx]
                sorted_x[final_original_idx] = col_idx
        
        return sorted_x, sorted_y, sorted_z
