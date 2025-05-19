import cv2
import numpy as np
import cupy as cp
from ultralytics import YOLO
from scipy.spatial import KDTree
from config import CAMERA_PARAMS, TRANSFORMATIONS, YOLO_PARAMS, GRID_PARAMS, IP_HOST_Csharp, IP_PORT_Csharp
from utils import send_detections_Csharp
import logging
import open3d as o3d
logger = logging.getLogger(__name__)

class PointDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.camera_matrix = np.array([[CAMERA_PARAMS['fx'], 0, CAMERA_PARAMS['cx']],
                                       [0, CAMERA_PARAMS['fy'], CAMERA_PARAMS['cy']],
                                       [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.array([CAMERA_PARAMS['k1'], CAMERA_PARAMS['k2'],
                                    CAMERA_PARAMS['p1'], CAMERA_PARAMS['p2'],
                                    CAMERA_PARAMS['k3']], dtype=np.float32)
        self.depth_scale = 0.001

    def _load_and_process_images(self, rgb_path, depth_path):
        if isinstance(rgb_path, str):
            color_img = cv2.imread(rgb_path)
            depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        else:
            color_img = rgb_path
            depth_img = depth_path
        if color_img is None or depth_img is None:
            raise ValueError(f"Failed to load images: {rgb_path} or {depth_path}")
        #depth_img[(depth_img > 2000) | (depth_img < 1000)] = 0
        #depth_img[(depth_img > 2500) | (depth_img < 1000)] = 0
        depth_img = depth_img.astype(np.float32)
        depth_img_norm = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX)
        depth_img_undist = cv2.medianBlur(depth_img_norm, 5)
        smooth_depth = cv2.bilateralFilter(depth_img_undist, d=9, sigmaColor=75, sigmaSpace=75)
        smooth_depth = cv2.normalize(smooth_depth, None, np.min(depth_img), np.max(depth_img), cv2.NORM_MINMAX)
        smooth_depth = smooth_depth.astype(np.uint16)

        color_img_undist = cv2.undistort(color_img, self.camera_matrix, self.dist_coeffs)
        depth_img_undist = cv2.undistort(smooth_depth, self.camera_matrix, self.dist_coeffs)
        return color_img_undist, depth_img_undist

    def process_images(self, rgb_low_path, depth_low_path, rgb_high_path, depth_high_path):
        color_low, depth_low = self._load_and_process_images(rgb_low_path, depth_low_path)
        color_high, depth_high = self._load_and_process_images(rgb_high_path, depth_high_path)
        low_points, low_colors = self._generate_point_cloud(color_low, depth_low, TRANSFORMATIONS['T2'])
        high_points, high_colors = self._generate_point_cloud(color_high, depth_high, TRANSFORMATIONS['T1'])
        return high_points, high_colors, low_points, low_colors

    def _generate_point_cloud(self, color_img, depth_img, T):
        h, w = depth_img.shape
        depth_img = cp.array(depth_img, dtype=cp.float32)
        color_img = cp.array(color_img, dtype=cp.float32)
        T = cp.array(T, dtype=cp.float32)

        u, v = cp.meshgrid(cp.arange(w), cp.arange(h))
        d = depth_img
        valid_mask = d > 0
        x_n = (u - CAMERA_PARAMS['cx']) / CAMERA_PARAMS['fx']
        y_n = (v - CAMERA_PARAMS['cy']) / CAMERA_PARAMS['fy']
        x_n, y_n, d = x_n[valid_mask], y_n[valid_mask], d[valid_mask]

        P_c = cp.vstack((x_n * d, y_n * d, d, cp.ones_like(d)))
        P_base = T @ P_c
        points = (P_base[:3, :].T) * self.depth_scale
        colors = (color_img[valid_mask] / 255.0).astype(cp.float32)
        return cp.asnumpy(points), cp.asnumpy(colors)

    def _project_and_detect(self, points, colors, label):
        W, H = 1920, 1080
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        z_min, z_max = points[:, 2].min(), points[:, 2].max()
        x_span, z_span = x_max - x_min, z_max - z_min

        padding = 0.1
        x_span_padded = x_span * (1 + padding)
        z_span_padded = z_span * (1 + padding)
        point_aspect, target_aspect = x_span_padded / z_span_padded, W / H

        if point_aspect > target_aspect:
            scale = W / x_span_padded
            scaled_z = z_span_padded * scale
            z_offset = (H - scaled_z) / 2
            x_offset = 0
        else:
            scale = H / z_span_padded
            scaled_x = x_span_padded * scale
            x_offset = (W - scaled_x) / 2
            z_offset = 0

        proj_params = {"x_min": x_min, "z_max": z_max, "scale": scale, "x_offset": x_offset, "z_offset": z_offset}
        proj_img = np.full((H, W, 3), 255, dtype=np.uint8)
        colors = (colors * 255).astype(np.uint8) if colors.max() <= 1.0 else colors
        depth_buffer = np.full((H, W), np.inf)

        radius = 3
        for idx in range(len(points)):
            X, Y, Z = points[idx]
            color = colors[idx]
            j = int((X - x_min) * scale + x_offset)
            i = int((z_max - Z) * scale + z_offset)
            if 0 <= i < H and 0 <= j < W:
                if Y < depth_buffer[i, j]:
                    cv2.circle(proj_img, (j, i), radius, color.tolist(), -1)
                    depth_buffer[i, j] = Y


        kernel = np.ones((3, 3), np.uint8)

        cv2.imshow('orgin',proj_img)

        key = cv2.waitKey(0)  # 等待按键；参数是等待毫秒数，0 表示无限等待
        if key == 27:  # 如果按下 ESC 键（ASCII 27），也可以做特殊处理
            print("ESC pressed, closing window")
        cv2.destroyAllWindows()  # 关闭所有 OpenCV 创建的窗口

        img_opened = cv2.morphologyEx(proj_img, cv2.MORPH_OPEN, kernel, iterations=1)
        img_opened = cv2.medianBlur(img_opened, 3)
        smoothed = cv2.bilateralFilter(img_opened, d=5, sigmaColor=35, sigmaSpace=20)
        laplacian = cv2.Laplacian(smoothed, cv2.CV_64F, ksize=3)
        proj_img = np.uint8(np.clip(smoothed - 0.4 * laplacian, 0, 255))

        cv2.imshow('after', proj_img)

        key = cv2.waitKey(0)  # 等待按键；参数是等待毫秒数，0 表示无限等待
        if key == 27:  # 如果按下 ESC 键（ASCII 27），也可以做特殊处理
            print("ESC pressed, closing window")
        cv2.destroyAllWindows()  # 关闭所有 OpenCV 创建的窗口

        results = self.model(proj_img, conf=YOLO_PARAMS['conf'], iou=YOLO_PARAMS['iou']) #yolo检测

        detected_boxes_info = []
        display_img = proj_img.copy()
        xz_points = points[:, [0, 2]]
        kdtree = KDTree(xz_points)

        for result in results:
            boxes = result.boxes
            if len(boxes) == 0:
                continue
            xyxyn = boxes.xyxyn.cpu().numpy()
            xywhn = boxes.xywhn.cpu().numpy()
            for box_xywhn, box_xyxyn in zip(xywhn, xyxyn):
                cx_pixel = int(box_xywhn[0] * W)
                cy_pixel = int(box_xywhn[1] * H)
                # if cy_pixel < 400 :
                #     continue
                w_pixel = int(box_xywhn[2] * W)
                h_pixel = int(box_xywhn[3] * H)
                x1 = cx_pixel - w_pixel // 2
                y1 = cy_pixel - h_pixel // 2
                x2 = cx_pixel + w_pixel // 2
                y2 = cy_pixel + h_pixel // 2

                cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.circle(display_img, (cx_pixel, cy_pixel), 3, (0, 255, 0), -1)
                label = f"({cx_pixel}, {cy_pixel})"
                cv2.putText(display_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                X_3d = (cx_pixel - proj_params["x_offset"]) / proj_params["scale"] + proj_params["x_min"]
                Z_3d = proj_params["z_max"] - (cy_pixel - proj_params["z_offset"]) / proj_params["scale"]

                k = 20
                distance, idx = kdtree.query([[X_3d, Z_3d]], k=k)
                neighbor_points = points[idx[0]]

                topk_y = 3
                order_y_desc = np.argsort(neighbor_points[:, 1])[::-1]
                topk_indices = order_y_desc[:topk_y]
                topk_points = neighbor_points[topk_indices]
                mean_point_k = np.mean(topk_points, axis=0)


                x1_norm, y1_norm, x2_norm, y2_norm = box_xyxyn
                x1_pixel, y1_pixel = int(x1_norm * W), int(y1_norm * H)
                x2_pixel, y2_pixel = int(x2_norm * W), int(y2_norm * H)

                X1_3d = (x1_pixel - proj_params["x_offset"]) / proj_params["scale"] + proj_params["x_min"]
                Z1_3d = proj_params["z_max"] - (y1_pixel - proj_params["z_offset"]) / proj_params["scale"]
                X2_3d = (x2_pixel - proj_params["x_offset"]) / proj_params["scale"] + proj_params["x_min"]
                Z2_3d = proj_params["z_max"] - (y2_pixel - proj_params["z_offset"]) / proj_params["scale"]

                threshold_ratio = 0.2  # 允许误差
                width_3d = abs(X2_3d - X1_3d)
                height_3d = abs(Z2_3d - Z1_3d)

                # 判断宽度和高度是否接近标准尺寸
                width_ok = abs(width_3d - GRID_PARAMS['x_spacing']) / GRID_PARAMS['x_spacing'] < threshold_ratio
                height_ok = abs(height_3d - GRID_PARAMS['z_spacing']) / GRID_PARAMS['z_spacing'] < threshold_ratio

                # 记录筛选结果
                if width_ok and height_ok:
                    logger.info(f"箱子通过阈值筛选 - 中心点: {mean_point_k}, 宽度: {width_3d:.2f}, 高度: {height_3d:.2f}")
                    detected_boxes_info.append({
                        'center_3d': mean_point_k,
                        'width_3d': width_3d,
                        'height_3d': height_3d
                    })
                else:
                    if not width_ok:
                        logger.warning(f"箱子宽度不符合要求 - 中心点: {mean_point_k}, 实际宽度: {width_3d:.2f}, 标准宽度: {GRID_PARAMS['x_spacing']:.2f}")
                    if not height_ok:
                        logger.warning(f"箱子高度不符合要求 - 中心点: {mean_point_k}, 实际高度: {height_3d:.2f}, 标准高度: {GRID_PARAMS['z_spacing']:.2f}")
            box_account = len(detected_boxes_info)
        #send_detections_Csharp(results, detected_boxes_info, proj_img,IP_HOST_Csharp,IP_PORT_Csharp)
        return detected_boxes_info, proj_img, display_img, box_account