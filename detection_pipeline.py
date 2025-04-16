import cv2
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List, Dict
import os
from datetime import datetime
import threading

from point_detector import PointDetector
from point_processor import PointProcessor
from point_adjuster import PointAdjuster
from utils import Visualizer, preprocess_point_cloud, create_boxes, assign_box_sides, create_tasks


# 配置参数
@dataclass
class FilterConfig:
    bin_width: float = 0.01
    y_range: float = 0.25
    outlier_nb_neighbors: int = 80
    outlier_std_ratio: float = 2.0


# 保存路径
SAVE_DIR = "images"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


def get_timestamp() -> str:
    """Generate timestamp string in format YYYYMMDDHHMMSS."""
    return datetime.now().strftime("%Y%m%d%H%M%S")


def save_image(image: np.ndarray, img_type: str):
    """Save image with timestamp."""
    timestamp = get_timestamp()
    filename = os.path.join(SAVE_DIR, f"{img_type}_{timestamp}.png")
    cv2.imwrite(filename, image)
    print(f"Saved image: {filename}")


def save_point_cloud_screenshot(pcd: o3d.geometry.PointCloud, pcd_type: str):
    """Save point cloud screenshot with timestamp."""
    timestamp = get_timestamp()
    filename = os.path.join(SAVE_DIR, f"{pcd_type}_{timestamp}.png")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=pcd_type, visible=False)  # Offscreen rendering
    vis.add_geometry(pcd)
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(filename, do_render=True)
    vis.destroy_window()
    print(f"Saved point cloud screenshot: {filename}")


def non_blocking_show_point_cloud(pcd: o3d.geometry.PointCloud, window_name: str):
    """Display point cloud in non-blocking mode using a separate thread."""

    def show_pcd():
        o3d.visualization.draw_geometries([pcd], window_name=window_name)

    thread = threading.Thread(target=show_pcd)
    thread.daemon = True  # Ensure thread exits when main program exits
    thread.start()


def non_blocking_show_image(image: np.ndarray, window_name: str):
    """Display image in non-blocking mode and save it."""
    save_image(image, window_name.lower().replace(" ", "_"))
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(window_name, image)
    cv2.waitKey(1)  # Non-blocking display


def filter_by_y_density_and_visualize(
        points: np.ndarray,
        colors: np.ndarray,
        config: FilterConfig = FilterConfig()
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter point cloud based on y-value density peak and remove outliers using statistical filtering.
    Save and visualize the retained point cloud with original colors in non-blocking mode.

    Args:
        points: 3D point cloud (N, 3).
        colors: Corresponding colors (N, 3).
        config: Configuration for filtering parameters.

    Returns:
        Filtered points and colors as NumPy arrays.
    """
    y_vals = points[:, 1]

    # Build y-value histogram
    hist, bin_edges = np.histogram(y_vals, bins=np.arange(y_vals.min(), y_vals.max(), config.bin_width))
    max_bin_idx = np.argmax(hist)

    # Find main peak center and set retention range
    y_center = (bin_edges[max_bin_idx] + bin_edges[max_bin_idx + 1]) / 2
    y_min, y_max = y_center - config.y_range / 2, y_center + config.y_range / 2

    print(f"Retaining points with Y values in [{y_min:.4f}, {y_max:.4f}]")

    # Initial y-range filtering
    mask = (y_vals >= y_min) & (y_vals <= y_max)
    filtered_points = points[mask]
    filtered_colors = colors[mask]

    # Create temporary point cloud for filtering
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_points)
    pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

    # Statistical outlier removal
    pcd_clean, ind = pcd.remove_statistical_outlier(
        nb_neighbors=config.outlier_nb_neighbors,
        std_ratio=config.outlier_std_ratio
    )

    print(f"Points after statistical filtering: {len(ind)} / Initial points: {len(filtered_points)}")

    # Save point cloud screenshot and display non-blocking
    save_point_cloud_screenshot(pcd_clean, "y_peak_filtered_point_cloud")
    non_blocking_show_point_cloud(pcd_clean, "Y Peak + Statistically Filtered Point Cloud (Original Colors)")

    return np.asarray(pcd_clean.points), np.asarray(pcd_clean.colors)


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
    Process point cloud detections and generate tasks for detected boxes.
    Save intermediate visualizations and ensure non-blocking display.

    Args:
        detector: PointDetector instance.
        processor: PointProcessor instance.
        adjuster: PointAdjuster instance.
        visualizer: Visualizer instance.
        color_low, depth_low, color_high, depth_high: Paths to input images.

    Returns:
        List of tasks for detected boxes.
    """
    # Process images to get point clouds
    high_points, high_colors, low_points, low_colors = detector.process_images(
        color_low, depth_low, color_high, depth_high
    )

    # Combine high and low point clouds
    all_points = np.vstack((high_points, low_points))
    all_colors = np.vstack((high_colors, low_colors))

    # Preprocess point cloud
    all_points, all_colors = preprocess_point_cloud(all_points, all_colors)

    # Visualize initial point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)
    save_point_cloud_screenshot(pcd, "initial_point_cloud")
    non_blocking_show_point_cloud(pcd, "Initial Point Cloud")

    # Filter point cloud
    all_points, all_colors = filter_by_y_density_and_visualize(all_points, all_colors)

    # Project and detect
    all_detected_info, all_proj_img, all_display_img = detector._project_and_detect(all_points, all_colors, "All")
    non_blocking_show_image(all_proj_img, "All XZ Projection")
    non_blocking_show_image(all_display_img, "All Detection Results")

    # Extract 3D centers
    all_points = np.array([box['center_3d'] for box in all_detected_info])
    indices = list(range(len(all_detected_info)))

    # Visualize points
    visualizer.visualize_points(all_points, len(all_detected_info))

    # Cluster and sort points
    points_2d = all_points[:, [0, 2]]
    x_labels, z_labels = processor.cluster_points(points_2d)
    sorted_x, sorted_z = processor.sort_labels(points_2d, x_labels, z_labels)

    # Find reference point
    second_row = np.where(sorted_z == 1)[0]
    if not len(second_row):
        raise ValueError("No points in first row")
    ref_idx = np.where(sorted_x[second_row] == 2)[0]
    if not len(ref_idx):
        raise ValueError("No points in third column of first row")
    ref_point = points_2d[second_row[ref_idx[0]]]

    # Adjust points
    grid_points, origin_x, origin_z = adjuster.generate_grid_points(ref_point)
    adjusted_points = adjuster.adjust_points(all_points, points_2d, sorted_x, sorted_z, origin_x, origin_z)
    visualizer.plot_all_2d(points_2d, grid_points, adjusted_points, sorted_x, sorted_z)

    # Create and assign boxes
    boxes = create_boxes(adjusted_points, all_detected_info, sorted_x, sorted_z, indices)
    assign_box_sides(boxes)

    # Generate tasks
    tasks = create_tasks(boxes)

    # Ensure OpenCV windows are closed
    cv2.destroyAllWindows()

    return tasks