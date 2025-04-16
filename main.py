from camera_handler import CameraHandler
from class_define import Box
from point_detector import PointDetector
from point_processor import PointProcessor
from point_adjuster import PointAdjuster
from utils import Visualizer, transmit_to_plc, create_tasks
from detection_pipeline import process_detections
import cv2
import logging

# manual_box_list = [
#     Box(
#         id=1,
#         row=1,
#         col=1,
#         side='left',
#         coords_3d=[-434.8, 1095.9, 150],
#         width_3d=401.2,
#         height_3d=196.3
#     ),
#     Box(
#         id=2,
#         row=1,
#         col=2,
#         side='left',
#         coords_3d=[-28.2, 1102.5, 150],
#         width_3d=419.9,
#         height_3d=203.5
#     ),
#     Box(
#         id=3,
#         row=1,
#         col=3,
#         side='left',
#         coords_3d=[377.3, 1099.9, 150],
#         width_3d=418.4,
#         height_3d=223.5
#     ),
#     Box(
#         id=4,
#         row=1,
#         col=4,
#         side='right',
#         coords_3d=[-415.5, 1100.6, 150],
#         width_3d=412.7,
#         height_3d=229.3
#     ),
#     Box(
#         id=5,
#         row=1,
#         col=5,
#         side='right',
#         coords_3d=[-8.3, 1084.1, 150],
#         width_3d=415.6,
#         height_3d=239.3
#     ),
#     Box(
#         id=6,
#         row=1,
#         col=6,
#         side='right',
#         coords_3d=[395.0, 1077.0, 150],
#         width_3d=402.7,
#         height_3d=266.5
#     )
# ]



# 配置日志
logging.basicConfig(
    filename='plc_camera_log.txt',
    level=logging.INFO,
    format='%(asctime)s: %(levelname)s: %(message)s'
)

def main():
    try:
        # 初始化相机和 PLC
        camera = CameraHandler(ip="192.168.1.30", plc_address="192.168.1.20.1.1")
        detector = PointDetector("E:\\Desktop\\car\\yolo11best.pt")
        processor = PointProcessor()
        adjuster = PointAdjuster()
        visualizer = Visualizer()

        if not camera.configure_camera() or not camera.start_plc():
            logging.critical("Failed to configure camera or PLC, exiting")
            print("Failed to configure camera or PLC, exiting")
            return

        while True:
            try:
                # 拍摄低位和高位
                color_low, depth_low, color_high, depth_high = camera.capture_images()
                if color_low is None or depth_low is None or color_high is None or depth_high is None:
                    logging.warning("Capture failed or user exited, skipping cycle")
                    print("拍摄失败或用户退出，跳过本次循环")
                    continue

                # 处理检测和任务
                tasks = process_detections(detector, processor, adjuster, visualizer, color_low, depth_low, color_high, depth_high)
                transmit_to_plc(tasks)
                logging.info("Detection and task transmission completed")

                print("按 Enter 键重新开始拍摄，或按 q/ESC 退出")
                key = cv2.waitKey(0)
                if key == ord('q') or key == CameraHandler.ESC_KEY:
                    logging.info("User requested program exit")
                    break
                elif key == CameraHandler.ENTER_KEY:
                    logging.info("Starting new capture cycle")
                    continue

            except Exception as e:
                logging.error(f"Error in capture cycle: {e}")
                print(f"循环中发生错误: {e}，继续下一次循环")
                continue

    except Exception as e:
        logging.critical(f"Fatal error: {e}")
        print(f"致命错误: {e}")
    finally:
        camera.stop()
        cv2.destroyAllWindows()
        logging.info("Program terminated, resources cleaned up")

if __name__ == "__main__":
    main()