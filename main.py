# manual_box_list = [
#     # 上一层 row = 1
#     Box(id=1, row=1, col=1, side='left',  coords_3d=[-379.9, 1100, 454.5], width_3d=387.2, height_3d=317.9),
#     Box(id=2, row=1, col=4, side='right', coords_3d=[-372.2, 1100, 458.1], width_3d=413.5, height_3d=315.5),
#     Box(id=3, row=1, col=2, side='left',  coords_3d=[23.7,   1100, 455.7], width_3d=423.1, height_3d=320.3),
#     Box(id=4, row=1, col=5, side='right', coords_3d=[36.8,   1100, 459.4], width_3d=413.5, height_3d=317.9),
#     Box(id=5, row=1, col=3, side='left',  coords_3d=[426.0,  1100, 456.9], width_3d=411.1, height_3d=320.3),
#     Box(id=6, row=1, col=6, side='right', coords_3d=[441.6,  1100, 460.6], width_3d=406.3, height_3d=322.7),
#
#     # 下一层 row = 2，z - 300
#     Box(id=7,  row=2, col=1, side='left',  coords_3d=[-379.9, 1100, 154.5], width_3d=387.2, height_3d=317.9),
#     Box(id=8,  row=2, col=4, side='right', coords_3d=[-372.2, 1100, 158.1], width_3d=413.5, height_3d=315.5),
#     Box(id=9,  row=2, col=2, side='left',  coords_3d=[23.7,   1100, 155.7], width_3d=423.1, height_3d=320.3),
#     Box(id=10, row=2, col=5, side='right', coords_3d=[36.8,   1100, 159.4], width_3d=413.5, height_3d=317.9),
#     Box(id=11, row=2, col=3, side='left',  coords_3d=[426.0,  1100, 156.9], width_3d=411.1, height_3d=320.3),
#     Box(id=12, row=2, col=6, side='right', coords_3d=[441.6,  1100, 160.6], width_3d=406.3, height_3d=322.7)
# ]

from camera_handler import CameraHandler, PLCConnectionLost
from class_define import Box, Tasks
from point_detector import PointDetector
from point_processor import PointProcessor
from point_adjuster import PointAdjuster
from utils import Visualizer, transmit_to_plc, create_tasks
from detection_pipeline import process_detections
import cv2
import logging
from config import IP_CAMERA, IP_PLC, MODEL

# 配置日志
logging.basicConfig(
    filename='plc_camera_log.txt',
    level=logging.INFO,
    format='%(asctime)s: %(levelname)s: %(message)s'
)

def main():
    """
    遇到 PLC 掉线或其它异常时，自动释放资源并从头开始运行。
    """
    while True:
        camera = None  # 先占位，方便 finally 中引用
        try:
            # ----------------- 以下代码与原逻辑保持一致 -----------------
            camera = CameraHandler(ip=IP_CAMERA, plc_address=IP_PLC)
            detector = PointDetector(MODEL)
            processor = PointProcessor()
            adjuster = PointAdjuster()
            visualizer = Visualizer(enable_display=False)

            if not camera.configure_camera() or not camera.start_plc():
                logging.critical("Failed to configure camera or PLC, exiting")
                print("Failed to configure camera or PLC, exiting")
                raise RuntimeError("Init failed")

            while True:
                try:
                    # 拍摄低位和高位
                    color_low, depth_low, color_high, depth_high = camera.capture_images()
                    if color_low is None or depth_low is None or color_high is None or depth_high is None:
                        logging.warning("Capture failed or user exited, skipping cycle")
                        print("拍摄失败或用户退出，跳过本次循环")
                        continue
                    # color_low='E:\\Desktop\\detect_final\\images\\Low_colour_20250427_164625.png'
                    # depth_low = 'E:\\Desktop\\detect_final\\images\\Low_depth_20250427_164625.png'
                    # color_high = 'E:\\Desktop\\detect_final\\images\\Low_colour_20250427_164625.png'
                    # depth_high = 'E:\\Desktop\\detect_final\\images\\Low_depth_20250427_164625.png'
                    # 处理检测和任务
                    tasks = process_detections(detector, processor, adjuster, visualizer, color_low, depth_low, color_high, depth_high)
                    # tasks = Tasks(manual_box_list, total_rows=2, total_cols=6)
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

                except PLCConnectionLost:
                    # 让外层 while 重新初始化
                    raise
                except Exception as e:
                    logging.error(f"Error in capture cycle: {e}")
                    print(f"循环中发生错误: {e}，继续下一次循环")
                    continue
            # ----------------- 原逻辑结束 -----------------
            break  # 用户正常退出时跳出外层 while

        except PLCConnectionLost:
            logging.warning("PLC disconnected, restarting main()")
            print("PLC 断线，正在重新启动 …")
            # 直接继续 while True，即重新初始化
            continue
        except Exception as e:
            logging.critical(f"Fatal error: {e}")
            print(f"致命错误: {e}，程序将重新启动 …")
            continue
        finally:
            if camera is not None:
                camera.stop()
            cv2.destroyAllWindows()
            logging.info("Resources cleaned up")

if __name__ == "__main__":
    main()
