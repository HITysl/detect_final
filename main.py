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

import gc
import sys
import time
import traceback

from camera_handler import CameraHandler, PLCConnectionLost, CameraConnectionLost
from class_define import Box, Tasks
from point_detector import PointDetector
from point_processor import PointProcessor
from point_adjuster import PointAdjuster
from utils import Visualizer, transmit_to_plc, create_tasks
from detection_pipeline import process_detections
import cv2
import logging
from config import IP_CAMERA, IP_PLC, MODEL
from logging.handlers import RotatingFileHandler
# 配置日志
logging.basicConfig(
    filename='plc_camera_log.txt',
    level=logging.INFO,
    format='%(asctime)s: %(levelname)s: [%(name)s]: %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """
    遇到 PLC 掉线或其它异常时，自动释放资源并从头开始运行。
    """
    MAX_RETRIES = 500
    retry_count = 0


    while True:
        camera = None  # 先占位，方便 finally 中引用
        try:
            retry_count = 0
            #camera = CameraHandler(ip=IP_CAMERA, plc_address=IP_PLC)
            detector = PointDetector(MODEL)
            processor = PointProcessor()
            adjuster = PointAdjuster()
            visualizer = Visualizer(enable_display=False)

            # if not camera.start_plc() or not camera.configure_camera() :
            #     logger.critical("Failed to configure camera or PLC, exiting")
            #     print("Failed to configure camera or PLC, exiting")
            #     raise RuntimeError("Init failed")

            while True:
                try:
                    # 拍摄低位和高位
                    # color_low, depth_low, color_high, depth_high = camera.capture_images()
                    # if color_low is None or depth_low is None or color_high is None or depth_high is None:
                    #     logger.warning("Capture failed or user exited, skipping cycle")
                    #     print("拍摄失败或用户退出，跳过本次循环")
                    #     continue
                    # 处理检测和任务
                    color_low = 'E:\\Desktop\\detect_final\\images\\Low_colour_20250519_093724.png'
                    depth_low = 'E:\\Desktop\\detect_final\\images\\Low_depth_20250519_093724.png'
                    color_high = 'E:\\Desktop\\detect_final\\images\\High_colour_20250519_093724.png'
                    depth_high = 'E:\\Desktop\\detect_final\\images\\High_depth_20250519_093724.png'
                    tasks = process_detections(detector, processor, adjuster, visualizer, color_low, depth_low, color_high, depth_high)
                    # tasks = Tasks(manual_box_list, total_rows=2, total_cols=6)
                    transmit_to_plc(tasks)
                    logger.info("Detection and task transmission completed")

                    print("按 Enter 键重新开始拍摄，或按 q/ESC 退出")
                    key = cv2.waitKey(0)
                    if key == ord('q') or key == CameraHandler.ESC_KEY:
                        logger.info("User requested program exit")
                        break
                    elif key == CameraHandler.ENTER_KEY:
                        logger.info("Starting new capture cycle")
                        continue

                except (PLCConnectionLost, CameraConnectionLost):
                    logger.warning('PLC or CAMERA disconnected')
                    raise
                except Exception as e:
                    logger.exception(f"Error in capture cycle: {e}")
                    print(f"循环中发生错误: {e}，继续下一次循环")
                    traceback.print_exc()
                    continue
            # ----------------- 原逻辑结束 -----------------
            break  # 用户正常退出时跳出外层 while

        except (PLCConnectionLost, CameraConnectionLost) as e:
            retry_count += 1
            retry_count += 1
            if retry_count >= MAX_RETRIES:
                logger.critical("Max retries reached for PLC connection, exiting")
                print(f"PLC重连失败，达到最大重试次数（{MAX_RETRIES}次），程序退出，请检查网络或硬件")
                sys.exit(1)
            logger.warning(f"PLC disconnected, restarting main() (Attempt {retry_count}/{MAX_RETRIES})")
            print(f"PLC断线，正在重新启动（第{retry_count}/{MAX_RETRIES}次）...")
            time.sleep(1)  # 添加1秒延迟，避免快速重试
            continue

        except Exception as e:
            retry_count += 1
            if retry_count >= MAX_RETRIES:
                logger.critical(f"Max retries reached: {e}, exiting")
                print(f"程序错误，达到最大重试次数（{MAX_RETRIES}次）: {e}，请联系技术支持")
                sys.exit(1)
            logger.critical(f"Fatal error: {e}")
            logger.exception(f"Fatal error: {e}")
            print(f"致命错误: {e}，正在重新启动（第{retry_count}/{MAX_RETRIES}次）...")
            traceback.print_exc()
            time.sleep(1)  # 添加1秒延迟
            continue
        finally:
            if camera is not None:
                try:
                    camera.stop()
                except Exception as e:
                    logger.error(f"Failed to stop camera: {e}")
            camera = None
            gc.collect()
            cv2.destroyAllWindows()
            print("All resources cleaned up")
            logger.info("All resources cleaned up")
if __name__ == "__main__":
    main()
