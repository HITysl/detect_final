import os
import time
import pyads
from ctypes import sizeof
from pyorbbecsdk import *
from utils_orb import frame_to_bgr_image
import cv2
import numpy as np
import logging
import sys

# 配置日志
logging.basicConfig(
    filename='plc_camera_log.txt',
    level=logging.INFO,
    format='%(asctime)s: %(levelname)s: %(message)s'
)

# 全局标志位
last_low_start_flag_value = False
last_high_start_flag_value = False
low_pos_done_flag = False
high_pos_done_flag = False

def retry(func, max_retries=3, retry_interval=5, error_msg="Operation failed"):
    """
    重试装饰器，用于相机初始化和 PLC 连接。
    """
    for attempt in range(1, max_retries + 1):
        try:
            result = func()
            logging.info(f"{func.__name__} succeeded on attempt {attempt}")
            return result
        except Exception as e:
            logging.error(f"{error_msg} on attempt {attempt}/{max_retries}: {e}")
            if attempt == max_retries:
                logging.critical(f"{error_msg} after {max_retries} attempts, exiting")
                print(f"{error_msg} after {max_retries} attempts, exiting")
                sys.exit(1)
            time.sleep(retry_interval)
    return False

class CameraHandler:
    ESC_KEY = 27
    ENTER_KEY = 13

    def __init__(self, ip="192.168.1.30", port=8090, plc_address="192.168.1.20.1.1", plc_port=851):
        self.ctx = Context()
        self.ip = ip
        self.port = port
        self.pipeline = None
        self.plc = pyads.Connection(plc_address, plc_port)
        self.low_notify_handle = None
        self.low_user_handle = None
        self.high_notify_handle = None
        self.high_user_handle = None

    def configure_camera(self):
        def try_configure():
            device = self.ctx.create_net_device(self.ip, self.port)
            config = Config()
            self.pipeline = Pipeline(device)

            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            color_profile = profile_list.get_video_stream_profile(1920, 1080, OBFormat.MJPG, 15)
            config.enable_stream(color_profile)

            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            depth_profile = profile_list.get_video_stream_profile(1024, 1024, OBFormat.Y16, 15)
            config.enable_stream(depth_profile)

            config.set_align_mode(OBAlignMode.HW_MODE)
            self.pipeline.start(config)
            return True

        return retry(
            try_configure,
            max_retries=3,
            retry_interval=5,
            error_msg="Failed to configure camera"
        )

    def start_plc(self):
        global last_low_start_flag_value, last_high_start_flag_value
        def try_start_plc():
            self.plc.open()
            logging.info("PLC connected")

            # 初始化变量值以规避首次回调
            last_low_start_flag_value = self.plc.read_by_name("Camera.bInspection_PosLow_Start", pyads.PLCTYPE_BOOL)
            last_high_start_flag_value = self.plc.read_by_name("Camera.bInspection_PosHigh_Start", pyads.PLCTYPE_BOOL)

            # 定义回调函数
            def camera_low_pos_start_callback(notification, data):
                global low_pos_done_flag, last_low_start_flag_value
                try:
                    data_type = position_low_start_tag[data]
                    handle, timestamp, value = self.plc.parse_notification(notification, data_type)
                    logging.info(f"Low position detection signal received: {value} at {timestamp}")
                    print(f"低位检测信号收到: {value} at {timestamp}")
                    if value != last_low_start_flag_value:
                        low_pos_done_flag = True
                        last_low_start_flag_value = value
                except pyads.ADSError as e:
                    logging.error(f"Low pos callback error: {e}")

            def camera_high_pos_start_callback(notification, data):
                global high_pos_done_flag, last_high_start_flag_value
                try:
                    data_type = position_high_start_tag[data]
                    handle, timestamp, value = self.plc.parse_notification(notification, data_type)
                    logging.info(f"High position detection signal received: {value} at {timestamp}")
                    print(f"高位检测信号收到: {value} at {timestamp}")
                    if value != last_high_start_flag_value:
                        high_pos_done_flag = True
                        last_high_start_flag_value = value
                except pyads.ADSError as e:
                    logging.error(f"High pos callback error: {e}")

            # 注册低位通知
            position_low_start_tag = {"Camera.bInspection_PosLow_Start": pyads.PLCTYPE_BOOL}
            attr = pyads.NotificationAttrib(sizeof(pyads.PLCTYPE_BOOL))
            self.low_notify_handle, self.low_user_handle = self.plc.add_device_notification(
                'Camera.bInspection_PosLow_Start', attr, camera_low_pos_start_callback)

            # 注册高位通知
            position_high_start_tag = {"Camera.bInspection_PosHigh_Start": pyads.PLCTYPE_BOOL}
            attr = pyads.NotificationAttrib(sizeof(pyads.PLCTYPE_BOOL))
            self.high_notify_handle, self.high_user_handle = self.plc.add_device_notification(
                'Camera.bInspection_PosHigh_Start', attr, camera_high_pos_start_callback)

            return True

        return retry(
            try_start_plc,
            max_retries=3,
            retry_interval=5,
            error_msg="Failed to start PLC"
        )

    def get_frames(self):
        if not self.pipeline:
            return None, None, None

        frames = self.pipeline.wait_for_frames(100)
        if frames is None:
            return None, None, None

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if color_frame is None or depth_frame is None:
            return None, None, None

        color_image = frame_to_bgr_image(color_frame)
        depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
        depth_data_out = depth_data.reshape((depth_frame.get_height(), depth_frame.get_width()))
        scale = depth_frame.get_depth_scale()
        depth_data = depth_data_out.astype(np.float32) * scale
        depth_image = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        return color_image, depth_image, depth_data_out

    def capture_images(self, timeout=30):
        global low_pos_done_flag, high_pos_done_flag
        # 确保 images 文件夹存在
        save_dir = "images"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 低位检测
        logging.info("Starting low position detection")
        print("开始低位检测")
        start_time = time.time()
        while True:
            color_image, depth_image, depth_data = self.get_frames()
            if color_image is None or depth_image is None:
                continue
            DISPLAY_WIDTH = 1280
            DISPLAY_HEIGHT = 800

            resized_color_image = cv2.resize(color_image, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            cv2.imshow("Low Preview", resized_color_image)
            key = cv2.waitKey(1)

            # 检查低位信号
            if low_pos_done_flag:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                color_filename = os.path.join(save_dir, f"Low_colour_{timestamp}.png")
                depth_filename = os.path.join(save_dir, f"Low_depth_{timestamp}.png")

                print(f"低位拍摄完成，保存为 {color_filename} 和 {depth_filename}")
                logging.info(f"Low position capture completed, saved as {color_filename} and {depth_filename}")
                cv2.destroyWindow("Low Preview")
                cv2.imwrite(color_filename, color_image)
                cv2.imwrite(depth_filename, depth_data)

                # 向 PLC 写入低位完成信号
                try:
                    self.plc.write_by_name("Camera.bInspection_PosLow_Done", True, pyads.PLCTYPE_BOOL)
                    logging.info("Low position inspection done")
                    low_pos_done_flag = False  # 重置标志位
                except pyads.ADSError as e:
                    logging.error(f"Failed to write low pos done: {e}")

                color_low = color_image
                depth_low = depth_data
                break

            # 检查超时
            if time.time() - start_time > timeout:
                print("低位拍摄超时")
                logging.warning(f"Low position capture timed out after {timeout} seconds")
                cv2.destroyWindow("Low Preview")
                return None, None, None, None

            # 检查退出键
            if key == ord('q') or key == self.ESC_KEY:
                print("用户退出低位拍摄")
                logging.info("User exited low position capture")
                cv2.destroyWindow("Low Preview")
                return None, None, None, None

        # 高位检测
        logging.info("Starting high position detection")
        print("开始高位检测")
        start_time = time.time()
        while True:
            color_image, depth_image, depth_data = self.get_frames()
            if color_image is None or depth_image is None:
                continue
            resized_color_image = cv2.resize(color_image, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            cv2.imshow("High Preview", resized_color_image)
            key = cv2.waitKey(1)

            if high_pos_done_flag:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                color_filename = os.path.join(save_dir, f"High_colour_{timestamp}.png")
                depth_filename = os.path.join(save_dir, f"High_depth_{timestamp}.png")

                print(f"高位拍摄完成，保存为 {color_filename} 和 {depth_filename}")
                logging.info(f"High position capture completed, saved as {color_filename} and {depth_filename}")
                cv2.destroyWindow("High Preview")
                cv2.imwrite(color_filename, color_image)
                cv2.imwrite(depth_filename, depth_data)

                # 向 PLC 写入高位完成信号
                try:
                    self.plc.write_by_name("Camera.bInspection_PosHigh_Done", True, pyads.PLCTYPE_BOOL)
                    logging.info("High position inspection done")
                    high_pos_done_flag = False  # 重置标志位
                except pyads.ADSError as e:
                    logging.error(f"Failed to write high pos done: {e}")

                return color_low, depth_low, color_image, depth_data

            # 检查超时
            if time.time() - start_time > timeout:
                print("高位拍摄超时")
                logging.warning(f"High position capture timed out after {timeout} seconds")
                cv2.destroyWindow("High Preview")
                return None, None, None, None

            # 检查退出键
            if key == ord('q') or key == self.ESC_KEY:
                print("用户退出高位拍摄")
                logging.info("User exited high position capture")
                cv2.destroyWindow("High Preview")
                return None, None, None, None

    def stop(self):
        # 清理相机资源
        if self.pipeline:
            self.pipeline.stop()
            logging.info("Camera pipeline stopped")
        # 清理 PLC 资源
        if self.plc:
            try:
                if self.low_notify_handle and self.low_user_handle:
                    self.plc.del_device_notification(self.low_notify_handle, self.low_user_handle)
                if self.high_notify_handle and self.high_user_handle:
                    self.plc.del_device_notification(self.high_notify_handle, self.high_user_handle)
                self.plc.close()
                logging.info("PLC connection closed")
            except pyads.ADSError as e:
                logging.error(f"PLC cleanup error: {e}")