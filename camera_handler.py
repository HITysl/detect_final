import os
import time
import threading            # 新增
import pyads
from ctypes import sizeof
from pyorbbecsdk import *
from utils_orb import frame_to_bgr_image
import cv2
import numpy as np
import logging
import sys

ENTER_KEY = 13
ESC_KEY   = 27               # 保留原有常量

# 配置日志
logger = logging.getLogger(__name__)
# 全局标志位
last_low_start_flag_value = False
last_high_start_flag_value = False
low_pos_done_flag = False
high_pos_done_flag = False

# ---------------- 新增：掉线异常 -----------------
class PLCConnectionLost(RuntimeError):
    """心跳失败后抛出的异常（由主程序捕获并重启）"""
    pass

class CameraConnectionLost(RuntimeError):
    """相机连接丢失异常"""
    pass
# ------------------------------------------------

def retry(func, retry_interval=1, max_retries=20, error_msg="Operation failed"):
    attempt = 1
    last_error_time = time.time()
    ERROR_CHECK_INTERVAL = 60
    while attempt <= max_retries:
        try:
            result = func()
            logger.info(f"{func.__name__} succeeded on attempt {attempt}")
            return result
        except (ConnectionError, pyads.ADSError, RuntimeError) as e:
            logger.error(f"{error_msg} on attempt {attempt}: {e}")
            current_time = time.time()
            if current_time - last_error_time >= ERROR_CHECK_INTERVAL:
                logger.error(f"{error_msg} after {attempt} attempts, continuing to retry")
                print(f"{error_msg} after {attempt} attempts, continuing to retry")
                last_error_time = current_time
            attempt += 1
            time.sleep(retry_interval)
    logger.critical(f"{error_msg}: Max retries ({max_retries}) exceeded")
    raise RuntimeError(f"{error_msg}: Max retries ({max_retries}) exceeded")

class CameraHandler:
    ESC_KEY = 27
    ENTER_KEY = 13

    def __init__(self, ip="192.168.1.30", port=8090, plc_address="192.168.1.20.1.1", plc_port=851):
        self.ctx = None
        self.ip = ip
        self.port = port
        self.pipeline = None
        self.plc = pyads.Connection(plc_address, plc_port)
        self.low_notify_handle = None
        self.low_user_handle = None
        self.high_notify_handle = None
        self.high_user_handle = None
        self.ctx = Context()
        # -------- 新增：心跳相关 ---------
        self._hb_thread = None
        self._hb_stop_event = threading.Event()
        self.plc_connection_lost = False         # 掉线标志
        self.heartbeat_tag = "Camera.bCameraHeartBeat" # 请在 PLC 中新建 BOOL 变量
        self.heartbeat_interval = 3
        self.disconnect_timeout = 30 # 秒
        # ---------------------------------

    # -------- 新增：启动心跳线程 ----------
    def _start_heartbeat(self):
        def _heartbeat_loop():
            while not self._hb_stop_event.is_set():
                try:
                    # 向 PLC 写 TRUE；写失败即认为掉线
                    self.plc.write_by_name(self.heartbeat_tag, True, pyads.PLCTYPE_BOOL)
                except pyads.ADSError as e:
                    logger.error(f"Heartbeat write failed: {e}")
                    self.plc_connection_lost = True
                    break
                time.sleep(self.heartbeat_interval)

        self._hb_thread = threading.Thread(target=_heartbeat_loop, daemon=True)
        self._hb_thread.start()
    # -------------------------------------

    def configure_camera(self):
        def try_configure():
            if self.pipeline:
                try:
                    self.pipeline.stop()
                    self.pipeline = None
                    logger.info("Existing pipeline stopped before reconfiguration")
                except Exception as e:
                    logger.error(f"Failed to stop existing pipeline: {e}")

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
            error_msg="Failed to configure camera"
        )

    def start_plc(self):
        global last_low_start_flag_value, last_high_start_flag_value
        def try_start_plc():
            self.plc.open()
            logger.info("PLC connected")

            # 初始化变量值以规避首次回调
            last_low_start_flag_value = self.plc.read_by_name("Camera.bInspection_PosLow_Start", pyads.PLCTYPE_BOOL)
            last_high_start_flag_value = self.plc.read_by_name("Camera.bInspection_PosHigh_Start", pyads.PLCTYPE_BOOL)

            # 定义回调函数
            def camera_low_pos_start_callback(notification, data):
                global low_pos_done_flag, last_low_start_flag_value
                try:
                    data_type = position_low_start_tag[data]
                    handle, timestamp, value = self.plc.parse_notification(notification, data_type)
                    logger.info(f"Low position detection signal received: {value} at {timestamp}")
                    print(f"低位检测信号收到: {value} at {timestamp}")
                    if value is True and last_low_start_flag_value is False:
                        low_pos_done_flag = True
                        logger.info("Low position signal changed from False to True, triggering capture")
                        print("低位信号从 False 变为 True，触发拍摄")
                    last_low_start_flag_value = value
                except Exception as e:
                    logger.error(f"Low pos callback error: {type(e).__name__}: {e}")

            def camera_high_pos_start_callback(notification, data):
                global high_pos_done_flag, last_high_start_flag_value
                try:
                    data_type = position_high_start_tag[data]
                    handle, timestamp, value = self.plc.parse_notification(notification, data_type)
                    logger.info(f"High position detection signal received: {value} at {timestamp}")
                    print(f"高位检测信号收到: {value} at {timestamp}")
                    if value is True and last_high_start_flag_value is False:
                        high_pos_done_flag = True
                        logger.info("High position signal changed from False to True, triggering capture")
                        print("高位信号从 False 变为 True，触发拍摄")
                    last_high_start_flag_value = value
                except pyads.ADSError as e:
                    logger.error(f"High pos callback error: {e}")

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

            # ---------- 新增：启动心跳 ----------
            self._start_heartbeat()
            # ----------------------------------

            return True

        return retry(
            try_start_plc,
            error_msg="Failed to start PLC"
        )

    def get_frames(self):
        if not self.pipeline:
            logger.error("Pipeline not initialized, raising CameraConnectionLost")
            raise CameraConnectionLost("Camera pipeline not initialized")
        first_failure_time = None
        while True:
            frames = self.pipeline.wait_for_frames(100)
            current_time = time.time()
            if frames is None:
                if first_failure_time is None:
                    first_failure_time = current_time
                    logger.warning("Failed to get frames, starting disconnect timer")
                elif current_time - first_failure_time >= self.disconnect_timeout:
                    logger.error(f"Camera connection lost: no frames for {self.disconnect_timeout} seconds")
                    raise CameraConnectionLost(f"No frames received for {self.disconnect_timeout} seconds")
                time.sleep(0.1)
                continue
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if color_frame is None or depth_frame is None:
                if first_failure_time is None:
                    first_failure_time = current_time
                    logger.warning("Invalid frame data, starting disconnect timer")
                elif current_time - first_failure_time >= self.disconnect_timeout:
                    logger.error(f"Camera connection lost: invalid frames for {self.disconnect_timeout} seconds")
                    raise CameraConnectionLost(f"Invalid frame data for {self.disconnect_timeout} seconds")
                time.sleep(0.1)
                continue
            first_failure_time = None
            color_image = frame_to_bgr_image(color_frame)
            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            depth_data_out = depth_data.reshape((depth_frame.get_height(), depth_frame.get_width()))
            scale = depth_frame.get_depth_scale()
            depth_data = depth_data_out.astype(np.float32) * scale
            depth_image = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            return color_image, depth_image, depth_data_out

    # def get_frames(self):
    #     if not self.pipeline:
    #         logger.error("Pipeline not initialized, raising CameraConnectionLost")
    #         raise CameraConnectionLost("Camera pipeline not initialized")
    #
    #     frames = self.pipeline.wait_for_frames(100)
    #     if frames is None:
    #         return None, None, None
    #
    #     color_frame = frames.get_color_frame()
    #     depth_frame = frames.get_depth_frame()
    #     if color_frame is None or depth_frame is None:
    #         return None, None, None
    #
    #     color_image = frame_to_bgr_image(color_frame)
    #     depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
    #     depth_data_out = depth_data.reshape((depth_frame.get_height(), depth_frame.get_width()))
    #     scale = depth_frame.get_depth_scale()
    #     depth_data = depth_data_out.astype(np.float32) * scale
    #     depth_image = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    #
    #     return color_image, depth_image, depth_data_out

    def capture_images(self, timeout=100):
        global low_pos_done_flag, high_pos_done_flag
        if self.plc_connection_lost:
            raise PLCConnectionLost("PLC connection lost")
        save_dir = "images"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        logger.info("Starting low position detection")
        print("开始低位检测")
        start_time = time.time()
        last_warning_time = start_time
        WARNING_CHECK_INTERVAL = 60
        DISPLAY_WIDTH = 1280
        DISPLAY_HEIGHT = 800
        while True:
            try:
                if self.plc_connection_lost:
                    raise PLCConnectionLost("PLC connection lost")
                color_image_low, depth_image_low, depth_data_low = self.get_frames()
                if color_image_low is None or depth_image_low is None:
                    warning_image = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
                    cv2.putText(warning_image, "Camera Disconnected", (50, DISPLAY_HEIGHT//2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow("Low Preview", warning_image)
                    cv2.waitKey(1)
                    logger.warning("Received None frames, continuing")
                    continue
                logger.debug("Low position frame acquired successfully")
                resized_color_image = cv2.resize(color_image_low, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
                cv2.imshow("Low Preview", resized_color_image)
                key = cv2.waitKey(1)
                if low_pos_done_flag or key == ENTER_KEY:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    color_filename = os.path.join(save_dir, f"Low_colour_{timestamp}.png")
                    depth_filename = os.path.join(save_dir, f"Low_depth_{timestamp}.png")
                    print(f"低位拍摄完成，保存为 {color_filename} 和 {depth_filename}")
                    logger.info(f"Low position capture completed, saved as {color_filename} and {depth_filename}")
                    cv2.destroyWindow("Low Preview")
                    cv2.imwrite(color_filename, color_image_low)
                    cv2.imwrite(depth_filename, depth_data_low)
                    try:
                        self.plc.write_by_name("Camera.bInspection_PosLow_Done", True, pyads.PLCTYPE_BOOL)
                        logger.info("Low position inspection done")
                        low_pos_done_flag = False
                    except pyads.ADSError as e:
                        logger.error(f"Failed to write low pos done: {e}")
                    break
                current_time = time.time()
                if current_time - last_warning_time >= WARNING_CHECK_INTERVAL:
                    logger.warning(f"Still waiting for low position signal after {int(current_time - start_time)} seconds")
                    print(f"仍在等待低位信号，已等待 {int(current_time - start_time)} 秒")
                    last_warning_time = current_time
                if key == ord('q') or key == self.ESC_KEY:
                    print("用户退出低位拍摄")
                    logger.info("User exited low position capture")
                    cv2.destroyWindow("Low Preview")
                    return None, None, None, None
            except CameraConnectionLost as e:
                logger.error(f"Camera connection lost during low position capture: {e}")
                print("相机断连，低位拍摄失败，正在重启...")
                raise
        logger.info("Starting high position detection")
        print("开始高位检测")
        start_time = time.time()
        last_warning_time = start_time
        while True:
            try:
                if self.plc_connection_lost:
                    raise PLCConnectionLost("PLC connection lost")
                color_image_high, depth_image_high, depth_data_high = self.get_frames()
                if color_image_high is None or depth_image_high is None:
                    warning_image = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
                    cv2.putText(warning_image, "Camera Disconnected", (50, DISPLAY_HEIGHT//2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow("High Preview", warning_image)
                    cv2.waitKey(1)
                    logger.warning("Received None frames, continuing")
                    continue
                logger.debug("High position frame acquired successfully")
                resized_color_image = cv2.resize(color_image_high, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
                cv2.imshow("High Preview", resized_color_image)
                key = cv2.waitKey(1)
                if high_pos_done_flag or key == ENTER_KEY:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    color_filename = os.path.join(save_dir, f"High_colour_{timestamp}.png")
                    depth_filename = os.path.join(save_dir, f"High_depth_{timestamp}.png")
                    print(f"高位拍摄完成，保存为 {color_filename} 和 {depth_filename}")
                    logger.info(f"High position capture completed, saved as {color_filename} and {depth_filename}")
                    cv2.destroyWindow("High Preview")
                    cv2.imwrite(color_filename, color_image_high)
                    cv2.imwrite(depth_filename, depth_data_high)
                    try:
                        self.plc.write_by_name("Camera.bInspection_PosHigh_Done", True, pyads.PLCTYPE_BOOL)
                        logger.info("High position inspection done")
                        high_pos_done_flag = False
                    except pyads.ADSError as e:
                        logger.error(f"Failed to write high pos done: {e}")
                    return color_image_low, depth_data_low, color_image_high, depth_data_high
                current_time = time.time()
                if current_time - last_warning_time >= WARNING_CHECK_INTERVAL:
                    logger.warning(f"Still waiting for high position signal after {int(current_time - start_time)} seconds")
                    print(f"仍在等待高位信号，已等待 {int(current_time - start_time)} 秒")
                    last_warning_time = current_time
                if key == ord('q') or key == self.ESC_KEY:
                    print("用户退出高位拍摄")
                    logger.info("User exited high position capture")
                    cv2.destroyWindow("High Preview")
                    return None, None, None, None
            except CameraConnectionLost as e:
                logger.error(f"Camera connection lost during high position capture: {e}")
                print('相机断连，高位拍摄失败，正在重启.')
                raise

    def stop(self):
        # -------- 新增：停止心跳 ----------
        self._hb_stop_event.set()
        if self._hb_thread and self._hb_thread.is_alive():
            self._hb_thread.join()
        # ----------------------------------

        # 清理相机资源
        if self.pipeline:
            self.pipeline.stop()
            logger.info("Camera pipeline stopped")
            self.pipeline = None
        if self.ctx:
            try:
                self.ctx = None  # 替换为 SDK 提供的销毁方法（如果有）
                logger.info("Camera context cleaned up")
            except Exception as e:
                logger.error(f"Failed to clean up context: {e}")

        # 清理 PLC 资源
        if self.plc:
            try:
                if self.low_notify_handle and self.low_user_handle:
                    self.plc.del_device_notification(self.low_notify_handle, self.low_user_handle)
                if self.high_notify_handle and self.high_user_handle:
                    self.plc.del_device_notification(self.high_notify_handle, self.high_user_handle)
                self.plc.close()
                logger.info("PLC connection closed")
            except pyads.ADSError as e:
                logger.error(f"PLC cleanup error: {e}")
