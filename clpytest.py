import pyads
import random

from _ctypes import sizeof

last_low_start_flag_value = False          # ADS设置通知，首次运行会运行一次回调读取变量值，设置一个标志位，记录上次回调值，来规避首次调用回调
last_high_start_flag_value = False          # ADS设置通知，首次运行会运行一次回调读取变量值，设置一个标志位，记录上次回调值，来规避首次调用回调

low_pos_done_flag = False                   # 低位检测完成标志位
high_pos_done_flag = False                  # 高位检测完成标志位

# 低位检测回调函数
def camera_low_pos_start_callback(notification, data):
    data_type = position_low_start_tag[data]
    handle, timestamp, value = plc.parse_notification(notification, data_type)
    print(f"bInspection_PosLow_Start: {value}:{timestamp}，第一次检测开始！")
    global low_pos_done_flag
    global last_low_start_flag_value

    if value == last_low_start_flag_value:
        return
    else:
        low_pos_done_flag = True
    last_low_value = value


# 高位检测回调函数
def camera_high_pos_start_callback(notification, data):
    data_type = position_high_start_tag[data]
    handle, timestamp, value = plc.parse_notification(notification, data_type)
    print(f"bInspection_PosHigh_Start: {value}:{timestamp}，第二次检测开始！")
    global high_pos_done_flag
    global last_high_start_flag_value
    if value ==  last_high_start_flag_value:
        return
    else:
        high_pos_done_flag = True
    last_high_value = value





if __name__ == "__main__":
    # ADS连接参数
    plc = pyads.Connection("192.168.1.20.1.1", 851)
    plc.open()



    # 龙门到低位，可以进行第一次拍摄，回调
    position_low_start_tag = {"Camera.bInspection_PosLow_Start": pyads.PLCTYPE_BOOL}                                                            # 定义变量
    attr = pyads.NotificationAttrib(sizeof(pyads.PLCTYPE_BOOL))                                                                                 # 回调函数传入变量的的大小
    camera_low_pos_notify_handle, camera_low_pos_user_handle = plc.add_device_notification('Camera.bInspection_PosLow_Start', attr, camera_low_pos_start_callback)       # 注册设备通知

    # 龙门到低位，可以进行第一次拍摄，回调
    position_high_start_tag = {"Camera.bInspection_PosHigh_Start": pyads.PLCTYPE_BOOL}                                                              # 定义关注的变量
    attr = pyads.NotificationAttrib(sizeof(pyads.PLCTYPE_BOOL))                                                                                     # 回调函数传入变量的的大小
    camera_high_pos_notify_handle, camera_high_pos_user_handle = plc.add_device_notification('Camera.bInspection_PosHigh_Start', attr, camera_high_pos_start_callback)        # 注册设备通知

    try:


        while True:

            if low_pos_done_flag:
                # 向PLC写入低位检测完成
                plc.write_by_name("Camera.bInspection_PosLow_Done", True, pyads.PLCTYPE_BOOL)
                low_pos_done_flag = False

            if high_pos_done_flag:
                # 向PLC写入高位检测完成
                plc.write_by_name("Camera.bInspection_PosHigh_Done", True, pyads.PLCTYPE_BOOL)
                high_pos_done_flag = False
            pass




    except pyads.ADSError as e:
        print(e)

    finally:
        # 删除回调的句柄
        plc.del_device_notification(camera_low_pos_notify_handle, camera_low_pos_user_handle)
        plc.del_device_notification(camera_high_pos_notify_handle, camera_high_pos_user_handle)
        plc.close()