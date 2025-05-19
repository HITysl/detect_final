import socket
import json
from datetime import datetime
from typing import List, Dict, Any
import numpy as np
import cv2

SERVER_HOST = "192.168.1.100"
SERVER_PORT = 5001

def send_detections(
    results: List[Any],
    extras: List[Dict[str, float]],
    img: np.ndarray,
    server_host: str = SERVER_HOST,
    server_port: int = SERVER_PORT
) -> None:
    """
    将 YOLO 检测结果 `results` 和对应的 `extras` 打包成 JSON，
    并通过 TCP 连同内存中的 `img`（numpy.ndarray）一起发送。
    """
    # —— 1) 将内存中的 img 编码为 JPEG 字节 ——
    success, img_encoded = cv2.imencode('.jpg', img)
    if not success:
        raise RuntimeError("Failed to encode image")
    img_bytes = img_encoded.tobytes()

    # —— 2) 构造 detections 列表 ——
    detections = []
    for res, extra in zip(results, extras):
        for bbox, conf in zip(res.boxes.xywh.cpu().numpy(), res.boxes.conf.cpu().numpy()):
            cx, cy, w, h = map(float, bbox)
            det = {
                "center_x":   cx,
                "center_y":   cy,
                "width":      w,
                "height":     h,
                "confidence": float(conf),
                **extra
            }
            detections.append(det)

    # —— 3) 排序 & 构造 JSON payload ——
    detections.sort(key=lambda d: (d["center_y"], d["center_x"]))
    payload = {
        "timestamp": datetime.now().isoformat(),
        "box_count": len(detections),
        "detections": detections
    }
    json_bytes = json.dumps(payload, indent=2).encode('utf-8')

    # —— 4) 发送 JSON 与 图像字节 ——
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((server_host, server_port))
        sock.sendall(len(json_bytes).to_bytes(4, 'big'))
        sock.sendall(json_bytes)
        sock.sendall(len(img_bytes).to_bytes(4, 'big'))
        sock.sendall(img_bytes)

    print(f"[send_detections] Sent {payload['box_count']} boxes at {payload['timestamp']}")

if __name__ == "__main__":
    from ultralytics import YOLO

    # —— 读取原图到内存 ——
    img = cv2.imread(r"E:\Desktop\detect_final\images\all_xz_projection_20250427_164704.png")

    # —— 模型推理 ——
    model   = YOLO(r"E:\Desktop\car\yolo11best.pt")
    results = model(img, conf=0.5, iou=0.3)

    # —— 构造 extras，与检测框数量一致 ——
    total_boxes = sum(len(res.boxes.xywh) for res in results)
    extras = [
        {
            "x3d":         0.0,
            "y3d":         0.0,
            "z3d":         0.0,
            "real_width":  0.0,
            "real_height": 0.0,
        }
        for _ in range(total_boxes)
    ]

    # —— 发送 ——
    send_detections(results, extras, img)
