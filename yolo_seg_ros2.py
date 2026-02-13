#!/usr/bin/env python3
"""
Ultralytics YOLO segmentation → ROS2: subscribe to a camera image topic,
run YOLO instance segmentation, publish the annotated image.

Supports YOLOv8, YOLO11, and YOLO26 (e.g. yolo26n-seg.pt). Models download
automatically on first use (COCO classes: person, car, etc.).

Usage:
  source /opt/ros/jazzy/setup.bash
  source .venv/bin/activate   # if you use the venv for ultralytics
  python source/yolo_seg_ros2.py --topic-in /camera/zed/image_raw
  # Publishes: /camera/zed/yolo_segmentation (view in FoxGlove Image panel)
"""

import argparse
import os
from threading import Lock

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False


# Models that auto-download (no Hugging Face access)
# YOLO26: edge-optimized, NMS-free; YOLO11/YOLOv8 also supported
DEFAULT_MODEL = "yolo26n-seg.pt"  # or "yolo11n-seg.pt", "yolo8n-seg.pt", "yolo26s-seg.pt", etc.


class YOLOSegRos2Node(Node):
    """Subscribes to an image topic, runs YOLO segmentation, publishes annotated image."""

    def __init__(
        self,
        model_name: str,
        topic_in: str,
        topic_out: str,
        process_hz: float = 5.0,
        conf: float = 0.25,
    ):
        super().__init__("yolo_seg_ros2")
        self.topic_in = topic_in
        self.topic_out = topic_out
        self.process_hz = process_hz
        self.conf = conf

        self._latest_msg = None
        self._lock = Lock()

        self.sub = self.create_subscription(Image, topic_in, self._image_cb, 10)
        self.pub = self.create_publisher(Image, topic_out, 10)
        self.timer = self.create_timer(1.0 / process_hz, self._process_and_publish)

        self._model = YOLO(model_name)
        self.get_logger().info(
            f"YOLO seg ROS2: sub={topic_in} -> pub={topic_out}, model={model_name}, {process_hz} Hz"
        )

    def _image_cb(self, msg: Image):
        with self._lock:
            self._latest_msg = msg

    def _msg_to_bgr(self, msg: Image) -> np.ndarray:
        h, w = msg.height, msg.width
        if msg.encoding in ("bgr8", "rgb8"):
            ch = 3
        else:
            ch = 1
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape((h, w, ch) if ch == 3 else (h, w))
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        elif msg.encoding == "rgb8":
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return arr

    def _bgr_to_msg(self, bgr: np.ndarray, stamp=None, frame_id: str = "yolo_seg") -> Image:
        out = Image()
        out.header = Header()
        out.header.stamp = stamp or self.get_clock().now().to_msg()
        out.header.frame_id = frame_id
        out.height, out.width = bgr.shape[0], bgr.shape[1]
        out.encoding = "bgr8"
        out.is_bigendian = 0
        out.step = int(bgr.shape[1] * bgr.shape[2])
        out.data = bgr.tobytes()
        return out

    def _process_and_publish(self):
        with self._lock:
            msg = self._latest_msg
            self._latest_msg = None
        if msg is None:
            return

        try:
            bgr = self._msg_to_bgr(msg)
            stamp = msg.header.stamp
            results = self._model.predict(bgr, conf=self.conf, verbose=False)
            if results and len(results) > 0:
                annotated = results[0].plot()  # BGR image with boxes/masks
            else:
                annotated = bgr
            out_msg = self._bgr_to_msg(annotated, stamp=stamp)
            self.pub.publish(out_msg)
        except Exception as e:
            self.get_logger().warn(f"YOLO step failed: {e}")


def main(args=None):
    parser = argparse.ArgumentParser(
        description="YOLO segmentation: subscribe image, publish annotated image to ROS2"
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Model name (e.g. yolo26n-seg.pt, yolo11n-seg.pt, yolo8n-seg.pt); downloads on first use")
    parser.add_argument("--topic-in", type=str, default="/camera/zed/image_raw")
    parser.add_argument("--topic-out", type=str, default="/camera/zed/yolo_segmentation")
    parser.add_argument("--hz", type=float, default=5.0)
    parser.add_argument("--conf", type=float, default=0.25)
    parsed, unknown = parser.parse_known_args(args)

    if not HAS_YOLO:
        print("Ultralytics not found. Install: pip install ultralytics  (or use the project .venv)")
        return 1

    rclpy.init(args=unknown)
    node = YOLOSegRos2Node(
        model_name=parsed.model,
        topic_in=parsed.topic_in,
        topic_out=parsed.topic_out,
        process_hz=parsed.hz,
        conf=parsed.conf,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    exit(main())
