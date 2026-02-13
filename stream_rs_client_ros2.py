#!/usr/bin/env python3
"""
RealSense (ZMQ) camera stream → ROS2 Image publisher.

Connects to the Unitree G1 RealSense ZMQ server and publishes each camera
as sensor_msgs/msg/Image on ROS2 topics for viewing in FoxGlove (or rqt_image_view).

Usage:
  source /opt/ros/<distro>/setup.bash
  python3 stream_rs_ros2.py --ip 192.168.50.251
  # Topic: /camera/realsense/image_raw (single topic; if multiple cameras, first by name is used)
"""

import base64
import cv2
import msgpack
import numpy as np
import zmq

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header

# === CHANGE THIS TO YOUR ROBOT'S IP ===
DEFAULT_ROBOT_IP = "192.168.50.251"
PORT = 5555
# ======================================


def decode_image(image_b64: str) -> np.ndarray:
    """Decode base64 JPEG image to numpy BGR."""
    color_data = base64.b64decode(image_b64)
    color_array = np.frombuffer(color_data, dtype=np.uint8)
    image = cv2.imdecode(color_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode image")
    # RealSense sends RGB; convert to BGR for standard OpenCV/ROS convention
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


class RealSenseRos2Bridge(Node):
    """Publishes RealSense ZMQ stream to ROS2 Image topic /camera/realsense/image_raw."""

    TOPIC = "/camera/realsense/image_raw"

    def __init__(self, robot_ip: str):
        super().__init__("realsense_ros2_bridge")
        self.robot_ip = robot_ip
        self.pub = self.create_publisher(Image, self.TOPIC, 10)
        self.declare_parameter("frame_id", "realsense")

    def image_to_msg(self, image: np.ndarray) -> Image:
        """Convert OpenCV BGR image to sensor_msgs/Image."""
        msg = Image()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.get_parameter("frame_id").value
        msg.height = image.shape[0]
        msg.width = image.shape[1]
        msg.encoding = "bgr8"
        msg.is_bigendian = 0
        msg.step = int(image.shape[1] * image.shape[2])
        msg.data = image.tobytes()
        return msg

    def run(self):
        """Connect to ZMQ and publish frames to ROS2."""
        self.get_logger().info(
            f"Connecting to camera server at tcp://{self.robot_ip}:{PORT}..."
        )

        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.setsockopt_string(zmq.SUBSCRIBE, "")
        socket.setsockopt(zmq.CONFLATE, True)
        socket.setsockopt(zmq.RCVHWM, 3)
        socket.connect(f"tcp://{self.robot_ip}:{PORT}")

        self.get_logger().info(f"Connected. Publishing to {self.TOPIC} (Ctrl+C to stop).")

        try:
            while rclpy.ok():
                packed = socket.recv()
                data = msgpack.unpackb(packed)
                images = data.get("images", {})
                if not images:
                    continue
                # Single topic: use first camera (sorted for deterministic choice if multiple)
                camera_name = next(iter(sorted(images.keys())))
                image_b64 = images[camera_name]
                try:
                    image = decode_image(image_b64)
                    msg = self.image_to_msg(image)
                    self.pub.publish(msg)
                except Exception as e:
                    self.get_logger().warn(
                        f"Skip frame from {camera_name}: {e}"
                    )
        finally:
            socket.close()
            context.term()


def main(args=None):
    import argparse
    parser = argparse.ArgumentParser(
        description="RealSense ZMQ stream → ROS2 Image topics"
    )
    parser.add_argument(
        "--ip", type=str, default=DEFAULT_ROBOT_IP,
        help="Robot / RealSense server IP"
    )
    parsed, unknown = parser.parse_known_args(args)

    rclpy.init(args=unknown)
    node = RealSenseRos2Bridge(robot_ip=parsed.ip)
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
