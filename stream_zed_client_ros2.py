#!/usr/bin/env python3
"""
ZED ZMQ client → ROS2 Image publisher.

Connects to a ZED stream server (e.g. stream_zed.py with --port 5556) and
publishes received frames to /camera/zed/image_raw (and optionally depth).
Use this when the ZED camera runs on another machine as the server.

Server (on machine with ZED):
  python3 stream_zed.py --port 5556

Client (this script, on any machine with ROS2):
  python3 stream_zed_client_ros2.py --host <server_ip>
  # Topics: /camera/zed/image_raw, /camera/zed/depth (if server sends depth)
"""

import argparse
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header

import msgpack
import zmq

try:
    import msgpack_numpy as m
    m.patch()
    HAS_MSGPACK_NUMPY = True
except ImportError:
    HAS_MSGPACK_NUMPY = False

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5556


class ZEDClientRos2Bridge(Node):
    """Subscribes to ZED ZMQ stream and publishes to ROS2 Image topics."""

    def __init__(
        self,
        host: str,
        port: int,
        topic_prefix: str = "/camera/zed",
        frame_id: str = "zed_camera",
    ):
        super().__init__("zed_client_ros2_bridge")
        self.topic_prefix = topic_prefix.rstrip("/")
        self.frame_id = frame_id
        self.image_pub = self.create_publisher(
            Image, f"{self.topic_prefix}/image_raw", 10
        )
        self.depth_pub = self.create_publisher(
            Image, f"{self.topic_prefix}/depth", 10
        )
        self.host = host
        self.port = port

    def image_to_msg(self, image: np.ndarray, stamp=None) -> Image:
        """Convert numpy RGB (H,W,3) to sensor_msgs/Image (bgr8)."""
        msg = Image()
        msg.header = Header()
        msg.header.stamp = stamp or self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.height = image.shape[0]
        msg.width = image.shape[1]
        msg.encoding = "bgr8"
        msg.is_bigendian = 0
        if image.shape[2] == 3:
            bgr = image[:, :, ::-1].copy()
        else:
            bgr = image
        msg.step = int(bgr.shape[1] * bgr.shape[2])
        msg.data = bgr.tobytes()
        return msg

    def depth_to_msg(self, depth: np.ndarray, stamp=None) -> Image:
        """Convert float depth (H,W) to sensor_msgs/Image (32FC1)."""
        msg = Image()
        msg.header = Header()
        msg.header.stamp = stamp or self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.height = depth.shape[0]
        msg.width = depth.shape[1]
        msg.encoding = "32fc1"
        msg.is_bigendian = 0
        msg.step = depth.shape[1] * 4
        msg.data = np.asarray(depth, dtype=np.float32).tobytes()
        return msg

    def run(self):
        if not HAS_MSGPACK_NUMPY:
            self.get_logger().error(
                "msgpack_numpy required. Install: pip install msgpack-numpy"
            )
            return

        addr = f"tcp://{self.host}:{self.port}"
        self.get_logger().info(f"Connecting to ZED stream at {addr}...")

        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.setsockopt_string(zmq.SUBSCRIBE, "")
        socket.setsockopt(zmq.CONFLATE, True)
        socket.setsockopt(zmq.RCVHWM, 3)
        socket.connect(addr)

        self.get_logger().info(
            f"Connected. Publishing to {self.topic_prefix}/image_raw (Ctrl+C to stop)."
        )

        try:
            while rclpy.ok():
                packed = socket.recv()
                try:
                    data = msgpack.unpackb(packed)
                except Exception:
                    data = msgpack.unpackb(packed, ext_hook=m.decode)

                if "image" not in data:
                    continue

                image = data["image"]
                if not isinstance(image, np.ndarray):
                    continue

                stamp = self.get_clock().now().to_msg()
                img_msg = self.image_to_msg(image, stamp=stamp)
                self.image_pub.publish(img_msg)

                if "depth" in data:
                    depth = data["depth"]
                    if isinstance(depth, np.ndarray):
                        depth_msg = self.depth_to_msg(depth, stamp=stamp)
                        self.depth_pub.publish(depth_msg)
        finally:
            socket.close()
            context.term()


def main(args=None):
    parser = argparse.ArgumentParser(
        description="ZED ZMQ client → ROS2 /camera/zed/image_raw"
    )
    parser.add_argument("--host", type=str, default=DEFAULT_HOST,
                        help="ZED stream server host (e.g. 192.168.50.251)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help="ZED stream server port")
    parser.add_argument("--topic-prefix", type=str, default="/camera/zed")
    parser.add_argument("--frame-id", type=str, default="zed_camera")
    parsed, unknown = parser.parse_known_args(args)

    rclpy.init(args=unknown)
    node = ZEDClientRos2Bridge(
        host=parsed.host,
        port=parsed.port,
        topic_prefix=parsed.topic_prefix,
        frame_id=parsed.frame_id,
    )
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
