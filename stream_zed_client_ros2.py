#!/usr/bin/env python3
"""
ZED ZMQ client → ROS2 Image (and optionally PointCloud2) publisher.

Connects to a ZED stream server (e.g. stream_zed.py with --port 5556) and
publishes received frames to <topic_prefix>/image_raw, depth, points, closest_distance
(default topic_prefix /humanoid/zed).

Server (on machine with ZED):
  python3 stream_zed.py --port 5556

Client (this script, on any machine with ROS2):
  python3 stream_zed_client_ros2.py --host <server_ip>
  # Topics: /humanoid/zed/image_raw, /humanoid/zed/depth, /humanoid/zed/closest_distance, /humanoid/zed/points (if --points)
"""

import argparse
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Float32, Header

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


def _depth_to_cloud(depth, fx, fy, cx, cy, rgb=None, depth_min=0.2, depth_max=50.0, step=2):
    """Unproject depth to (x,y,z) and optionally pack rgb. Returns (xyz, rgb_packed or None)."""
    h, w = depth.shape[:2]
    u = np.arange(0, w, step, dtype=np.float32)
    v = np.arange(0, h, step, dtype=np.float32)
    u, v = np.meshgrid(u, v)
    ui = u.astype(int)
    vi = v.astype(int)
    z = depth[vi, ui]
    valid = np.isfinite(z) & (z >= depth_min) & (z <= depth_max) & (z > 0)
    u, v, z = u[valid], v[valid], z[valid]
    vi, ui = vi[valid], ui[valid]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    xyz = np.stack([x, y, z], axis=1).astype(np.float32)
    if rgb is not None and rgb.shape[:2] == (h, w):
        r = rgb[vi, ui, 0].astype(np.uint32)
        g = rgb[vi, ui, 1].astype(np.uint32)
        b = rgb[vi, ui, 2].astype(np.uint32)
        rgb_packed = (r << 16 | g << 8 | b).view(np.float32)
        return xyz, rgb_packed
    return xyz, None


class ZEDClientRos2Bridge(Node):
    """Subscribes to ZED ZMQ stream and publishes to ROS2 Image (and optionally PointCloud2) topics."""

    def __init__(
        self,
        host: str,
        port: int,
        topic_prefix: str = "/humanoid/zed",
        frame_id: str = "zed_camera",
        publish_points: bool = True,
        fx: float = 700.0,
        fy: float = 700.0,
        cx: float | None = None,
        cy: float | None = None,
        pointcloud_step: int = 2,
        depth_min: float = 0.2,
        depth_max: float = 50.0,
    ):
        super().__init__("zed_client_ros2_bridge")
        self.topic_prefix = topic_prefix.rstrip("/")
        self.frame_id = frame_id
        self.publish_points = publish_points
        self._fx, self._fy = fx, fy
        self._cx, self._cy = cx, cy
        self._pointcloud_step = pointcloud_step
        self._depth_min, self._depth_max = depth_min, depth_max

        self.image_pub = self.create_publisher(
            Image, f"{self.topic_prefix}/image_raw", 10
        )
        self.depth_pub = self.create_publisher(
            Image, f"{self.topic_prefix}/depth", 10
        )
        if publish_points:
            self.points_pub = self.create_publisher(
                PointCloud2, f"{self.topic_prefix}/points", 10
            )
        else:
            self.points_pub = None
        self.closest_dist_pub = self.create_publisher(
            Float32, f"{self.topic_prefix}/closest_distance", 10
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

    def _build_pointcloud(self, depth: np.ndarray, image: np.ndarray | None, stamp) -> PointCloud2 | None:
        """Build PointCloud2 from depth and optional RGB image."""
        h, w = depth.shape[:2]
        cx = self._cx if self._cx is not None else (w - 1) / 2.0
        cy = self._cy if self._cy is not None else (h - 1) / 2.0
        rgb = image if image is not None and image.ndim == 3 and image.shape[2] >= 3 else None
        xyz, rgb_packed = _depth_to_cloud(
            depth, self._fx, self._fy, cx, cy,
            rgb=rgb, depth_min=self._depth_min, depth_max=self._depth_max, step=self._pointcloud_step,
        )
        if xyz.size == 0:
            return None
        # Rotate 90° around Y: (x,y,z) → (-z, y, x)
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        # xyz = np.column_stack([-z, y, x]).astype(np.float32)
        xyz = np.column_stack([x, z, -y]).astype(np.float32)
        n = xyz.shape[0]
        header = Header()
        header.stamp = stamp
        header.frame_id = self.frame_id
        if rgb_packed is not None:
            fields = [
                PointField(name="x", offset=0, datatype=7, count=1),
                PointField(name="y", offset=4, datatype=7, count=1),
                PointField(name="z", offset=8, datatype=7, count=1),
                PointField(name="rgb", offset=12, datatype=6, count=1),
            ]
            point_step = 16
            arr = np.empty((n,), dtype=[("x", np.float32), ("y", np.float32), ("z", np.float32), ("rgb", np.uint32)])
            arr["x"] = xyz[:, 0]
            arr["y"] = xyz[:, 1]
            arr["z"] = xyz[:, 2]
            arr["rgb"] = rgb_packed.view(np.uint32)
        else:
            fields = [
                PointField(name="x", offset=0, datatype=7, count=1),
                PointField(name="y", offset=4, datatype=7, count=1),
                PointField(name="z", offset=8, datatype=7, count=1),
            ]
            point_step = 12
            arr = np.empty((n,), dtype=[("x", np.float32), ("y", np.float32), ("z", np.float32)])
            arr["x"] = xyz[:, 0]
            arr["y"] = xyz[:, 1]
            arr["z"] = xyz[:, 2]
        cloud = PointCloud2()
        cloud.header = header
        cloud.height = 1
        cloud.width = n
        cloud.fields = fields
        cloud.is_bigendian = False
        cloud.point_step = point_step
        cloud.row_step = point_step * n
        cloud.data = arr.tobytes()
        cloud.is_dense = True
        return cloud

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
            f"Connected. Publishing to {self.topic_prefix}/image_raw"
            + (f" and {self.topic_prefix}/points (3D)" if self.points_pub else "")
            + " (Ctrl+C to stop)."
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
                        # Closest object distance = min valid depth (meters)
                        valid = np.isfinite(depth) & (depth >= self._depth_min) & (depth <= self._depth_max) & (depth > 0)
                        if np.any(valid):
                            closest = float(np.min(depth[valid]))
                            self.closest_dist_pub.publish(Float32(data=closest))
                        if self.points_pub is not None:
                            cloud = self._build_pointcloud(depth, image, stamp)
                            if cloud is not None:
                                self.points_pub.publish(cloud)
        finally:
            socket.close()
            context.term()


def main(args=None):
    parser = argparse.ArgumentParser(
        description="ZED ZMQ client → ROS2 image, depth, closest_distance, points (default prefix /humanoid/zed)"
    )
    parser.add_argument("--host", type=str, default=DEFAULT_HOST,
                        help="ZED stream server host (e.g. 192.168.50.251)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help="ZED stream server port")
    parser.add_argument("--topic-prefix", type=str, default="/humanoid/zed")
    parser.add_argument("--frame-id", type=str, default="zed_camera")
    parser.add_argument("--no-points", action="store_true", help="Do not publish PointCloud2 from depth")
    parser.add_argument("--fx", type=float, default=700.0, help="Focal x for point cloud (use ZED intrinsics if known)")
    parser.add_argument("--fy", type=float, default=700.0, help="Focal y for point cloud")
    parser.add_argument("--cx", type=float, default=None, help="Principal point x (default: width/2)")
    parser.add_argument("--cy", type=float, default=None, help="Principal point y (default: height/2)")
    parser.add_argument("--pointcloud-step", type=int, default=2, help="Subsample step for points (2=every 2nd pixel)")
    parser.add_argument("--depth-min", type=float, default=0.2)
    parser.add_argument("--depth-max", type=float, default=50.0)
    parsed, unknown = parser.parse_known_args(args)

    rclpy.init(args=unknown)
    node = ZEDClientRos2Bridge(
        host=parsed.host,
        port=parsed.port,
        topic_prefix=parsed.topic_prefix,
        frame_id=parsed.frame_id,
        publish_points=not parsed.no_points,
        fx=parsed.fx,
        fy=parsed.fy,
        cx=parsed.cx,
        cy=parsed.cy,
        pointcloud_step=parsed.pointcloud_step,
        depth_min=parsed.depth_min,
        depth_max=parsed.depth_max,
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
