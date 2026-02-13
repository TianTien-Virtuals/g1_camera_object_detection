#!/usr/bin/env python3
"""
Ultralytics SAM 3 → ROS2: subscribe to a camera image topic, run concept
segmentation with text prompts, publish the annotated image for FoxGlove.

Requires: ultralytics >= 8.3.237, and SAM 3 weights (sam3.pt).
See https://docs.ultralytics.com/models/sam-3/

Usage:
  source /opt/ros/jazzy/setup.bash
  python3 source/sam3_ros2.py --model sam3.pt --topic-in /camera/zed/image_raw --text person car
  # Publishes: /camera/zed/sam3_segmentation (view in FoxGlove Image panel)
"""

import argparse
import os
import tempfile
from threading import Lock

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header

# Optional: Ultralytics SAM 3
try:
    from ultralytics.models.sam import SAM3SemanticPredictor
    HAS_SAM3 = True
except ImportError:
    HAS_SAM3 = False


class SAM3Ros2Node(Node):
    """Subscribes to an image topic, runs SAM 3 concept segmentation, publishes annotated image."""

    def __init__(
        self,
        model_path: str,
        topic_in: str,
        topic_out: str,
        text_prompts: list,
        process_hz: float = 2.0,
        conf: float = 0.25,
    ):
        super().__init__("sam3_ros2")
        self.model_path = os.path.abspath(model_path)
        self.topic_in = topic_in
        self.topic_out = topic_out
        self.text_prompts = text_prompts if text_prompts else ["person"]
        self.process_hz = process_hz
        self.conf = conf

        self._latest_msg = None
        self._lock = Lock()

        self.sub = self.create_subscription(Image, topic_in, self._image_cb, 10)
        self.pub = self.create_publisher(Image, topic_out, 10)
        self.timer = self.create_timer(1.0 / process_hz, self._process_and_publish)

        self._predictor = None
        self.get_logger().info(
            f"SAM 3 ROS2: sub={topic_in} -> pub={topic_out}, text={self.text_prompts}, {process_hz} Hz"
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

    def _bgr_to_msg(self, bgr: np.ndarray, stamp=None, frame_id: str = "sam3") -> Image:
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

        if self._predictor is None:
            return

        try:
            bgr = self._msg_to_bgr(msg)
            stamp = msg.header.stamp
            # SAM 3: set_image accepts path; use temp file for numpy frame
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                tmp = f.name
            cv2.imwrite(tmp, bgr)
            try:
                self._predictor.set_image(tmp)
                results = self._predictor(text=self.text_prompts)
            finally:
                try:
                    os.unlink(tmp)
                except Exception:
                    pass

            # Overlay masks on image for visualization (structure may vary by ultralytics version)
            out_bgr = bgr
            try:
                from ultralytics.utils.plotting import Annotator, colors
                res = results[0] if isinstance(results, (list, tuple)) else results
                if hasattr(res, "masks") and res.masks is not None:
                    annotator = Annotator(bgr.copy(), pil=False)
                    if hasattr(res.masks, "data"):
                        masks_np = res.masks.data.cpu().numpy()
                    else:
                        masks_np = getattr(res.masks, "xy", [])
                    if len(masks_np):
                        color_list = [colors(i, True) for i in range(len(masks_np))]
                        annotator.masks(masks_np, color_list)
                    out_bgr = annotator.result()
            except Exception:
                pass

            out_msg = self._bgr_to_msg(out_bgr, stamp=stamp)
            self.pub.publish(out_msg)
        except Exception as e:
            self.get_logger().warn(f"SAM 3 step failed: {e}")


def main(args=None):
    parser = argparse.ArgumentParser(
        description="SAM 3 concept segmentation: subscribe image, publish segmented image to ROS2"
    )
    parser.add_argument("--model", type=str, default="sam3.pt",
                        help="Path to sam3.pt (request from Hugging Face SAM 3 model page)")
    parser.add_argument("--topic-in", type=str, default="/camera/zed/image_raw",
                        help="ROS2 Image topic to subscribe to")
    parser.add_argument("--topic-out", type=str, default="/camera/zed/sam3_segmentation",
                        help="ROS2 Image topic to publish annotated result")
    parser.add_argument("--text", type=str, nargs="+", default=["person"],
                        help="Text prompts for concept segmentation (e.g. person car)")
    parser.add_argument("--hz", type=float, default=2.0,
                        help="Processing rate (Hz); lower saves GPU/CPU")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold for SAM 3")
    parsed, unknown = parser.parse_known_args(args)

    if not HAS_SAM3:
        print("Ultralytics SAM 3 not found. Install: pip install -U ultralytics  # >= 8.3.237")
        print("Then request sam3.pt from: https://huggingface.co/models?search=sam3")
        return 1

    if not os.path.isfile(parsed.model):
        print(f"Model file not found: {parsed.model}")
        print("Download sam3.pt from Hugging Face (SAM 3 model page), then pass --model /path/to/sam3.pt")
        return 1

    rclpy.init(args=unknown)
    node = SAM3Ros2Node(
        model_path=parsed.model,
        topic_in=parsed.topic_in,
        topic_out=parsed.topic_out,
        text_prompts=parsed.text,
        process_hz=parsed.hz,
        conf=parsed.conf,
    )

    overrides = dict(
        conf=parsed.conf,
        task="segment",
        mode="predict",
        model=parsed.model,
        half=True,
        verbose=False,
        save=False,  # do not save images to runs/
    )
    try:
        node._predictor = SAM3SemanticPredictor(overrides=overrides)
    except Exception as e:
        print(f"Failed to load SAM 3 model: {e}")
        node.destroy_node()
        rclpy.shutdown()
        return 1

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
