#!/usr/bin/env python3
"""
SAM 3 segmentation on streaming video → ROS2.

Designed for live camera stream: subscribes to an image topic (streaming video), runs SAM on the
latest frame at --hz, publishes segmented frames and shows an OpenCV window. Not for offline video files.

Backends:
  --use-hf       : Hugging Face SAM 3 (point click). Click sets the point to segment on the stream.
  --tracking     : With --use-hf, track object across frames by re-prompting at previous mask center each frame.
  --model-dir    : With --use-hf, load from local dir (e.g. ./sam3_pt); no Hugging Face access needed.
  default        : Ultralytics sam3.pt + text prompt. Click is display-only.

Requires: ultralytics + sam3.pt (default), or transformers + accelerate (--use-hf).
Usage (streaming video from camera topic):
  python3 sam3_ros2_point_click.py --use-hf --model-dir ./sam3_pt --topic-in /camera/realsense/image_raw --hz 1
  python3 sam3_ros2_point_click.py --use-hf --tracking --model-dir ./sam3_pt ...   # track object across frames
  python3 sam3_ros2_point_click.py --model sam3.pt --text chair --topic-in /camera/realsense/image_raw --hz 1
"""

import argparse
import os
import threading
from threading import Lock

import cv2
import numpy as np
import torch
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header

try:
    from ultralytics.models.sam import SAM3SemanticPredictor
    from ultralytics.utils.plotting import Annotator, colors
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False

try:
    from transformers import Sam3TrackerProcessor, Sam3TrackerModel
    from PIL import Image as PILImage
    HAS_HF_SAM3 = True
except ImportError:
    HAS_HF_SAM3 = False


class SAM3PointClickNode(Node):
    """Segment streaming video: subscribe to camera image topic, run SAM on latest frame at process_hz, publish result."""

    def __init__(
        self,
        topic_in: str,
        topic_out: str,
        model_path: str,
        text_prompts: list,
        use_hf: bool = False,
        tracking: bool = False,
        conf: float = 0.25,
        process_hz: float = 1.0,
        debug: bool = False,
    ):
        super().__init__("sam3_point_click")
        self._latest_bgr = None
        self._latest_stamp = None
        self._lock = Lock()
        self._processing = False
        self._predictor = None
        self._hf_processor = None
        self._hf_model = None
        self._use_hf = use_hf
        self._tracking = tracking and use_hf  # only with HF point-click
        self._text_prompts = text_prompts if text_prompts else ["person"]
        self._conf = conf
        self._process_hz = process_hz
        self._debug = debug
        self._frame_count = 0
        self._result_bgr = None
        self._result_stamp = None
        self._target_point = None  # (x, y) or None
        self._target_bbox = None  # (x1, y1, x2, y2) or None; F key sets bbox around center
        self._tracked_point = None  # (x, y) centroid of last mask; used next frame when tracking
        self._last_click_display = None  # (x, y) last click coords for debug text
        self._click_count = 0
        self._cursor_xy = None  # (x, y) keyboard crosshair; None = use image center until first frame

        self.sub = self.create_subscription(Image, topic_in, self._image_cb, 10)
        self.pub = self.create_publisher(Image, topic_out, 10)
        self.timer_process = self.create_timer(1.0 / process_hz, self._process_continuous)
        self.timer_stream = self.create_timer(1.0 / 30.0, self._publish_stream)

        mode = "point-click (HF)" if use_hf else f"text={self._text_prompts}"
        if use_hf and self._tracking:
            mode += ", tracking across frames"
        self.get_logger().info(
            f"SAM 3: sub={topic_in} -> pub={topic_out} @ {process_hz} Hz, {mode} (click/S=point F=bbox c=clear)"
        )

    def _image_cb(self, msg: Image):
        """Store latest frame from streaming video (camera topic)."""
        bgr = self._msg_to_bgr(msg)
        with self._lock:
            self._latest_bgr = bgr
            self._latest_stamp = msg.header.stamp
            self._frame_count += 1
        if self._debug and self._frame_count == 1:
            self.get_logger().info(f"[debug] First frame from stream: {bgr.shape[1]}x{bgr.shape[0]}")

    def _msg_to_bgr(self, msg: Image) -> np.ndarray:
        h, w = msg.height, msg.width
        ch = 3 if msg.encoding in ("bgr8", "rgb8") else 1
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

    def set_target(self, x: int, y: int, from_click: bool = False):
        with self._lock:
            self._target_point = (x, y)
            self._target_bbox = None
            self._tracked_point = None
            if from_click:
                self._last_click_display = (x, y)
                self._click_count += 1
        if self._debug:
            self.get_logger().info(f"[debug] Target set at ({x}, {y})" + (" [click]" if from_click else " [S key]"))

    def set_target_bbox_center(self, half_size: int = 250):
        """Set a bounding box around image center (like S but bbox). F key."""
        with self._lock:
            bgr = self._latest_bgr
        if bgr is None:
            return False
        h, w = bgr.shape[:2]
        cx, cy = w // 2, h // 2
        x1 = max(0, cx - half_size)
        y1 = max(0, cy - half_size)
        x2 = min(w, cx + half_size)
        y2 = min(h, cy + half_size)
        with self._lock:
            self._target_bbox = (x1, y1, x2, y2)
            self._target_point = None
            self._tracked_point = None
        if self._debug:
            self.get_logger().info(f"[debug] Target bbox at center: ({x1},{y1})-({x2},{y2})")
        return True

    def clear_target(self):
        with self._lock:
            self._target_point = None
            self._target_bbox = None
            self._tracked_point = None
            self._result_bgr = None
            self._result_stamp = None
        if self._debug:
            self.get_logger().info("[debug] Target cleared")

    def move_cursor(self, dx: int, dy: int):
        """Move keyboard crosshair by (dx, dy); bounds checked in get_display_frame."""
        with self._lock:
            bgr = self._latest_bgr
        if bgr is None:
            return
        h, w = bgr.shape[:2]
        with self._lock:
            cur = self._cursor_xy
        if cur is None:
            cur = (w // 2, h // 2)
        self._cursor_xy = (max(0, min(w - 1, cur[0] + dx)), max(0, min(h - 1, cur[1] + dy)))

    def set_cursor(self, x: int, y: int):
        with self._lock:
            self._cursor_xy = (x, y)

    def set_target_at_cursor(self):
        """Set target to current keyboard cursor position. Returns True if set."""
        with self._lock:
            bgr = self._latest_bgr
            cur = self._cursor_xy
        if bgr is None:
            return False
        h, w = bgr.shape[:2]
        if cur is None:
            cur = (w // 2, h // 2)
        x, y = max(0, min(w - 1, cur[0])), max(0, min(h - 1, cur[1]))
        self.set_target(x, y, from_click=False)
        return True

    def _publish_stream(self):
        with self._lock:
            bgr = self._result_bgr if self._result_bgr is not None else self._latest_bgr
            stamp = self._result_stamp or self._latest_stamp
        if bgr is None or stamp is None:
            return
        self.pub.publish(self._bgr_to_msg(bgr, stamp=stamp))

    def get_display_frame(self):
        """Return BGR to show; draw target point or keyboard crosshair."""
        with self._lock:
            bgr = self._result_bgr if self._result_bgr is not None else self._latest_bgr
        if bgr is None:
            return None
        disp = bgr.copy()
        h, w = disp.shape[:2]
        with self._lock:
            cur = self._cursor_xy
            if cur is None:
                self._cursor_xy = (w // 2, h // 2)
                cur = self._cursor_xy
        cx, cy = max(0, min(w - 1, cur[0])), max(0, min(h - 1, cur[1]))
        if self._tracking and self._tracked_point is not None:
            px, py = int(round(self._tracked_point[0])), int(round(self._tracked_point[1]))
            px, py = max(0, min(w - 1, px)), max(0, min(h - 1, py))
            cv2.circle(disp, (px, py), 6, (0, 255, 0), 2)
        elif self._target_point is not None:
            px, py = self._target_point
            cv2.circle(disp, (px, py), 6, (0, 255, 0), 2)
        else:
            # No target: draw keyboard crosshair so user can move with arrows and press Enter
            cv2.circle(disp, (cx, cy), 8, (0, 255, 255), 1)
            cv2.drawMarker(disp, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 12, 1)
        with self._lock:
            bbox = self._target_bbox
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(disp, (x1, y1), (x2, y2), (255, 0, 255), 2)
        font_scale = 0.6
        thickness = 1
        hint = "Right-click or S=point F=bbox c=clear" + (" (tracking)" if self._tracking else "") if self._use_hf else f"Segment: {self._text_prompts}"
        cv2.putText(disp, f"{hint} | c=clear q=quit",
                    (10, disp.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        with self._lock:
            last_click = self._last_click_display
            nclicks = self._click_count
        if last_click is not None:
            lx, ly = last_click
            cv2.putText(disp, f"Point set: ({lx}, {ly})  [{nclicks}]",
                        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)
        return disp

    def _process_continuous(self):
        """Run SAM on latest frame from the stream at process_hz (streaming video, not offline file)."""
        if self._processing:
            return
        with self._lock:
            if self._latest_bgr is None:
                return
            if self._use_hf:
                if self._hf_model is None or self._hf_processor is None:
                    return
                bbox = self._target_bbox
                target = self._target_point
                tracked = self._tracked_point
                if bbox is not None:
                    pass  # use bbox below
                elif self._tracking and tracked is not None:
                    pass  # use tracked point below
                elif target is not None:
                    pass  # use target point below
                else:
                    return  # no prompt set yet: show raw stream
                bgr = self._latest_bgr.copy()
                stamp = self._latest_stamp
                h, w = bgr.shape[:2]
                if bbox is not None:
                    x, y, use_bbox = None, None, True
                else:
                    use_bbox = False
                    if self._tracking and tracked is not None:
                        x, y = int(round(tracked[0])), int(round(tracked[1]))
                        x, y = max(0, min(w - 1, x)), max(0, min(h - 1, y))
                    else:
                        x, y = target
            else:
                if self._predictor is None:
                    return
                bgr = self._latest_bgr.copy()
                stamp = self._latest_stamp
        self._processing = True
        try:
            if self._use_hf:
                self._run_sam_hf(bgr, stamp, x, y, bbox=bbox if bbox is not None else None)
            else:
                self._run_sam_and_publish(bgr, stamp)
        finally:
            self._processing = False

    def _run_sam_hf(self, bgr: np.ndarray, stamp, x: int = None, y: int = None, bbox=None):
        """Hugging Face SAM 3: point or bbox via Sam3TrackerProcessor + Sam3TrackerModel."""
        try:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(rgb)
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                input_boxes = [[[float(x1), float(y1), float(x2), float(y2)]]]
                inputs = self._hf_processor(
                    images=pil_image,
                    input_boxes=input_boxes,
                    return_tensors="pt",
                ).to(self._hf_model.device)
            else:
                input_points = [[[[x, y]]]]
                input_labels = [[[1]]]  # 1 = foreground
                inputs = self._hf_processor(
                    images=pil_image,
                    input_points=input_points,
                    input_labels=input_labels,
                    return_tensors="pt",
                ).to(self._hf_model.device)
            with torch.no_grad():
                outputs = self._hf_model(**inputs)
            # post_process_masks returns list per image; we have one image
            masks = self._hf_processor.post_process_masks(
                outputs.pred_masks.cpu(), inputs["original_sizes"]
            )[0]
            # masks: (num_masks, H, W); take best (e.g. first) or by score if available
            if masks is not None and masks.numel() > 0:
                mask_np = masks[0].numpy()  # (H, W)
                if mask_np.ndim == 3:
                    mask_np = mask_np[0]
                # Resize to bgr size if needed (processor may have resized)
                if mask_np.shape[:2] != (bgr.shape[0], bgr.shape[1]):
                    mask_np = cv2.resize(
                        mask_np.astype(np.uint8),
                        (bgr.shape[1], bgr.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                mask_bool = mask_np > 0.5
                out_bgr = bgr.copy()
                out_bgr[mask_bool] = [0, 255, 0]
                # Tracking: update point to mask centroid for next frame so object is tracked across frames
                if self._tracking and mask_bool.any():
                    ys, xs = np.where(mask_bool)
                    cy, cx = float(ys.mean()), float(xs.mean())
                    with self._lock:
                        self._tracked_point = (cx, cy)
                if self._debug:
                    self.get_logger().info(
                        f"[debug] HF SAM 3: 1 mask" + (f" bbox={bbox}" if bbox else f" at ({x},{y})")
                    )
            else:
                out_bgr = bgr.copy()
                if self._tracking:
                    with self._lock:
                        self._tracked_point = None  # lost track
                if self._debug:
                    self.get_logger().info("[debug] HF SAM 3: no masks")
            with self._lock:
                self._result_bgr = out_bgr
                self._result_stamp = stamp
            self.pub.publish(self._bgr_to_msg(out_bgr, stamp=stamp))
        except Exception as e:
            self.get_logger().warn(f"SAM 3 (HF) segment failed: {e}")
            if self._debug:
                import traceback
                self.get_logger().info(f"[debug] {traceback.format_exc()}")

    def _run_sam_and_publish(self, bgr: np.ndarray, stamp):
        """SAM 3: text prompt (same as sam3_ros2). Avoids point inference which can trigger torch.cat error."""
        try:
            with torch.no_grad():
                self._predictor.set_image(bgr)
                results = self._predictor(text=self._text_prompts)
            out_bgr = bgr.copy()
            res = results[0] if isinstance(results, (list, tuple)) else results
            if self._debug:
                has_m = hasattr(res, "masks") and res.masks is not None
                self.get_logger().info(f"[debug] Inference done; result has masks: {has_m}")
            if hasattr(res, "masks") and res.masks is not None:
                annotator = Annotator(bgr.copy(), pil=False)
                if hasattr(res.masks, "data"):
                    masks_np = res.masks.data.clone().cpu().numpy()
                else:
                    masks_np = getattr(res.masks, "xy", [])
                if len(masks_np):
                    color_list = [colors(i, True) for i in range(len(masks_np))]
                    annotator.masks(masks_np, color_list)
                    out_bgr = annotator.result()
                    if self._debug:
                        self.get_logger().info(f"[debug] Got {len(masks_np)} mask(s)")
            with self._lock:
                self._result_bgr = out_bgr
                self._result_stamp = stamp
            self.pub.publish(self._bgr_to_msg(out_bgr, stamp=stamp))
        except Exception as e:
            self.get_logger().warn(f"SAM 3 segment failed: {e}")
            if self._debug:
                import traceback
                self.get_logger().info(f"[debug] {traceback.format_exc()}")


def main(args=None):
    parser = argparse.ArgumentParser(
        description="SAM 3 on streaming video: subscribe to camera topic, segment latest frame, show window, publish."
    )
    parser.add_argument("--use-hf", action="store_true",
                        help="Use Hugging Face SAM 3 for point-click segmentation")
    parser.add_argument("--tracking", action="store_true",
                        help="Track object across frames (re-prompt at previous mask center each frame; requires --use-hf)")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Local dir for HF model (e.g. ./sam3_pt). If set with --use-hf, load from here (no Hugging Face access).")
    parser.add_argument("--model", type=str, default="sam3.pt", help="Path to sam3.pt (Ultralytics; ignored if --use-hf)")
    parser.add_argument("--topic-in", type=str, default="/camera/realsense/image_raw")
    parser.add_argument("--topic-out", type=str, default="/camera/realsense/sam3_segmentation")
    parser.add_argument("--text", type=str, nargs="+", default=["chair"], help="Text prompts when not --use-hf")
    parser.add_argument("--hz", type=float, default=1.0, help="Segmentation rate (Hz)")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--debug", action="store_true")
    parsed, unknown = parser.parse_known_args(args)

    use_hf = parsed.use_hf
    if use_hf:
        if not HAS_HF_SAM3:
            print("Hugging Face SAM 3 not found. pip install transformers accelerate")
            return 1
    else:
        if not HAS_ULTRALYTICS:
            print("Ultralytics SAM 3 not found. pip install -U ultralytics  # >= 8.3.237")
            return 1
        if not os.path.isfile(parsed.model):
            print(f"Model not found: {parsed.model}")
            return 1

    rclpy.init(args=unknown)
    node = SAM3PointClickNode(
        topic_in=parsed.topic_in,
        topic_out=parsed.topic_out,
        model_path=parsed.model,
        text_prompts=parsed.text,
        use_hf=use_hf,
        tracking=parsed.tracking,
        conf=parsed.conf,
        process_hz=parsed.hz,
        debug=parsed.debug,
    )

    if use_hf:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        hf_path = parsed.model_dir
        if hf_path and not os.path.isdir(hf_path):
            print(f"Model dir not found: {hf_path}")
            node.destroy_node()
            rclpy.shutdown()
            return 1
        if hf_path:
            hf_path = os.path.abspath(hf_path)
        try:
            load_from = hf_path if hf_path else "facebook/sam3"
            node._hf_model = Sam3TrackerModel.from_pretrained(load_from).to(device)
            node._hf_processor = Sam3TrackerProcessor.from_pretrained(load_from)
        except Exception as e:
            print(f"Failed to load Hugging Face SAM 3: {e}")
            if hf_path:
                print(f"  Ensure {hf_path} contains config.json and model.safetensors (full HF snapshot).")
            node.destroy_node()
            rclpy.shutdown()
            return 1
    else:
        overrides = dict(
            conf=parsed.conf,
            task="segment",
            mode="predict",
            model=parsed.model,
            half=True,
            verbose=False,
            save=False,
        )
        try:
            node._predictor = SAM3SemanticPredictor(overrides=overrides)
        except Exception as e:
            print(f"Failed to load Ultralytics SAM 3: {e}")
            node.destroy_node()
            rclpy.shutdown()
            return 1

    stream_win = "SAM 3 — streaming video (point-click)" if use_hf else "SAM 3 — streaming video (text)"
    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(placeholder, "Waiting... Right-click or W/A/D/E + Enter or S=center", (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.namedWindow(stream_win)
    cv2.imshow(stream_win, placeholder)
    # Give Qt time to create the window before attaching mouse callback
    for _ in range(20):
        if cv2.waitKey(50) >= 0:
            break

    def on_mouse(event, mx, my, _flags, param):
        # event: 1=LEFT_DOWN, 2=RIGHT_DOWN on OpenCV. Right often works when left doesn't (Qt/X11).
        if event == 1:  # EVENT_LBUTTONDOWN
            print(f"[mouse] LEFT at ({mx}, {my}) -> set target")
            if param is not None:
                param.set_target(mx, my, from_click=True)
        elif event == 2:  # EVENT_RBUTTONDOWN
            print(f"[mouse] RIGHT at ({mx}, {my}) -> set target")
            if param is not None:
                param.set_target(mx, my, from_click=True)
        elif event in (1, 2, 3):
            print(f"[mouse] event={event} at ({mx}, {my})")

    try:
        cv2.setMouseCallback(stream_win, on_mouse, node)
        print("[mouse] Callback attached. Right-click or left-click in window to set point.")
    except cv2.error as e:
        print(f"[mouse] setMouseCallback failed: {e}. Use arrows + Enter or S instead.")

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        while rclpy.ok():
            disp = node.get_display_frame()
            if disp is not None:
                cv2.imshow(stream_win, disp)
            else:
                cv2.imshow(stream_win, placeholder)
            key = cv2.waitKeyEx(50) if hasattr(cv2, "waitKeyEx") else cv2.waitKey(50)
            if key < 0:
                continue
            key_low = key & 0xFF
            if key_low == ord("q") or key_low == 27:
                break
            if key_low == ord("c"):
                node.clear_target()
            elif key_low == ord("s"):
                with node._lock:
                    latest = node._latest_bgr
                if latest is not None:
                    h, w = latest.shape[:2]
                    node.set_target(w // 2, h // 2, from_click=False)
            elif key_low == ord("f"):
                if node.set_target_bbox_center():
                    print("[key] F -> bbox at center")
            elif key_low == 13 or key_low == 10:  # Enter
                if node.set_target_at_cursor():
                    print("[key] Enter -> target set at cursor")
            # Arrow keys (waitKeyEx) or W/A/D/E (up/left/right/down; S=center)
            elif key_low == ord("a") or key == 63234:  # Left
                node.move_cursor(-15, 0)
            elif key_low == ord("d") or key == 63235:  # Right
                node.move_cursor(15, 0)
            elif key_low == ord("w") or key == 63232:  # Up
                node.move_cursor(0, -15)
            elif key_low == ord("e") or key == 63233:  # Down (E or Down arrow)
                node.move_cursor(0, 15)
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    exit(main())
