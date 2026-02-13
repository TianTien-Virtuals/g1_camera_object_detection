#!/usr/bin/env python3
"""
Fake camera server for testing: emits random images on ZMQ like the real sources.

- Port 5555: RealSense-style stream (msgpack with base64 JPEG in "images" dict).
  Use with stream_rs.py or stream_rs_ros2.py: connect to tcp://<this_host>:5555
- Port 5556: ZED-style stream (msgpack with numpy image + timestamp).
  Use with a ZED subscriber or stream_zed client connecting to tcp://<this_host>:5556

Run on the machine that will act as the source (e.g. 192.168.50.251):
  python3 fake_camera_server.py --host 0.0.0.0 --port-rs 5555 --port-zed 5556

Or locally:
  python3 fake_camera_server.py
  # Then in another terminal: stream_rs.py with ROBOT_IP=127.0.0.1
"""

import argparse
import base64
import io
import threading
import time

import cv2
import msgpack
import numpy as np
import zmq

try:
    import msgpack_numpy as m
    m.patch()
    HAS_MSGPACK_NUMPY = True
except ImportError:
    HAS_MSGPACK_NUMPY = False


def make_random_frame(width: int = 640, height: int = 480, frame_id: int = 0) -> np.ndarray:
    """Generate a deterministic-but-varying RGB image (uint8 HWC)."""
    np.random.seed(frame_id % (2**32))
    img = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    # Add a visible label
    cv2.putText(
        img, f"Fake camera #{frame_id}", (20, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2
    )
    cv2.putText(
        img, f"{width}x{height}", (20, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1
    )
    return img


def run_realsense_server(host: str, port: int, fps: float, width: int, height: int):
    """Publish RealSense-style msgpack: { images: { camera_name: base64_jpeg } }."""
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    addr = f"tcp://{host}:{port}"
    socket.bind(addr)
    print(f"[RealSense] Bound {addr} (fps={fps})")

    frame_id = 0
    interval = 1.0 / fps if fps > 0 else 0.033
    # Simulate two cameras like some G1 setups
    camera_names = ["left", "right"]

    while True:
        t0 = time.monotonic()
        images = {}
        rgb = make_random_frame(width, height, frame_id)
        # RealSense-style: RGB JPEG per camera (slightly different per name for variety)
        for i, name in enumerate(camera_names):
            # Slight variation per camera
            cam_rgb = np.clip(rgb.astype(np.int32) + (i * 10), 0, 255).astype(np.uint8)
            _, jpeg = cv2.imencode(".jpg", cv2.cvtColor(cam_rgb, cv2.COLOR_RGB2BGR))
            images[name] = base64.b64encode(jpeg.tobytes()).decode("ascii")
        payload = {"images": images}
        socket.send(msgpack.packb(payload))
        frame_id += 1
        elapsed = time.monotonic() - t0
        time.sleep(max(0, interval - elapsed))


def run_zed_server(host: str, port: int, fps: float, width: int, height: int):
    """Publish ZED-style msgpack: { timestamp, image (numpy RGB) }."""
    if not HAS_MSGPACK_NUMPY:
        print(
            f"[ZED] Port {port} NOT started: msgpack_numpy is required to send numpy arrays.\n"
            "      Install with: pip install msgpack-numpy  (or use a venv if pip complains)"
        )
        return

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    addr = f"tcp://{host}:{port}"
    socket.bind(addr)
    print(f"[ZED] Bound {addr} (fps={fps})")

    frame_id = 0
    interval = 1.0 / fps if fps > 0 else 0.033

    while True:
        t0 = time.monotonic()
        rgb = make_random_frame(width, height, frame_id)
        payload = {
            "timestamp": time.time(),
            "image": rgb,
        }
        socket.send(msgpack.packb(payload, default=m.encode))
        frame_id += 1
        elapsed = time.monotonic() - t0
        time.sleep(max(0, interval - elapsed))


def main():
    parser = argparse.ArgumentParser(
        description="Fake camera server: random images on ZMQ (RealSense + ZED format)"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Bind address (0.0.0.0 for all interfaces)")
    parser.add_argument("--port-rs", type=int, default=5555,
                        help="Port for RealSense-style stream")
    parser.add_argument("--port-zed", type=int, default=5556,
                        help="Port for ZED-style stream")
    parser.add_argument("--fps", type=float, default=15.0,
                        help="Target FPS per stream")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--rs-only", action="store_true",
                        help="Only run RealSense-style server (port 5555)")
    parser.add_argument("--zed-only", action="store_true",
                        help="Only run ZED-style server (port 5556)")
    args = parser.parse_args()

    if args.rs_only:
        run_realsense_server(args.host, args.port_rs, args.fps, args.width, args.height)
        return
    if args.zed_only:
        run_zed_server(args.host, args.port_zed, args.fps, args.width, args.height)
        return

    t_rs = threading.Thread(
        target=run_realsense_server,
        args=(args.host, args.port_rs, args.fps, args.width, args.height),
        daemon=True,
    )
    t_zed = threading.Thread(
        target=run_zed_server,
        args=(args.host, args.port_zed, args.fps, args.width, args.height),
        daemon=True,
    )
    t_rs.start()
    t_zed.start()
    print("Fake camera server running. Ctrl+C to stop.")
    try:
        t_rs.join()
        t_zed.join()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
