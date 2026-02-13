#!/usr/bin/env python3
"""
RealSense ZMQ stream server. Continuously streams frames on port 5555.

Message format (same as g1-camera): msgpack with {"images": {camera_name: base64_jpeg}}.
Run on the robot: python3 stream_rs.py
Then from your PC: python3 stream_rs_client_ros2.py --ip <robot_ip>
"""
import argparse
import base64
import sys
import time

try:
    import cv2
    import msgpack
    import numpy as np
    import zmq
except ImportError as e:
    print(f"✗ Missing dependency: {e}")
    sys.exit(1)

try:
    import pyrealsense2 as rs
except ImportError:
    print("✗ pyrealsense2 not installed")
    sys.exit(1)


def list_devices(ctx: rs.context()) -> list:
    devices = ctx.query_devices()
    print(f"\nFound {len(devices)} RealSense device(s):")
    for i, dev in enumerate(devices):
        print(f"  [{i}] {dev.get_info(rs.camera_info.name)} - S/N: {dev.get_info(rs.camera_info.serial_number)}")
    return devices


def run_stream_server(pipeline: rs.pipeline(), config: rs.config(), host: str, port: int, fps: int):
    """Run ZMQ PUB server: capture RealSense frames and send msgpack { images: { name: base64_jpeg } }."""
    pipeline.start(config)
    print("✓ Pipeline started successfully!")

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    addr = f"tcp://{host}:{port}"
    socket.bind(addr)
    print(f"Streaming on {addr} (fps={fps}). Connect with stream_rs_client_ros2.py --ip <this_host>")
    print("Press Ctrl+C to stop.")

    frame_interval = 1.0 / fps if fps > 0 else 0.033
    frame_id = 0

    try:
        while True:
            t0 = time.monotonic()
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            # RealSense color is RGB when using rs.format.rgb8
            img = np.asanyarray(color_frame.get_data())
            _, jpeg = cv2.imencode(".jpg", img)
            b64 = base64.b64encode(jpeg.tobytes()).decode("ascii")
            payload = {"images": {"realsense": b64}}
            socket.send(msgpack.packb(payload))
            frame_id += 1
            if frame_id % 100 == 0:
                print(f"  Sent {frame_id} frames")
            elapsed = time.monotonic() - t0
            time.sleep(max(0, frame_interval - elapsed))
    except KeyboardInterrupt:
        print("\nStopping stream...")
    finally:
        pipeline.stop()
        socket.close()
        context.term()


def main():
    parser = argparse.ArgumentParser(description="RealSense ZMQ stream server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5555, help="ZMQ port (default: 5555)")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS (default: 30)")
    parser.add_argument("--width", type=int, default=640, help="Color stream width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Color stream height (default: 480)")
    args = parser.parse_args()

    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        print("✗ No RealSense devices found!")
        sys.exit(1)
    list_devices(ctx)

    print("\nStarting camera pipeline...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.rgb8, args.fps)

    try:
        run_stream_server(pipeline, config, args.host, args.port, args.fps)
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
