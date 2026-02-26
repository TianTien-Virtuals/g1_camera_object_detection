#!/usr/bin/env python3
"""
ZED Camera Streaming Script for Unitree G1

This script provides a standalone ZED camera streamer that can be used
for testing or as an alternative to the integrated camera service.

Usage:
    python3 stream_zed.py --port 5556 --fps 30
    python3 stream_zed.py --display  # Show local preview
    python3 stream_zed.py --save /path/to/output.mp4  # Record video
    python3 stream_zed.py --livekit-url ws://localhost:7880 --livekit-api-key devkey --livekit-api-secret secret --livekit-room robot-stream  # LiveKit

If you get "Failed to open ZED camera": another process (often g1-camera) is
using the ZED. Stop only the ZED part if your setup allows it, else stop the
whole camera service:
    # If you have a ZED-only service (e.g. g1-camera-zed):
    sudo systemctl stop g1-camera-zed   # RealSense keeps running
    # If you have a single g1-camera service (ZED + RealSense):
    sudo systemctl stop g1-camera       # stops both; restart when done
    python3 stream_zed.py --port 5556


I run : python ./scripts/stream_zed_livekit.py --livekit-url ws://192.168.123.222:7881 --livekit-api-key devkey --livekit-api-secret secret --livekit-room robot-stream
"""

import argparse
import asyncio
import time
import sys
from typing import Optional

import numpy as np

try:
    import pyzed.sl as sl
except ImportError:
    print("Error: pyzed not installed. Please install ZED SDK first.")
    print("Run: cd /usr/local/zed && python3 get_python_api.py")
    sys.exit(1)

try:
    import cv2
except ImportError:
    print("Error: opencv-python not installed.")
    print("Run: pip3 install opencv-python")
    sys.exit(1)

try:
    import zmq
    import msgpack
    import msgpack_numpy as m
    m.patch()
except ImportError:
    zmq = None
    print("Warning: zmq/msgpack not installed. Network streaming disabled.")

try:
    from livekit import rtc, api
    LIVEKIT_AVAILABLE = True
except ImportError:
    LIVEKIT_AVAILABLE = False


class ZEDStreamer:
    """ZED camera streamer with optional ZMQ publishing."""

    def __init__(
        self,
        resolution: str = "720p",
        fps: int = 30,
        enable_depth: bool = False,
        serial_number: Optional[int] = None,
    ):
        """Initialize ZED streamer.

        Args:
            resolution: Resolution string ('720p', '1080p', '2k', 'vga')
            fps: Target frames per second
            enable_depth: Whether to capture depth data
            serial_number: Optional ZED serial number for multi-camera
        """
        self.fps = fps
        self.enable_depth = enable_depth
        self.running = False

        # Initialize ZED camera
        self.zed = sl.Camera()

        # Configure initialization parameters
        init_params = sl.InitParameters()
        init_params.camera_resolution = self._parse_resolution(resolution)
        init_params.camera_fps = fps
        init_params.depth_mode = sl.DEPTH_MODE.ULTRA if enable_depth else sl.DEPTH_MODE.NONE
        init_params.coordinate_units = sl.UNIT.METER
        init_params.sdk_verbose = 0

        if serial_number is not None:
            init_params.set_from_serial_number(serial_number)

        # Open camera
        status = self.zed.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED camera: {status}")

        # Get camera info
        cam_info = self.zed.get_camera_information()
        self.width = cam_info.camera_configuration.resolution.width
        self.height = cam_info.camera_configuration.resolution.height
        print(f"ZED Camera initialized:")
        print(f"  Model: {cam_info.camera_model}")
        print(f"  Serial: {cam_info.serial_number}")
        print(f"  Resolution: {self.width}x{self.height} @ {fps}fps")
        print(f"  Depth: {'Enabled' if enable_depth else 'Disabled'}")

        # Create image containers
        self.image = sl.Mat()
        self.depth_map = sl.Mat() if enable_depth else None

        # Runtime parameters
        self.runtime_params = sl.RuntimeParameters()
        self.runtime_params.enable_fill_mode = True

        # ZMQ publisher (optional)
        self.zmq_context = None
        self.zmq_socket = None

        # Video writer (optional)
        self.video_writer = None

    def _parse_resolution(self, resolution: str) -> sl.RESOLUTION:
        """Parse resolution string to ZED enum."""
        resolutions = {
            "2k": sl.RESOLUTION.HD2K,
            "1080p": sl.RESOLUTION.HD1080,
            "720p": sl.RESOLUTION.HD720,
            "vga": sl.RESOLUTION.VGA,
        }
        return resolutions.get(resolution.lower(), sl.RESOLUTION.HD720)

    def start_zmq_server(self, port: int = 5556):
        """Start ZMQ publisher for network streaming."""
        if zmq is None:
            print("Error: ZMQ not available. Install with: pip3 install pyzmq msgpack msgpack-numpy")
            return False

        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.PUB)
        self.zmq_socket.bind(f"tcp://*:{port}")
        print(f"ZMQ publisher started on port {port}")
        return True

    def start_video_writer(self, output_path: str, codec: str = "mp4v"):
        """Start video recording."""
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.video_writer = cv2.VideoWriter(
            output_path, fourcc, self.fps, (self.width, self.height)
        )
        print(f"Recording to: {output_path}")

    def grab_frame(self) -> Optional[dict]:
        """Grab a single frame from ZED camera.

        Returns:
            Dictionary with 'image', 'timestamp', and optionally 'depth'
        """
        status = self.zed.grab(self.runtime_params)
        if status != sl.ERROR_CODE.SUCCESS:
            return None

        # Get RGB image
        self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
        image_data = self.image.get_data()

        # Convert BGRA to RGB
        rgb_image = image_data[:, :, :3][:, :, ::-1].copy()

        result = {
            "image": rgb_image,
            "timestamp": time.time(),
        }

        # Get depth if enabled
        if self.enable_depth and self.depth_map is not None:
            self.zed.retrieve_measure(self.depth_map, sl.MEASURE.DEPTH)
            result["depth"] = self.depth_map.get_data().copy()

        return result

    def publish_frame(self, frame: dict):
        """Publish frame over ZMQ."""
        if self.zmq_socket is None:
            return

        message = {
            "timestamp": frame["timestamp"],
            "image": frame["image"],
        }
        if "depth" in frame:
            message["depth"] = frame["depth"]

        packed = msgpack.packb(message, default=m.encode)
        self.zmq_socket.send(packed)

    def run(self, display: bool = False, duration: Optional[float] = None):
        """Run the streaming loop.

        Args:
            display: Show local preview window
            duration: Optional duration in seconds (None = run forever)
        """
        self.running = True
        start_time = time.time()
        frame_count = 0
        fps_time = time.time()

        print("Streaming started. Press Ctrl+C to stop.")

        try:
            while self.running:
                # Check duration
                if duration and (time.time() - start_time) > duration:
                    break

                # Grab frame
                frame = self.grab_frame()
                if frame is None:
                    continue

                frame_count += 1

                # Publish over ZMQ
                if self.zmq_socket:
                    self.publish_frame(frame)

                # Write to video file
                if self.video_writer:
                    # Convert RGB to BGR for OpenCV
                    bgr_frame = frame["image"][:, :, ::-1]
                    self.video_writer.write(bgr_frame)

                # Display preview
                if display:
                    # Convert RGB to BGR for OpenCV display
                    bgr_frame = frame["image"][:, :, ::-1]
                    cv2.imshow("ZED Stream", bgr_frame)

                    if self.enable_depth and "depth" in frame:
                        # Normalize depth for visualization
                        depth_vis = frame["depth"].copy()
                        depth_vis = np.nan_to_num(depth_vis, nan=0.0, posinf=10.0, neginf=0.0)
                        depth_vis = np.clip(depth_vis, 0, 10)  # Clip to 10m max
                        depth_vis = (depth_vis / 10.0 * 255).astype(np.uint8)
                        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                        cv2.imshow("ZED Depth", depth_color)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                # Print FPS every 2 seconds
                if time.time() - fps_time >= 2.0:
                    fps = frame_count / (time.time() - fps_time)
                    print(f"FPS: {fps:.1f}")
                    frame_count = 0
                    fps_time = time.time()

        except KeyboardInterrupt:
            print("\nStopping...")

        self.running = False

    async def run_livekit(
        self,
        url: str,
        api_key: str,
        api_secret: str,
        room_name: str,
        identity: str = "robot",
        name: str = "Robot Camera",
    ):
        """Stream ZED frames to a LiveKit room (replaces or complements ZMQ).

        Runs until self.running is False (e.g. KeyboardInterrupt).
        """
        if not LIVEKIT_AVAILABLE:
            raise RuntimeError("LiveKit SDK not installed. pip install livekit-api livekit-rtc")

        token = (
            api.AccessToken(api_key, api_secret)
            .with_identity(identity)
            .with_name(name)
            .with_grants(api.VideoGrants(room_join=True, room=room_name))
            .to_jwt()
        )

        room = rtc.Room()
        await room.connect(url, token)
        print(f"[LiveKit] Connected to room: {room.name}")

        source = rtc.VideoSource(self.width, self.height)
        track = rtc.LocalVideoTrack.create_video_track("zed-cam", source)
        await room.local_participant.publish_track(
            track,
            rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA),
        )
        print("[LiveKit] Publishing ZED camera track...")

        self.running = True
        interval = 1.0 / self.fps

        try:
            while self.running:
                # Grab frame in thread so we don't block the event loop
                frame = await asyncio.to_thread(self.grab_frame)
                if frame is None:
                    await asyncio.sleep(0.01)
                    continue

                rgb = frame["image"]
                if rgb.shape[0] != self.height or rgb.shape[1] != self.width:
                    rgb = cv2.resize(rgb, (self.width, self.height))
                rgba = cv2.cvtColor(rgb, cv2.COLOR_RGB2RGBA)

                source.capture_frame(
                    rtc.VideoFrame(
                        width=self.width,
                        height=self.height,
                        type=rtc.VideoBufferType.RGBA,
                        data=rgba.tobytes(),
                    )
                )

                # Optionally publish same frame to ZMQ if server is running
                if self.zmq_socket is not None:
                    self.publish_frame(frame)

                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            pass
        finally:
            await room.disconnect()
            print("[LiveKit] Disconnected")

    def close(self):
        """Clean up resources."""
        if self.video_writer:
            self.video_writer.release()
            print("Video saved")

        if self.zmq_socket:
            self.zmq_socket.close()
        if self.zmq_context:
            self.zmq_context.term()

        self.zed.close()
        cv2.destroyAllWindows()
        print("ZED camera closed")


def main():
    parser = argparse.ArgumentParser(description="ZED Camera Streamer for Unitree G1")
    parser.add_argument("--resolution", type=str, default="720p",
                        choices=["vga", "720p", "1080p", "2k"],
                        help="Camera resolution")
    parser.add_argument("--fps", type=int, default=30,
                        help="Target frames per second")
    parser.add_argument("--depth", action="store_true",
                        help="Enable depth capture")
    parser.add_argument("--serial", type=int, default=None,
                        help="ZED camera serial number (for multi-camera)")
    parser.add_argument("--port", type=int, default=5556,
                        help="ZMQ publisher port (0 to disable)")
    parser.add_argument("--display", action="store_true",
                        help="Show local preview window")
    parser.add_argument("--save", type=str, default=None,
                        help="Save video to file path")
    parser.add_argument("--duration", type=float, default=None,
                        help="Recording duration in seconds")
    # LiveKit (optional)
    parser.add_argument("--livekit-url", type=str, default=None,
                        help="LiveKit server URL (e.g. ws://localhost:7880). If set, stream to LiveKit instead of/in addition to ZMQ.")
    parser.add_argument("--livekit-api-key", type=str, default="devkey",
                        help="LiveKit API key")
    parser.add_argument("--livekit-api-secret", type=str, default="secret",
                        help="LiveKit API secret")
    parser.add_argument("--livekit-room", type=str, default="robot-stream",
                        help="LiveKit room name")
    parser.add_argument("--livekit-identity", type=str, default="robot",
                        help="LiveKit participant identity")
    parser.add_argument("--livekit-name", type=str, default="Robot Camera",
                        help="LiveKit participant display name")

    args = parser.parse_args()

    # Create streamer (ZED can only be opened by one process at a time)
    try:
        streamer = ZEDStreamer(
            resolution=args.resolution,
            fps=args.fps,
            enable_depth=args.depth,
            serial_number=args.serial,
        )
    except RuntimeError as e:
        if "Failed to open ZED" in str(e):
            print("ZED camera failed to start. If g1-camera service is running, stop it first:")
            print("  sudo systemctl stop g1-camera")
            print("Then run this script again. Restart the service when done: sudo systemctl start g1-camera")
        raise

    try:
        # Start ZMQ server if port specified (used for LiveKit+ZMQ or ZMQ-only)
        if args.port > 0:
            streamer.start_zmq_server(args.port)

        # Start video recording if path specified
        if args.save:
            streamer.start_video_writer(args.save)

        if args.livekit_url:
            if not LIVEKIT_AVAILABLE:
                print("Error: LiveKit not installed. pip install livekit-api livekit-rtc")
                sys.exit(1)
            # Stream to LiveKit (optionally also publishes to ZMQ if --port > 0)
            asyncio.run(
                streamer.run_livekit(
                    url=args.livekit_url,
                    api_key=args.livekit_api_key,
                    api_secret=args.livekit_api_secret,
                    room_name=args.livekit_room,
                    identity=args.livekit_identity,
                    name=args.livekit_name,
                )
            )
        else:
            # Run normal ZMQ/display/save loop
            streamer.run(display=args.display, duration=args.duration)

    finally:
        streamer.close()


if __name__ == "__main__":
    main()


# 16095190 On 24

