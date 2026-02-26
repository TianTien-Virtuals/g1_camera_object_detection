#!/usr/bin/env python3
"""
ZED stream → browser view (MJPEG).

Supports two image sources:
  1. ZMQ: connect to stream_zed.py at host:port (default).
  2. LiveKit: join a room and subscribe to the first video track (e.g. robot publishing ZED).

No ROS2 required.

ZMQ source:
  Server: python3 stream_zed.py --port 5556
  Viewer: python3 stream_image_web_view.py --host <ip> --port 5556 --web-port 8080

LiveKit source:
  Server: robot runs stream_zed_livekit.py (publishes ZED to a room).
  Token:  get JWT from your auth (e.g. GET http://host:8080/token) or use LIVEKIT_TOKEN env.
  Viewer: python3 stream_image_web_view.py --source livekit --livekit-url ws://host:7880 --livekit-token <JWT> --web-port 8080
  # Then open http://localhost:8080/ in a browser.
"""

import argparse
import asyncio
import os
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import msgpack
import zmq

try:
    from livekit import rtc
    LIVEKIT_AVAILABLE = True
except ImportError:
    LIVEKIT_AVAILABLE = False

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    import msgpack_numpy as m
    m.patch()
    HAS_MSGPACK_NUMPY = True
except ImportError:
    HAS_MSGPACK_NUMPY = False

DEFAULT_STREAM_HOST = "127.0.0.1"
DEFAULT_STREAM_PORT = 5556
DEFAULT_WEB_PORT = 8080

MJPEG_BOUNDARY = b"streamframe"


class ReuseAddrHTTPServer(HTTPServer):
    """Allow binding to the same port shortly after the previous process exited."""

    def server_bind(self):
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        super().server_bind()


def _make_mjpeg_handler(get_latest_frame):
    """Build a request handler that serves / and /stream (MJPEG). get_latest_frame() -> (bytes|None)."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_GET(self):
            if self.path == "/" or self.path == "/index.html":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(
                    b"<!DOCTYPE html><html><head><title>ZED stream</title></head>"
                    b"<body><h1>Live stream</h1><img src='/stream' alt='Live stream' /></body></html>"
                )
                return
            if self.path == "/stream":
                frame = get_latest_frame()
                if frame is None:
                    self.send_response(503)
                    self.send_header("Content-Type", "text/plain")
                    self.end_headers()
                    self.wfile.write(b"No frame yet")
                    return
                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    "multipart/x-mixed-replace; boundary=" + MJPEG_BOUNDARY.decode(),
                )
                self.end_headers()
                try:
                    while True:
                        frame = get_latest_frame()
                        if frame is None:
                            break
                        self.wfile.write(
                            b"--" + MJPEG_BOUNDARY + b"\r\nContent-Type: image/jpeg\r\n\r\n"
                        )
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                        time.sleep(1.0 / 30.0)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            self.send_response(404)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Not found")

    return Handler


def _frame_to_jpeg(frame) -> bytes | None:
    """Convert LiveKit VideoFrame to JPEG bytes (BGR for cv2).

    LiveKit often delivers decoded frames as NV12 (YUV 4:2:0): buffer size = width*height*3/2.
    Sender may publish RGBA; the server can transcode to NV12 for transport.
    """
    if cv2 is None or np is None:
        return None
    try:
        w, h = frame.width, frame.height
        buf = bytes(frame.data) if hasattr(frame.data, "__bytes__") else frame.data
        if not buf or w <= 0 or h <= 0:
            return None
        n = len(buf)

        # RGBA: exact match
        if n == h * w * 4:
            arr = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        # RGB
        elif n == h * w * 3:
            arr = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 3))
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        # NV12 (YUV 4:2:0): common for decoded WebRTC. size = w*h*3/2 (e.g. 1280*720*1.5 = 1382400)
        elif n == (h * w * 3) // 2:
            nv12 = np.frombuffer(buf, dtype=np.uint8).reshape((h * 3 // 2, w))
            bgr = cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)
        else:
            return None
        ok, jpeg = cv2.imencode(".jpg", bgr)
        return jpeg.tobytes() if ok else None
    except Exception as e:
        if not hasattr(_frame_to_jpeg, "_logged"):
            print("[LiveKit] Frame to JPEG failed: %s" % e, file=sys.stderr)
            _frame_to_jpeg._logged = True
        return None


async def _run_livekit_source(
    livekit_url: str,
    token: str,
    latest_jpeg_ref: list,
    lock: threading.Lock,
    web_port: int = 8080,
):
    """Connect to LiveKit room, subscribe to first video track, push frames into latest_jpeg_ref."""
    room = rtc.Room()
    track_received = asyncio.Event()
    video_track_ref = []

    @room.on("track_subscribed")
    def _on_track(track, publication, participant):
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            video_track_ref.append(track)
            track_received.set()

    await room.connect(livekit_url, token)
    print("[LiveKit] Connected to room, waiting for video track (start robot stream if not already)...")

    # Wait for track_subscribed or check existing participants
    for _ in range(300):  # ~30s
        if video_track_ref:
            break
        for participant in room.remote_participants.values():
            for pub in participant.track_publications.values():
                if pub.subscribed and pub.track and pub.track.kind == rtc.TrackKind.KIND_VIDEO:
                    video_track_ref.append(pub.track)
                    break
            if video_track_ref:
                break
        if video_track_ref:
            break
        await asyncio.sleep(0.1)

    if not video_track_ref:
        print("[LiveKit] No video track after 30s. Is the robot publishing to this room? (e.g. stream_zed_livekit.py)", file=sys.stderr)
        await room.disconnect()
        return

    video_track = video_track_ref[0]
    stream = rtc.VideoStream.from_track(track=video_track)
    print("[LiveKit] Subscribed to video, streaming to MJPEG...")
    first = [True]
    try:
        async for frame_event in stream:
            frame = frame_event.frame
            jpeg = _frame_to_jpeg(frame)
            if jpeg:
                with lock:
                    latest_jpeg_ref[0] = jpeg
                if first[0]:
                    first[0] = False
                    print("[LiveKit] First frame received. View at http://0.0.0.0:%d/ or /stream" % web_port)
    except asyncio.CancelledError:
        pass
    finally:
        await room.disconnect()
        print("[LiveKit] Disconnected")


def run_web_server(web_port: int, get_latest_frame):
    """Start HTTP server in current thread (blocks). get_latest_frame() -> bytes|None."""
    handler = _make_mjpeg_handler(get_latest_frame)
    try:
        httpd = ReuseAddrHTTPServer(("0.0.0.0", web_port), handler)
    except OSError as e:
        if e.errno == 98:
            print("Error: port %d already in use. Try --web-port N (e.g. 8081)." % web_port, file=sys.stderr)
        else:
            print("Error binding to 0.0.0.0:%d: %s" % (web_port, e), file=sys.stderr)
        sys.exit(1)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    print("Web view: http://0.0.0.0:%d/ and http://0.0.0.0:%d/stream" % (web_port, web_port))
    return httpd


def run_zmq(stream_host: str, stream_port: int, web_port: int):
    """Use ZMQ as image source (stream_zed.py)."""
    if cv2 is None or np is None:
        print("Error: opencv-python and numpy required.", file=sys.stderr)
        sys.exit(1)
    if not HAS_MSGPACK_NUMPY:
        print("Warning: msgpack_numpy recommended. pip install msgpack-numpy", file=sys.stderr)

    latest_jpeg_ref = [None]
    lock = threading.Lock()

    def get_latest():
        with lock:
            return latest_jpeg_ref[0]

    run_web_server(web_port, get_latest)
    print("Connecting to ZED stream at tcp://%s:%d ... (Ctrl+C to stop)" % (stream_host, stream_port))
    print("Waiting for first frame... (ensure stream_zed.py or stream_zed_both.py is running on the robot)")

    context = zmq.Context()
    sock = context.socket(zmq.SUB)
    sock.setsockopt_string(zmq.SUBSCRIBE, "")
    sock.setsockopt(zmq.CONFLATE, True)
    sock.setsockopt(zmq.RCVHWM, 3)
    sock.connect("tcp://%s:%d" % (stream_host, stream_port))

    def decode(packed):
        try:
            return msgpack.unpackb(packed)
        except Exception:
            return msgpack.unpackb(packed, ext_hook=m.decode if HAS_MSGPACK_NUMPY else None)

    first = [True]
    try:
        while True:
            packed = sock.recv()
            data = decode(packed)
            if "image" not in data:
                continue
            image = data["image"]
            if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
                continue
            bgr = image[:, :, ::-1].copy()
            ok, jpeg = cv2.imencode(".jpg", bgr)
            if ok:
                with lock:
                    latest_jpeg_ref[0] = jpeg.tobytes()
                if first[0]:
                    first[0] = False
                    print("First frame received. View at http://0.0.0.0:%d/ or http://0.0.0.0:%d/stream" % (web_port, web_port))
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
        context.term()


def run_livekit(livekit_url: str, token: str, web_port: int):
    """Use LiveKit room as image source (subscribe to first video track)."""
    if not LIVEKIT_AVAILABLE:
        print("Error: LiveKit required. pip install livekit", file=sys.stderr)
        sys.exit(1)
    if cv2 is None or np is None:
        print("Error: opencv-python and numpy required for JPEG encoding.", file=sys.stderr)
        sys.exit(1)

    latest_jpeg_ref = [None]
    lock = threading.Lock()

    def get_latest():
        with lock:
            return latest_jpeg_ref[0]

    run_web_server(web_port, get_latest)
    print("Connecting to LiveKit at %s ... (Ctrl+C to stop)" % livekit_url)
    try:
        asyncio.run(_run_livekit_source(livekit_url, token, latest_jpeg_ref, lock, web_port))
    except KeyboardInterrupt:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Connect to ZED stream (ZMQ or LiveKit) and serve live view in browser (no ROS2)"
    )
    parser.add_argument(
        "--source",
        choices=("zmq", "livekit"),
        default="zmq",
        help="Image source: zmq (stream_zed) or livekit (room video track)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_STREAM_HOST,
        help="ZED ZMQ server host (for --source zmq)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_STREAM_PORT,
        help="ZED ZMQ server port (for --source zmq)",
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=DEFAULT_WEB_PORT,
        help="HTTP port for browser view",
    )
    parser.add_argument(
        "--livekit-url",
        type=str,
        default=None,
        help="LiveKit server URL (e.g. ws://192.168.1.100:7880). Required for --source livekit.",
    )
    parser.add_argument(
        "--livekit-token",
        type=str,
        default=None,
        help="LiveKit JWT token. Or set env LIVEKIT_TOKEN. Required for --source livekit.",
    )
    parsed = parser.parse_args()

    if parsed.source == "livekit":
        token = parsed.livekit_token or os.environ.get("LIVEKIT_TOKEN", "")
        if not parsed.livekit_url or not token:
            print("Error: --source livekit requires --livekit-url and --livekit-token (or LIVEKIT_TOKEN env)", file=sys.stderr)
            sys.exit(1)
        run_livekit(parsed.livekit_url, token, parsed.web_port)
    else:
        run_zmq(parsed.host, parsed.port, parsed.web_port)


if __name__ == "__main__":
    main()
