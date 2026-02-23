#!/usr/bin/env python3
"""
ZED ZMQ stream → browser view.

Connects directly to a ZED stream server (e.g. stream_zed.py) at host:port and
serves the latest frame as MJPEG at http://<host>:<web-port>/ so you can view
the stream in a browser. No ROS2 required.

Server (on machine with ZED):
  python3 stream_zed.py --port 5556

This script (on any machine):
  python3 stream_image_web_view.py --host <stream_server_ip> --port 5556 --web-port 8080
  # Then open http://localhost:8080/ in a browser.
"""

import argparse
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import msgpack
import zmq

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


def run(stream_host: str, stream_port: int, web_port: int):
    if cv2 is None:
        print("Error: opencv-python required. pip install opencv-python", file=sys.stderr)
        sys.exit(1)
    if np is None:
        print("Error: numpy required. pip install numpy", file=sys.stderr)
        sys.exit(1)
    if not HAS_MSGPACK_NUMPY:
        print("Warning: msgpack_numpy recommended for numpy arrays. pip install msgpack-numpy", file=sys.stderr)

    latest_jpeg = None
    lock = threading.Lock()

    def get_latest():
        with lock:
            return latest_jpeg

    handler = _make_mjpeg_handler(get_latest)
    httpd = HTTPServer(("0.0.0.0", web_port), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    print("Web view: http://0.0.0.0:%d/ and http://0.0.0.0:%d/stream" % (web_port, web_port))
    print("Connecting to ZED stream at tcp://%s:%d ... (Ctrl+C to stop)" % (stream_host, stream_port))

    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    socket.setsockopt(zmq.CONFLATE, True)
    socket.setsockopt(zmq.RCVHWM, 3)
    socket.connect("tcp://%s:%d" % (stream_host, stream_port))

    def decode(packed):
        try:
            return msgpack.unpackb(packed)
        except Exception:
            return msgpack.unpackb(packed, ext_hook=m.decode if HAS_MSGPACK_NUMPY else None)

    try:
        while True:
            packed = socket.recv()
            data = decode(packed)
            if "image" not in data:
                continue
            image = data["image"]
            if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
                continue
            # Stream sends RGB; OpenCV encode expects BGR
            bgr = image[:, :, ::-1].copy()
            ok, jpeg = cv2.imencode(".jpg", bgr)
            if ok:
                with lock:
                    latest_jpeg = jpeg.tobytes()
    except KeyboardInterrupt:
        pass
    finally:
        socket.close()
        context.term()


def main():
    parser = argparse.ArgumentParser(
        description="Connect to ZED ZMQ stream and serve live view in browser (no ROS2)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_STREAM_HOST,
        help="ZED stream server host (e.g. 192.168.50.251)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_STREAM_PORT,
        help="ZED stream server port",
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=DEFAULT_WEB_PORT,
        help="HTTP port for browser view",
    )
    parsed = parser.parse_args()
    run(stream_host=parsed.host, stream_port=parsed.port, web_port=parsed.web_port)


if __name__ == "__main__":
    main()
