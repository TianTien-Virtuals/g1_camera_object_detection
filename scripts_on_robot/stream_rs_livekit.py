#!/usr/bin/env python3
"""
RealSense LiveKit stream server.

Streams RealSense color frames to a LiveKit room (same pattern as stream_zed_livekit.py).
Optionally also publishes to ZMQ on a separate port (stream_rs.py format).

Usage:
  python3 stream_rs_livekit.py --livekit-url ws://localhost:7880 --livekit-room robot-stream
  python3 stream_rs_livekit.py --livekit-url ws://192.168.1.100:7880 --port 5555  # LiveKit + ZMQ

Requires: pyrealsense2, opencv-python, numpy, livekit
"""

import argparse
import asyncio
import sys
import time

try:
    import cv2
    import numpy as np
except ImportError as e:
    print("Error: %s" % e)
    sys.exit(1)

try:
    import pyrealsense2 as rs
except ImportError:
    print("Error: pyrealsense2 not installed.")
    sys.exit(1)

try:
    from livekit import rtc, api
    LIVEKIT_AVAILABLE = True
except ImportError:
    LIVEKIT_AVAILABLE = False

try:
    import zmq
    import msgpack
    import base64
    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False


def run_livekit(
    pipeline: rs.pipeline,
    width: int,
    height: int,
    fps: int,
    url: str,
    api_key: str,
    api_secret: str,
    room_name: str,
    identity: str = "robot",
    name: str = "RealSense Camera",
    zmq_socket=None,
):
    """Capture RealSense frames and stream to LiveKit (and optionally ZMQ)."""

    def get_frame():
        try:
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            color_frame = frames.get_color_frame()
            if not color_frame:
                return None
            # RealSense color is RGB when using rs.format.rgb8
            img = np.asanyarray(color_frame.get_data())
            if img.ndim != 3 or img.shape[2] != 3:
                return None
            return img
        except RuntimeError:
            return None

    async def run():
        if not LIVEKIT_AVAILABLE:
            raise RuntimeError("LiveKit SDK not installed. pip install livekit")

        token = (
            api.AccessToken(api_key, api_secret)
            .with_identity(identity)
            .with_name(name)
            .with_grants(api.VideoGrants(room_join=True, room=room_name))
            .to_jwt()
        )

        room = rtc.Room()
        await room.connect(url, token)
        print("[LiveKit] Connected to room: %s" % room.name)

        source = rtc.VideoSource(width, height)
        track = rtc.LocalVideoTrack.create_video_track("realsense-cam", source)
        await room.local_participant.publish_track(
            track,
            rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA),
        )
        print("[LiveKit] Publishing RealSense camera track...")

        interval = 1.0 / fps if fps > 0 else 0.033
        frame_id = 0

        try:
            while True:
                frame = await asyncio.to_thread(get_frame)
                if frame is None:
                    await asyncio.sleep(0.01)
                    continue

                if frame.shape[0] != height or frame.shape[1] != width:
                    frame = cv2.resize(frame, (width, height))

                rgba = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
                source.capture_frame(
                    rtc.VideoFrame(
                        width=width,
                        height=height,
                        type=rtc.VideoBufferType.RGBA,
                        data=rgba.tobytes(),
                    )
                )

                if zmq_socket is not None:
                    _, jpeg = cv2.imencode(".jpg", frame)
                    b64 = base64.b64encode(jpeg.tobytes()).decode("ascii")
                    payload = {"images": {"realsense": b64}}
                    try:
                        zmq_socket.send(msgpack.packb(payload))
                    except Exception:
                        pass

                frame_id += 1
                if frame_id % 100 == 0:
                    print("  Sent %d frames" % frame_id)

                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            pass
        finally:
            await room.disconnect()
            print("[LiveKit] Disconnected")

    return asyncio.run(run())


def main():
    parser = argparse.ArgumentParser(
        description="RealSense LiveKit stream server (stream to room; optional ZMQ)"
    )
    parser.add_argument("--host", default="0.0.0.0", help="ZMQ bind address (if --port used)")
    parser.add_argument("--port", type=int, default=0,
                        help="ZMQ port (0 = disable). If > 0, also publish msgpack to ZMQ.")
    parser.add_argument("--width", type=int, default=640, help="Color stream width")
    parser.add_argument("--height", type=int, default=480, help="Color stream height")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS")
    parser.add_argument("--livekit-url", type=str, required=True,
                        help="LiveKit server URL (e.g. ws://localhost:7880)")
    parser.add_argument("--livekit-api-key", type=str, default="devkey")
    parser.add_argument("--livekit-api-secret", type=str, default="secret")
    parser.add_argument("--livekit-room", type=str, default="robot-stream")
    parser.add_argument("--livekit-identity", type=str, default="robot")
    parser.add_argument("--livekit-name", type=str, default="RealSense Camera")
    args = parser.parse_args()

    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        print("No RealSense devices found.")
        sys.exit(1)
    print("RealSense device(s) found: %d" % len(devices))

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.rgb8, args.fps)

    zmq_socket = None
    if args.port > 0 and ZMQ_AVAILABLE:
        context = zmq.Context()
        zmq_socket = context.socket(zmq.PUB)
        zmq_socket.bind("tcp://%s:%d" % (args.host, args.port))
        print("ZMQ publishing on tcp://%s:%d" % (args.host, args.port))
    elif args.port > 0:
        print("Warning: ZMQ disabled (zmq/msgpack not installed)")

    try:
        pipeline.start(config)
        print("Pipeline started: %dx%d @ %d fps" % (args.width, args.height, args.fps))
        run_livekit(
            pipeline,
            width=args.width,
            height=args.height,
            fps=args.fps,
            url=args.livekit_url,
            api_key=args.livekit_api_key,
            api_secret=args.livekit_api_secret,
            room_name=args.livekit_room,
            identity=args.livekit_identity,
            name=args.livekit_name,
            zmq_socket=zmq_socket,
        )
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print("Error: %s" % e)
        sys.exit(1)
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass
        if zmq_socket is not None:
            zmq_socket.close()
            zmq_socket.context.term()
        print("RealSense pipeline stopped.")


if __name__ == "__main__":
    main()
