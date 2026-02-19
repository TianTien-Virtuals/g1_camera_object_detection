If not setup

Run Virtuals-ShoonKit setup_g1
verify_g1
sudo systemctl stop g1-camera
run stream_zed.py and stream_rs.py


# Viewing camera streams in FoxGlove (ROS2)

These scripts publish camera images as ROS2 **Image** topics so you can view them in [FoxGlove Studio](https://foxglove.dev/) (or `rqt_image_view`).

## Quick test (no camera)

To confirm ROS2 and FoxGlove work without any hardware:

1. **Terminal 1 – start test publisher**
   ```bash
   cd /home/robo/g1_camera_object_detection
   source /opt/ros/humble/setup.bash
   python3 source/test_ros2_image.py
   ```
   You should see: `Publishing test images on /camera/test/image_raw`

2. **Terminal 2 – check topic**
   ```bash
   source /opt/ros/humble/setup.bash
   ros2 topic list
   ```
   You should see `/camera/test/image_raw`.

3. **View the image**
   - **Option A – rqt:** `ros2 run rqt_image_view rqt_image_view` → in the dropdown, select **Topic** → pick `/camera/test/image_raw`. If the list is empty, click the refresh icon or run the publisher first and wait a few seconds.
   - **Option B – FoxGlove:** You must use **Foxglove WebSocket** and run the bridge (see below). Then in FoxGlove: add **Image** panel → in the panel settings, select topic `/camera/test/image_raw`.

Stop the test with Ctrl+C in Terminal 1.

### If rqt_image_view or FoxGlove show no image

1. **Confirm the topic is publishing**
   ```bash
   ros2 topic list
   ros2 topic hz /camera/test/image_raw
   ```
   You should see `/camera/test/image_raw` and a ~30 Hz rate. If `topic list` doesn’t show it, the publisher isn’t running or discovery failed (same terminal/shell: source ROS2 and run the script again).

2. **rqt_image_view**
   - Select the topic in the **Topic** dropdown (don’t leave it on “—”).
   - Or start rqt and add the plugin: `rqt` → Plugins → Visualization → Image View.

3. **FoxGlove**
   - **You must run the FoxGlove bridge** so Studio can see ROS2. In a terminal (with system ROS sourced, not only a workspace):
     ```bash
     source /opt/ros/jazzy/setup.bash
     ros2 launch foxglove_bridge foxglove_bridge_launch.xml port:=8765
     ```
   - In FoxGlove Studio: **Open connection** → **Foxglove WebSocket** → URL `ws://localhost:8765` (or `ws://<your_ip>:8765` from another machine) → **Open**.
   - Add a panel → **Image** → in the sidebar, set the topic to `/camera/test/image_raw`.
   - The native “ROS 2” connection in FoxGlove often does not discover your ROS2 daemon; prefer **Foxglove WebSocket** + bridge.

4. **Same machine and ROS domain**  
   All of these must use the same `ROS_DOMAIN_ID` (default 0). If you set it elsewhere, run `export ROS_DOMAIN_ID=0` (or the same value) in every terminal where you run the publisher, bridge, or tools.

## Fake camera server (ZMQ at 5555 / 5556)

To test the RealSense and ZED pipelines without hardware, run the fake server so something is emitting on the same ports:

```bash
# On the machine that will act as the source (e.g. 192.168.50.251), or use 0.0.0.0 to bind all interfaces
python3 source/fake_camera_server.py --host 0.0.0.0 --port-rs 5555 --port-zed 5556
```

- **5555**: RealSense-style stream (msgpack with base64 JPEG in `images` dict). Use `stream_rs.py` or `stream_rs_ros2.py` with `--ip <this_machine>`.
- **5556**: ZED-style stream (msgpack with numpy image + timestamp). Use a ZED client subscribing to this host:5556.

Options: `--rs-only` / `--zed-only`, `--fps`, `--width`, `--height`. Install `msgpack-numpy` for the ZED-style stream.

## Prerequisites

- ROS2 installed and sourced, e.g.:
  ```bash
  source /opt/ros/humble/setup.bash   # or iron, jazzy, etc.
  ```
- Python dependencies: `rclpy`, `sensor_msgs` (from your ROS2 install), plus existing deps for each stream (ZMQ/msgpack for RealSense, pyzed/opencv for ZED).

## 1. RealSense (robot ZMQ stream)

Publishes each camera from the Unitree G1 RealSense server to separate topics.

```bash
cd /home/robo/g1_camera_object_detection
source /opt/ros/humble/setup.bash
python3 source/stream_rs_ros2.py --ip 192.168.50.251
```

**Topics:**  
`/humanoid/realsense/image_raw` (single topic; if the server sends multiple cameras, the first by name is used)

**Where is the RealSense stream coming from?**  
The RealSense frames are **streamed from the robot** (or whatever host runs the Unitree camera stack) over **ZMQ** on **port 5555**:

- **Address:** `tcp://<robot_ip>:5555` (e.g. `tcp://192.168.50.251:5555`)
- **Configured in:** `test/stream_rs.py` (`ROBOT_IP`, `PORT`) and `stream_rs_client_ros2.py` (`DEFAULT_ROBOT_IP`, `PORT`). Override with `--ip <host>` when running the client.

**How to check it:**

1. **See which host/port your client uses**  
   Run the client with the same IP you use for the robot, e.g.  
   `python3 stream_rs_client_ros2.py --ip 192.168.50.251`  
   It will connect to `tcp://192.168.50.251:5555`.

2. **Check if something is listening on the robot**  
   On the robot (or over SSH):  
   `ss -tlnp | grep 5555`  
   If the RealSense ZMQ server is running, you should see port 5555 in LISTEN.

3. **Quick connectivity test from your PC**  
   `nc -zv 192.168.50.251 5555`  
   (Replace with your robot IP.) If the port is open, the connection succeeds.

4. **Receive frames**  
   Run `python3 test/stream_rs.py` (and set `ROBOT_IP` in the file if needed, or add `--ip` support). If you get a window with the camera image, the stream is at that IP and port 5555.

## 2. ZED camera
Run the ZED stream server on the machine with the camera, then run the client where you need ROS2 topics:

```bash
# On machine with ZED (server):
python3 source/stream_zed.py --port 5556

# On machine with ROS2 (client):
source /opt/ros/humble/setup.bash
python3 source/stream_zed_client_ros2.py --host <zed_server_ip>
# e.g. python3 source/stream_zed_client_ros2.py --host 192.168.50.251
```

**Topics (both options):**  
- `/camera/zed/image_raw` — color image  
- `/camera/zed/depth` — depth (if server sends depth / you use `--depth` on direct mode)

You can change prefix/frame with `--topic-prefix` and `--frame-id`. The client needs `msgpack-numpy` (`pip install msgpack-numpy`).

### ZED client → FoxGlove (step-by-step)

To get data from `stream_zed_client_ros2.py` and view it in FoxGlove:

1. **Start a ZED stream server** (pick one):
   - **Fake server (no camera):**  
     `python3 source/fake_camera_server.py --host 0.0.0.0 --port-zed 5556`  
     (Or run both RS and ZED: `--port-rs 5555 --port-zed 5556`. Needs `msgpack-numpy` for ZED.)
   - **Real ZED:**  
     On the machine with the ZED camera: `python3 source/stream_zed.py --port 5556`

2. **Start the ZED → ROS2 client** (on the machine where you want ROS2/FoxGlove):
   ```bash
   cd /home/robo/g1_camera_object_detection
   source /opt/ros/jazzy/setup.bash
   python3 source/stream_zed_client_ros2.py --host localhost --port 5556
   ```
   Use `--host <server_ip>` if the ZED server is on another machine. The client publishes `/camera/zed/image_raw` (and `/camera/zed/depth` if the server sends depth).

3. **Start the FoxGlove bridge** (so FoxGlove can see ROS2 topics):
   ```bash
   source /opt/ros/jazzy/setup.bash
   ros2 launch foxglove_bridge foxglove_bridge_launch.xml port:=8765
   ```
   Leave it running.

4. **Open FoxGlove Studio** (desktop or https://foxglove.dev/studio):
   - **Open connection** → **Foxglove WebSocket** → URL `ws://localhost:8765` → **Open**.
   - **Add panel** → **Image**.
   - In the Image panel settings, set the topic to **`/camera/zed/image_raw`** (or **`/camera/zed/depth`** for depth).

Order: start (1) and (2) first so the topic exists, then (3) bridge, then (4) connect FoxGlove so the topic appears in the list.

### View depth as 3D in Foxglove

To see depth as a **3D point cloud** in Foxglove:

1. Ensure the ZED server sends depth (e.g. `stream_zed.py` with depth enabled) and the client publishes **`/camera/zed/depth`** (and **`/camera/zed/image_raw`** for color).
2. Run the depth→pointcloud node:
   ```bash
   python3 depth_to_pointcloud_ros2.py
   ```
   This subscribes to `/camera/zed/depth` and optionally `/camera/zed/image_raw`, and publishes **`/camera/zed/points`** (PointCloud2).
3. In Foxglove: **Add panel** → **3D**. In the 3D panel settings, add a **PointCloud** layer and set the topic to **`/camera/zed/points`**.

You can tune intrinsics if needed: `--fx`, `--fy`, `--cx`, `--cy` (defaults assume ~720p; use your ZED resolution). Use `--step 4` to reduce points for performance.

**If `ros2 topic list` doesn’t show `/camera/zed/image_raw` or `ros2 topic hz /camera/zed/image_raw` shows no messages:**

- The client only publishes when it **receives and decodes** a frame from the ZED server. If the server isn’t running or the client can’t connect, no messages are published.
- Start the **ZED server first** (fake_camera_server or stream_zed on port 5556), then start the client with the correct `--host` (e.g. `localhost` if same machine).
- In the client terminal you should see `Published 1 frames...` then `Published 100 frames...` etc. If you never see that, the client isn’t getting data (check server, host, port, and that the server has `msgpack-numpy` for the ZED stream).
- Use the same ROS2 environment in all terminals (`source /opt/ros/jazzy/setup.bash`) and same `ROS_DOMAIN_ID` so `ros2 topic list` sees the node.

## Viewing in FoxGlove

1. **Open FoxGlove Studio** (desktop or https://foxglove.dev/studio).
2. **Connect to ROS2:**  
   - **Live connection:** choose **Foxglove WebSocket** or **ROS 2** (native).  
   - For **ROS 2**: ensure FoxGlove is using the same DDS/ROS_DOMAIN_ID as your scripts (default 0).  
   - For **Foxglove WebSocket**: run a bridge that subscribes to ROS2 and forwards to FoxGlove, e.g. [foxglove_bridge](https://github.com/foxglove/ros_foxglove_bridge).
3. **Add an Image panel:**  
   - Add panel → **Image**.  
   - Select topic:  
     - RealSense: `/humanoid/realsense/image_raw`  
     - ZED: `/camera/zed/image_raw` (or `/camera/zed/depth` for depth).

### Using foxglove_bridge (recommended for FoxGlove Studio)

If you prefer to connect via FoxGlove’s own protocol:

1. **Install the bridge** (if needed):  
   `sudo apt install ros-jazzy-foxglove-bridge`  (or `ros-humble-foxglove-bridge` for Humble)

2. **Run the bridge** in a terminal where **system ROS 2 is visible**.  
   If you use a local workspace (`ros2_jazzy`), `ros2 launch` only sees packages in that workspace, so the apt-installed `foxglove_bridge` won’t be found. Fix by sourcing **system ROS 2 first** (and optionally your workspace after):

```bash
# Use system Jazzy so foxglove_bridge is found (apt install puts it here)
source /opt/ros/jazzy/setup.bash
# Optional: then source your workspace if you need it for other nodes
# source /home/robo/ros2_jazzy/install/setup.bash

ros2 launch foxglove_bridge foxglove_bridge_launch.xml port:=8765
```

Then in FoxGlove Studio connect to **Foxglove WebSocket** with the URL shown by the bridge (e.g. `ws://localhost:8765`). Your Image topics will appear in the topic list; add an **Image** panel and pick the topic.

## SAM 3 (Ultralytics) concept segmentation

[Ultralytics SAM 3](https://docs.ultralytics.com/models/sam-3/) does **Promptable Concept Segmentation**: segment all instances of a concept given text prompts (e.g. "person", "car") or image exemplars. You can plug it into your camera pipeline and view the segmented image in FoxGlove.

### Setup

**Option A – Virtual environment (recommended if `pip install` gives "externally-managed-environment")**

The venv **must use the same Python version as ROS** (Jazzy uses Python 3.12). If your default `python3` is something else (e.g. Miniconda’s 3.13), rclpy’s C extension will not load. Create the venv with **Python 3.12** explicitly:

```bash
cd /home/robo/g1_camera_object_detection
# Use the same Python as ROS (3.12). If you use Miniconda, call system Python:
/usr/bin/python3.12 -m venv .venv --system-site-packages
# Or if python3.12 is on PATH:  python3.12 -m venv .venv --system-site-packages
source .venv/bin/activate
pip install -U ultralytics
# Optional: fix CLIP if you hit SimpleTokenizer error later
# pip uninstall clip -y && pip install git+https://github.com/ultralytics/CLIP.git
deactivate
```

If you already created `.venv` with another Python (e.g. 3.13), remove it and recreate with 3.12:

```bash
rm -rf .venv
/usr/bin/python3.12 -m venv .venv --system-site-packages
source .venv/bin/activate
pip install -U ultralytics
deactivate
```

Then run the SAM 3 node with that venv **after** sourcing ROS:

```bash
source /opt/ros/jazzy/setup.bash
source /home/robo/g1_camera_object_detection/.venv/bin/activate
python source/sam3_ros2.py --model sam3.pt --topic-in /camera/zed/image_raw --text person car --hz 2
```

Check that the venv is Python 3.12: `python --version` after activate should show `Python 3.12.x`.

**Option B – System or user install**

If your system allows it:

```bash
pip install -U ultralytics
# or: pip install --user -U ultralytics
```

2. **Get SAM 3 weights**: Request access on the [SAM 3 model page on Hugging Face](https://huggingface.co/models?search=sam3), then download `sam3.pt` and place it in your project (e.g. `g1_camera_object_detection/sam3.pt`) or pass its path with `--model`.
3. **CLIP dependency**: If you see `TypeError: 'SimpleTokenizer' object is not callable`, run (inside the same env where you installed ultralytics):
   ```bash
   pip uninstall clip -y
   pip install git+https://github.com/ultralytics/CLIP.git
   ```

### Run SAM 3 on a camera topic

The node subscribes to an image topic, runs SAM 3 with text prompts, and publishes the **annotated image** (masks overlaid) to a new topic so you can view it in FoxGlove.

```bash
cd /home/robo/g1_camera_object_detection
source /opt/ros/jazzy/setup.bash
python3 source/sam3_ros2.py --model sam3.pt --topic-in /camera/zed/image_raw --text person car --hz 2
```

- **`--topic-in`**: Image topic to segment (e.g. `/humanoid/zed/image_raw` or `/humanoid/realsense/image_raw`).
- **`--topic-out`**: Where to publish the result (default: `/camera/zed/sam3_segmentation`).
- **`--text`**: Space-separated concept prompts (default: `person`).
- **`--hz`**: Processing rate in Hz (default 2); lower saves GPU/CPU.

In FoxGlove, add an **Image** panel and select **`/camera/zed/sam3_segmentation`** (or whatever you set with `--topic-out`).

### Alternatives if you don’t have SAM 3 weights

If you can’t get access to `sam3.pt`, you can still run segmentation/detection with models that **auto-download**:

| Option | Model | Weights | What it does |
|--------|--------|--------|----------------|
| **YOLO segmentation** | `yolo11n-seg.pt` / `yolo8n-seg.pt` | Auto-download | Instance segmentation with **fixed COCO classes** (person, car, bicycle, etc.). No text prompts; detects 80 classes. |
| **SAM 2** | `sam2.pt` (if available in Ultralytics) | Check Ultralytics docs | Visual prompts (points/boxes); no text-concept segmentation. |

**YOLO segmentation (recommended alternative)** – same venv, no extra access:

```bash
source /opt/ros/jazzy/setup.bash
source /home/robo/g1_camera_object_detection/.venv/bin/activate
python source/yolo_seg_ros2.py --topic-in /camera/zed/image_raw --hz 5
```

- **`--model`**: e.g. `yolo11n-seg.pt` (nano, fast), `yolo8n-seg.pt`, `yolo11s-seg.pt` (small). Weights download on first run.
- **`--topic-out`**: default `/camera/zed/yolo_segmentation`. View this topic in FoxGlove.

You get masks and boxes for COCO classes (person, car, dog, etc.) without any manual weight download or Hugging Face access.

## Quick check with ROS2 CLI

```bash
source /opt/ros/humble/setup.bash
ros2 topic list | grep camera
ros2 topic echo /camera/zed/image_raw --once   # or realsense topic
```

You can also use **rqt_image_view** to confirm images:

```bash
ros2 run rqt_image_view rqt_image_view
# Choose the image topic from the dropdown.
```
