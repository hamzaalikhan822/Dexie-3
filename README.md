# DEXI 3 Precision Landing (laptop)

Takes off to N m, yaws 360°, centres on an ArUco marker with the downward camera, lands on it.
Everything runs on the laptop over Wi-Fi: MAVLink via MAVSDK, video via RTSP.

## Setup

1. `pip install -r requirements.txt`
   The `mavsdk` package is pinned below version 4 because this code uses the gRPC MAVSDK-Python API (the pip package was renamed `mavsdk-grpc`; versions 4 and up are a different native binding and will not work).
2. Marker: ArUco `DICT_4X4_50`, id 0, printed **35 cm** square on matte paper/board, white border
   at least 5 cm. Generate with:
   ```
   python -c "import cv2;cv2.imwrite('marker0.png',cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),0,1000))"
   ```
   If the marker is not detected from 5 m in `--dry`, print 50 cm and pass `--marker-size 0.5`.
3. VOXL2: enable `voxl-streamer` for the downward (tracking or hires) camera. Note the RTSP URL,
   e.g. `rtsp://192.168.8.1:8900/hires`. Forward MAVLink to the laptop IP on UDP 14540 in
   `voxl-mavlink-server` / `voxl-vision-px4` config.
4. Calibrate once, at the same stream resolution the mission uses:
   `python calibrate.py rtsp://<voxl-ip>:8900/hires --cols 9 --rows 6 --square 0.025`
   Capture ~20 frames with the board at different tilts and positions. Reprojection error should be < 1 px.

## Flight test order

Each step needs an RC pilot on the sticks with a mode switch set to Position mode. Switching mode
at any time kicks PX4 out of OFFBOARD; the script notices and exits. Before each run, have the RC mode switch in Position mode and the vehicle in HOLD/Loiter, not in a manual mode, because the script treats a manual mode seen during the climb as a pilot takeover and exits.

1. Bench: `python mission.py --rtsp <url> --dry`. Check frame age < 0.5 s and the marker is seen.
2. Hand-held: lift the drone 1 m, move it right. `right` must go negative. If `fwd` changes instead,
   add `--cam-yaw 90` (or `-90`) until only `right` changes. Keep that flag for every run.
3. `python mission.py --rtsp <url> --alt 2 --kp 0` : takeoff, yaw, straight descent, land.
4. `python mission.py --rtsp <url> --alt 2` : full loop at 2 m.
5. `python mission.py --rtsp <url> --alt 5` : the real thing.

## Safety behaviour

- Marker lost: holds position (zero velocity). Lost > 10 s: lands where it is.
- Any error or Ctrl-C: lands.
- Velocities are clamped to `--vmax` (0.3 m/s) horizontally and downward.
