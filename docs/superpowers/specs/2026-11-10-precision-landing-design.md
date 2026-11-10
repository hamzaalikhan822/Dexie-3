# DEXI 3 Precision Landing — Design

Date: 2026-11-10

## Goal

Fly the DEXI 3 from a laptop: take off to 5 m, rotate 360°, then land back on the
takeoff spot, which is marked with an ArUco marker seen by the drone's
downward-facing camera.

## Decisions already made

- Control from a laptop over MAVLink using MAVSDK-Python. No code runs on the drone.
- Camera frames reach the laptop as an RTSP stream from the VOXL2 (`voxl-streamer`).
- Centering over the marker is a closed loop on the laptop: detect marker, send
  offboard body-velocity setpoints. PX4's built-in precision landing is not used.
- Camera intrinsics come from a one-off checkerboard calibration script in this repo.
- Testing is on the real drone from the first flight, with an RC pilot ready to
  take over. No simulator.

## Layout

```
requirements.txt    mavsdk, opencv-contrib-python, numpy, pyyaml
calibrate.py        checkerboard calibration of the RTSP stream -> camera.yaml
aruco_pose.py       detect one marker in a frame -> (x, y, z) metres in camera frame, or None
mission.py          the flight state machine
README.md           marker printing, PX4 params, calibration, how to run
```

Three files of code. No packages, no classes beyond what MAVSDK needs.

## calibrate.py

- Opens the RTSP URL (CLI arg), shows frames, grabs a frame on keypress when a
  checkerboard (inner-corner size as CLI args, default 9x6, square size in metres)
  is found.
- After ~20 captures, runs `cv2.calibrateCamera` and writes `camera.yaml` with
  `width`, `height`, `camera_matrix`, `dist_coeffs`, and the reprojection error.
- The stream resolution used here must be the one `mission.py` uses. README says so.

## aruco_pose.py

- `load_camera(path) -> (K, dist)`.
- `marker_offset(frame, K, dist, marker_id, marker_size_m) -> (x, y, z) | None`.
  Detects with `cv2.aruco` (dictionary `DICT_4X4_50`, configurable), picks the
  marker with the requested id, runs `solvePnP` on its four corners, returns the
  translation vector in metres in the OpenCV camera frame (x right, y down, z
  forward = toward the ground for a downward camera).
- Pure function. No drone dependency, so it is testable on a saved image.

## mission.py

CLI flags: `--url` (MAVLink, default `udp://:14540`), `--rtsp`, `--camera camera.yaml`,
`--marker-id`, `--marker-size`, `--alt 5`, `--dry`.

A background thread reads the RTSP stream and keeps only the latest frame plus its
timestamp (OpenCV buffers otherwise lag by seconds).

Flight sequence, run with `asyncio` and MAVSDK:

1. **Connect and check**: connect, wait for `health.is_global_position_ok` and
   `is_home_position_ok`. Confirm the marker is visible on the ground before
   arming; abort if not, since the drone starts on the marker.
2. **Takeoff**: set `MIS_TAKEOFF_ALT` to `--alt`, arm, `action.takeoff()`, wait
   until relative altitude is within 0.3 m of target.
3. **Rotate 360°**: start offboard with `PositionNedYaw` at the current NED
   position. Advance the yaw setpoint 1° every 0.1 s (about 36 s per turn) until
   it has moved 360°, then hold the original heading for 2 s. Yaw-rate is kept
   low so the camera stays usable.
4. **Centre**: loop at ~10 Hz. Get the latest frame, call `marker_offset`. Map
   camera offset to body velocity: forward = -y_cam * Kp, right = x_cam * Kp
   (sign confirmed on the first flight with `--dry` holding the drone by hand),
   clamp to 0.3 m/s. Send `VelocityBodyYawspeed(fwd, right, 0, 0)`. Advance when
   the horizontal offset stays below 0.15 m for 1 s.
5. **Descend**: same loop with down velocity 0.3 m/s, still correcting
   horizontally, until relative altitude < 1.0 m.
6. **Land**: stop offboard, `action.land()`, wait for `in_air == False`, disarm.

Camera-to-body mapping assumes the camera's image "up" points toward the drone's
nose. The README has a one-line check for this using `--dry`; if the camera is
mounted rotated, a `--cam-yaw` flag (degrees) rotates the offset before use.

### Safety

- Marker not seen in steps 4 or 5: send zero velocity (hold) at once. Not seen
  for > 10 s: `action.land()` where it is, log it, exit.
- Any exception, `KeyboardInterrupt`, or offboard refused: `action.land()`, exit.
- RC pilot switching mode drops PX4 out of offboard; the script detects the
  `flight_mode` change, stops sending setpoints, and exits without touching the
  drone further.
- Velocity clamp 0.3 m/s horizontal, 0.3 m/s down. Gains start conservative
  (Kp = 0.5) and are CLI flags.

### --dry mode

Connects to the drone and the stream, never arms. Prints the marker offset and the
velocity command it would send, at 2 Hz. Used for: checking the stream, checking
calibration, confirming the camera axis mapping by moving the drone by hand.

## Marker and PX4 setup (README)

- Marker: `DICT_4X4_50`, id 0, printed 35 cm, on a flat matte surface. Must be
  large enough to be found from 5 m with the VOXL downward cam; if not, a 50 cm
  print.
- PX4: offboard must be allowed from a companion link (`COM_RC_IN_MODE` or
  `COM_OBL_RC_ACT` set so losing RC does not fight the test; left at defaults
  unless the first flight shows a problem). `MIS_TAKEOFF_ALT` set by the script.
- MAVLink to laptop: VOXL2 forwards MAVLink over Wi-Fi UDP to the laptop IP on
  14540 (`voxl-vision-px4` / `voxl-mavlink-server` config).

## Testing

1. `aruco_pose.py` self-check: a saved frame with a known marker returns a
   sensible z close to the measured camera height. One assert-based check at
   the bottom of the file, run with `python aruco_pose.py image.jpg`.
2. `--dry` on the ground, then with the drone lifted and moved by hand to
   confirm offset signs.
3. First flight: `--alt 2 --kp 0`, so centre/descend become a plain vertical
   descent and only takeoff, rotation and landing are exercised. Then the full
   sequence at 2 m, then at 5 m.

## Out of scope

Onboard execution on the VOXL2, PX4 precision-landing mode, multiple markers,
ROS 2, simulation, Kalman filtering of the marker pose, wind compensation.
