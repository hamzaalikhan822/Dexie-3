# DEXI 3 Precision Landing (laptop)

Takes off to N m, yaws 360°, centres on an ArUco marker with the downward camera, lands on it.
Everything runs on the laptop over the drone's Wi-Fi: MAVLink via MAVSDK through the Pi's
mavlink-router, video from the ROS 2 camera topic through rosbridge.

## Setup

1. `pip install -r requirements.txt`
   The `mavsdk` package is pinned below version 4 because this code uses the gRPC MAVSDK-Python API (the pip package was renamed `mavsdk-grpc`; versions 4 and up are a different native binding and will not work).
2. Marker: the DroneBlocks **DEXI landing pad** sheet (AprilTag 36h11, id 0, black square 14.0 cm),
   taped flat. These are the defaults (`--dict DICT_APRILTAG_36h11 --marker-id 0 --marker-size 0.14`).
   A 14 cm tag is only detectable from about 2 m down with the 640x480 camera, so after the 360° the
   mission first returns to `--acquire-alt` (1.5 m) over the takeoff spot on PX4's position estimate,
   then starts looking for the tag. On the ground the camera is 4 cm from the pad and cannot see
   the tag, so there is no marker check before arming, only a live-stream check. A larger print of the same tag lets you raise that altitude.
3. Connect to DEXI 3 (Raspberry Pi CM5 companion, DEXI OS):
   - Power the drone. Join its Wi-Fi hotspot `dexi_XXXX` (XXXX = last 4 of its MAC), password `droneblocks`.
   - The Pi is `192.168.4.1`. Dashboard: http://192.168.4.1 (check the camera shows there). SSH: `ssh dexi@192.168.4.1`, password `dexi`.
   - MAVLink: mavlink-router on the Pi listens as a UDP server on **14540** (for MAVSDK) and **14550**
     (for QGroundControl). The laptop connects out: `udpout://192.168.4.1:14540` (the default). Do not
     run QGroundControl on 14540 at the same time.
   - Camera: DEXI OS publishes `/cam0/image_raw/compressed` (640x480 JPEG) and runs rosbridge on
     port 9090. The default camera URL `ws://192.168.4.1:9090` subscribes to that topic. Any
     `cv2.VideoCapture` URL (RTSP, HTTP MJPEG) also works in the same field.
   - Optional, to join your own Wi-Fi instead of the hotspot: on the Pi, `sudo dexi-wifi 'SSID' 'password'`,
     then replace 192.168.4.1 with the address your router gives it.
   - The stock DEXI 3 camera is forward-facing. This project needs it pointing down at the marker.
   - DEXI 3 has no GPS. Height above the floor is PX4's local Z re-zeroed on the downward
     rangefinder (MAVLink `DISTANCE_SENSOR`) whenever that is in range; local Z alone is relative
     to the boot spot and drifts. The window shows which source is live next to `alt`. If it says
     "NO rangefinder", PX4 is not sending a usable reading: check `listener distance_sensor` in
     the PX4 shell before flying. The mission arms only when PX4's own pre-arm check passes ("PX4 pre-arm OK" in the
     window). Right after the drone is picked up or set down the check fails for a few seconds
     ("horizontal velocity unstable"); let it sit still and it clears.
4. Camera model: `python calibrate.py --nominal` writes `camera.yaml` with the focal length measured
   on DEXI 3 (878 px at 640x480: the stream is a tighter crop than the full sensor), centre at the
   image middle, no distortion. That is enough for the closed-loop landing. To check it on your unit:
   hold the drone still over the pad, measure lens-to-pad distance, and compare with the `z` the
   window shows; they should agree within a few cm. A full calibration with a printed checkerboard
   (`python calibrate.py --cols 9 --rows 6 --square 0.022`) also models lens distortion.

### Optical-flow orientation (check on every unit)

With the drone held ~50 cm over a textured floor and moved 1 m forward, PX4's local position must
move +1 m forward (and +1 m right for a right move). On the DEXI 3 used here both axes read
inverted with the shipped `SENS_FLOW_ROT` = 4 (180°); the camera was also mounted 180° from the
expected orientation. Setting `SENS_FLOW_ROT` = 0 fixed it. A wrong value makes Position mode run
away (positive feedback), which looks like violent drift on takeoff.

### Sensor calibration and indoor limits

`python px4_cal.py accel` (props off, hold on each of six sides), then `gyro`, then `level` with
the drone still on a truly level surface. DEXI 3 has no magnetometer, so PX4's "magnetometer
calibration not OK" flag is harmless.

The vendor manual limits suit outdoor flying. On the DEXI 3 used here they were lowered on
2026-11-14 (old value in brackets): `MPC_Z_VEL_MAX_UP` 1 m/s (3), `MPC_VEL_MANUAL` 1 m/s (4),
`MPC_MAN_TILT_MAX` 15° (25). The mode switch (channel 6) is Stabilized / Position / Altitude;
the third slot, `COM_FLTMODE6`, was Precision Land (13) and is now Altitude (1). Do not hover below about 0.3 m: the drone sits in its own ground
effect and optical flow cannot focus under 8 cm, so it slides and Position mode has no velocity.

### PX4 failsafe

Set `COM_OF_LOSS_T` (offboard loss timeout, e.g. 1 s) and `COM_OBL_RC_ACT` so that offboard loss with RC
present falls back to Position mode. If Wi-Fi drops, the script exits without sending commands; PX4's
failsafe and the RC pilot take over.

## Control window

`python ui.py` opens a window with the live camera view, the marker readout (this replaces
`--dry`), the mission fields, `Start mission`, and three emergency buttons that work at any
moment, mission running or not:

- **HOLD**: leaves offboard, PX4 holds position.
- **LAND**: leaves offboard and lands where it is.
- **KILL (motor cutoff)**: two clicks within 3 s. Cuts the motors at once; the drone falls.

Pressing any of them also cancels a running mission. The RC pilot's switch still overrides
everything.

## Taking off

PX4 with optical flow only (no GPS) often refuses Position, Takeoff and Offboard modes while the
drone is on the ground ("Preflight Fail: horizontal velocity unstable") and accepts them once it
is airborne. So the normal procedure on DEXI 3 is a **pilot-assisted takeoff**:

1. Pilot arms in Stabilized or Altitude mode and lifts off by hand to about 1 m over the pad.
2. Pilot switches to Position mode. The drone should hold position on its own.
3. Press `Start mission`. The script sees the drone airborne in Position mode and takes over:
   Offboard, climb to Altitude, 360°, return to the acquire altitude, centre on the tag, land, disarm.

If `Start mission` is pressed with the drone on the ground, it tries the automatic takeoff instead
(arm + PX4 Takeoff mode), which only works if PX4's pre-arm checks pass on the ground.

## Flight test order

Each step needs an RC pilot on the sticks with a mode switch set to Position mode. Switching mode
at any time kicks PX4 out of OFFBOARD; the script notices and exits. Before each run, have the RC mode switch in Position mode and the vehicle in HOLD/Loiter, not in a manual mode, because the script treats a manual mode seen during the climb as a pilot takeover and exits.

1. Bench: `python mission.py --dry` (or `python ui.py` and Connect). Check frame age < 0.5 s and the marker is seen.
2. Hand-held: lift the drone 1 m, move it right: only `right` must change, and go negative; move it
   forward: only `fwd` must change, and go negative. On the DEXI 3 used here the camera is mounted
   rotated 180° (both went positive at 0), so the default `--cam-yaw` is 180. If your readings differ,
   try 0, 90 or -90 until both are correct, and keep that value for every run.
3. Pilot hovers at 1 m in Position mode, then `python mission.py --alt 2 --kp 0` (or the window with
   Altitude 2, Kp 0): climb, yaw, straight descent, land.
4. Same with Kp 0.5: full loop at 2 m.
5. `--alt 5`: the real thing.

## Safety behaviour

- Marker lost: sends zero velocity (the vehicle may drift). Lost > 10 s: lands where it is.
- Any error or Ctrl-C: lands. If the pilot is in a manual mode, the script sends nothing.
- If the land command fails, the vehicle is left in HOLD and the pilot must take over.
- Every scripted move (climb, descent, centring) is capped at `--vmax` (0.1 m/s; "Max speed" in
  the window). Altitude changes slide the setpoint at that speed instead of stepping it.
