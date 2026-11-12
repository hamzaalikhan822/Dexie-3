# Project context

Where the project stands as of 2026-11-12. `README.md` is the operating manual (setup, flight
procedure, safety); this file is the background a newcomer needs before touching the code.

## Goal

A DroneBlocks DEXI 3 takes off, climbs to a set height, yaws 360°, finds an AprilTag landing pad
with its downward camera, centres on it and lands on it. Everything runs on a laptop over the
drone's Wi-Fi. Nothing is installed on the drone.

## Hardware and links

| Part | Detail |
|------|--------|
| Vehicle | DEXI 3, PX4 airframe 4701, Raspberry Pi CM5 companion running DEXI OS |
| Position | No GPS. Optical flow plus a downward rangefinder, fused by PX4's EKF2 |
| Camera | 640x480, re-mounted to point down (stock is forward), image rotated 180° from the nose |
| Landing pad | DroneBlocks DEXI pad: AprilTag 36h11, id 0, black square 14.0 cm |
| MAVLink | mavlink-router on the Pi, UDP 14540 for MAVSDK, 14550 for QGroundControl |
| Video | ROS 2 topic `/cam0/image_raw/compressed` through rosbridge on port 9090 |
| RC | Mode switch on channel 6 (Position in the middle), arm on 5, kill on 8 |

## Code map

| File | Role |
|------|------|
| `mission.py` | The mission: telemetry cache, camera thread, flight sequence, abort handling. Also a CLI (`--dry` never arms) |
| `ui.py` | Tkinter control window: live video, marker readout, mission fields, Start, HOLD / LAND / KILL |
| `aruco_pose.py` | Detects one tag and returns its (x, y, z) offset from the camera in metres |
| `calibrate.py` | Writes `camera.yaml`, either nominal (`--nominal`) or from a checkerboard |
| `tests/` | 33 pytest tests driven by fakes; no drone or display needed |
| `docs/superpowers/` | The original design specs and implementation plan |

`camera.yaml` is git-ignored. Create it with `python calibrate.py --nominal` (focal length 878 px
at 640x480, measured on this unit).

## Mission sequence (`mission.fly`)

1. Wait for PX4's own pre-arm check (`health.is_armable`) and a live camera stream.
2. If the drone is already airborne in Position, Altitude or Hold mode, take over from the pilot's
   hover. Otherwise arm and use PX4 Takeoff mode.
3. Enter Offboard holding the current spot, slide up to `--alt`, yaw 360° at 5°/s.
4. Slide down to `--acquire-alt` (1.5 m) over the same spot, because the 14 cm tag is only
   detectable from about 2 m down.
5. Centre on the tag with a proportional velocity command (skipped when `--kp 0`).
6. Descend over the tag to 1 m, then hand over to PX4 Land and disarm.

## Design decisions and why

- **Laptop-side, MAVSDK gRPC API.** `mavsdk` is pinned below 4: the pip package was renamed and
  versions 4 and up are a different binding.
- **Arm gate is PX4's pre-arm verdict**, not MAVSDK's local-position flag, which guesses from GPS
  and flow health bits and disagreed with PX4 on this vehicle.
- **No marker check before arming.** On the ground the camera is 4 cm above the pad and cannot
  see the tag.
- **Pilot-assisted takeoff is the normal procedure.** With flow only, PX4 often refuses Position,
  Takeoff and Offboard on the ground ("horizontal velocity unstable") and accepts them airborne.
- **Height is local Z re-zeroed on the rangefinder.** Local Z alone is relative to the boot spot
  and drifts indoors. `Telem.gnd` holds the floor's local-frame "down"; above the sensor's range
  the last floor reference is kept.
- **Every scripted move is capped at `--vmax` (0.1 m/s).** Altitude changes slide the position
  setpoint at that speed; a stepped setpoint makes PX4 move at its own limit.
- **The pilot always wins.** Any manual mode, a PX4 failsafe mode, or a lost link makes the
  script stop sending commands. Other errors land in place.
- **The window detects the tag at about 3 Hz.** At 10 Hz the detector starved the MAVSDK
  telemetry thread.

## Findings on this unit

- `SENS_FLOW_ROT` shipped as 4 (180°) and both flow axes read inverted, which makes Position mode
  run away. It was set to 0. Later hand-moved re-tests gave inconsistent signs and magnitudes, so
  the flow orientation is **not yet confirmed**.
- The first manual flight drifted heavily. It was flown in Stabilized mode, which does not hold
  position, so that flight says nothing about the flow sensor.
- On 2026-11-11 the PX4 shell showed `dist_bottom_valid: False` while the rangefinder read 0.51 m.
  Whether PX4 fuses the rangefinder properly is **unresolved**.
- Vendor parameters: `EKF2_OF_CTRL=1`, `EKF2_RNG_CTRL=2`, `EKF2_HGT_REF=2`, `EKF2_EV_CTRL=0`,
  `EKF2_GPS_CTRL=0`. Parameter changes need a reboot.
- Ceiling at the test site is 2.7 m, so indoor tests stay at 2 m or below.

## Status

- Not yet flown through the mission. The automatic sequence has never run in the air.
- The rangefinder height and slow-move changes (commit `fd99151`) pass their tests but have not
  been tried on the drone.

## Next steps

1. Bench: connect, check the `alt` line says `(rangefinder)` on the pad and at about 50 cm. If it
   says "NO rangefinder", run `listener distance_sensor` in the PX4 shell.
2. Settle the optical-flow orientation with the displacement check in the README.
3. Pilot hovers at 1 m in Position mode and confirms it holds on its own.
4. Mission with Altitude 2, Kp 0: climb, yaw, straight descent, land.
5. Same with Kp 0.5, then higher only with a clear ceiling.

## Open questions

- The rangefinder's maximum range is unknown; at 2 m it may already be out of range.
- The 360° yaw, PX4 Land below 1 m and PX4 Takeoff are not covered by `--vmax`.
- `mavsdk` 3.x is deprecated upstream in favour of `mavsdk-grpc`.
