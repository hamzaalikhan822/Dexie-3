# DEXI 3 Precision Landing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A laptop-run Python script that takes the DEXI 3 off to 5 m, yaws 360°, centres over an ArUco marker seen by the downward camera, and lands on it.

**Architecture:** Three flat Python files. `aruco_pose.py` is a pure function from frame to marker offset. `calibrate.py` produces `camera.yaml` once. `mission.py` is an asyncio state machine over MAVSDK-Python with a background RTSP reader thread; `--dry` runs everything except arming.

**Tech Stack:** Python 3.10+, mavsdk (MAVSDK-Python), opencv-contrib-python (>= 4.7 for `ArucoDetector`), numpy, pyyaml, pytest.

## Global Constraints

- Windows laptop, PowerShell. Use `python` and `pip` from the shell.
- Spec: `docs/superpowers/specs/2026-11-10-precision-landing-design.md`.
- Velocity clamp 0.3 m/s horizontal and 0.3 m/s down. Default Kp = 0.5.
- Marker: `DICT_4X4_50`, id 0, 0.35 m.
- Marker not seen: hold (zero velocity) at once. Not seen > 10 s: land in place. Any error / Ctrl-C: land.
- No classes beyond what the code needs. No extra features. Flat layout at repo root.

---

### Task 1: Repo scaffold and `aruco_pose.py`

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `aruco_pose.py`
- Create: `tests/test_aruco_pose.py`

**Interfaces:**
- Produces: `load_camera(path: str) -> tuple[np.ndarray, np.ndarray]` returning `(K 3x3 float64, dist 1xN float64)`.
- Produces: `marker_offset(frame: np.ndarray, K, dist, marker_id: int, marker_size_m: float, dictionary: int = cv2.aruco.DICT_4X4_50) -> tuple[float, float, float] | None` returning `(x, y, z)` metres in the OpenCV camera frame (x right, y down, z forward).

- [ ] **Step 1: Init repo and dependencies**

```powershell
git init
```

`requirements.txt`:
```
mavsdk>=2.0
opencv-contrib-python>=4.8
numpy
pyyaml
pytest
```

`.gitignore`:
```
__pycache__/
*.pyc
.pytest_cache/
camera.yaml
*.jpg
*.png
```

```powershell
pip install -r requirements.txt
```

- [ ] **Step 2: Write the failing test**

`tests/test_aruco_pose.py`:
```python
import cv2
import numpy as np

from aruco_pose import marker_offset

MARKER_PX = 200
FX = 600.0
K = np.array([[FX, 0, 320.0], [0, FX, 240.0], [0, 0, 1]], dtype=np.float64)
DIST = np.zeros((1, 5))


def synthetic_frame(cx=320, cy=240):
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker = cv2.aruco.generateImageMarker(d, 0, MARKER_PX)
    frame = np.full((480, 640), 255, dtype=np.uint8)
    y0, x0 = cy - MARKER_PX // 2, cx - MARKER_PX // 2
    frame[y0:y0 + MARKER_PX, x0:x0 + MARKER_PX] = marker
    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)


def test_centered_marker_gives_zero_xy_and_expected_z():
    x, y, z = marker_offset(synthetic_frame(), K, DIST, 0, 0.35)
    # pinhole: z = f * size / pixel_size
    assert abs(z - FX * 0.35 / MARKER_PX) < 0.05
    assert abs(x) < 0.02 and abs(y) < 0.02


def test_marker_right_of_centre_gives_positive_x():
    x, y, z = marker_offset(synthetic_frame(cx=420), K, DIST, 0, 0.35)
    assert x > 0.1 and abs(y) < 0.02


def test_missing_marker_returns_none():
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    assert marker_offset(frame, K, DIST, 0, 0.35) is None


def test_wrong_id_returns_none():
    assert marker_offset(synthetic_frame(), K, DIST, 7, 0.35) is None
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_aruco_pose.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aruco_pose'`

- [ ] **Step 4: Write `aruco_pose.py`**

```python
"""Detect one ArUco marker and return its offset from the camera in metres."""
import sys

import cv2
import numpy as np
import yaml


def load_camera(path):
    with open(path) as f:
        c = yaml.safe_load(f)
    return np.array(c["camera_matrix"], dtype=np.float64), np.array(c["dist_coeffs"], dtype=np.float64)


def marker_offset(frame, K, dist, marker_id, marker_size_m, dictionary=cv2.aruco.DICT_4X4_50):
    """Return (x, y, z) of the marker centre in the camera frame, or None if not seen."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(dictionary),
                                       cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return None
    hits = [c for c, i in zip(corners, ids.flatten()) if i == marker_id]
    if not hits:
        return None
    s = marker_size_m / 2
    obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float64)
    ok, _, tvec = cv2.solvePnP(obj, hits[0].reshape(4, 2), K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    x, y, z = tvec.flatten()
    return float(x), float(y), float(z)


if __name__ == "__main__":
    # Self-check on a saved frame: python aruco_pose.py frame.jpg camera.yaml [marker_id] [size_m]
    frame = cv2.imread(sys.argv[1])
    K, dist = load_camera(sys.argv[2])
    mid = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    size = float(sys.argv[4]) if len(sys.argv) > 4 else 0.35
    print(marker_offset(frame, K, dist, mid, size))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_aruco_pose.py -v`
Expected: 4 PASS

- [ ] **Step 6: Commit**

```powershell
git add requirements.txt .gitignore aruco_pose.py tests/test_aruco_pose.py
git commit -m "feat: aruco marker offset with synthetic-frame tests"
```

---

### Task 2: `calibrate.py`

**Files:**
- Create: `calibrate.py`
- Create: `tests/test_calibrate.py`

**Interfaces:**
- Produces: `camera.yaml` with keys `width`, `height`, `camera_matrix` (3x3 list), `dist_coeffs` (1x5 list), `reprojection_error` (float). Read by `aruco_pose.load_camera`.
- Produces: `calibrate(objpoints, imgpoints, size) -> dict` (pure, tested) and `write_camera_yaml(path, result)`.

- [ ] **Step 1: Write the failing test**

`tests/test_calibrate.py`:
```python
import cv2
import numpy as np

from aruco_pose import load_camera
from calibrate import calibrate, write_camera_yaml


def test_calibrate_recovers_known_intrinsics(tmp_path):
    # Project a 9x6 board through a known camera from several poses, calibrate, compare.
    K = np.array([[600.0, 0, 320.0], [0, 600.0, 240.0], [0, 0, 1]])
    dist = np.zeros(5)
    board = np.zeros((9 * 6, 3), np.float32)
    board[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * 0.025
    objpoints, imgpoints = [], []
    rng = np.random.default_rng(0)
    for i in range(12):
        rvec = rng.normal(0, 0.3, 3)
        tvec = np.array([-0.1, -0.07, 0.6]) + rng.normal(0, 0.05, 3)
        pts, _ = cv2.projectPoints(board, rvec, tvec, K, dist)
        objpoints.append(board)
        imgpoints.append(pts.astype(np.float32))
    result = calibrate(objpoints, imgpoints, (640, 480))
    out = tmp_path / "camera.yaml"
    write_camera_yaml(str(out), result)
    K2, dist2 = load_camera(str(out))
    assert np.allclose(K2, K, atol=5)
    assert result["reprojection_error"] < 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_calibrate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'calibrate'`

- [ ] **Step 3: Write `calibrate.py`**

```python
"""One-off checkerboard calibration of the RTSP stream -> camera.yaml.

Usage: python calibrate.py rtsp://<voxl-ip>:8900/hires --cols 9 --rows 6 --square 0.025
Press SPACE to capture when corners are drawn, q to finish (needs >= 10 captures).
"""
import argparse

import cv2
import numpy as np
import yaml


def calibrate(objpoints, imgpoints, size):
    err, K, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, size, None, None)
    return {"width": size[0], "height": size[1], "camera_matrix": K.tolist(),
            "dist_coeffs": dist.reshape(1, -1).tolist(), "reprojection_error": float(err)}


def write_camera_yaml(path, result):
    with open(path, "w") as f:
        yaml.safe_dump(result, f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("rtsp")
    p.add_argument("--cols", type=int, default=9, help="inner corners across")
    p.add_argument("--rows", type=int, default=6, help="inner corners down")
    p.add_argument("--square", type=float, default=0.025, help="square size, metres")
    p.add_argument("--out", default="camera.yaml")
    a = p.parse_args()

    board = np.zeros((a.cols * a.rows, 3), np.float32)
    board[:, :2] = np.mgrid[0:a.cols, 0:a.rows].T.reshape(-1, 2) * a.square
    objpoints, imgpoints = [], []
    cap = cv2.VideoCapture(a.rtsp)
    size = None
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        size = (frame.shape[1], frame.shape[0])
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, (a.cols, a.rows))
        if found:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                       (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            cv2.drawChessboardCorners(frame, (a.cols, a.rows), corners, found)
        cv2.putText(frame, f"captures: {len(imgpoints)}  SPACE=capture q=finish", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("calibrate", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" ") and found:
            objpoints.append(board)
            imgpoints.append(corners)
        elif key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    if len(imgpoints) < 10:
        raise SystemExit(f"only {len(imgpoints)} captures, need >= 10")
    result = calibrate(objpoints, imgpoints, size)
    write_camera_yaml(a.out, result)
    print(f"wrote {a.out}, reprojection error {result['reprojection_error']:.3f} px at {size}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```powershell
git add calibrate.py tests/test_calibrate.py
git commit -m "feat: checkerboard calibration script writing camera.yaml"
```

---

### Task 3: `mission.py` — stream reader, offset-to-velocity, dry mode

**Files:**
- Create: `mission.py`
- Create: `tests/test_mission.py`

**Interfaces:**
- Consumes: `aruco_pose.load_camera`, `aruco_pose.marker_offset`.
- Produces: `offset_to_velocity(offset, kp, vmax, cam_yaw_deg) -> tuple[float, float]` returning `(forward, right)` body m/s.
- Produces: `LatestFrame` thread with `.get() -> tuple[np.ndarray | None, float]` (frame, monotonic timestamp).
- Produces: `parse_args()` and `dry_run(args, drone, frames, K, dist)`; Task 4 adds `fly(args, drone, frames, K, dist)` beside them.

- [ ] **Step 1: Write the failing test**

`tests/test_mission.py`:
```python
import math

from mission import offset_to_velocity


def test_marker_right_moves_right():
    fwd, right = offset_to_velocity((0.4, 0.0, 3.0), kp=0.5, vmax=0.3, cam_yaw_deg=0)
    assert right > 0 and abs(fwd) < 1e-9


def test_marker_low_in_image_moves_backward():
    fwd, right = offset_to_velocity((0.0, 0.4, 3.0), kp=0.5, vmax=0.3, cam_yaw_deg=0)
    assert fwd < 0 and abs(right) < 1e-9


def test_clamped_to_vmax():
    fwd, right = offset_to_velocity((5.0, 5.0, 3.0), kp=0.5, vmax=0.3, cam_yaw_deg=0)
    assert math.hypot(fwd, right) <= 0.3 + 1e-9


def test_cam_yaw_90_swaps_axes():
    fwd, right = offset_to_velocity((0.4, 0.0, 3.0), kp=0.5, vmax=0.3, cam_yaw_deg=90)
    assert abs(right) < 1e-9 and abs(fwd) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_mission.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mission'`

- [ ] **Step 3: Write `mission.py` (stream, mapping, dry mode)**

```python
"""DEXI 3 precision landing from a laptop over MAVSDK.

python mission.py --rtsp rtsp://<voxl-ip>:8900/hires --dry      # never arms
python mission.py --rtsp rtsp://<voxl-ip>:8900/hires --alt 2 --kp 0
python mission.py --rtsp rtsp://<voxl-ip>:8900/hires --alt 5
"""
import argparse
import asyncio
import math
import threading
import time

import cv2

from aruco_pose import load_camera, marker_offset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="udp://:14540")
    p.add_argument("--rtsp", required=True)
    p.add_argument("--camera", default="camera.yaml")
    p.add_argument("--marker-id", type=int, default=0)
    p.add_argument("--marker-size", type=float, default=0.35)
    p.add_argument("--alt", type=float, default=5.0)
    p.add_argument("--kp", type=float, default=0.5)
    p.add_argument("--vmax", type=float, default=0.3)
    p.add_argument("--cam-yaw", type=float, default=0.0, help="deg, camera image-up vs nose")
    p.add_argument("--dry", action="store_true", help="connect and print offsets, never arm")
    return p.parse_args()


class LatestFrame(threading.Thread):
    """Keeps only the newest RTSP frame so the control loop never reads stale video."""

    def __init__(self, url):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(url)
        self.lock = threading.Lock()
        self.frame, self.stamp = None, 0.0

    def run(self):
        while True:
            ok, f = self.cap.read()
            if ok:
                with self.lock:
                    self.frame, self.stamp = f, time.monotonic()

    def get(self):
        with self.lock:
            return self.frame, self.stamp


def offset_to_velocity(offset, kp, vmax, cam_yaw_deg):
    """Camera-frame marker offset (x right, y down) -> body (forward, right) m/s."""
    x, y, _ = offset
    a = math.radians(cam_yaw_deg)
    xr = x * math.cos(a) - y * math.sin(a)
    yr = x * math.sin(a) + y * math.cos(a)
    fwd, right = -kp * yr, kp * xr
    n = math.hypot(fwd, right)
    if n > vmax:
        fwd, right = fwd * vmax / n, right * vmax / n
    return fwd, right


async def dry_run(args, drone, frames, K, dist):
    print("DRY: not arming. Ctrl-C to stop.")
    while True:
        frame, stamp = frames.get()
        off = None if frame is None else marker_offset(frame, K, dist, args.marker_id, args.marker_size)
        age = time.monotonic() - stamp if stamp else float("inf")
        if off is None:
            print(f"frame age {age:.2f}s  marker: not seen")
        else:
            fwd, right = offset_to_velocity(off, args.kp, args.vmax, args.cam_yaw)
            print(f"frame age {age:.2f}s  marker x={off[0]:+.2f} y={off[1]:+.2f} z={off[2]:.2f}"
                  f"  -> fwd={fwd:+.2f} right={right:+.2f}")
        await asyncio.sleep(0.5)


async def main():
    args = parse_args()
    K, dist = load_camera(args.camera)
    frames = LatestFrame(args.rtsp)
    frames.start()

    from mavsdk import System
    drone = System()
    await drone.connect(system_address=args.url)
    async for s in drone.core.connection_state():
        if s.is_connected:
            break
    print("connected")

    if args.dry:
        await dry_run(args, drone, frames, K, dist)
    else:
        await fly(args, drone, frames, K, dist)


if __name__ == "__main__":
    asyncio.run(main())
```

`fly` is defined in Task 4. Until then `--dry` works and a non-dry run raises `NameError`, which is acceptable for this commit.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests -v`
Expected: 9 PASS

- [ ] **Step 5: Hardware check (optional now, required before Task 4 flights)**

With the drone powered on the bench and streaming:
```powershell
python mission.py --rtsp rtsp://<voxl-ip>:8900/hires --dry
```
Expected: `connected`, then lines showing frame age under 0.5 s and the marker offset when the marker is under the camera. Lift the drone and move it right: `x` should go negative (marker moves left in the image) and `right` negative. If it is `fwd` that changes instead, the camera is rotated: use `--cam-yaw 90` or `-90` until moving the drone right changes only `right`.

- [ ] **Step 6: Commit**

```powershell
git add mission.py tests/test_mission.py
git commit -m "feat: mission stream reader, offset mapping and dry mode"
```

---

### Task 4: `mission.py` — flight state machine

**Files:**
- Modify: `mission.py` (add telemetry cache, `fly`, safety wrapper)
- Modify: `tests/test_mission.py` (add `heading_delta` test)

**Interfaces:**
- Consumes: `LatestFrame`, `offset_to_velocity`, `marker_offset`.
- Produces: `fly(args, drone, frames, K, dist)` and `heading_delta(start_deg, now_deg) -> float` in `[0, 360)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_mission.py`:
```python
from mission import heading_delta


def test_heading_delta_wraps():
    assert heading_delta(350, 10) == 20
    assert heading_delta(10, 350) == 340
    assert heading_delta(90, 90) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_mission.py -v`
Expected: FAIL with `ImportError: cannot import name 'heading_delta'`

- [ ] **Step 3: Add telemetry cache, `heading_delta`, and `fly` to `mission.py`**

Insert after `offset_to_velocity`:

```python
def heading_delta(start_deg, now_deg):
    """Degrees turned clockwise from start to now, in [0, 360)."""
    return (now_deg - start_deg) % 360


class Telem:
    """Latest values from the MAVSDK telemetry streams, filled by background tasks."""

    def __init__(self, drone):
        self.rel_alt = None
        self.ned = None          # (north, east, down)
        self.yaw = None          # degrees
        self.mode = None
        self.in_air = None
        self.tasks = [asyncio.create_task(c) for c in (
            self._pos(drone), self._ned(drone), self._att(drone), self._mode(drone), self._air(drone))]

    async def _pos(self, d):
        async for p in d.telemetry.position():
            self.rel_alt = p.relative_altitude_m

    async def _ned(self, d):
        async for p in d.telemetry.position_velocity_ned():
            self.ned = (p.position.north_m, p.position.east_m, p.position.down_m)

    async def _att(self, d):
        async for a in d.telemetry.attitude_euler():
            self.yaw = a.yaw_deg

    async def _mode(self, d):
        async for m in d.telemetry.flight_mode():
            self.mode = m

    async def _air(self, d):
        async for a in d.telemetry.in_air():
            self.in_air = a

    async def ready(self):
        while None in (self.rel_alt, self.ned, self.yaw, self.mode, self.in_air):
            await asyncio.sleep(0.1)


class PilotTookOver(Exception):
    pass


def check_offboard(t):
    """Raise if the RC pilot switched PX4 out of OFFBOARD."""
    if t.mode.name != "OFFBOARD":
        raise PilotTookOver(f"flight mode is {t.mode.name}")


async def centre_and_descend(args, drone, frames, K, dist, t, descend):
    """Hold over the marker (descend=False) or descend over it (descend=True).

    Returns when settled (hold) or below 1 m (descend). Lost marker: hold, land after 10 s.
    """
    from mavsdk.offboard import VelocityBodyYawspeed
    settled_since = None
    last_seen = time.monotonic()
    while True:
        check_offboard(t)
        frame, _ = frames.get()
        off = None if frame is None else marker_offset(frame, K, dist, args.marker_id, args.marker_size)
        now = time.monotonic()
        if off is not None:
            last_seen = now
            fwd, right = offset_to_velocity(off, args.kp, args.vmax, args.cam_yaw)
            horiz = math.hypot(off[0], off[1])
        else:
            fwd, right, horiz = 0.0, 0.0, None
            if now - last_seen > 10:
                raise RuntimeError("marker lost > 10 s")
        down = args.vmax if (descend and off is not None) else 0.0
        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(fwd, right, down, 0.0))

        if descend:
            if t.rel_alt is not None and t.rel_alt < 1.0:
                return
        elif horiz is not None and horiz < 0.15:
            settled_since = settled_since or now
            if now - settled_since > 1.0:
                return
        else:
            settled_since = None
        await asyncio.sleep(0.1)


async def fly(args, drone, frames, K, dist):
    from mavsdk.offboard import OffboardError, PositionNedYaw
    t = Telem(drone)
    try:
        # 1. connect and check
        async for h in drone.telemetry.health():
            if h.is_global_position_ok and h.is_home_position_ok:
                break
            print("waiting for position health...")
            await asyncio.sleep(1)
        await t.ready()
        frame, _ = frames.get()
        if frame is None or marker_offset(frame, K, dist, args.marker_id, args.marker_size) is None:
            raise RuntimeError("marker not visible on the ground; aborting before arm")

        # 2. takeoff
        await drone.param.set_param_float("MIS_TAKEOFF_ALT", args.alt)
        await drone.action.arm()
        await drone.action.takeoff()
        while abs(t.rel_alt - args.alt) > 0.3:
            await asyncio.sleep(0.2)
        print(f"at {t.rel_alt:.1f} m")

        # 3. rotate 360
        n, e, d = t.ned
        start_yaw = t.yaw
        await drone.offboard.set_position_ned(PositionNedYaw(n, e, d, start_yaw))
        try:
            await drone.offboard.start()
        except OffboardError as ex:
            raise RuntimeError(f"offboard refused: {ex._result.result}")
        # telemetry lags the mode switch; keep streaming setpoints until PX4 reports OFFBOARD
        for _ in range(30):
            if t.mode.name == "OFFBOARD":
                break
            await drone.offboard.set_position_ned(PositionNedYaw(n, e, d, start_yaw))
            await asyncio.sleep(0.1)
        else:
            raise RuntimeError(f"never entered OFFBOARD, mode is {t.mode.name}")
        turned = 0.0
        while turned < 360:
            check_offboard(t)
            turned += 10
            await drone.offboard.set_position_ned(PositionNedYaw(n, e, d, (start_yaw + turned) % 360))
            await asyncio.sleep(0.1)
        for _ in range(20):
            check_offboard(t)
            await drone.offboard.set_position_ned(PositionNedYaw(n, e, d, start_yaw))
            await asyncio.sleep(0.1)
        print(f"rotation done, heading {t.yaw:.0f} deg (delta {heading_delta(start_yaw, t.yaw):.0f})")

        # 4. centre, 5. descend
        await centre_and_descend(args, drone, frames, K, dist, t, descend=False)
        print("centred")
        await centre_and_descend(args, drone, frames, K, dist, t, descend=True)
        print(f"below 1 m at {t.rel_alt:.2f} m, landing")

        # 6. land
        await drone.offboard.stop()
        await drone.action.land()
    except PilotTookOver as ex:
        print(f"pilot took over: {ex}; exiting without further commands")
        return
    except (Exception, asyncio.CancelledError, KeyboardInterrupt) as ex:
        print(f"ABORT: {ex!r}; landing in place")
        try:
            await drone.offboard.stop()
        except Exception:
            pass
        await drone.action.land()
    while t.in_air:
        await asyncio.sleep(0.5)
    try:
        await drone.action.disarm()
    except Exception:
        pass
    print("landed and disarmed")
```

Also change the end of `main()` so Ctrl-C during the flight lands:

```python
    if args.dry:
        await dry_run(args, drone, frames, K, dist)
    else:
        try:
            await fly(args, drone, frames, K, dist)
        except KeyboardInterrupt:
            await drone.action.land()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests -v`
Expected: 10 PASS

- [ ] **Step 5: Commit**

```powershell
git add mission.py tests/test_mission.py
git commit -m "feat: takeoff, 360 yaw, marker centring, descent and landing"
```

---

### Task 5: README and flight test checklist

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`**

````markdown
# DEXI 3 Precision Landing (laptop)

Takes off to N m, yaws 360°, centres on an ArUco marker with the downward camera, lands on it.
Everything runs on the laptop over Wi-Fi: MAVLink via MAVSDK, video via RTSP.

## Setup

1. `pip install -r requirements.txt`
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
at any time kicks PX4 out of OFFBOARD; the script notices and exits.

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
````

- [ ] **Step 2: Commit**

```powershell
git add README.md
git commit -m "docs: setup and flight test checklist"
```
