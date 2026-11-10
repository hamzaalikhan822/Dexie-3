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
        frame, stamp = frames.get()
        off = (None if frame is None or time.monotonic() - stamp > 1.0
               else marker_offset(frame, K, dist, args.marker_id, args.marker_size))
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
        frame, stamp = frames.get()
        if (frame is None or time.monotonic() - stamp > 1.0
                or marker_offset(frame, K, dist, args.marker_id, args.marker_size) is None):
            raise RuntimeError("marker not visible on the ground; aborting before arm")

        # 2. takeoff
        await drone.param.set_param_float("MIS_TAKEOFF_ALT", args.alt)
        await drone.action.arm()
        await drone.action.takeoff()
        climb_start = time.monotonic()
        while abs(t.rel_alt - args.alt) > 0.3:
            if t.mode.name not in ("TAKEOFF", "HOLD"):
                raise PilotTookOver(f"flight mode is {t.mode.name}")
            if time.monotonic() - climb_start > 30:
                raise RuntimeError("takeoff timed out")
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
        if t.mode is not None and t.mode.name in ("POSCTL", "ALTCTL", "MANUAL", "STABILIZED", "ACRO"):
            print(f"ABORT: {ex!r}; pilot has control ({t.mode.name}), sending nothing")
            return
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
        try:
            await fly(args, drone, frames, K, dist)
        except KeyboardInterrupt:
            await drone.action.land()


if __name__ == "__main__":
    asyncio.run(main())
