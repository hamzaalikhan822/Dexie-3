"""DEXI 3 precision landing from a laptop over MAVSDK.

python mission.py --dry                 # never arms; DEXI 3 defaults below
python mission.py --alt 2 --kp 0
python mission.py --alt 5
"""
import argparse
import asyncio
import base64
import json
import math
import threading
import time

import cv2
import numpy as np

from aruco_pose import load_camera, marker_offset

log = print  # ui.py replaces this to show mission output in the window


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="udpout://192.168.4.1:14540", help="mavlink-router MAVSDK endpoint")
    p.add_argument("--rtsp", default="ws://192.168.4.1:9090",
                   help="camera source: ws://host:9090 (rosbridge) or any cv2.VideoCapture URL")
    p.add_argument("--topic", default="/cam0/image_raw/compressed", help="ROS 2 topic for ws:// sources")
    p.add_argument("--camera", default="camera.yaml")
    p.add_argument("--marker-id", type=int, default=0)
    p.add_argument("--marker-size", type=float, default=0.35)
    p.add_argument("--alt", type=float, default=5.0)
    p.add_argument("--kp", type=float, default=0.5)
    p.add_argument("--vmax", type=float, default=0.3)
    p.add_argument("--cam-yaw", type=float, default=0.0, help="deg, camera image-up vs nose")
    p.add_argument("--dry", action="store_true", help="connect and print offsets, never arm")
    return p.parse_args(argv)


def rosbridge_frame(msg):
    """Decode a rosbridge CompressedImage publish message to a BGR frame, or None."""
    if msg.get("op") != "publish":
        return None
    buf = np.frombuffer(base64.b64decode(msg["msg"]["data"]), np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


class LatestFrame(threading.Thread):
    """Keeps only the newest camera frame so the control loop never reads stale video.

    url ws://host:9090 subscribes to `topic` over rosbridge (DEXI OS has no RTSP server);
    anything else is opened with cv2.VideoCapture.
    """

    def __init__(self, url, topic="/cam0/image_raw/compressed"):
        super().__init__(daemon=True)
        self.url, self.topic = url, topic
        self.lock = threading.Lock()
        self.frame, self.stamp = None, 0.0

    def _put(self, f):
        if f is not None:
            with self.lock:
                self.frame, self.stamp = f, time.monotonic()

    def run(self):
        if self.url.startswith("ws://"):
            import websocket  # websocket-client
            while True:
                try:
                    ws = websocket.create_connection(self.url, timeout=5)
                    ws.send(json.dumps({"op": "subscribe", "topic": self.topic,
                                        "type": "sensor_msgs/msg/CompressedImage",
                                        "queue_length": 1, "throttle_rate": 100}))
                    while True:
                        self._put(rosbridge_frame(json.loads(ws.recv())))
                except Exception as ex:  # reconnect; a dead stream shows as stale frames
                    log(f"camera stream: {ex!r}; reconnecting")
                    time.sleep(1)
        cap = cv2.VideoCapture(self.url)
        while True:
            ok, f = cap.read()
            self._put(f if ok else None)
            if not ok:
                time.sleep(0.01)

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
        self.rel_alt = None      # metres above the EKF origin (boot spot); DEXI 3 has no GPS
        self.ned = None          # (north, east, down)
        self.yaw = None          # degrees
        self.mode = None
        self.in_air = None
        self.armed = None
        self.local_ok = None     # PX4 local position estimate valid (flow + range indoors)
        self.att_stamp = None    # monotonic time of the last attitude update
        self.tasks = [asyncio.create_task(c) for c in (
            self._ned(drone), self._att(drone), self._mode(drone), self._air(drone),
            self._armed(drone), self._health(drone))]

    async def _ned(self, d):
        async for p in d.telemetry.position_velocity_ned():
            self.ned = (p.position.north_m, p.position.east_m, p.position.down_m)
            self.rel_alt = -p.position.down_m

    async def _health(self, d):
        async for h in d.telemetry.health():
            self.local_ok = h.is_local_position_ok

    async def _att(self, d):
        async for a in d.telemetry.attitude_euler():
            self.yaw = a.yaw_deg
            self.att_stamp = time.monotonic()

    async def _mode(self, d):
        async for m in d.telemetry.flight_mode():
            self.mode = m

    async def _air(self, d):
        async for a in d.telemetry.in_air():
            self.in_air = a

    async def _armed(self, d):
        async for a in d.telemetry.armed():
            self.armed = a

    async def ready(self):
        while None in (self.ned, self.yaw, self.mode, self.in_air, self.armed, self.local_ok):
            await asyncio.sleep(0.1)


class PilotTookOver(Exception):
    pass


MANUAL = {"MANUAL", "ALTCTL", "POSCTL", "STABILIZED", "ACRO", "RATTITUDE"}
FAILSAFE = {"RETURN_TO_LAUNCH", "LAND"}


def link_lost(t):
    """A telemetry task died, or attitude (10+ Hz) has been silent for 2 s."""
    return (any(x.done() for x in t.tasks)
            or t.att_stamp is None or time.monotonic() - t.att_stamp > 2.0)


def check_offboard(t):
    """Raise if the RC pilot switched PX4 out of OFFBOARD or the link dropped."""
    if link_lost(t):
        raise PilotTookOver("link lost")
    if t.mode.name != "OFFBOARD":
        raise PilotTookOver(f"flight mode is {t.mode.name}")


async def abort_action(drone, kind):
    """kill: cut motors now. hold: leave offboard (PX4 Hold). land: stop offboard, land, retry once."""
    if kind == "kill":
        await drone.action.kill()
        return True
    if kind == "hold":
        await drone.offboard.stop()
        return True
    try:
        await drone.offboard.stop()
    except Exception:
        pass
    for _ in range(2):
        try:
            await drone.action.land()
            return True
        except Exception:
            await asyncio.sleep(1)
    log("VEHICLE IN HOLD - PILOT TAKE OVER")
    return False


async def connect(url):
    from mavsdk import System
    drone = System()
    await drone.connect(system_address=url)
    async for s in drone.core.connection_state():
        if s.is_connected:
            break
    log("connected")
    return drone


async def centre_and_descend(args, drone, frames, K, dist, t, descend):
    """Hold over the marker (descend=False) or descend over it (descend=True).

    Returns when settled (hold) or below 1 m (descend). Lost marker: hold, land after 10 s.
    """
    from mavsdk.offboard import VelocityBodyYawspeed
    settled_since = None
    start = last_seen = time.monotonic()
    while True:
        check_offboard(t)
        frame, stamp = frames.get()
        off = (None if frame is None or time.monotonic() - stamp > 1.0
               else marker_offset(frame, K, dist, args.marker_id, args.marker_size))
        now = time.monotonic()
        if not descend and now - start > 30:
            raise RuntimeError("centre timed out")
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


async def fly(args, drone, frames, K, dist, ctl=None):
    from mavsdk.offboard import OffboardError, PositionNedYaw
    t = Telem(drone)
    armed = False
    try:
        # 1. connect and check
        await t.ready()
        while not t.local_ok:
            log("waiting for PX4 local position (flow + range) to be valid...")
            await asyncio.sleep(1)
        frame, stamp = frames.get()
        if (frame is None or time.monotonic() - stamp > 1.0
                or marker_offset(frame, K, dist, args.marker_id, args.marker_size) is None):
            raise RuntimeError("marker not visible on the ground; aborting before arm")

        # 2. takeoff
        await drone.param.set_param_float("MIS_TAKEOFF_ALT", args.alt)
        armed = True
        await drone.action.arm()
        await drone.action.takeoff()
        climb_start = time.monotonic()
        while abs(t.rel_alt - args.alt) > 0.3:
            if t.mode.name in MANUAL or t.mode.name in FAILSAFE:
                raise PilotTookOver(f"flight mode is {t.mode.name}")
            if time.monotonic() - climb_start > 30:
                raise RuntimeError("takeoff timed out")
            await asyncio.sleep(0.2)
        log(f"at {t.rel_alt:.1f} m")

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
            turned += 1
            await drone.offboard.set_position_ned(PositionNedYaw(n, e, d, (start_yaw + turned) % 360))
            await asyncio.sleep(0.1)
        log(f"sweep done, measured yaw delta {heading_delta(start_yaw, t.yaw):.0f} deg")
        for _ in range(20):
            check_offboard(t)
            await drone.offboard.set_position_ned(PositionNedYaw(n, e, d, start_yaw))
            await asyncio.sleep(0.1)
        log(f"rotation done, heading {t.yaw:.0f} deg (delta {heading_delta(start_yaw, t.yaw):.0f})")

        # 4. centre, 5. descend
        if args.kp > 0:
            await centre_and_descend(args, drone, frames, K, dist, t, descend=False)
            log("centred")
        await centre_and_descend(args, drone, frames, K, dist, t, descend=True)
        log(f"below 1 m at {t.rel_alt:.2f} m, landing")

        # 6. land
        await drone.offboard.stop()
        await drone.action.land()
    except PilotTookOver as ex:
        log(f"pilot took over: {ex}; exiting without further commands")
        return
    except (Exception, asyncio.CancelledError, KeyboardInterrupt) as ex:
        if not armed:
            log(f"ABORT before arm: {ex!r}; sending nothing")
            return
        if link_lost(t):
            log(f"ABORT: {ex!r}; link lost, sending nothing (PX4 failsafe)")
            return
        if t.mode is not None and (t.mode.name in MANUAL or t.mode.name in FAILSAFE):
            log(f"ABORT: {ex!r}; pilot or PX4 failsafe has control ({t.mode.name}), sending nothing")
            return
        kind = ctl.get("abort", "land") if ctl else "land"
        log(f"ABORT: {ex!r}; {kind}")
        if not await abort_action(drone, kind) or kind != "land":
            return
    while t.in_air:
        await asyncio.sleep(0.5)
    try:
        await drone.action.disarm()
    except Exception:
        pass
    log("landed and disarmed")


async def dry_run(args, drone, frames, K, dist):
    log("DRY: not arming. Ctrl-C to stop.")
    while True:
        frame, stamp = frames.get()
        off = None if frame is None else marker_offset(frame, K, dist, args.marker_id, args.marker_size)
        age = time.monotonic() - stamp if stamp else float("inf")
        if off is None:
            log(f"frame age {age:.2f}s  marker: not seen")
        else:
            fwd, right = offset_to_velocity(off, args.kp, args.vmax, args.cam_yaw)
            log(f"frame age {age:.2f}s  marker x={off[0]:+.2f} y={off[1]:+.2f} z={off[2]:.2f}"
                  f"  -> fwd={fwd:+.2f} right={right:+.2f}")
        await asyncio.sleep(0.5)


async def main():
    args = parse_args()
    K, dist = load_camera(args.camera)
    frames = LatestFrame(args.rtsp, args.topic)
    frames.start()

    drone = await connect(args.url)

    if args.dry:
        await dry_run(args, drone, frames, K, dist)
    else:
        await fly(args, drone, frames, K, dist)


if __name__ == "__main__":
    asyncio.run(main())
