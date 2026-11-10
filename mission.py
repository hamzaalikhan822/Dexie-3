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
