"""Mini 3 precision landing: the laptop detects and commands, the phone relays (android/Relay.kt).

The pilot takes off by hand, hovers about 2 m above the tag with the gimbal dial fully down, then:
python dji_mission.py 192.168.43.1 --dry       # video and offsets, never sends a command
python dji_mission.py 192.168.43.1 --kp 0      # takes control without steering: tests the hand-offs
python dji_mission.py 192.168.43.1
Keys in the video window: s start, h hold (control back to the RC), l land now, q quit.
"""
import argparse
import math
import time

import cv2
import numpy as np

from aruco_pose import load_camera, marker_offset
from dji_link import DjiLink
from mission import offset_to_velocity

HOVER = {"fwd": 0.0, "right": 0.0, "down": 0.0}


def tick(m, off, vs, now, args):
    """Advance the mission one step and return the message for the relay, or None.

    m: {"phase", "since", "last_seen", "settled"}; off: marker offset or None; vs: Virtual Stick on.
    """
    if m["phase"] == "arming":
        if vs:
            m.update(phase="centre", last_seen=now, settled=None)
        elif now - m["since"] > 3.0:
            m["phase"] = "idle"
        return None
    if m["phase"] not in ("centre", "descend"):
        return None
    if not vs:  # the pilot, or DJI itself, took control away
        m["phase"] = "hold"
        return None
    if off is None:
        m["settled"] = None
        if now - m["last_seen"] > args.lost:
            m["phase"] = "hold"
            return {"cmd": "disable"}
        return dict(HOVER)
    m["last_seen"] = now
    fwd, right = offset_to_velocity(off, args.kp, args.vmax, args.cam_yaw)
    horiz = math.hypot(off[0], off[1])
    if m["phase"] == "centre":
        if horiz < args.tol:
            m["settled"] = m["settled"] or now
            if now - m["settled"] > 1.0:
                m["phase"] = "descend"
        else:
            m["settled"] = None
        return {"fwd": fwd, "right": right, "down": 0.0}
    if off[2] < args.land_alt:
        m["phase"] = "landing"
        return {"cmd": "land"}
    # descend only while centred, so the tag stays in frame as the view narrows
    return {"fwd": fwd, "right": right, "down": args.vmax if horiz < args.tol else 0.0}


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("host", help="phone IP: the laptop's default gateway on the phone hotspot")
    p.add_argument("--camera", default="camera_mini3.yaml")
    p.add_argument("--marker-id", type=int, default=0)
    p.add_argument("--marker-size", type=float, default=0.14, help="outer black square, metres")
    p.add_argument("--dict", default="DICT_APRILTAG_36h11", help="cv2.aruco dictionary name")
    p.add_argument("--kp", type=float, default=0.3)
    p.add_argument("--vmax", type=float, default=0.1, help="m/s cap on centring and descent")
    p.add_argument("--cam-yaw", type=float, default=0.0, help="deg, image-up vs nose with the gimbal down")
    p.add_argument("--tol", type=float, default=0.15, help="m, horizontal error counted as centred")
    p.add_argument("--land-alt", type=float, default=0.8, help="m above the tag where DJI auto-land takes over")
    p.add_argument("--lost", type=float, default=5.0, help="s without the tag before giving control back")
    p.add_argument("--dry", action="store_true", help="show video and offsets, never send a command")
    args = p.parse_args(argv)
    args.dict = getattr(cv2.aruco, args.dict)
    return args


def draw(frame, K, off, lines):
    img = frame.copy()
    h, w = img.shape[:2]
    cv2.drawMarker(img, (w // 2, h // 2), (0, 255, 0), cv2.MARKER_CROSS, 40, 2)
    if off is not None:
        x, y, z = off
        cv2.circle(img, (int(K[0, 0] * x / z + K[0, 2]), int(K[1, 1] * y / z + K[1, 2])), 20, (0, 0, 255), 3)
    for i, line in enumerate(lines):
        cv2.putText(img, line, (10, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return img


def main():
    args = parse_args()
    K, dist = load_camera(args.camera)
    link = DjiLink(args.host)
    link.start()
    m = {"phase": "idle", "since": 0.0, "last_seen": 0.0, "settled": None}
    blank = np.zeros((480, 640, 3), np.uint8)
    while True:
        frame, stamp = link.get()
        now = time.monotonic()
        age = now - stamp
        # a frame older than 0.5 s is not where the drone is now: treat it as tag not seen
        off = (marker_offset(frame, K, dist, args.marker_id, args.marker_size, args.dict)
               if frame is not None and age < 0.5 else None)
        vs, flying = bool(link.state.get("vs")), bool(link.state.get("flying"))
        before = m["phase"]
        msg = tick(m, off, vs, now, args)
        if msg is not None and not args.dry:
            link.send(**msg)
        if m["phase"] == "landing" and not flying:
            m["phase"] = "idle"
        if m["phase"] != before:
            print(f"{time.strftime('%H:%M:%S')} {before} -> {m['phase']}")

        lines = [f"phase {m['phase']}{'  DRY' if args.dry else ''}   flying {flying}   virtual stick {vs}",
                 "no video" if frame is None else f"frame {age:.2f} s old",
                 "tag not seen" if off is None else f"tag x={off[0]:+.2f} y={off[1]:+.2f} z={off[2]:.2f} m",
                 "" if msg is None else f"cmd {msg}"]
        cv2.imshow("Mini 3 precision landing", draw(blank if frame is None else frame, K, off, lines))
        key = cv2.waitKey(100) & 0xFF  # also paces the loop at 10 Hz
        if key == ord("s") and not args.dry and m["phase"] in ("idle", "hold"):
            if flying and off is not None:
                link.send(cmd="enable")
                m.update(phase="arming", since=now)
            else:
                print("start refused: need a flying aircraft with the tag in view")
        elif key == ord("h") and not args.dry:
            link.send(cmd="disable")
            m["phase"] = "hold"
        elif key == ord("l") and not args.dry:
            link.send(cmd="land")
            m["phase"] = "landing"
        elif key == ord("q"):
            if m["phase"] in ("arming", "centre", "descend"):
                link.send(cmd="disable")
                time.sleep(0.3)
            break


if __name__ == "__main__":
    main()
