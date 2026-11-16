"""Laptop end of the phone relay (android/Relay.kt): video in, commands out, one TCP socket.

python dji_link.py 192.168.43.1                 # live video, never sends a command
python dji_link.py 192.168.43.1 --nudge fwd     # IN FLIGHT: 0.1 m/s along one axis for 2 s
"""
import argparse
import json
import socket
import struct
import threading
import time

import cv2
import numpy as np

PORT = 9000
MAX_PART = 16 << 20  # no state header or JPEG is ever this big; a larger length is a corrupt stream


def read_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("relay closed the connection")
        buf += chunk
    return bytes(buf)


def read_part(sock):
    (n,) = struct.unpack(">I", read_exact(sock, 4))
    if n > MAX_PART:
        raise ValueError(f"relay sent a {n}-byte part; stream is corrupt")
    return read_exact(sock, n)


def read_packet(sock):
    """One relay packet -> (state dict, BGR frame or None for a heartbeat)."""
    state = json.loads(read_part(sock))
    jpeg = read_part(sock)
    return state, (cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR) if jpeg else None)


class DjiLink(threading.Thread):
    """Same get() as mission.LatestFrame, plus the relay's state and a command sender."""

    def __init__(self, host, port=PORT):
        super().__init__(daemon=True)
        self.addr = (host, port)
        self.lock = threading.Lock()
        self.frame, self.stamp, self.state = None, 0.0, {}
        self.sock = None

    def run(self):
        while True:
            try:
                s = socket.create_connection(self.addr, timeout=5)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.sock = s
                while True:
                    state, frame = read_packet(s)
                    with self.lock:
                        self.state = state
                        if frame is not None:
                            self.frame, self.stamp = frame, time.monotonic()
            except (OSError, ValueError) as ex:  # reconnect; a dead link shows as stale frames
                self.sock = None
                with self.lock:
                    self.state = {}
                print(f"relay link: {ex!r}; reconnecting")
                time.sleep(1)

    def get(self):
        with self.lock:
            return self.frame, self.stamp

    def send(self, **msg):
        """False if the link is down; the relay then hovers on its own watchdog."""
        s = self.sock
        if s is None:
            return False
        try:
            s.sendall(json.dumps(msg).encode() + b"\n")
            return True
        except OSError:
            return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("host")
    p.add_argument("--nudge", choices=["fwd", "right", "down"],
                   help="take control, move 0.1 m/s along this axis for 2 s, give control back")
    a = p.parse_args()
    link = DjiLink(a.host)
    link.start()
    if a.nudge:
        while not link.state.get("flying"):
            print("waiting for a flying aircraft...")
            time.sleep(1)
        link.send(cmd="enable")
        time.sleep(1.5)
        if not link.state.get("vs"):
            raise SystemExit("Virtual Stick did not enable")
        v = {"fwd": 0.0, "right": 0.0, "down": 0.0, a.nudge: 0.1}
        for _ in range(20):
            link.send(**v)
            time.sleep(0.1)
        link.send(cmd="disable")
        time.sleep(0.5)
        print(f"nudged {a.nudge}: the drone should have moved about 0.2 m that way")
        return
    while True:
        frame, stamp = link.get()
        if frame is not None:
            cv2.putText(frame, f"{link.state}  {time.monotonic() - stamp:.2f} s old", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("dji_link (q quits)", frame)
        if cv2.waitKey(50) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()
