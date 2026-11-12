"""Tkinter control window for the precision-landing mission.

python ui.py
Connect, watch the live marker readout (replaces --dry), Start mission, and at any moment
HOLD / LAND / KILL. KILL cuts the motors at once: the drone falls.
"""
import asyncio
import queue
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

import cv2

import mission
from aruco_pose import load_camera, marker_offset

VIEW_W = 640


class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("DEXI 3 precision landing")
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()
        self.drone = self.frames = self.telem = self.task = None
        self.K = self.dist = None
        self.ctl = {"abort": "land"}
        self.kill_armed_at = 0.0
        self.logq = queue.Queue()
        self.ticks = 0
        self.last_off = None
        mission.log = self.log

        # a Label sizes in characters unless it holds an image; start with a blank one
        self.photo = tk.PhotoImage(width=VIEW_W, height=480)
        self.video = tk.Label(self.root, bg="black", image=self.photo)
        self.video.grid(row=0, column=0, rowspan=3, padx=4, pady=4)

        form = ttk.Frame(self.root)
        form.grid(row=0, column=1, sticky="new", padx=4, pady=4)
        self.fields = {}
        for i, (key, label, default) in enumerate([
                ("url", "MAVLink URL", "udpout://192.168.4.1:14540"),
                ("rtsp", "Camera URL", "ws://192.168.4.1:9090"),
                ("topic", "Camera topic", "/cam0/image_raw/compressed"),
                ("camera", "camera.yaml", "camera.yaml"),
                ("alt", "Altitude m", "2"),
                ("acquire_alt", "Acquire alt m", "1.5"),
                ("kp", "Kp", "0.5"),
                ("vmax", "Max speed m/s", "0.1"),
                ("cam_yaw", "Cam yaw deg", "180"),
                ("marker_size", "Tag size m", "0.14"),
                ("dict", "Tag dictionary", "DICT_APRILTAG_36h11")]):
            ttk.Label(form, text=label).grid(row=i, column=0, sticky="w")
            e = ttk.Entry(form, width=28)
            e.insert(0, default)
            e.grid(row=i, column=1, pady=1)
            self.fields[key] = e
        self.connect_btn = ttk.Button(form, text="Connect", command=self.connect)
        self.connect_btn.grid(row=len(self.fields), column=0, columnspan=2, pady=4, sticky="ew")

        self.status = tk.Label(self.root, justify="left", anchor="nw", font=("Consolas", 10),
                               text="not connected")
        self.status.grid(row=1, column=1, sticky="new", padx=4)

        btns = ttk.Frame(self.root)
        btns.grid(row=2, column=1, sticky="sew", padx=4, pady=4)
        self.start_btn = ttk.Button(btns, text="Start mission", command=self.start, state="disabled")
        self.start_btn.pack(fill="x", pady=2)
        self.hold_btn = tk.Button(btns, text="HOLD", bg="#ffd966", height=2,
                                  command=lambda: self.emergency("hold"))
        self.hold_btn.pack(fill="x", pady=2)
        self.land_btn = tk.Button(btns, text="LAND", bg="#f4b183", height=2,
                                  command=lambda: self.emergency("land"))
        self.land_btn.pack(fill="x", pady=2)
        self.kill_btn = tk.Button(btns, text="KILL (motor cutoff)", bg="#c00000", fg="white",
                                  height=2, command=self.kill)
        self.kill_btn.pack(fill="x", pady=2)

        self.logbox = tk.Text(self.root, height=8, state="disabled", font=("Consolas", 9))
        self.logbox.grid(row=3, column=0, columnspan=2, sticky="ew", padx=4, pady=4)

        self.root.after(100, self.tick)

    # --- helpers ---------------------------------------------------------
    def log(self, *parts):
        # called from the asyncio thread too; Tk widgets are only touched in tick()
        self.logq.put(time.strftime("%H:%M:%S ") + " ".join(str(p) for p in parts))

    def submit(self, coro):
        fut = asyncio.run_coroutine_threadsafe(coro, self.loop)

        def done(f):
            if not f.cancelled() and f.exception() is not None:
                self.log(f"error: {f.exception()!r}")
        fut.add_done_callback(done)
        return fut

    def args(self):
        f = {k: e.get() for k, e in self.fields.items()}
        return mission.parse_args([
            "--url", f["url"], "--rtsp", f["rtsp"], "--topic", f["topic"], "--camera", f["camera"],
            "--alt", f["alt"], "--acquire-alt", f["acquire_alt"], "--kp", f["kp"], "--vmax", f["vmax"],
            "--cam-yaw", f["cam_yaw"], "--marker-size", f["marker_size"], "--dict", f["dict"]])

    # --- buttons ---------------------------------------------------------
    def connect(self):
        a = self.args()
        self.K, self.dist = load_camera(a.camera)
        self.frames = mission.LatestFrame(a.rtsp, a.topic)
        self.frames.start()
        self.connect_btn.configure(state="disabled")

        async def go():
            self.drone = await mission.connect(a.url)
            self.telem = mission.Telem(self.drone)
        self.submit(go())

    def start(self):
        if self.drone is None or self.task is not None:
            return
        a = self.args()
        if not messagebox.askokcancel("Start mission",
                                      f"Climb to {a.alt:.1f} m, rotate 360, return to {a.acquire_alt:.1f} m, "
                                      f"land on the tag (Kp {a.kp}), all moves at {a.vmax} m/s.\n\nIs the ceiling clear above {a.alt:.1f} m?"):
            return
        self.ctl["abort"] = "land"

        async def go():
            self.task = asyncio.current_task()
            try:
                await mission.fly(a, self.drone, self.frames, self.K, self.dist, self.ctl)
            finally:
                self.task = None
        self.submit(go())

    def emergency(self, kind):
        if self.drone is None:
            self.log("not connected")
            return
        self.log(f"EMERGENCY: {kind}")
        self.ctl["abort"] = kind
        self.submit(mission.abort_action(self.drone, kind))  # direct, even if no mission runs
        if self.task is not None:
            self.loop.call_soon_threadsafe(self.task.cancel)

    def kill(self):
        now = time.monotonic()
        if now - self.kill_armed_at > 3.0:
            self.kill_armed_at = now
            self.kill_btn.configure(text="CONFIRM KILL (3 s)", bg="#ff0000")
            self.root.after(3000, self.reset_kill)
            return
        self.reset_kill()
        self.emergency("kill")

    def reset_kill(self):
        self.kill_armed_at = 0.0
        self.kill_btn.configure(text="KILL (motor cutoff)", bg="#c00000")

    # --- periodic display ------------------------------------------------
    def tick(self):
        while not self.logq.empty():
            self.logbox.configure(state="normal")
            self.logbox.insert("end", self.logq.get() + "\n")
            self.logbox.see("end")
            self.logbox.configure(state="disabled")
        self.start_btn.configure(state="normal" if self.drone and self.task is None else "disabled")
        lines = []
        t = self.telem
        if t is not None:
            lines += [f"mode   {getattr(t.mode, 'name', '?')}",
                      f"armed  {t.armed}   in air {t.in_air}   PX4 pre-arm {'OK' if t.armable else 'FAIL'}",
                      f"alt    {t.rel_alt if t.rel_alt is None else f'{t.rel_alt:.2f} m'}"
                      f"  ({'rangefinder' if t.rng_live else 'PX4 local Z, NO rangefinder'})",
                      f"link   {'LOST' if mission.link_lost(t) else 'ok'}"]
        elif self.drone is not None:
            lines.append("connected, waiting for telemetry")
        else:
            lines.append("not connected")

        if self.frames is not None:
            frame, stamp = self.frames.get()
            if frame is not None:
                try:
                    a = self.args()
                    # detect at ~3 Hz: at 10 Hz the detector starves the MAVSDK telemetry thread
                    if self.ticks % 3 == 0:
                        self.last_off = marker_offset(frame, self.K, self.dist, a.marker_id, a.marker_size, a.dict)
                    off = self.last_off
                except (ValueError, SystemExit, AttributeError):  # bad field or unknown dictionary
                    off, a = None, None
                    lines.append("check the fields")
                lines.append(f"frame  {time.monotonic() - stamp:.2f} s old")
                if off is None:
                    lines.append("marker not seen")
                else:
                    fwd, right = mission.offset_to_velocity(off, a.kp, a.vmax, a.cam_yaw)
                    lines.append(f"marker x={off[0]:+.2f} y={off[1]:+.2f} z={off[2]:.2f} m")
                    lines.append(f"cmd    fwd={fwd:+.2f} right={right:+.2f} m/s")
                self.show(frame, off)
        self.status.configure(text="\n".join(lines))
        self.ticks += 1
        self.root.after(100, self.tick)

    def show(self, frame, off):
        h, w = frame.shape[:2]
        s = VIEW_W / w
        img = cv2.resize(frame, (VIEW_W, int(h * s)))
        cx, cy = VIEW_W // 2, int(h * s) // 2
        cv2.drawMarker(img, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 30, 1)
        if off is not None:
            x, y, z = off
            u = int((self.K[0, 0] * x / z + self.K[0, 2]) * s)
            v = int((self.K[1, 1] * y / z + self.K[1, 2]) * s)
            cv2.circle(img, (u, v), 12, (0, 0, 255), 2)
        # ponytail: PPM bytes straight into Tk avoids a Pillow dependency; fine at 10 fps
        self.photo = tk.PhotoImage(data=cv2.imencode(".ppm", img)[1].tobytes())
        self.video.configure(image=self.photo, width=img.shape[1], height=img.shape[0])


if __name__ == "__main__":
    App().root.mainloop()
