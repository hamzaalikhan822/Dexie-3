# Mini 3 Precision Landing (Laptop Processing) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land a DJI Mini 3 on the ArUco/AprilTag pad, with tag detection and velocity commands computed on the laptop and an Android phone acting only as a relay.

**Architecture:** The phone (plugged into the RC-N1) runs DJI's official sample app plus one added file, `Relay.kt`, which opens a TCP server: it streams JPEG video frames and a small state header to the laptop and applies the velocity commands the laptop sends back through DJI Virtual Stick. The laptop runs two new Python files: `dji_link.py` (the socket client, same `get()` interface as `mission.LatestFrame`) and `dji_mission.py` (mission loop and an OpenCV window). Detection (`aruco_pose.marker_offset`) and the control law (`mission.offset_to_velocity`) are reused unchanged.

**Tech Stack:** Python 3.10+, opencv-contrib-python, numpy, pyyaml, pytest (all already in `requirements.txt`). Android Studio, Kotlin, DJI Mobile SDK v5 (5.20.1 sample, `com.dji:dji-sdk-v5-aircraft`).

## Global Constraints

- Windows laptop, PowerShell. Use `python` from the shell.
- No new Python dependencies. No changes to `mission.py`, `ui.py` or `aruco_pose.py`: the DEXI 3 path must keep working and its 33 tests must stay green.
- Phone and laptop talk over the phone's Wi-Fi hotspot. The phone's USB port is taken by the RC-N1 in flight.
- Wire format, phone to laptop, repeated: `[4-byte big-endian length][state JSON][4-byte big-endian length][JPEG, may be empty]`. State JSON is `{"flying": bool, "vs": bool}` (`vs` = Virtual Stick enabled).
- Wire format, laptop to phone: one JSON object per line. Velocity: `{"fwd": m/s, "right": m/s, "down": m/s}`. Commands: `{"cmd": "enable"}`, `{"cmd": "disable"}`, `{"cmd": "land"}`.
- TCP port 9000.
- Default speed cap 0.1 m/s on every scripted move (same as the DEXI 3). The phone hard-clamps to 0.5 m/s whatever the laptop sends.
- Phone watchdog: no velocity message for 0.5 s means hover (zero velocity). Laptop socket closed means Virtual Stick disabled, control back to the RC.
- Tag not seen: zero velocity at once. Not seen for 5 s: give control back to the RC (the Mini 3 hovers on GPS). Never land blind.
- The pilot takes off and points the gimbal by hand: manual takeoff, hover about 2 m over the tag, gimbal dial rolled fully down. The script takes over from that hover.
- Tag defaults match the existing pad: `DICT_APRILTAG_36h11`, id 0, 0.14 m.
- All flight tests outdoors, calm wind, pilot holding the RC-N1. The RC pause button or flight-mode switch cancels Virtual Stick at any time.

## File Structure

| File | Responsibility |
|---|---|
| `dji_link.py` (new) | Socket client. Parses relay packets, keeps the newest frame and state, sends commands. Also a bench tool: live view and a single-axis nudge. |
| `tests/test_dji_link.py` (new) | Packet parsing and command framing, over a local socket pair. |
| `android/Relay.kt` (new) | The whole phone side. Copied into DJI's sample app. |
| `calibrate.py` (modify, lines 94) | Accept `--rtsp dji://HOST` so the existing checkerboard calibration works on the Mini 3 stream. |
| `dji_mission.py` (new) | `tick()` state machine (pure, tested) and the OpenCV window loop. |
| `tests/test_dji_mission.py` (new) | `tick()` behaviour. |
| `README.md` (modify) | Mini 3 section: setup, checks, flight procedure. |

---

### Task 0: Hardware gate (manual, no code)

Nothing below is worth building until this passes.

- [ ] **Step 1: Pair the RC-N1 (RC231) with the Mini 3**

In DJI Fly: camera view, `⋯`, Control, Re-pair to Aircraft, then hold the drone's power button four seconds. Take any firmware update offered. Then force-close DJI Fly.

- [ ] **Step 2: Build DJI's sample app**

Clone `https://github.com/dji-sdk/Mobile-SDK-Android-V5` (branch `dev-sdk-main`) somewhere outside this repo. Create an App Key at developer.dji.com for the sample's package name, put it where the sample's README says, open `SampleCode-V5/android-sdk-v5-sample` in Android Studio, and run it on the phone.

- [ ] **Step 3: Check the three things**

Phone plugged into the RC-N1, drone on:
1. The sample shows the aircraft as connected and registration succeeded.
2. Its camera/live-view page shows video.
3. Its Virtual Stick page can enable Virtual Stick and move the drone in a low outdoor hover.

Expected: all three pass. If 2 or 3 fails on the Mini 3, stop: the plan is not viable.

---

### Task 1: `dji_link.py`, the laptop end of the link

**Files:**
- Create: `dji_link.py`
- Test: `tests/test_dji_link.py`

**Interfaces:**
- Consumes: nothing from this repo.
- Produces:
  - `dji_link.PORT = 9000`
  - `dji_link.read_packet(sock) -> (state: dict, frame: np.ndarray | None)`
  - `dji_link.DjiLink(host, port=PORT)`, a daemon `threading.Thread` with `.get() -> (frame | None, monotonic stamp)`, `.state: dict`, `.send(**msg) -> bool`

- [ ] **Step 1: Write the failing tests**

`tests/test_dji_link.py`:

```python
import json
import socket
import struct

import cv2
import numpy as np
import pytest

import dji_link


def packet(state, jpeg=b""):
    h = json.dumps(state).encode()
    return struct.pack(">I", len(h)) + h + struct.pack(">I", len(jpeg)) + jpeg


def test_read_packet_decodes_state_and_jpeg():
    img = np.zeros((8, 8, 3), np.uint8)
    img[:, :, 2] = 255  # red in BGR
    a, b = socket.socketpair()
    a.sendall(packet({"flying": True, "vs": False}, cv2.imencode(".jpg", img)[1].tobytes()))
    state, frame = dji_link.read_packet(b)
    assert state == {"flying": True, "vs": False}
    assert frame.shape == (8, 8, 3) and frame[4, 4, 2] > 200 and frame[4, 4, 0] < 50


def test_heartbeat_packet_has_no_frame():
    a, b = socket.socketpair()
    a.sendall(packet({"flying": False, "vs": False}))
    assert dji_link.read_packet(b) == ({"flying": False, "vs": False}, None)


def test_closed_socket_raises():
    a, b = socket.socketpair()
    a.close()
    with pytest.raises(ConnectionError):
        dji_link.read_packet(b)


def test_oversized_length_is_rejected():
    a, b = socket.socketpair()
    a.sendall(struct.pack(">I", 1 << 30))
    with pytest.raises(ValueError):
        dji_link.read_packet(b)


def test_send_writes_one_json_line():
    a, b = socket.socketpair()
    link = dji_link.DjiLink("unused")
    link.sock = a
    assert link.send(fwd=0.1, right=0.0, down=0.0) is True
    raw = b.recv(200)
    assert raw.endswith(b"\n") and json.loads(raw) == {"fwd": 0.1, "right": 0.0, "down": 0.0}


def test_send_without_a_link_returns_false():
    assert dji_link.DjiLink("unused").send(cmd="land") is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_dji_link.py -v`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'dji_link'`

- [ ] **Step 3: Write `dji_link.py`**

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_dji_link.py -v`
Expected: 6 passed

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest -q`
Expected: 39 passed (33 existing + 6 new)

- [ ] **Step 6: Commit**

```bash
git add dji_link.py tests/test_dji_link.py
git commit -m "feat: dji_link, laptop client for the Mini 3 phone relay"
```

---

### Task 2: `android/Relay.kt`, the phone side

**Files:**
- Create: `android/Relay.kt` (kept in this repo, copied into the DJI sample)
- Modify (in the DJI sample checkout, not this repo): `SampleCode-V5/android-sdk-v5-sample/src/main/java/dji/sampleV5/aircraft/DJIMainActivity.kt`, one added line

**Interfaces:**
- Consumes: the wire format in Global Constraints.
- Produces: a TCP server on port 9000 speaking that format; `Relay.start()` (idempotent).

No unit test: this file is glue around DJI calls that only run on the phone with an aircraft. Its check is Step 5, on real hardware.

- [ ] **Step 1: Write `android/Relay.kt`**

```kotlin
package dji.sampleV5.aircraft

import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.util.Log
import dji.sdk.keyvalue.key.FlightControllerKey
import dji.sdk.keyvalue.key.KeyTools
import dji.sdk.keyvalue.value.common.ComponentIndexType
import dji.sdk.keyvalue.value.common.EmptyMsg
import dji.sdk.keyvalue.value.flightcontroller.FlightControlAuthorityChangeReason
import dji.sdk.keyvalue.value.flightcontroller.FlightCoordinateSystem
import dji.sdk.keyvalue.value.flightcontroller.RollPitchControlMode
import dji.sdk.keyvalue.value.flightcontroller.VerticalControlMode
import dji.sdk.keyvalue.value.flightcontroller.VirtualStickFlightControlParam
import dji.sdk.keyvalue.value.flightcontroller.YawControlMode
import dji.v5.common.callback.CommonCallbacks
import dji.v5.common.error.IDJIError
import dji.v5.manager.KeyManager
import dji.v5.manager.aircraft.virtualstick.VirtualStickManager
import dji.v5.manager.aircraft.virtualstick.VirtualStickState
import dji.v5.manager.aircraft.virtualstick.VirtualStickStateListener
import dji.v5.manager.datacenter.MediaDataCenter
import dji.v5.manager.interfaces.ICameraStreamManager
import org.json.JSONException
import org.json.JSONObject
import java.io.BufferedOutputStream
import java.io.ByteArrayOutputStream
import java.io.DataOutputStream
import java.io.IOException
import java.net.ServerSocket
import java.net.Socket
import java.util.concurrent.atomic.AtomicReference
import kotlin.concurrent.thread

/** Phone relay for the laptop (dji_link.py): video and state out, velocity commands in. */
object Relay {
    private const val TAG = "Relay"
    private const val PORT = 9000
    private const val MAX_SPEED = 0.5      // m/s, hard cap whatever the laptop asks for
    private const val WATCHDOG_MS = 500L   // no velocity message for this long: hover
    private const val FRAME_GAP_MS = 100L  // 10 fps is all the 10 Hz control loop can use

    private var started = false
    private val jpeg = AtomicReference<ByteArray?>(null)
    @Volatile private var lastFrameMs = 0L
    @Volatile private var lastCmdMs = 0L
    @Volatile private var vsOn = false
    @Volatile private var vel = doubleArrayOf(0.0, 0.0, 0.0)  // fwd, right, down

    private val stick get() = VirtualStickManager.getInstance()

    @Synchronized
    fun start() {
        if (started) return
        started = true
        MediaDataCenter.getInstance().cameraStreamManager.addFrameListener(
            ComponentIndexType.LEFT_OR_MAIN, ICameraStreamManager.FrameFormat.NV21,
            object : ICameraStreamManager.CameraFrameListener {
                override fun onFrame(frameData: ByteArray, offset: Int, length: Int, width: Int,
                                     height: Int, format: ICameraStreamManager.FrameFormat) {
                    val now = System.currentTimeMillis()
                    if (now - lastFrameMs < FRAME_GAP_MS) return
                    lastFrameMs = now
                    val nv21 = if (offset == 0 && length == frameData.size) frameData
                               else frameData.copyOfRange(offset, offset + length)
                    val out = ByteArrayOutputStream()
                    // ponytail: full-resolution JPEG; downscale here if the Wi-Fi hop lags
                    YuvImage(nv21, ImageFormat.NV21, width, height, null)
                        .compressToJpeg(Rect(0, 0, width, height), 70, out)
                    jpeg.set(out.toByteArray())
                }
            })
        stick.setVirtualStickStateListener(object : VirtualStickStateListener {
            override fun onVirtualStickStateUpdate(stickState: VirtualStickState) {
                vsOn = stickState.isVirtualStickEnable
            }
            override fun onChangeReasonUpdate(reason: FlightControlAuthorityChangeReason) {
                Log.i(TAG, "control authority changed: $reason")
            }
        })
        thread(isDaemon = true, name = "relay-server") { serve() }
        thread(isDaemon = true, name = "relay-stick") { stickLoop() }
        Log.i(TAG, "listening on port $PORT")
    }

    private fun done(what: String) = object : CommonCallbacks.CompletionCallback {
        override fun onSuccess() { Log.i(TAG, "$what ok") }
        override fun onFailure(error: IDJIError) { Log.e(TAG, "$what failed: $error") }
    }

    private fun release(why: String) {
        Log.i(TAG, "releasing Virtual Stick: $why")
        vel = doubleArrayOf(0.0, 0.0, 0.0)
        stick.disableVirtualStick(done("disable"))
    }

    private fun flying(): Boolean =
        KeyManager.getInstance().getValue(KeyTools.createKey(FlightControllerKey.KeyIsFlying)) ?: false

    private fun serve() {
        val server = ServerSocket(PORT)
        while (true) {
            val sock = server.accept()
            sock.tcpNoDelay = true
            Log.i(TAG, "laptop connected: ${sock.inetAddress}")
            thread(isDaemon = true, name = "relay-commands") { readCommands(sock) }
            try {
                val out = DataOutputStream(BufferedOutputStream(sock.getOutputStream()))
                while (!sock.isClosed) {
                    val state = JSONObject().put("flying", flying()).put("vs", vsOn)
                        .toString().toByteArray()
                    val img = jpeg.getAndSet(null) ?: ByteArray(0)
                    out.writeInt(state.size); out.write(state)
                    out.writeInt(img.size); out.write(img)
                    out.flush()
                    Thread.sleep(50)
                }
            } catch (e: IOException) {
                Log.i(TAG, "laptop link: $e")
            } finally {
                sock.close()
                release("laptop link closed")
            }
        }
    }

    private fun readCommands(sock: Socket) {
        try {
            sock.getInputStream().bufferedReader().forEachLine { line ->
                try {
                    val m = JSONObject(line)
                    when (m.optString("cmd")) {
                        "enable" -> {
                            lastCmdMs = 0L  // start in hover until the first velocity arrives
                            stick.enableVirtualStick(done("enable"))
                            stick.setVirtualStickAdvancedModeEnabled(true)
                        }
                        "disable" -> release("laptop asked")
                        "land" -> {
                            // ponytail: no tag corrections during DJI's own descent; add them only if
                            // the measured landing error from the hand-off height is too large
                            release("landing")
                            KeyManager.getInstance().performAction(
                                KeyTools.createKey(FlightControllerKey.KeyStartAutoLanding),
                                object : CommonCallbacks.CompletionCallbackWithParam<EmptyMsg> {
                                    override fun onSuccess(t: EmptyMsg?) { Log.i(TAG, "auto-land started") }
                                    override fun onFailure(error: IDJIError) { Log.e(TAG, "auto-land failed: $error") }
                                })
                        }
                        else -> {
                            vel = doubleArrayOf(clamp(m.getDouble("fwd")), clamp(m.getDouble("right")),
                                                clamp(m.getDouble("down")))
                            lastCmdMs = System.currentTimeMillis()
                        }
                    }
                } catch (e: JSONException) {
                    Log.e(TAG, "bad command ignored: $line")
                }
            }
        } catch (e: IOException) {
            Log.i(TAG, "command stream: $e")
        } finally {
            sock.close()  // also ends the sender loop in serve(), which releases Virtual Stick
        }
    }

    private fun clamp(v: Double) = if (v.isNaN()) 0.0 else v.coerceIn(-MAX_SPEED, MAX_SPEED)

    private fun stickLoop() {
        while (true) {
            if (vsOn) {
                val stale = System.currentTimeMillis() - lastCmdMs > WATCHDOG_MS
                val v = if (stale) doubleArrayOf(0.0, 0.0, 0.0) else vel
                stick.sendVirtualStickAdvancedParam(VirtualStickFlightControlParam().apply {
                    rollPitchCoordinateSystem = FlightCoordinateSystem.BODY
                    rollPitchControlMode = RollPitchControlMode.VELOCITY
                    yawControlMode = YawControlMode.ANGULAR_VELOCITY
                    verticalControlMode = VerticalControlMode.VELOCITY
                    // axis mapping is confirmed in flight by Task 5 Step 2; swap these two if wrong
                    roll = v[0]                // forward, m/s
                    pitch = v[1]               // right, m/s
                    yaw = 0.0
                    verticalThrottle = -v[2]   // SDK is up-positive; the laptop sends down-positive
                })
            }
            Thread.sleep(50)  // 20 Hz, inside DJI's recommended 5 to 25 Hz
        }
    }
}
```

- [ ] **Step 2: Copy it into the sample and start it**

Copy `android/Relay.kt` to `SampleCode-V5/android-sdk-v5-sample/src/main/java/dji/sampleV5/aircraft/Relay.kt` in the DJI checkout. In `DJIMainActivity.kt`, search for the registration-success branch (the one that shows the "Register Success" toast) and add one line after the toast:

```kotlin
Relay.start()
```

- [ ] **Step 3: Build, and fix any name the compiler rejects**

Run the sample from Android Studio onto the phone.
Expected: builds and installs. These names were taken from DJI's API reference and sample but could not be compiled here; if the compiler rejects one, open the named sample file and copy its spelling:

| Name in `Relay.kt` | Where the sample uses it |
|---|---|
| `stickState.isVirtualStickEnable`, `VirtualStickStateListener` | `models/VirtualStickVM.kt` |
| `ComponentIndexType.LEFT_OR_MAIN`, `CameraFrameListener.onFrame` | `models/CameraStreamDetailVM.kt` |
| `KeyManager.getInstance().performAction(...)`, `getValue(...)` | any file under `models/` that imports `KeyManager` |

- [ ] **Step 4: Network setup**

Turn on the phone's Wi-Fi hotspot and join it from the laptop. The phone's address is the laptop's default gateway:

```bash
ipconfig
```

Use that address as `PHONE_IP` below.

- [ ] **Step 5: Bench check, drone on the ground, motors off**

Phone in the RC-N1, drone on, sample app open and registered. On the laptop:

```bash
python dji_link.py PHONE_IP
```

Expected: a window with the Mini 3's live camera view, overlay text `{'flying': False, 'vs': False}` and a frame age under about 0.3 s. Wave a hand in front of the camera and judge the lag by eye. If the age climbs steadily, lower the JPEG quality (70) in `Relay.kt` or downscale.

- [ ] **Step 6: Commit**

```bash
git add android/Relay.kt
git commit -m "feat: Relay.kt, phone relay for Mini 3 video and Virtual Stick"
```

---

### Task 3: Calibrate the Mini 3 camera through the relay

**Files:**
- Modify: `calibrate.py:94`
- Create (generated): `camera_mini3.yaml`

**Interfaces:**
- Consumes: `dji_link.DjiLink(host)` and its `.get()`.
- Produces: `camera_mini3.yaml`, same format as `camera.yaml`, at the relay stream's resolution.

- [ ] **Step 1: Let `calibrate.py` read from the relay**

Replace line 94 of `calibrate.py`:

```python
    frames = LatestFrame(a.rtsp, a.topic)
```

with:

```python
    if a.rtsp.startswith("dji://"):
        from dji_link import DjiLink
        frames = DjiLink(a.rtsp[len("dji://"):])
    else:
        frames = LatestFrame(a.rtsp, a.topic)
```

- [ ] **Step 2: Check the existing tests still pass**

Run: `python -m pytest -q`
Expected: 39 passed

- [ ] **Step 3: Run the checkerboard calibration**

Print `checkerboard_9x6_22mm.png` at 100 % scale if no print exists. Drone on the ground, relay running. Hold the board in front of the camera and move and tilt it until 20 captures are taken:

```bash
python calibrate.py --rtsp dji://PHONE_IP --cols 9 --rows 6 --square 0.022 --out camera_mini3.yaml
```

Expected: `wrote camera_mini3.yaml, reprojection error` below 1.0 px, and a principal point within about 10 % of the image centre. Do not use `--tag` or `--nominal`: those are tuned to the DEXI 3 camera.

- [ ] **Step 4: Commit**

```bash
git add calibrate.py camera_mini3.yaml
git commit -m "feat: calibrate over the Mini 3 relay; camera_mini3.yaml"
```

---

### Task 4: `dji_mission.py`, mission logic and window

**Files:**
- Create: `dji_mission.py`
- Test: `tests/test_dji_mission.py`

**Interfaces:**
- Consumes: `dji_link.DjiLink` (`.get()`, `.state`, `.send(**msg)`), `aruco_pose.load_camera`, `aruco_pose.marker_offset`, `mission.offset_to_velocity(offset, kp, vmax, cam_yaw_deg) -> (fwd, right)`.
- Produces: `dji_mission.tick(m, off, vs, now, args) -> dict | None`, where `m` is `{"phase", "since", "last_seen", "settled"}` and phase is one of `idle`, `arming`, `centre`, `descend`, `landing`, `hold`.

- [ ] **Step 1: Write the failing tests**

`tests/test_dji_mission.py`:

```python
from types import SimpleNamespace

import pytest

import dji_mission

ARGS = SimpleNamespace(kp=0.5, vmax=0.1, cam_yaw=0.0, tol=0.15, land_alt=0.8, lost=5.0)
CENTRED = (0.0, 0.0, 2.0)


def state(phase, **kw):
    return {"phase": phase, "since": 0.0, "last_seen": 100.0, "settled": None, **kw}


def test_idle_sends_nothing():
    m = state("idle")
    assert dji_mission.tick(m, CENTRED, False, 100.0, ARGS) is None and m["phase"] == "idle"


def test_arming_waits_for_virtual_stick_then_centres():
    m = state("arming", since=100.0)
    assert dji_mission.tick(m, CENTRED, False, 101.0, ARGS) is None and m["phase"] == "arming"
    assert dji_mission.tick(m, CENTRED, True, 101.5, ARGS) is None and m["phase"] == "centre"
    assert m["last_seen"] == 101.5


def test_arming_gives_up_after_3_s():
    m = state("arming", since=100.0)
    dji_mission.tick(m, CENTRED, False, 103.5, ARGS)
    assert m["phase"] == "idle"


def test_centre_steers_toward_the_tag_without_descending():
    m = state("centre")
    msg = dji_mission.tick(m, (0.4, 0.0, 2.0), True, 100.0, ARGS)
    assert msg == pytest.approx({"fwd": 0.0, "right": 0.1, "down": 0.0})  # 0.5 * 0.4 clamped to vmax
    assert m["phase"] == "centre" and m["settled"] is None


def test_centre_moves_to_descend_after_1_s_settled():
    m = state("centre")
    dji_mission.tick(m, CENTRED, True, 100.0, ARGS)
    assert m["phase"] == "centre"
    dji_mission.tick(m, CENTRED, True, 101.1, ARGS)
    assert m["phase"] == "descend"


def test_descend_goes_down_only_while_centred():
    m = state("descend")
    assert dji_mission.tick(m, CENTRED, True, 100.0, ARGS)["down"] == 0.1
    assert dji_mission.tick(m, (0.4, 0.0, 2.0), True, 100.1, ARGS)["down"] == 0.0


def test_descend_hands_off_to_land_below_land_alt():
    m = state("descend")
    assert dji_mission.tick(m, (0.0, 0.0, 0.7), True, 100.0, ARGS) == {"cmd": "land"}
    assert m["phase"] == "landing"


def test_tag_lost_hovers_then_gives_control_back():
    m = state("descend")
    assert dji_mission.tick(m, None, True, 102.0, ARGS) == {"fwd": 0.0, "right": 0.0, "down": 0.0}
    assert m["phase"] == "descend"
    assert dji_mission.tick(m, None, True, 105.1, ARGS) == {"cmd": "disable"}
    assert m["phase"] == "hold"


def test_pilot_taking_over_stops_the_mission():
    m = state("centre")
    assert dji_mission.tick(m, CENTRED, False, 100.0, ARGS) is None and m["phase"] == "hold"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_dji_mission.py -v`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'dji_mission'`

- [ ] **Step 3: Write `dji_mission.py`**

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_dji_mission.py -v`
Expected: 9 passed

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest -q`
Expected: 48 passed (39 + 9)

- [ ] **Step 6: Commit**

```bash
git add dji_mission.py tests/test_dji_mission.py
git commit -m "feat: dji_mission, Mini 3 precision landing driven from the laptop"
```

---

### Task 5: Hardware checks, first landings, README

**Files:**
- Modify: `README.md` (append a "DJI Mini 3" section)
- Possibly modify: `android/Relay.kt` (two lines, Step 2), `dji_mission.py` defaults (Step 5)

Each step is a separate flight or bench session. Stop at the first failure and fix it before going on.

- [ ] **Step 1: Bench check of detection and camera direction (drone on the ground)**

Stand the drone on a box so the camera looks down at the tag from 0.5 m or more, gimbal dial fully down.

```bash
python dji_mission.py PHONE_IP --dry
```

Expected: `tag x=... y=... z=...` with z close to the measured height. Slide the tag toward the drone's nose: the red circle must move up the image and y must go negative. Slide it to the drone's right: x must go positive. If the image is rotated, set `--cam-yaw` to 90, 180 or 270 until both hold, and use that value in every later step.

- [ ] **Step 2: Axis check in flight**

Pilot hovers at about 2 m in an open area, thumbs on the sticks.

```bash
python dji_link.py PHONE_IP --nudge fwd
```

Expected: the drone moves about 0.2 m toward its nose. Repeat with `--nudge right` (moves right) and `--nudge down` (descends). If `fwd` and `right` are swapped, swap the `roll = v[0]` and `pitch = v[1]` lines in `Relay.kt`; if one direction is reversed, negate that line. Rebuild and repeat until all three are right.

- [ ] **Step 3: Hand-off test, no steering**

Pilot hovers at about 2 m, directly over the tag.

```bash
python dji_mission.py PHONE_IP --kp 0
```

Press `s`. Expected sequence in the terminal: `idle -> arming`, `arming -> centre`, `centre -> descend` after one second centred, a slow descent, `descend -> landing` below 0.8 m, DJI auto-land, then `landing -> idle`. Also check each abort once: `h` mid-descent (drone hovers, RC has control), the RC pause button (phase goes to `hold`), and closing the laptop lid (drone hovers).

If DJI auto-land stops and waits low to the ground, the Mini 3 wants a landing confirmation. Pull the throttle stick down to finish this flight, then add confirmation to `Relay.kt` using `FlightControllerKey.KeyIsLandingConfirmationNeeded` and `KeyConfirmLanding`.

- [ ] **Step 4: First steered landing**

```bash
python dji_mission.py PHONE_IP
```

Start from a hover 0.3 to 0.5 m off-centre. Expected: the drone slides over the tag, settles, descends and lands. Measure from the tag centre to the point under the camera.

- [ ] **Step 5: Tune and record**

Repeat Step 4 five times. If the drone oscillates over the tag, halve `--kp`; if it is sluggish, raise it by 0.1. If the tag drops out of frame during descent, lower `--tol`. Record the five landing errors and the final `--kp`, `--cam-yaw`, `--tol` and `--land-alt`, and make those the defaults in `dji_mission.py`.

- [ ] **Step 6: Document in `README.md`**

Append a "DJI Mini 3 (laptop processing)" section containing: the hardware (Mini 3, RC-N1 RC231, Android phone, laptop on the phone hotspot); the one-line sample-app change from Task 2 Step 2; the calibration command from Task 3 Step 3; the flight procedure (manual takeoff, hover 2 m over the tag, gimbal down, `python dji_mission.py PHONE_IP`, press `s`); the keys; the three safety behaviours (watchdog hover, link-closed release, tag-lost release); and the measured results from Step 5.

- [ ] **Step 7: Run the suite and commit**

Run: `python -m pytest -q`
Expected: 48 passed

```bash
git add README.md dji_mission.py android/Relay.kt
git commit -m "docs: Mini 3 laptop-processing setup, checks and measured landing error"
```

---

## Known limits of this plan

- `Relay.kt` has not been compiled. Its DJI names come from the API reference and the sample; Task 2 Step 3 lists where to check each one.
- The meaning of `roll` and `pitch` in body velocity mode is not stated in the pages that could be read. Task 5 Step 2 settles it in flight before any closed-loop control.
- There are no tag corrections during DJI's own landing from 0.8 m. If the measured error is too large, the next step is steering during the landing or lowering `--land-alt`.
- No Tk control window as on the DEXI 3; the OpenCV window and four keys replace it.
- The gimbal is pointed down by hand, not by the script.
