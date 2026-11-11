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


from mission import heading_delta


def test_heading_delta_wraps():
    assert heading_delta(350, 10) == 20
    assert heading_delta(10, 350) == 340
    assert heading_delta(90, 90) == 0


# --- centre_and_descend / link_lost, driven by fakes and a controllable clock ---
import asyncio
import sys
import time
from collections import namedtuple
from types import SimpleNamespace

import pytest

import mission

Vel = namedtuple("Vel", "fwd right down yaw")


def rig(monkeypatch, off=(0.0, 0.0, 3.0), mode="OFFBOARD", rel_alt=5.0, frame_age=0.0):
    """Fake drone/frames/telemetry; asyncio.sleep advances the fake clock and keeps telemetry fresh."""
    clk = [1000.0]
    t = SimpleNamespace(mode=SimpleNamespace(name=mode), rel_alt=rel_alt, tasks=[], att_stamp=clk[0])
    sent = []

    async def set_velocity_body(v):
        sent.append(v)

    async def sleep(dt):
        clk[0] += dt
        t.att_stamp = clk[0]

    monkeypatch.setitem(sys.modules, "mavsdk.offboard", SimpleNamespace(VelocityBodyYawspeed=Vel))
    monkeypatch.setattr(mission, "time", SimpleNamespace(monotonic=lambda: clk[0]))
    monkeypatch.setattr(mission.asyncio, "sleep", sleep)
    monkeypatch.setattr(mission, "marker_offset", lambda *a: off)
    args = SimpleNamespace(kp=0.5, vmax=0.3, cam_yaw=0.0, marker_id=0, marker_size=0.35)
    drone = SimpleNamespace(offboard=SimpleNamespace(set_velocity_body=set_velocity_body))
    frames = SimpleNamespace(get=lambda: (object(), clk[0] - frame_age))
    return args, drone, frames, t, sent


def run(monkeypatch, descend=False, **kw):
    args, drone, frames, t, sent = rig(monkeypatch, **kw)
    asyncio.run(mission.centre_and_descend(args, drone, frames, None, None, t, descend))
    return sent


def test_hold_settles_and_returns(monkeypatch):
    assert run(monkeypatch, off=(0.05, 0.05, 3.0))


def test_marker_not_seen_sends_zero_then_raises(monkeypatch):
    args, drone, frames, t, sent = rig(monkeypatch, off=None)
    with pytest.raises(RuntimeError, match="marker lost"):
        asyncio.run(mission.centre_and_descend(args, drone, frames, None, None, t, False))
    assert sent and all(v == Vel(0.0, 0.0, 0.0, 0.0) for v in sent)


def test_stale_frame_counts_as_not_seen(monkeypatch):
    with pytest.raises(RuntimeError, match="marker lost"):
        run(monkeypatch, off=(0.05, 0.05, 3.0), frame_age=1.5)


def test_posctl_raises_pilot_took_over(monkeypatch):
    with pytest.raises(mission.PilotTookOver):
        run(monkeypatch, mode="POSCTL")


def test_descend_returns_below_1m(monkeypatch):
    sent = run(monkeypatch, descend=True, rel_alt=0.9)
    assert sent[0].down == 0.3


def test_hold_times_out_when_never_settled(monkeypatch):
    with pytest.raises(RuntimeError, match="centre timed out"):
        run(monkeypatch, off=(0.5, 0.0, 3.0))


def test_link_lost():
    now = time.monotonic()
    fresh = SimpleNamespace(tasks=[SimpleNamespace(done=lambda: False)], att_stamp=now)
    assert not mission.link_lost(fresh)
    assert mission.link_lost(SimpleNamespace(tasks=[SimpleNamespace(done=lambda: True)], att_stamp=now))
    assert mission.link_lost(SimpleNamespace(tasks=[], att_stamp=now - 3))
    assert mission.link_lost(SimpleNamespace(tasks=[], att_stamp=None))


# --- abort_action ---------------------------------------------------------

def fake_drone(land_fails=0):
    calls = []

    async def rec(name):
        calls.append(name)

    async def land():
        calls.append("land")
        if calls.count("land") <= land_fails:
            raise RuntimeError("land refused")

    drone = SimpleNamespace(
        action=SimpleNamespace(kill=lambda: rec("kill"), land=land),
        offboard=SimpleNamespace(stop=lambda: rec("stop")))
    return drone, calls


def test_abort_kill_cuts_motors_only():
    drone, calls = fake_drone()
    assert asyncio.run(mission.abort_action(drone, "kill")) is True
    assert calls == ["kill"]


def test_abort_hold_stops_offboard_only():
    drone, calls = fake_drone()
    assert asyncio.run(mission.abort_action(drone, "hold")) is True
    assert calls == ["stop"]


def test_abort_land_stops_then_lands():
    drone, calls = fake_drone()
    assert asyncio.run(mission.abort_action(drone, "land")) is True
    assert calls == ["stop", "land"]


def test_abort_land_failing_twice_logs_hold(monkeypatch):
    logged = []
    monkeypatch.setattr(mission, "log", logged.append)

    async def no_sleep(dt):
        pass
    monkeypatch.setattr(mission.asyncio, "sleep", no_sleep)
    drone, calls = fake_drone(land_fails=2)
    assert asyncio.run(mission.abort_action(drone, "land")) is False
    assert calls == ["stop", "land", "land"]
    assert logged == ["VEHICLE IN HOLD - PILOT TAKE OVER"]
