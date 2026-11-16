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
