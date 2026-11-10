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
